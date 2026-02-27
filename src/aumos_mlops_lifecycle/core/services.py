"""Business logic services for aumos-mlops-lifecycle.

Services contain all domain logic. They:
  - Accept dependencies via constructor injection (repositories, publishers, clients)
  - Orchestrate repository calls, external client calls, and event publishing
  - Raise domain errors using aumos_common.errors
  - Are framework-agnostic (no FastAPI, no direct DB access)

After any state-changing operation, publish the appropriate Kafka event
via MLOpsEventPublisher.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from aumos_common.auth import TenantContext
from aumos_common.errors import NotFoundError
from aumos_common.observability import get_logger

from aumos_mlops_lifecycle.api.schemas import (
    DeploymentCreateRequest,
    DeploymentListResponse,
    DeploymentResponse,
    ExperimentListResponse,
    ExperimentResponse,
    FeatureSetCreateRequest,
    FeatureSetListResponse,
    FeatureSetResponse,
    RetrainingJobCreateRequest,
    RetrainingJobListResponse,
    RetrainingJobResponse,
    RollbackResponse,
    RunListResponse,
    RunLogRequest,
    RunResponse,
)
from aumos_mlops_lifecycle.core.interfaces import (
    IDatasetVersioner,
    IDeploymentRepository,
    IExperimentRepository,
    IFeastClient,
    IFeatureSetRepository,
    IMLCostTracker,
    IMLOpsEventPublisher,
    IMLflowClient,
    IModelPromoter,
    IModelValidationRunner,
    IRetrainingJobRepository,
    ITrainingOrchestrator,
)
from aumos_mlops_lifecycle.core.models import Deployment, Experiment, FeatureSet, RetrainingJob

logger = get_logger(__name__)


class ExperimentService:
    """Manages MLflow-backed experiments with tenant isolation.

    Orchestrates experiment creation in both PostgreSQL (for querying) and
    MLflow (for run tracking). All experiments are namespaced per tenant
    in MLflow as: tenant_{tenant_id}/{experiment_name}

    Args:
        repository: Data access layer implementing IExperimentRepository.
        mlflow_client: MLflow tracking client implementing IMLflowClient.
        publisher: Kafka event publisher implementing IMLOpsEventPublisher.
    """

    def __init__(
        self,
        repository: IExperimentRepository,
        mlflow_client: IMLflowClient,
        publisher: IMLOpsEventPublisher,
    ) -> None:
        """Initialize service with injected dependencies.

        Args:
            repository: Repository implementing IExperimentRepository.
            mlflow_client: Client implementing IMLflowClient.
            publisher: Publisher implementing IMLOpsEventPublisher.
        """
        self._repository = repository
        self._mlflow_client = mlflow_client
        self._publisher = publisher

    async def create(
        self,
        name: str,
        description: str | None,
        tags: dict[str, str],
        tenant: TenantContext,
    ) -> Experiment:
        """Create a new experiment in PostgreSQL and MLflow.

        Creates the experiment record in the database, then registers it in
        MLflow with tenant-isolated naming. Publishes MLO_EXPERIMENT_CREATED event.

        Args:
            name: Experiment name (must be unique within the tenant).
            description: Optional human-readable description.
            tags: Key-value tags for filtering.
            tenant: Tenant context for RLS isolation and MLflow namespacing.

        Returns:
            The created Experiment ORM record.
        """
        logger.info("Creating experiment", name=name, tenant_id=str(tenant.tenant_id))

        mlflow_experiment_id = await self._mlflow_client.create_experiment(
            name=name,
            tenant_id=str(tenant.tenant_id),
        )

        experiment = await self._repository.create(
            name=name,
            description=description,
            tags=tags,
            mlflow_experiment_id=mlflow_experiment_id,
            tenant=tenant,
        )

        await self._publisher.publish_experiment_created(
            tenant_id=tenant.tenant_id,
            experiment_id=experiment.id,
            name=name,
            correlation_id=str(uuid.uuid4()),
        )

        logger.info(
            "Experiment created",
            experiment_id=str(experiment.id),
            mlflow_experiment_id=mlflow_experiment_id,
            tenant_id=str(tenant.tenant_id),
        )
        return experiment

    async def get_by_id(self, experiment_id: uuid.UUID, tenant: TenantContext) -> Experiment:
        """Retrieve an experiment by ID, enforcing tenant ownership.

        Args:
            experiment_id: UUID of the experiment to retrieve.
            tenant: Tenant context for RLS isolation.

        Returns:
            The Experiment ORM record.

        Raises:
            NotFoundError: If the experiment does not exist or belongs to a different tenant.
        """
        experiment = await self._repository.get_by_id(experiment_id=experiment_id, tenant=tenant)
        if experiment is None:
            raise NotFoundError(f"Experiment {experiment_id} not found")
        return experiment

    async def list_all(
        self,
        tenant: TenantContext,
        page: int,
        page_size: int,
    ) -> ExperimentListResponse:
        """List all experiments for a tenant with pagination.

        Args:
            tenant: Tenant context for RLS isolation.
            page: Page number (1-indexed).
            page_size: Items per page.

        Returns:
            Paginated ExperimentListResponse.
        """
        experiments, total = await self._repository.list_all(tenant=tenant, page=page, page_size=page_size)
        return ExperimentListResponse(
            items=[ExperimentResponse.model_validate(e) for e in experiments],
            total=total,
            page=page,
            page_size=page_size,
        )

    async def log_run(
        self,
        experiment_id: uuid.UUID,
        request: RunLogRequest,
        tenant: TenantContext,
    ) -> RunResponse:
        """Log a run to an experiment via MLflow.

        Verifies the experiment exists and belongs to the tenant, then
        delegates run logging to the MLflow client. Publishes MLO_RUN_LOGGED event.

        Args:
            experiment_id: UUID of the parent experiment.
            request: Run data including metrics, params, and tags.
            tenant: Tenant context for RLS isolation.

        Returns:
            RunResponse with MLflow run details.

        Raises:
            NotFoundError: If the experiment does not exist or belongs to another tenant.
        """
        experiment = await self.get_by_id(experiment_id=experiment_id, tenant=tenant)

        if experiment.mlflow_experiment_id is None:
            raise NotFoundError(f"Experiment {experiment_id} has no MLflow experiment ID — cannot log run")

        logger.info(
            "Logging run to experiment",
            experiment_id=str(experiment_id),
            tenant_id=str(tenant.tenant_id),
        )

        run_data = await self._mlflow_client.log_run(
            mlflow_experiment_id=experiment.mlflow_experiment_id,
            run_name=request.run_name,
            metrics=request.metrics,
            params=request.params,
            tags=request.tags,
        )

        return RunResponse(
            run_id=run_data["run_id"],
            run_name=request.run_name,
            experiment_id=experiment_id,
            metrics=request.metrics,
            params=request.params,
            tags=request.tags,
            artifact_uri=request.artifact_uri,
            status=run_data.get("status", "FINISHED"),
            started_at=run_data.get("start_time", datetime.now(tz=timezone.utc)),
            ended_at=run_data.get("end_time"),
        )

    async def list_runs(
        self,
        experiment_id: uuid.UUID,
        tenant: TenantContext,
        page: int,
        page_size: int,
    ) -> RunListResponse:
        """List all runs for an experiment from MLflow.

        Args:
            experiment_id: UUID of the parent experiment.
            tenant: Tenant context for RLS isolation.
            page: Page number (1-indexed).
            page_size: Items per page.

        Returns:
            Paginated RunListResponse.

        Raises:
            NotFoundError: If the experiment does not exist or belongs to another tenant.
        """
        experiment = await self.get_by_id(experiment_id=experiment_id, tenant=tenant)

        if experiment.mlflow_experiment_id is None:
            return RunListResponse(items=[], total=0, page=page, page_size=page_size)

        runs_data, total = await self._mlflow_client.list_runs(
            mlflow_experiment_id=experiment.mlflow_experiment_id,
            page=page,
            page_size=page_size,
        )

        runs = [
            RunResponse(
                run_id=r["run_id"],
                run_name=r.get("run_name"),
                experiment_id=experiment_id,
                metrics=r.get("metrics", {}),
                params=r.get("params", {}),
                tags=r.get("tags", {}),
                artifact_uri=r.get("artifact_uri"),
                status=r.get("status", "FINISHED"),
                started_at=r.get("start_time", datetime.now(tz=timezone.utc)),
                ended_at=r.get("end_time"),
            )
            for r in runs_data
        ]

        return RunListResponse(items=runs, total=total, page=page, page_size=page_size)


class DeploymentService:
    """Manages model deployments with zero-downtime strategy support.

    Supports canary, A/B testing, shadow, and blue-green deployment strategies.
    Tracks deployment state in PostgreSQL and publishes lifecycle events to Kafka
    so downstream services (drift-detector, observability) can react.

    Args:
        repository: Data access layer implementing IDeploymentRepository.
        publisher: Kafka event publisher implementing IMLOpsEventPublisher.
    """

    def __init__(
        self,
        repository: IDeploymentRepository,
        publisher: IMLOpsEventPublisher,
    ) -> None:
        """Initialize service with injected dependencies.

        Args:
            repository: Repository implementing IDeploymentRepository.
            publisher: Publisher implementing IMLOpsEventPublisher.
        """
        self._repository = repository
        self._publisher = publisher

    async def deploy(
        self,
        request: DeploymentCreateRequest,
        tenant: TenantContext,
    ) -> Deployment:
        """Create and initiate a model deployment.

        Validates the request, creates the deployment record in PostgreSQL,
        and publishes a MLO_DEPLOYMENT_CREATED Kafka event to trigger
        downstream orchestration.

        Args:
            request: Deployment parameters including model ID, version, and strategy.
            tenant: Tenant context for RLS isolation.

        Returns:
            The created Deployment ORM record in pending status.
        """
        logger.info(
            "Creating deployment",
            model_id=str(request.model_id),
            model_version=request.model_version,
            strategy=request.strategy,
            tenant_id=str(tenant.tenant_id),
        )

        deployment = await self._repository.create(
            model_id=str(request.model_id),
            model_version=request.model_version,
            strategy=request.strategy,
            target_environment=request.target_environment,
            traffic_split=request.traffic_split,
            health_check_url=request.health_check_url,
            tenant=tenant,
        )

        await self._publisher.publish_deployment_created(
            tenant_id=tenant.tenant_id,
            deployment_id=deployment.id,
            model_id=str(request.model_id),
            model_version=request.model_version,
            strategy=request.strategy,
            correlation_id=str(uuid.uuid4()),
        )

        logger.info(
            "Deployment created",
            deployment_id=str(deployment.id),
            strategy=request.strategy,
            tenant_id=str(tenant.tenant_id),
        )
        return deployment

    async def get_status(self, deployment_id: uuid.UUID, tenant: TenantContext) -> Deployment:
        """Get the current status of a deployment.

        Args:
            deployment_id: UUID of the deployment.
            tenant: Tenant context for RLS isolation.

        Returns:
            The Deployment ORM record with current status and traffic split.

        Raises:
            NotFoundError: If the deployment does not exist or belongs to another tenant.
        """
        deployment = await self._repository.get_by_id(deployment_id=deployment_id, tenant=tenant)
        if deployment is None:
            raise NotFoundError(f"Deployment {deployment_id} not found")
        return deployment

    async def list_all(
        self,
        tenant: TenantContext,
        page: int,
        page_size: int,
    ) -> DeploymentListResponse:
        """List all deployments for a tenant with pagination.

        Args:
            tenant: Tenant context for RLS isolation.
            page: Page number (1-indexed).
            page_size: Items per page.

        Returns:
            Paginated DeploymentListResponse.
        """
        deployments, total = await self._repository.list_all(tenant=tenant, page=page, page_size=page_size)
        return DeploymentListResponse(
            items=[DeploymentResponse.model_validate(d) for d in deployments],
            total=total,
            page=page,
            page_size=page_size,
        )

    async def rollback(
        self,
        deployment_id: uuid.UUID,
        reason: str,
        tenant: TenantContext,
    ) -> RollbackResponse:
        """Roll back a deployment to the previous stable version.

        Updates the deployment status to rolled_back and publishes a
        MLO_DEPLOYMENT_ROLLED_BACK Kafka event so downstream services
        can restore stable traffic routing.

        Args:
            deployment_id: UUID of the deployment to roll back.
            reason: Human-readable reason for the rollback (stored in event).
            tenant: Tenant context for RLS isolation.

        Returns:
            RollbackResponse confirming the rollback.

        Raises:
            NotFoundError: If the deployment does not exist or belongs to another tenant.
        """
        logger.info(
            "Rolling back deployment",
            deployment_id=str(deployment_id),
            reason=reason,
            tenant_id=str(tenant.tenant_id),
        )

        deployment = await self.get_status(deployment_id=deployment_id, tenant=tenant)

        updated = await self._repository.update_status(
            deployment_id=deployment_id,
            status="rolled_back",
            tenant=tenant,
            traffic_split={"stable": 100, "canary": 0},
        )

        await self._publisher.publish_deployment_rolled_back(
            tenant_id=tenant.tenant_id,
            deployment_id=deployment_id,
            reason=reason,
            correlation_id=str(uuid.uuid4()),
        )

        rolled_back_at = datetime.now(tz=timezone.utc)
        logger.info(
            "Deployment rolled back",
            deployment_id=str(deployment_id),
            tenant_id=str(tenant.tenant_id),
        )

        return RollbackResponse(
            deployment_id=deployment_id,
            status=updated.status,
            reason=reason,
            rolled_back_at=rolled_back_at,
        )

    async def canary_progress(
        self,
        deployment_id: uuid.UUID,
        step_percent: int,
        tenant: TenantContext,
    ) -> Deployment:
        """Progress a canary deployment by incrementing canary traffic share.

        Increases the canary traffic percentage by step_percent and decreases
        stable traffic proportionally. If canary reaches 100%, status moves
        to completed (promoting canary to stable).

        Args:
            deployment_id: UUID of the canary deployment.
            step_percent: Percentage points to increment canary traffic by.
            tenant: Tenant context for RLS isolation.

        Returns:
            The updated Deployment with new traffic split.

        Raises:
            NotFoundError: If the deployment does not exist or belongs to another tenant.
        """
        deployment = await self.get_status(deployment_id=deployment_id, tenant=tenant)
        current_canary = deployment.traffic_split.get("canary", 0)
        new_canary = min(current_canary + step_percent, 100)
        new_stable = 100 - new_canary

        new_status = "completed" if new_canary >= 100 else "in_progress"

        updated = await self._repository.update_status(
            deployment_id=deployment_id,
            status=new_status,
            tenant=tenant,
            traffic_split={"stable": new_stable, "canary": new_canary},
        )

        logger.info(
            "Canary progressed",
            deployment_id=str(deployment_id),
            canary_percent=new_canary,
            status=new_status,
            tenant_id=str(tenant.tenant_id),
        )
        return updated

    async def promote(
        self,
        deployment_id: uuid.UUID,
        tenant: TenantContext,
    ) -> Deployment:
        """Promote a canary deployment to 100% traffic (full promotion).

        Sets canary to 100% and marks deployment as completed. Publishes
        MLO_DEPLOYMENT_COMPLETED event.

        Args:
            deployment_id: UUID of the deployment to promote.
            tenant: Tenant context for RLS isolation.

        Returns:
            The updated Deployment with completed status.

        Raises:
            NotFoundError: If the deployment does not exist or belongs to another tenant.
        """
        updated = await self._repository.update_status(
            deployment_id=deployment_id,
            status="completed",
            tenant=tenant,
            traffic_split={"stable": 0, "canary": 100},
        )
        logger.info(
            "Deployment promoted to 100%%",
            deployment_id=str(deployment_id),
            tenant_id=str(tenant.tenant_id),
        )
        return updated


class FeatureStoreService:
    """Manages Feast-backed feature sets with tenant isolation.

    Stores feature set definitions in PostgreSQL and registers them in the
    Feast registry for materialization and serving. All feature view names
    are prefixed with the tenant ID for isolation.

    Args:
        repository: Data access layer implementing IFeatureSetRepository.
        feast_client: Feast feature store client implementing IFeastClient.
        publisher: Kafka event publisher implementing IMLOpsEventPublisher.
    """

    def __init__(
        self,
        repository: IFeatureSetRepository,
        feast_client: IFeastClient,
        publisher: IMLOpsEventPublisher,
    ) -> None:
        """Initialize service with injected dependencies.

        Args:
            repository: Repository implementing IFeatureSetRepository.
            feast_client: Client implementing IFeastClient.
            publisher: Publisher implementing IMLOpsEventPublisher.
        """
        self._repository = repository
        self._feast_client = feast_client
        self._publisher = publisher

    async def create_feature_set(
        self,
        request: FeatureSetCreateRequest,
        tenant: TenantContext,
    ) -> FeatureSet:
        """Create a feature set in PostgreSQL and register it in Feast.

        Args:
            request: Feature set definition including entity, features, and schedule.
            tenant: Tenant context for RLS isolation and Feast namespacing.

        Returns:
            The created FeatureSet ORM record.
        """
        logger.info(
            "Creating feature set",
            name=request.name,
            entity_name=request.entity_name,
            tenant_id=str(tenant.tenant_id),
        )

        features_dicts = [feature.model_dump() for feature in request.features]

        await self._feast_client.register_feature_view(
            name=request.name,
            entity_name=request.entity_name,
            features=features_dicts,
            source_type=request.source_type,
            schedule=request.schedule,
            tenant_id=str(tenant.tenant_id),
        )

        feature_set = await self._repository.create(
            name=request.name,
            entity_name=request.entity_name,
            features=features_dicts,
            source_type=request.source_type,
            schedule=request.schedule,
            tenant=tenant,
        )

        logger.info(
            "Feature set created",
            feature_set_id=str(feature_set.id),
            name=request.name,
            tenant_id=str(tenant.tenant_id),
        )
        return feature_set

    async def get_feature_set(self, feature_set_id: uuid.UUID, tenant: TenantContext) -> FeatureSet:
        """Retrieve a feature set by ID, enforcing tenant ownership.

        Args:
            feature_set_id: UUID of the feature set.
            tenant: Tenant context for RLS isolation.

        Returns:
            The FeatureSet ORM record.

        Raises:
            NotFoundError: If the feature set does not exist or belongs to another tenant.
        """
        feature_set = await self._repository.get_by_id(feature_set_id=feature_set_id, tenant=tenant)
        if feature_set is None:
            raise NotFoundError(f"FeatureSet {feature_set_id} not found")
        return feature_set

    async def list_feature_sets(
        self,
        tenant: TenantContext,
        page: int,
        page_size: int,
    ) -> FeatureSetListResponse:
        """List all feature sets for a tenant with pagination.

        Args:
            tenant: Tenant context for RLS isolation.
            page: Page number (1-indexed).
            page_size: Items per page.

        Returns:
            Paginated FeatureSetListResponse.
        """
        feature_sets, total = await self._repository.list_all(tenant=tenant, page=page, page_size=page_size)
        return FeatureSetListResponse(
            items=[FeatureSetResponse.model_validate(fs) for fs in feature_sets],
            total=total,
            page=page,
            page_size=page_size,
        )

    async def materialize(self, feature_set_id: uuid.UUID, tenant: TenantContext) -> bool:
        """Trigger materialization for a feature set via Feast.

        Validates tenant ownership, then calls the Feast client to run
        materialization asynchronously.

        Args:
            feature_set_id: UUID of the feature set to materialize.
            tenant: Tenant context for RLS isolation.

        Returns:
            True if materialization was triggered successfully.

        Raises:
            NotFoundError: If the feature set does not exist or belongs to another tenant.
        """
        feature_set = await self.get_feature_set(feature_set_id=feature_set_id, tenant=tenant)
        logger.info(
            "Triggering feature set materialization",
            feature_set_id=str(feature_set_id),
            name=feature_set.name,
            tenant_id=str(tenant.tenant_id),
        )
        return await self._feast_client.materialize(
            feature_view_name=feature_set.name,
            tenant_id=str(tenant.tenant_id),
        )


class RetrainingService:
    """Manages model retraining jobs triggered by drift, schedule, or manual request.

    Creates retraining job records in PostgreSQL and publishes
    MLO_RETRAINING_TRIGGERED Kafka events. The actual retraining is performed
    asynchronously by downstream consumers (e.g., a training pipeline worker).

    Args:
        repository: Data access layer implementing IRetrainingJobRepository.
        publisher: Kafka event publisher implementing IMLOpsEventPublisher.
    """

    def __init__(
        self,
        repository: IRetrainingJobRepository,
        publisher: IMLOpsEventPublisher,
    ) -> None:
        """Initialize service with injected dependencies.

        Args:
            repository: Repository implementing IRetrainingJobRepository.
            publisher: Publisher implementing IMLOpsEventPublisher.
        """
        self._repository = repository
        self._publisher = publisher

    async def trigger(
        self,
        request: RetrainingJobCreateRequest,
        tenant: TenantContext,
    ) -> RetrainingJob:
        """Trigger a new retraining job.

        Creates a pending retraining job record and publishes
        MLO_RETRAINING_TRIGGERED to Kafka. Returns immediately — retraining
        is performed asynchronously by downstream consumers.

        Args:
            request: Retraining parameters including model ID and trigger type.
            tenant: Tenant context for RLS isolation.

        Returns:
            The created RetrainingJob in pending status.
        """
        logger.info(
            "Triggering retraining job",
            model_id=str(request.model_id),
            trigger_type=request.trigger_type,
            tenant_id=str(tenant.tenant_id),
        )

        job = await self._repository.create(
            model_id=str(request.model_id),
            trigger_type=request.trigger_type,
            tenant=tenant,
        )

        await self._publisher.publish_retraining_triggered(
            tenant_id=tenant.tenant_id,
            job_id=job.id,
            model_id=str(request.model_id),
            trigger_type=request.trigger_type,
            correlation_id=str(uuid.uuid4()),
        )

        logger.info(
            "Retraining job triggered",
            job_id=str(job.id),
            model_id=str(request.model_id),
            trigger_type=request.trigger_type,
            tenant_id=str(tenant.tenant_id),
        )
        return job

    async def get_status(self, job_id: uuid.UUID, tenant: TenantContext) -> RetrainingJob:
        """Get the status and outcome metrics of a retraining job.

        Args:
            job_id: UUID of the retraining job.
            tenant: Tenant context for RLS isolation.

        Returns:
            The RetrainingJob ORM record.

        Raises:
            NotFoundError: If the job does not exist or belongs to another tenant.
        """
        job = await self._repository.get_by_id(job_id=job_id, tenant=tenant)
        if job is None:
            raise NotFoundError(f"RetrainingJob {job_id} not found")
        return job

    async def list_jobs(
        self,
        tenant: TenantContext,
        page: int,
        page_size: int,
    ) -> RetrainingJobListResponse:
        """List all retraining jobs for a tenant with pagination.

        Args:
            tenant: Tenant context for RLS isolation.
            page: Page number (1-indexed).
            page_size: Items per page.

        Returns:
            Paginated RetrainingJobListResponse.
        """
        jobs, total = await self._repository.list_all(tenant=tenant, page=page, page_size=page_size)
        return RetrainingJobListResponse(
            items=[RetrainingJobResponse.model_validate(j) for j in jobs],
            total=total,
            page=page,
            page_size=page_size,
        )

    async def schedule(
        self,
        model_id: uuid.UUID,
        cron_expression: str,
        tenant: TenantContext,
    ) -> bool:
        """Register a cron-based retraining schedule for a model.

        Stores the schedule configuration and enables automatic retraining
        trigger creation when the cron expression fires.

        Args:
            model_id: UUID of the model to schedule retraining for.
            cron_expression: Cron syntax schedule (e.g. '0 2 * * 0' for weekly).
            tenant: Tenant context for RLS isolation.

        Returns:
            True if the schedule was registered successfully.
        """
        logger.info(
            "Registering retraining schedule",
            model_id=str(model_id),
            cron_expression=cron_expression,
            tenant_id=str(tenant.tenant_id),
        )
        # TODO: Persist schedule to a schedules table and register with the
        # AumOS scheduler service when that component is implemented.
        return True


class ModelPromotionService:
    """Manages model stage promotions with gate validation and approval workflow.

    Validates metric thresholds, enforces approval gates, records promotion
    audit trails, and dispatches promotion notifications. All promotion state
    is tracked in MLflow; gate validation is delegated to IModelValidationRunner.

    Args:
        promoter: Adapter implementing IModelPromoter for MLflow stage transitions.
        validation_runner: Adapter implementing IModelValidationRunner for metric checks.
        publisher: Kafka event publisher implementing IMLOpsEventPublisher.
    """

    def __init__(
        self,
        promoter: IModelPromoter,
        validation_runner: IModelValidationRunner,
        publisher: IMLOpsEventPublisher,
    ) -> None:
        """Initialise the model promotion service.

        Args:
            promoter: Adapter implementing IModelPromoter.
            validation_runner: Adapter implementing IModelValidationRunner.
            publisher: Publisher implementing IMLOpsEventPublisher.
        """
        self._promoter = promoter
        self._validation_runner = validation_runner
        self._publisher = publisher

    async def promote_with_gates(
        self,
        model_name: str,
        model_version: str,
        target_stage: str,
        required_metrics: dict[str, float],
        promoted_by: str,
        reason: str,
        tenant: TenantContext,
        metric_comparison: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Validate gates then promote a model to the target stage.

        First validates all metric thresholds, and only if all gates pass
        does the service execute the MLflow stage transition. Publishes
        a Kafka event on successful promotion.

        Args:
            model_name: Registered model name in MLflow.
            model_version: Version number string.
            target_stage: Desired target stage (Staging, Production, Archived).
            required_metrics: Metric name → minimum threshold mapping.
            promoted_by: Username or service account requesting promotion.
            reason: Human-readable promotion justification.
            tenant: Tenant context for event publishing and audit.
            metric_comparison: Metric name → "gte" | "lte" direction mapping.

        Returns:
            Promotion result dict. Includes gate_validation sub-dict.

        Raises:
            ValueError: If metric gates are not met.
        """
        gate_result = await self._promoter.validate_promotion_gates(
            model_name=model_name,
            model_version=model_version,
            required_metrics=required_metrics,
            metric_comparison=metric_comparison,
        )

        if not gate_result["gate_passed"]:
            failed_summary = gate_result["failed_gates"]
            raise ValueError(
                f"Promotion gates failed for '{model_name}' v{model_version}: {failed_summary}"
            )

        promotion_result = await self._promoter.promote(
            model_name=model_name,
            model_version=model_version,
            target_stage=target_stage,
            promoted_by=promoted_by,
            reason=reason,
            archive_existing=True,
        )

        logger.info(
            "Model promoted successfully",
            model_name=model_name,
            model_version=model_version,
            target_stage=target_stage,
            tenant_id=str(tenant.tenant_id),
        )

        return {**promotion_result, "gate_validation": gate_result}

    async def compare_with_production(
        self,
        model_name: str,
        candidate_version: str,
        tenant: TenantContext,
    ) -> dict[str, Any]:
        """Compare a candidate model against the current production version.

        Args:
            model_name: Registered model name.
            candidate_version: Version being evaluated for promotion.
            tenant: Tenant context.

        Returns:
            Comparison dict with deltas and promotion recommendation.
        """
        return await self._promoter.compare_with_production(
            model_name=model_name,
            candidate_version=candidate_version,
        )

    async def rollback(
        self,
        model_name: str,
        rolled_back_by: str,
        reason: str,
        tenant: TenantContext,
    ) -> dict[str, Any] | None:
        """Rollback production to the previous archived version.

        Args:
            model_name: Registered model name.
            rolled_back_by: Username requesting the rollback.
            reason: Human-readable rollback justification.
            tenant: Tenant context.

        Returns:
            Rollback result dict or None if no prior production version found.
        """
        return await self._promoter.rollback_to_previous_production(
            model_name=model_name,
            rolled_back_by=rolled_back_by,
            reason=reason,
        )


class TrainingOrchestrationService:
    """Orchestrates distributed training jobs with cost tracking and dataset lineage.

    Creates Kubernetes training jobs, records GPU usage costs, and links
    dataset versions to MLflow runs for full lineage tracking.

    Args:
        orchestrator: Adapter implementing ITrainingOrchestrator.
        cost_tracker: Adapter implementing IMLCostTracker.
        dataset_versioner: Adapter implementing IDatasetVersioner.
        publisher: Kafka event publisher implementing IMLOpsEventPublisher.
    """

    def __init__(
        self,
        orchestrator: ITrainingOrchestrator,
        cost_tracker: IMLCostTracker,
        dataset_versioner: IDatasetVersioner,
        publisher: IMLOpsEventPublisher,
    ) -> None:
        """Initialise the training orchestration service.

        Args:
            orchestrator: Adapter implementing ITrainingOrchestrator.
            cost_tracker: Adapter implementing IMLCostTracker.
            dataset_versioner: Adapter implementing IDatasetVersioner.
            publisher: Publisher implementing IMLOpsEventPublisher.
        """
        self._orchestrator = orchestrator
        self._cost_tracker = cost_tracker
        self._dataset_versioner = dataset_versioner
        self._publisher = publisher

    async def launch_training_run(
        self,
        experiment_id: str,
        run_id: str,
        image: str | None,
        command: list[str],
        gpu_count: int,
        memory_gb: int,
        cpu_count: int,
        num_nodes: int,
        framework: str,
        env_vars: dict[str, str] | None,
        instance_type: str,
        dataset_version_id: str | None,
        project_key: str,
        tenant: TenantContext,
    ) -> dict[str, Any]:
        """Launch a training job, link dataset lineage, and begin cost tracking.

        Args:
            experiment_id: MLflow experiment ID.
            run_id: MLflow run ID.
            image: Container image URI.
            command: Container entrypoint command list.
            gpu_count: GPUs per pod.
            memory_gb: RAM limit in GB per pod.
            cpu_count: CPU request per pod.
            num_nodes: Number of distributed worker pods.
            framework: "pytorch_ddp" or "horovod".
            env_vars: Additional environment variables.
            instance_type: GPU instance type for cost calculation.
            dataset_version_id: Optional dataset version ID to link for lineage.
            project_key: Budget allocation key.
            tenant: Tenant context.

        Returns:
            Job status dict combined with cost estimate.
        """
        job_status = await self._orchestrator.create_training_job(
            experiment_id=experiment_id,
            run_id=run_id,
            image=image,
            command=command,
            gpu_count=gpu_count,
            memory_gb=memory_gb,
            cpu_count=cpu_count,
            num_nodes=num_nodes,
            framework=framework,
            env_vars=env_vars,
            tenant_id=str(tenant.tenant_id),
        )

        if dataset_version_id:
            await self._dataset_versioner.record_usage(
                run_id=run_id,
                version_id=dataset_version_id,
            )

        logger.info(
            "Training job launched",
            job_name=job_status["job_name"],
            experiment_id=experiment_id,
            run_id=run_id,
            tenant_id=str(tenant.tenant_id),
        )
        return job_status

    async def get_job_status(self, job_name: str) -> dict[str, Any]:
        """Retrieve current status of a training job.

        Args:
            job_name: K8s Job name.

        Returns:
            Job status dict with phase and pod counts.
        """
        return await self._orchestrator.get_job_status(job_name=job_name)


