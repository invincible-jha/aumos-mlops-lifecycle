"""Abstract interfaces (Protocol classes) for aumos-mlops-lifecycle.

Defining interfaces as Protocol classes enables:
  - Dependency injection in services
  - Easy mocking in tests
  - Clear contracts between layers

Services depend on interfaces, not concrete implementations.
All interface methods are async — I/O must never block.
"""

import uuid
from typing import Any, Protocol, runtime_checkable

from aumos_common.auth import TenantContext

from aumos_mlops_lifecycle.core.models import Deployment, Experiment, FeatureSet, RetrainingJob


@runtime_checkable
class IExperimentRepository(Protocol):
    """Repository interface for Experiment records."""

    async def get_by_id(self, experiment_id: uuid.UUID, tenant: TenantContext) -> Experiment | None:
        """Retrieve an experiment by ID within tenant scope.

        Args:
            experiment_id: UUID of the experiment.
            tenant: Tenant context for RLS isolation.

        Returns:
            The Experiment if found, None otherwise.
        """
        ...

    async def list_all(
        self, tenant: TenantContext, page: int, page_size: int
    ) -> tuple[list[Experiment], int]:
        """List experiments for a tenant with pagination.

        Args:
            tenant: Tenant context for RLS isolation.
            page: Page number (1-indexed).
            page_size: Items per page.

        Returns:
            Tuple of (experiments list, total count).
        """
        ...

    async def create(
        self,
        name: str,
        description: str | None,
        tags: dict[str, str],
        mlflow_experiment_id: str | None,
        tenant: TenantContext,
    ) -> Experiment:
        """Create a new experiment record.

        Args:
            name: Experiment name.
            description: Optional description.
            tags: Key-value tags.
            mlflow_experiment_id: Corresponding MLflow experiment ID.
            tenant: Tenant context for RLS isolation.

        Returns:
            The created Experiment record.
        """
        ...


@runtime_checkable
class IDeploymentRepository(Protocol):
    """Repository interface for Deployment records."""

    async def get_by_id(self, deployment_id: uuid.UUID, tenant: TenantContext) -> Deployment | None:
        """Retrieve a deployment by ID within tenant scope.

        Args:
            deployment_id: UUID of the deployment.
            tenant: Tenant context for RLS isolation.

        Returns:
            The Deployment if found, None otherwise.
        """
        ...

    async def list_all(
        self, tenant: TenantContext, page: int, page_size: int
    ) -> tuple[list[Deployment], int]:
        """List deployments for a tenant with pagination.

        Args:
            tenant: Tenant context for RLS isolation.
            page: Page number (1-indexed).
            page_size: Items per page.

        Returns:
            Tuple of (deployments list, total count).
        """
        ...

    async def create(
        self,
        model_id: str,
        model_version: str,
        strategy: str,
        target_environment: str,
        traffic_split: dict[str, int],
        health_check_url: str | None,
        tenant: TenantContext,
    ) -> Deployment:
        """Create a new deployment record.

        Args:
            model_id: UUID string of the model from aumos-model-registry.
            model_version: Model version to deploy.
            strategy: Deployment strategy (canary, ab, shadow, blue-green).
            target_environment: Target environment (development, staging, production).
            traffic_split: Traffic percentage allocation.
            health_check_url: Optional health check URL.
            tenant: Tenant context for RLS isolation.

        Returns:
            The created Deployment record.
        """
        ...

    async def update_status(
        self,
        deployment_id: uuid.UUID,
        status: str,
        tenant: TenantContext,
        traffic_split: dict[str, int] | None = None,
    ) -> Deployment:
        """Update a deployment's status and optionally its traffic split.

        Args:
            deployment_id: UUID of the deployment to update.
            status: New status value.
            tenant: Tenant context for RLS isolation.
            traffic_split: Optional new traffic split to apply.

        Returns:
            The updated Deployment record.
        """
        ...


@runtime_checkable
class IFeatureSetRepository(Protocol):
    """Repository interface for FeatureSet records."""

    async def get_by_id(self, feature_set_id: uuid.UUID, tenant: TenantContext) -> FeatureSet | None:
        """Retrieve a feature set by ID within tenant scope.

        Args:
            feature_set_id: UUID of the feature set.
            tenant: Tenant context for RLS isolation.

        Returns:
            The FeatureSet if found, None otherwise.
        """
        ...

    async def list_all(
        self, tenant: TenantContext, page: int, page_size: int
    ) -> tuple[list[FeatureSet], int]:
        """List feature sets for a tenant with pagination.

        Args:
            tenant: Tenant context for RLS isolation.
            page: Page number (1-indexed).
            page_size: Items per page.

        Returns:
            Tuple of (feature sets list, total count).
        """
        ...

    async def create(
        self,
        name: str,
        entity_name: str,
        features: list[dict[str, Any]],
        source_type: str,
        schedule: str | None,
        tenant: TenantContext,
    ) -> FeatureSet:
        """Create a new feature set record.

        Args:
            name: Feature set name.
            entity_name: Entity associated with this feature set.
            features: List of feature definitions.
            source_type: Data source type (batch, stream, request).
            schedule: Optional cron materialization schedule.
            tenant: Tenant context for RLS isolation.

        Returns:
            The created FeatureSet record.
        """
        ...


@runtime_checkable
class IRetrainingJobRepository(Protocol):
    """Repository interface for RetrainingJob records."""

    async def get_by_id(self, job_id: uuid.UUID, tenant: TenantContext) -> RetrainingJob | None:
        """Retrieve a retraining job by ID within tenant scope.

        Args:
            job_id: UUID of the retraining job.
            tenant: Tenant context for RLS isolation.

        Returns:
            The RetrainingJob if found, None otherwise.
        """
        ...

    async def list_all(
        self, tenant: TenantContext, page: int, page_size: int
    ) -> tuple[list[RetrainingJob], int]:
        """List retraining jobs for a tenant with pagination.

        Args:
            tenant: Tenant context for RLS isolation.
            page: Page number (1-indexed).
            page_size: Items per page.

        Returns:
            Tuple of (jobs list, total count).
        """
        ...

    async def create(
        self,
        model_id: str,
        trigger_type: str,
        tenant: TenantContext,
    ) -> RetrainingJob:
        """Create a new retraining job record in pending status.

        Args:
            model_id: UUID string of the model to retrain.
            trigger_type: What triggered retraining (drift, scheduled, manual).
            tenant: Tenant context for RLS isolation.

        Returns:
            The created RetrainingJob in pending status.
        """
        ...

    async def count_running_for_tenant(self, tenant: TenantContext) -> int:
        """Count active retraining jobs for a tenant.

        Used to enforce max_concurrent_retraining_jobs limit.

        Args:
            tenant: Tenant context for RLS isolation.

        Returns:
            Number of jobs with status 'pending' or 'running'.
        """
        ...


@runtime_checkable
class IMLflowClient(Protocol):
    """Interface for the MLflow tracking client adapter."""

    async def create_experiment(self, name: str, tenant_id: str) -> str:
        """Create an MLflow experiment namespaced by tenant.

        Args:
            name: Experiment name (will be prefixed with tenant_id).
            tenant_id: Tenant UUID string for namespace isolation.

        Returns:
            The MLflow experiment ID string.
        """
        ...

    async def log_run(
        self,
        mlflow_experiment_id: str,
        run_name: str | None,
        metrics: dict[str, float],
        params: dict[str, str],
        tags: dict[str, str],
    ) -> dict[str, Any]:
        """Log a run to an MLflow experiment.

        Args:
            mlflow_experiment_id: MLflow experiment ID.
            run_name: Optional run name.
            metrics: Numeric metric values to log.
            params: Hyperparameter key-value pairs.
            tags: Run-level tags.

        Returns:
            Run data dict including run_id, artifact_uri, and status.
        """
        ...

    async def list_runs(
        self,
        mlflow_experiment_id: str,
        page: int,
        page_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """List runs for an MLflow experiment.

        Args:
            mlflow_experiment_id: MLflow experiment ID.
            page: Page number (1-indexed).
            page_size: Items per page.

        Returns:
            Tuple of (run dicts list, total count).
        """
        ...


@runtime_checkable
class IFeastClient(Protocol):
    """Interface for the Feast feature store client adapter."""

    async def register_feature_view(
        self,
        name: str,
        entity_name: str,
        features: list[dict[str, Any]],
        source_type: str,
        schedule: str | None,
        tenant_id: str,
    ) -> bool:
        """Register a feature view in the Feast registry.

        Args:
            name: Feature view name.
            entity_name: Associated entity name.
            features: Feature definitions.
            source_type: Data source type.
            schedule: Optional cron materialization schedule.
            tenant_id: Tenant ID for namespace isolation.

        Returns:
            True if registration succeeded.
        """
        ...

    async def materialize(self, feature_view_name: str, tenant_id: str) -> bool:
        """Trigger materialization for a feature view.

        Args:
            feature_view_name: Name of the feature view to materialize.
            tenant_id: Tenant ID for namespace isolation.

        Returns:
            True if materialization was triggered successfully.
        """
        ...


@runtime_checkable
class IMLOpsEventPublisher(Protocol):
    """Interface for the MLOps Kafka event publisher."""

    async def publish_experiment_created(
        self,
        tenant_id: uuid.UUID,
        experiment_id: uuid.UUID,
        name: str,
        correlation_id: str,
    ) -> None:
        """Publish an experiment created event.

        Args:
            tenant_id: Owning tenant UUID.
            experiment_id: Created experiment UUID.
            name: Experiment name.
            correlation_id: Request correlation ID for distributed tracing.
        """
        ...

    async def publish_deployment_created(
        self,
        tenant_id: uuid.UUID,
        deployment_id: uuid.UUID,
        model_id: str,
        model_version: str,
        strategy: str,
        correlation_id: str,
    ) -> None:
        """Publish a deployment created event.

        Args:
            tenant_id: Owning tenant UUID.
            deployment_id: Created deployment UUID.
            model_id: Model being deployed.
            model_version: Model version.
            strategy: Deployment strategy.
            correlation_id: Request correlation ID.
        """
        ...

    async def publish_deployment_rolled_back(
        self,
        tenant_id: uuid.UUID,
        deployment_id: uuid.UUID,
        reason: str,
        correlation_id: str,
    ) -> None:
        """Publish a deployment rolled back event.

        Args:
            tenant_id: Owning tenant UUID.
            deployment_id: Rolled-back deployment UUID.
            reason: Rollback reason.
            correlation_id: Request correlation ID.
        """
        ...

    async def publish_retraining_triggered(
        self,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        model_id: str,
        trigger_type: str,
        correlation_id: str,
    ) -> None:
        """Publish a retraining triggered event.

        Args:
            tenant_id: Owning tenant UUID.
            job_id: Created retraining job UUID.
            model_id: Model being retrained.
            trigger_type: What triggered retraining.
            correlation_id: Request correlation ID.
        """
        ...


__all__ = [
    "IDeploymentRepository",
    "IExperimentRepository",
    "IFeastClient",
    "IFeatureSetRepository",
    "IMLOpsEventPublisher",
    "IMLflowClient",
    "IRetrainingJobRepository",
]
