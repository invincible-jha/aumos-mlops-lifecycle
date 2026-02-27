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


@runtime_checkable
class IExperimentTracker(Protocol):
    """Interface for the hyperparameter and artifact experiment tracker."""

    async def create_experiment(
        self, name: str, tenant_id: str, tags: dict[str, str] | None, artifact_location: str | None
    ) -> str:
        """Create a namespaced experiment and return its MLflow experiment ID.

        Args:
            name: Human-readable experiment name.
            tenant_id: Tenant UUID string for namespace isolation.
            tags: Optional key-value tags applied on creation.
            artifact_location: Optional artifact storage URI.

        Returns:
            MLflow experiment ID string.
        """
        ...

    async def start_run(
        self, mlflow_experiment_id: str, run_name: str | None, tags: dict[str, str] | None
    ) -> str:
        """Start a new MLflow run and return the run ID.

        Args:
            mlflow_experiment_id: MLflow experiment to associate the run with.
            run_name: Optional human-readable run name.
            tags: Optional key-value tags for the run.

        Returns:
            MLflow run ID string.
        """
        ...

    async def log_hyperparameters(self, run_id: str, params: dict[str, str]) -> None:
        """Log hyperparameter key-value pairs to an active run.

        Args:
            run_id: MLflow run ID.
            params: Hyperparameter name-to-string-value mapping.
        """
        ...

    async def log_metrics(
        self, run_id: str, metrics: dict[str, float], step: int | None, timestamp: Any | None
    ) -> None:
        """Log metric values (optionally per epoch step) to an active run.

        Args:
            run_id: MLflow run ID.
            metrics: Metric name to float value mapping.
            step: Optional epoch or global step index.
            timestamp: Optional UTC datetime for the metric values.
        """
        ...

    async def end_run(self, run_id: str, status: str) -> None:
        """Terminate an active MLflow run.

        Args:
            run_id: MLflow run ID to terminate.
            status: Terminal status (FINISHED, FAILED, or KILLED).
        """
        ...

    async def compare_runs(
        self, mlflow_experiment_id: str, metric_keys: list[str], max_runs: int
    ) -> list[dict[str, Any]]:
        """Compare runs within an experiment across specified metrics.

        Args:
            mlflow_experiment_id: MLflow experiment to search within.
            metric_keys: Metric names to include in the comparison.
            max_runs: Maximum number of runs to return.

        Returns:
            List of run comparison dicts sorted by the first metric descending.
        """
        ...


@runtime_checkable
class IDatasetVersioner(Protocol):
    """Interface for the DVC-backed dataset version manager."""

    async def register_dataset(
        self, name: str, file_path: str, metadata: dict[str, Any], tenant_id: str
    ) -> dict[str, Any]:
        """Register a dataset and return a version record.

        Args:
            name: Human-readable dataset name.
            file_path: Absolute path to the dataset file.
            metadata: Arbitrary metadata dict.
            tenant_id: Tenant UUID string.

        Returns:
            Version record dict with version_id, hash, and metadata.
        """
        ...

    async def diff_versions(
        self, name: str, version_a_id: str, version_b_id: str, tenant_id: str
    ) -> dict[str, Any]:
        """Compute a diff summary between two dataset versions.

        Args:
            name: Dataset name.
            version_a_id: Earlier version ID.
            version_b_id: Later version ID.
            tenant_id: Tenant UUID string.

        Returns:
            Diff dict with hash_changed and metadata_diff fields.
        """
        ...

    async def record_usage(self, run_id: str, version_id: str) -> None:
        """Record that a training run consumed a specific dataset version.

        Args:
            run_id: MLflow run ID that consumed the dataset.
            version_id: Dataset version ID that was used.
        """
        ...


@runtime_checkable
class ITrainingOrchestrator(Protocol):
    """Interface for the Kubernetes distributed training orchestrator."""

    async def create_training_job(
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
        tenant_id: str,
    ) -> dict[str, Any]:
        """Create a Kubernetes Job for a distributed training workload.

        Args:
            experiment_id: Owning MLflow experiment ID.
            run_id: MLflow run ID for this training job.
            image: Container image URI.
            command: Container entrypoint command list.
            gpu_count: Number of NVIDIA GPUs per pod.
            memory_gb: RAM limit in gigabytes per pod.
            cpu_count: CPU request per pod.
            num_nodes: Number of parallel worker pods.
            framework: "pytorch_ddp" or "horovod".
            env_vars: Additional environment variables.
            tenant_id: Tenant UUID string.

        Returns:
            Job status dict with job_name, namespace, uid, and created_at.
        """
        ...

    async def get_job_status(self, job_name: str) -> dict[str, Any]:
        """Retrieve current status of a Kubernetes training job.

        Args:
            job_name: K8s Job name.

        Returns:
            Status dict with active, succeeded, failed pod counts, and phase.
        """
        ...

    async def delete_job(self, job_name: str) -> bool:
        """Delete a training job and its pods.

        Args:
            job_name: K8s Job name to delete.

        Returns:
            True if deletion was accepted.
        """
        ...


@runtime_checkable
class IHyperoptAdapter(Protocol):
    """Interface for the hyperparameter optimisation adapter."""

    async def create_study(
        self,
        study_name: str,
        tenant_id: str,
        direction: str | list[str],
        pruner_type: str,
        sampler_type: str,
    ) -> str:
        """Create a new Optuna study and return its namespaced study name.

        Args:
            study_name: Human-readable study name.
            tenant_id: Tenant UUID string for namespace isolation.
            direction: "minimize", "maximize", or a list for multi-objective.
            pruner_type: "median", "percentile", or "hyperband".
            sampler_type: "tpe", "cma", or "random".

        Returns:
            Namespaced study name (tenant_{tenant_id}/{study_name}).
        """
        ...

    async def get_best_trial(self, study_name: str) -> dict[str, Any] | None:
        """Retrieve the best trial from a completed single-objective study.

        Args:
            study_name: Namespaced study name.

        Returns:
            Best trial dict or None if no completed trials exist.
        """
        ...

    async def get_optimisation_history(self, study_name: str) -> list[dict[str, Any]]:
        """Return the full optimisation history for a study.

        Args:
            study_name: Namespaced study name.

        Returns:
            List of trial dicts sorted by trial number ascending.
        """
        ...


@runtime_checkable
class IModelPackager(Protocol):
    """Interface for the Docker-based model packager."""

    async def serialise_model(
        self, model: Any, output_dir: str, format: str, model_name: str
    ) -> str:
        """Serialise a model to the specified format and return the output path.

        Args:
            model: Python model object to serialise.
            output_dir: Directory where the artefact is written.
            format: "pickle", "safetensors", or "onnx".
            model_name: Base filename for the artefact.

        Returns:
            Absolute path to the serialised model file.
        """
        ...

    async def generate_dockerfile(
        self, model_name: str, model_version: str, framework: str, output_dir: str
    ) -> str:
        """Generate a multi-stage Dockerfile for model serving.

        Args:
            model_name: Model name for the image label.
            model_version: Semantic version string.
            framework: ML framework identifier.
            output_dir: Directory where Dockerfile is written.

        Returns:
            Absolute path to the generated Dockerfile.
        """
        ...

    async def build_image(
        self, build_context_dir: str, model_name: str, model_version: str, tenant_id: str
    ) -> str:
        """Build a Docker image and return the full image tag.

        Args:
            build_context_dir: Directory containing the Dockerfile.
            model_name: Model name used to construct the image tag.
            model_version: Model version tag.
            tenant_id: Tenant UUID string.

        Returns:
            Full image tag string.
        """
        ...

    async def push_image(self, image_tag: str) -> str:
        """Push a Docker image to the registry and return the digest.

        Args:
            image_tag: Full image tag string.

        Returns:
            Image digest string (sha256:...).
        """
        ...


@runtime_checkable
class IDeploymentAutomator(Protocol):
    """Interface for the canary and A/B deployment automator."""

    async def create_canary_deployment(
        self,
        model_name: str,
        stable_version: str,
        canary_version: str,
        initial_canary_pct: int,
        tenant_id: str,
        image: str,
        replicas: int,
    ) -> dict[str, Any]:
        """Create a canary deployment with initial traffic split.

        Args:
            model_name: Logical model name.
            stable_version: Current production version tag.
            canary_version: New candidate version tag.
            initial_canary_pct: Initial canary traffic percentage.
            tenant_id: Tenant UUID string.
            image: Container image URI for the canary.
            replicas: Number of canary replica pods.

        Returns:
            Deployment state dict with deployment_id and traffic_split.
        """
        ...

    async def shift_traffic(self, deployment_id: str, canary_pct: int) -> dict[str, Any]:
        """Shift canary traffic to a new percentage.

        Args:
            deployment_id: Deployment state ID.
            canary_pct: New canary traffic percentage (0–100).

        Returns:
            Updated deployment state dict.
        """
        ...

    async def rollback_deployment(self, deployment_id: str, reason: str) -> dict[str, Any]:
        """Roll back a deployment by restoring 100% stable traffic.

        Args:
            deployment_id: Deployment state ID.
            reason: Human-readable rollback reason.

        Returns:
            Updated deployment state dict with status "rolled_back".
        """
        ...

    async def get_deployment_status(self, deployment_id: str) -> dict[str, Any] | None:
        """Get the current state of a deployment.

        Args:
            deployment_id: Deployment state ID.

        Returns:
            Deployment state dict or None if not found.
        """
        ...


@runtime_checkable
class IMLCostTracker(Protocol):
    """Interface for the ML compute cost tracker."""

    async def record_gpu_usage(
        self,
        experiment_id: str,
        run_id: str,
        instance_type: str,
        gpu_count: int,
        duration_seconds: float,
        tenant_id: str,
        project_key: str,
    ) -> dict[str, Any]:
        """Record GPU compute usage for a training run.

        Args:
            experiment_id: MLflow experiment ID.
            run_id: MLflow run ID.
            instance_type: GPU instance type key.
            gpu_count: Number of GPUs used.
            duration_seconds: Training session wall-clock duration.
            tenant_id: Tenant UUID string.
            project_key: Budget allocation key.

        Returns:
            Usage record dict with cost_usd and budget_remaining_usd.
        """
        ...

    async def check_budget(self, project_key: str) -> dict[str, Any]:
        """Check current budget utilisation for a project.

        Args:
            project_key: Budget allocation key.

        Returns:
            Budget status dict with limit_usd, spent_usd, and remaining_usd.
        """
        ...

    async def generate_cost_report(self, tenant_id: str, report_period_label: str) -> dict[str, Any]:
        """Generate a cost allocation report for a tenant.

        Args:
            tenant_id: Tenant UUID string.
            report_period_label: Label for the report period.

        Returns:
            Report dict with per-project breakdowns and totals.
        """
        ...


@runtime_checkable
class IModelValidationRunner(Protocol):
    """Interface for the model evaluation and validation runner."""

    async def cross_validate(
        self,
        model: Any,
        features: Any,
        labels: Any,
        n_folds: int,
        stratified: bool,
        scoring: list[str] | None,
    ) -> dict[str, Any]:
        """Run k-fold cross-validation and return metric results.

        Args:
            model: Scikit-learn compatible estimator.
            features: Feature matrix.
            labels: Target vector.
            n_folds: Number of CV folds.
            stratified: Use stratified folds if True.
            scoring: List of sklearn scoring metric names.

        Returns:
            CV result dict with per-metric mean, std, and fold scores.
        """
        ...

    async def evaluate_holdout(
        self, model: Any, test_features: Any, test_labels: Any, task_type: str
    ) -> dict[str, Any]:
        """Evaluate a model on a held-out test set.

        Args:
            model: Fitted model.
            test_features: Held-out feature matrix.
            test_labels: Held-out true labels.
            task_type: "classification", "regression", or "ranking".

        Returns:
            Metric dict appropriate for the task type.
        """
        ...

    async def significance_test(
        self, scores_model_a: list[float], scores_model_b: list[float], alpha: float
    ) -> dict[str, Any]:
        """Paired t-test for statistical significance between two models.

        Args:
            scores_model_a: Metric scores for model A.
            scores_model_b: Metric scores for model B.
            alpha: Significance level.

        Returns:
            Result with t_statistic, p_value, is_significant, and winner.
        """
        ...

    async def detect_data_leakage(
        self, train_features: Any, test_features: Any, threshold_duplicate_pct: float
    ) -> dict[str, Any]:
        """Detect data leakage between train and test sets.

        Args:
            train_features: Training feature matrix.
            test_features: Test feature matrix.
            threshold_duplicate_pct: Percentage above which leakage is high risk.

        Returns:
            Leakage report with overlap_count, overlap_pct, and risk_level.
        """
        ...


@runtime_checkable
class IModelPromoter(Protocol):
    """Interface for the MLflow model stage promoter."""

    async def promote(
        self,
        model_name: str,
        model_version: str,
        target_stage: str,
        promoted_by: str,
        reason: str,
        archive_existing: bool,
    ) -> dict[str, Any]:
        """Promote a model version to a new stage.

        Args:
            model_name: Registered model name in MLflow.
            model_version: Version number string.
            target_stage: Target stage (Staging, Production, Archived, None).
            promoted_by: Username triggering the promotion.
            reason: Human-readable justification.
            archive_existing: Archive current production version if True.

        Returns:
            Promotion result dict with from_stage, to_stage, and promoted_at.
        """
        ...

    async def validate_promotion_gates(
        self,
        model_name: str,
        model_version: str,
        required_metrics: dict[str, float],
        metric_comparison: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Validate that a model meets all promotion metric thresholds.

        Args:
            model_name: Registered model name.
            model_version: Version number string.
            required_metrics: Metric name → threshold value mapping.
            metric_comparison: Metric name → "gte" | "lte" direction mapping.

        Returns:
            Gate validation result with gate_passed, passed_gates, and failed_gates.
        """
        ...

    async def rollback_to_previous_production(
        self, model_name: str, rolled_back_by: str, reason: str
    ) -> dict[str, Any] | None:
        """Roll back to the most recent previously-active production version.

        Args:
            model_name: Registered model name.
            rolled_back_by: Username requesting the rollback.
            reason: Human-readable rollback justification.

        Returns:
            Rollback result dict or None if no prior production version exists.
        """
        ...

    async def compare_with_production(
        self, model_name: str, candidate_version: str
    ) -> dict[str, Any]:
        """Compare a candidate model against the current production version.

        Args:
            model_name: Registered model name.
            candidate_version: Version being evaluated for promotion.

        Returns:
            Comparison dict with deltas and promotion recommendation.
        """
        ...

    async def get_audit_log(
        self, model_name: str | None, limit: int
    ) -> list[dict[str, Any]]:
        """Retrieve the promotion audit trail.

        Args:
            model_name: Optional filter for a specific model.
            limit: Maximum number of events to return.

        Returns:
            List of audit log dicts sorted by promoted_at descending.
        """
        ...


__all__ = [
    "IDatasetVersioner",
    "IDeploymentAutomator",
    "IDeploymentRepository",
    "IExperimentRepository",
    "IExperimentTracker",
    "IFeastClient",
    "IFeatureSetRepository",
    "IHyperoptAdapter",
    "IMLCostTracker",
    "IMLOpsEventPublisher",
    "IMLflowClient",
    "IModelPackager",
    "IModelPromoter",
    "IModelValidationRunner",
    "IRetrainingJobRepository",
    "ITrainingOrchestrator",
]
