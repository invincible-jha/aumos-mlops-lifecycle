"""Kafka event publishing for aumos-mlops-lifecycle.

Defines typed publish methods for every MLOps lifecycle domain event.
All events include tenant_id and correlation_id for distributed tracing.

Events published by this service:
  - MLO_EXPERIMENT_CREATED  — new experiment registered
  - MLO_RUN_LOGGED          — run metrics logged to experiment
  - MLO_DEPLOYMENT_CREATED  — model deployment initiated
  - MLO_DEPLOYMENT_COMPLETED — deployment finished successfully
  - MLO_DEPLOYMENT_ROLLED_BACK — deployment rolled back
  - MLO_FEATURE_SET_CREATED — feature set registered in Feast
  - MLO_RETRAINING_TRIGGERED — retraining job started
  - MLO_RETRAINING_COMPLETED — retraining job finished
"""

import uuid

from aumos_common.events import EventPublisher, Topics
from aumos_common.observability import get_logger

from aumos_mlops_lifecycle.core.interfaces import IMLOpsEventPublisher

logger = get_logger(__name__)


class MLOpsEventPublisher(IMLOpsEventPublisher):
    """Publisher for aumos-mlops-lifecycle domain events.

    Wraps EventPublisher from aumos-common with typed methods for each
    event type produced by the MLOps lifecycle service.

    Args:
        publisher: The underlying EventPublisher from aumos-common.
                   Defaults to a new EventPublisher() if not provided.
    """

    def __init__(self, publisher: EventPublisher | None = None) -> None:
        """Initialize with the shared event publisher.

        Args:
            publisher: Configured EventPublisher instance. If None, a new
                       instance is created using aumos-common defaults.
        """
        self._publisher = publisher or EventPublisher()

    async def publish_experiment_created(
        self,
        tenant_id: uuid.UUID,
        experiment_id: uuid.UUID,
        name: str,
        correlation_id: str,
    ) -> None:
        """Publish an ExperimentCreated event to Kafka.

        Downstream consumers (e.g., observability service) subscribe to this
        event to begin tracking experiment-level metrics.

        Args:
            tenant_id: Owning tenant UUID.
            experiment_id: Created experiment UUID.
            name: Experiment name for human-readable event context.
            correlation_id: Request correlation ID for distributed tracing.
        """
        await self._publisher.publish(
            Topics.MLO_EXPERIMENT_CREATED,
            {
                "event_type": "experiment_created",
                "tenant_id": str(tenant_id),
                "experiment_id": str(experiment_id),
                "name": name,
                "correlation_id": correlation_id,
            },
        )
        logger.info(
            "Published ExperimentCreated event",
            tenant_id=str(tenant_id),
            experiment_id=str(experiment_id),
        )

    async def publish_run_logged(
        self,
        tenant_id: uuid.UUID,
        experiment_id: uuid.UUID,
        run_id: str,
        metrics: dict[str, float],
        correlation_id: str,
    ) -> None:
        """Publish a RunLogged event to Kafka.

        Args:
            tenant_id: Owning tenant UUID.
            experiment_id: Parent experiment UUID.
            run_id: MLflow run ID.
            metrics: Key-value metrics logged in this run.
            correlation_id: Request correlation ID for distributed tracing.
        """
        await self._publisher.publish(
            Topics.MLO_RUN_LOGGED,
            {
                "event_type": "run_logged",
                "tenant_id": str(tenant_id),
                "experiment_id": str(experiment_id),
                "run_id": run_id,
                "metrics": metrics,
                "correlation_id": correlation_id,
            },
        )
        logger.info(
            "Published RunLogged event",
            tenant_id=str(tenant_id),
            experiment_id=str(experiment_id),
            run_id=run_id,
        )

    async def publish_deployment_created(
        self,
        tenant_id: uuid.UUID,
        deployment_id: uuid.UUID,
        model_id: str,
        model_version: str,
        strategy: str,
        correlation_id: str,
    ) -> None:
        """Publish a DeploymentCreated event to Kafka.

        Downstream consumers (drift-detector, observability) subscribe to begin
        monitoring the new deployment.

        Args:
            tenant_id: Owning tenant UUID.
            deployment_id: Created deployment UUID.
            model_id: Model being deployed (from aumos-model-registry).
            model_version: Model version string.
            strategy: Deployment strategy (canary, ab, shadow, blue-green).
            correlation_id: Request correlation ID for distributed tracing.
        """
        await self._publisher.publish(
            Topics.MLO_DEPLOYMENT_CREATED,
            {
                "event_type": "deployment_created",
                "tenant_id": str(tenant_id),
                "deployment_id": str(deployment_id),
                "model_id": model_id,
                "model_version": model_version,
                "strategy": strategy,
                "correlation_id": correlation_id,
            },
        )
        logger.info(
            "Published DeploymentCreated event",
            tenant_id=str(tenant_id),
            deployment_id=str(deployment_id),
            strategy=strategy,
        )

    async def publish_deployment_completed(
        self,
        tenant_id: uuid.UUID,
        deployment_id: uuid.UUID,
        status: str,
        target_environment: str,
        correlation_id: str,
    ) -> None:
        """Publish a DeploymentCompleted event to Kafka.

        Args:
            tenant_id: Owning tenant UUID.
            deployment_id: Completed deployment UUID.
            status: Final deployment status.
            target_environment: Environment where the model was deployed.
            correlation_id: Request correlation ID for distributed tracing.
        """
        await self._publisher.publish(
            Topics.MLO_DEPLOYMENT_COMPLETED,
            {
                "event_type": "deployment_completed",
                "tenant_id": str(tenant_id),
                "deployment_id": str(deployment_id),
                "status": status,
                "target_environment": target_environment,
                "correlation_id": correlation_id,
            },
        )
        logger.info(
            "Published DeploymentCompleted event",
            tenant_id=str(tenant_id),
            deployment_id=str(deployment_id),
            status=status,
        )

    async def publish_deployment_rolled_back(
        self,
        tenant_id: uuid.UUID,
        deployment_id: uuid.UUID,
        reason: str,
        correlation_id: str,
    ) -> None:
        """Publish a DeploymentRolledBack event to Kafka.

        Downstream consumers restore stable traffic routing and may trigger
        alerting workflows.

        Args:
            tenant_id: Owning tenant UUID.
            deployment_id: Rolled-back deployment UUID.
            reason: Human-readable rollback reason for audit trail.
            correlation_id: Request correlation ID for distributed tracing.
        """
        await self._publisher.publish(
            Topics.MLO_DEPLOYMENT_ROLLED_BACK,
            {
                "event_type": "deployment_rolled_back",
                "tenant_id": str(tenant_id),
                "deployment_id": str(deployment_id),
                "reason": reason,
                "correlation_id": correlation_id,
            },
        )
        logger.info(
            "Published DeploymentRolledBack event",
            tenant_id=str(tenant_id),
            deployment_id=str(deployment_id),
            reason=reason,
        )

    async def publish_feature_set_created(
        self,
        tenant_id: uuid.UUID,
        feature_set_id: uuid.UUID,
        name: str,
        correlation_id: str,
    ) -> None:
        """Publish a FeatureSetCreated event to Kafka.

        Args:
            tenant_id: Owning tenant UUID.
            feature_set_id: Created feature set UUID.
            name: Feature set name.
            correlation_id: Request correlation ID for distributed tracing.
        """
        await self._publisher.publish(
            Topics.MLO_FEATURE_SET_CREATED,
            {
                "event_type": "feature_set_created",
                "tenant_id": str(tenant_id),
                "feature_set_id": str(feature_set_id),
                "name": name,
                "correlation_id": correlation_id,
            },
        )
        logger.info(
            "Published FeatureSetCreated event",
            tenant_id=str(tenant_id),
            feature_set_id=str(feature_set_id),
        )

    async def publish_retraining_triggered(
        self,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        model_id: str,
        trigger_type: str,
        correlation_id: str,
    ) -> None:
        """Publish a RetrainingTriggered event to Kafka.

        Downstream training pipeline workers consume this event to start
        the actual retraining computation.

        Args:
            tenant_id: Owning tenant UUID.
            job_id: Created retraining job UUID.
            model_id: Model being retrained.
            trigger_type: What triggered retraining (drift, scheduled, manual).
            correlation_id: Request correlation ID for distributed tracing.
        """
        await self._publisher.publish(
            Topics.MLO_RETRAINING_TRIGGERED,
            {
                "event_type": "retraining_triggered",
                "tenant_id": str(tenant_id),
                "job_id": str(job_id),
                "model_id": model_id,
                "trigger_type": trigger_type,
                "correlation_id": correlation_id,
            },
        )
        logger.info(
            "Published RetrainingTriggered event",
            tenant_id=str(tenant_id),
            job_id=str(job_id),
            trigger_type=trigger_type,
        )

    async def publish_retraining_completed(
        self,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        status: str,
        metrics: dict[str, float],
        correlation_id: str,
    ) -> None:
        """Publish a RetrainingCompleted event to Kafka.

        Args:
            tenant_id: Owning tenant UUID.
            job_id: Completed retraining job UUID.
            status: Final job status (completed or failed).
            metrics: Outcome metrics from the retraining run.
            correlation_id: Request correlation ID for distributed tracing.
        """
        await self._publisher.publish(
            Topics.MLO_RETRAINING_COMPLETED,
            {
                "event_type": "retraining_completed",
                "tenant_id": str(tenant_id),
                "job_id": str(job_id),
                "status": status,
                "metrics": metrics,
                "correlation_id": correlation_id,
            },
        )
        logger.info(
            "Published RetrainingCompleted event",
            tenant_id=str(tenant_id),
            job_id=str(job_id),
            status=status,
        )
