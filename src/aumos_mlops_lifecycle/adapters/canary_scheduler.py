"""APScheduler-based canary deployment progression daemon.

Runs a background task that periodically checks all active canary deployments
and automatically progresses traffic or triggers rollback based on error rates
from aumos-observability.

GAP-159: Auto-Scaling Canary Logic
"""

from __future__ import annotations

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class CanaryScheduler:
    """Background scheduler that automatically progresses or rolls back canary deployments.

    Every check_interval_minutes the scheduler:
      1. Lists all active canary deployments across all tenants
      2. Queries aumos-observability for the error rate in the last window
      3. If error rate >= threshold: triggers rollback
      4. If canary traffic + increment >= 100%: promotes to stable
      5. Otherwise: increments canary traffic by the configured step

    Args:
        deployment_service: Service handling deployment state transitions.
        observability_client: Client for querying production error rates.
        check_interval_minutes: How often to evaluate canary health.
        traffic_increment_percent: Traffic step size per successful check.
        error_rate_threshold: Error rate above which rollback is triggered.
    """

    def __init__(
        self,
        deployment_service: object,
        observability_client: object,
        check_interval_minutes: int = 5,
        traffic_increment_percent: int = 10,
        error_rate_threshold: float = 0.05,
    ) -> None:
        """Initialise the canary scheduler with its dependencies.

        Args:
            deployment_service: DeploymentService instance.
            observability_client: ObservabilityClient instance.
            check_interval_minutes: Polling interval in minutes.
            traffic_increment_percent: Canary traffic increment per step.
            error_rate_threshold: Maximum tolerable error rate before rollback.
        """
        self._deployment_service = deployment_service
        self._observability_client = observability_client
        self._check_interval_minutes = check_interval_minutes
        self._traffic_increment = traffic_increment_percent
        self._error_threshold = error_rate_threshold
        self._scheduler: object | None = None

    def start(self) -> None:
        """Start the canary progression background scheduler."""
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import-untyped]
            from apscheduler.triggers.interval import IntervalTrigger  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("apscheduler is required for canary scheduling. Install with: pip install apscheduler") from exc

        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            self._check_all_canaries,
            trigger=IntervalTrigger(minutes=self._check_interval_minutes),
            id="canary_progression",
            replace_existing=True,
        )
        scheduler.start()
        self._scheduler = scheduler
        logger.info("canary_scheduler_started", interval_minutes=self._check_interval_minutes)

    def stop(self) -> None:
        """Gracefully stop the canary scheduler without waiting for running jobs."""
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)  # type: ignore[union-attr]
            logger.info("canary_scheduler_stopped")

    async def _check_all_canaries(self) -> None:
        """Evaluate all active canary deployments and take corrective action."""
        active_deployments = await self._deployment_service.list_active_canary_deployments()  # type: ignore[union-attr]
        logger.info("canary_check_started", deployment_count=len(active_deployments))
        for deployment in active_deployments:
            try:
                await self._evaluate_canary(deployment)
            except Exception as exc:
                logger.error(
                    "canary_check_failed",
                    deployment_id=str(getattr(deployment, "id", "unknown")),
                    error=str(exc),
                )

    async def _evaluate_canary(self, deployment: object) -> None:
        """Evaluate a single canary deployment and progress, promote, or rollback.

        Args:
            deployment: Deployment ORM model instance.
        """
        deployment_id = getattr(deployment, "id", None)
        tenant_id = getattr(deployment, "tenant_id", None)
        traffic_split: dict[str, int] = getattr(deployment, "traffic_split", {}) or {}
        canary_pct = traffic_split.get("canary", 0)

        error_rate: float = await self._observability_client.get_error_rate(  # type: ignore[union-attr]
            model_id=str(getattr(deployment, "model_id", "")),
            deployment_id=str(deployment_id),
            window_minutes=self._check_interval_minutes,
        )

        bound_log = logger.bind(
            deployment_id=str(deployment_id),
            canary_pct=canary_pct,
            error_rate=error_rate,
        )

        if error_rate >= self._error_threshold:
            bound_log.warning("canary_auto_rollback_triggered")
            await self._deployment_service.rollback(  # type: ignore[union-attr]
                deployment_id=deployment_id,
                tenant_id=tenant_id,
                reason=(
                    f"Auto-rollback: error rate {error_rate:.2%} >= "
                    f"threshold {self._error_threshold:.2%}"
                ),
            )
        elif canary_pct + self._traffic_increment >= 100:
            bound_log.info("canary_promoting_to_stable")
            await self._deployment_service.promote(  # type: ignore[union-attr]
                deployment_id=deployment_id,
                tenant_id=tenant_id,
            )
        else:
            bound_log.info("canary_traffic_incremented")
            await self._deployment_service.canary_progress(  # type: ignore[union-attr]
                deployment_id=deployment_id,
                tenant_id=tenant_id,
                increment=self._traffic_increment,
            )
