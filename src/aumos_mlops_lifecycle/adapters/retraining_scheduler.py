"""Cron-based retraining scheduler for aumos-mlops-lifecycle.

Reads feature sets with cron schedules and automatically triggers
retraining jobs when the schedule is due. Runs as a background task
within the FastAPI lifespan.

GAP-164: Cron Scheduler Integration
"""

from __future__ import annotations

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class RetrainingScheduler:
    """Background scheduler that triggers retraining jobs from feature set cron schedules.

    Reads FeatureSet records that have a non-null `schedule` cron expression,
    determines which are due based on `last_materialized_at`, and triggers
    new retraining jobs for overdue schedules.

    Args:
        feature_store_service: Service for listing schedulable feature sets.
        retraining_service: Service for triggering retraining jobs.
        check_interval_minutes: How often to poll for due schedules.
    """

    def __init__(
        self,
        feature_store_service: object,
        retraining_service: object,
        check_interval_minutes: int = 1,
    ) -> None:
        """Initialise the retraining scheduler.

        Args:
            feature_store_service: FeatureStoreService instance.
            retraining_service: RetrainingService instance.
            check_interval_minutes: Polling interval in minutes.
        """
        self._feature_store_service = feature_store_service
        self._retraining_service = retraining_service
        self._check_interval_minutes = check_interval_minutes
        self._scheduler: object | None = None

    def start(self) -> None:
        """Start the retraining cron scheduler."""
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import-untyped]
            from apscheduler.triggers.interval import IntervalTrigger  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "apscheduler is required for retraining scheduling. "
                "Install with: pip install apscheduler"
            ) from exc

        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            self._check_due_schedules,
            trigger=IntervalTrigger(minutes=self._check_interval_minutes),
            id="retraining_schedule_check",
            replace_existing=True,
        )
        scheduler.start()
        self._scheduler = scheduler
        logger.info("retraining_scheduler_started", interval_minutes=self._check_interval_minutes)

    def stop(self) -> None:
        """Stop the retraining scheduler gracefully."""
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)  # type: ignore[union-attr]
            logger.info("retraining_scheduler_stopped")

    async def _check_due_schedules(self) -> None:
        """Check all schedulable feature sets and trigger overdue retraining jobs."""
        try:
            feature_sets = await self._feature_store_service.list_schedulable()  # type: ignore[union-attr]
        except Exception as exc:
            logger.error("retraining_scheduler_list_failed", error=str(exc))
            return

        for feature_set in feature_sets:
            try:
                schedule = getattr(feature_set, "schedule", None)
                last_materialized = getattr(feature_set, "last_materialized_at", None)
                if schedule and self._is_due(schedule, last_materialized):
                    model_id = str(getattr(feature_set, "model_id", ""))
                    tenant_id = str(getattr(feature_set, "tenant_id", ""))
                    await self._retraining_service.trigger_scheduled(  # type: ignore[union-attr]
                        model_id=model_id,
                        tenant_id=tenant_id,
                        feature_set_id=str(getattr(feature_set, "id", "")),
                    )
                    logger.info(
                        "retraining_triggered_by_schedule",
                        feature_set_id=str(getattr(feature_set, "id", "")),
                        model_id=model_id,
                        schedule=schedule,
                    )
            except Exception as exc:
                logger.error(
                    "retraining_schedule_check_failed",
                    feature_set_id=str(getattr(feature_set, "id", "unknown")),
                    error=str(exc),
                )

    def _is_due(self, cron_expression: str, last_run_at: object | None) -> bool:
        """Determine if a cron schedule is due for execution.

        Uses croniter to compute the next scheduled run time after last_run_at
        and checks if that time has passed.

        Args:
            cron_expression: Cron string (e.g. '0 2 * * 1' for Monday 2 AM).
            last_run_at: datetime of the last successful run, or None if never run.

        Returns:
            True if the schedule is due, False otherwise.
        """
        from datetime import datetime, timezone

        try:
            from croniter import croniter  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("croniter_not_installed", cron_expression=cron_expression)
            return False

        now = datetime.now(timezone.utc)

        if last_run_at is None:
            # Never run before — treat as due immediately
            return True

        # Ensure last_run_at is timezone-aware
        if hasattr(last_run_at, "tzinfo") and last_run_at.tzinfo is None:  # type: ignore[union-attr]
            last_run_at = last_run_at.replace(tzinfo=timezone.utc)  # type: ignore[union-attr]

        try:
            cron = croniter(cron_expression, last_run_at)
            next_run = cron.get_next(datetime)
            if next_run.tzinfo is None:
                next_run = next_run.replace(tzinfo=timezone.utc)
            return now >= next_run
        except Exception as exc:
            logger.warning(
                "retraining_cron_parse_failed",
                cron_expression=cron_expression,
                error=str(exc),
            )
            return False
