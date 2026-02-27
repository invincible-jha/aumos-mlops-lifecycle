"""Model promoter adapter for aumos-mlops-lifecycle.

Manages MLflow model stage transitions (None → Staging → Production → Archived),
enforces promotion gate thresholds, integrates with the approval workflow,
dispatches promotion notifications, and supports rollback to a previous
production model.

Configuration:
    AUMOS_MLOPS_MLFLOW_TRACKING_URI — MLflow tracking server URL
"""

import asyncio
import uuid
from datetime import datetime, timezone
from functools import partial
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_STAGE_ORDER = ["None", "Staging", "Production", "Archived"]
_VALID_TRANSITIONS: dict[str, list[str]] = {
    "None": ["Staging"],
    "Staging": ["Production", "None"],
    "Production": ["Archived", "Staging"],
    "Archived": ["None"],
}


class ModelPromoter:
    """MLflow model stage promotion manager with gate validation.

    Handles stage transitions for registered MLflow models, enforces metric
    thresholds before promoting to production, records an audit trail, and
    dispatches notifications via a pluggable notification callback.

    Args:
        tracking_uri: MLflow tracking server URL.
        notification_callback: Optional async callable invoked after each
                               promotion. Receives a notification dict as its
                               sole argument.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        notification_callback: Any | None = None,
    ) -> None:
        """Initialise the model promoter.

        Args:
            tracking_uri: MLflow tracking server URL. Uses the MLFLOW_TRACKING_URI
                          environment variable if None.
            notification_callback: Async callable(notification: dict) called after
                                   successful promotions. None disables notifications.
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient()
        self._notification_callback = notification_callback
        # Audit log: list of promotion event dicts
        self._audit_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Stage management                                                     #
    # ------------------------------------------------------------------ #

    async def get_model_stage(self, model_name: str, model_version: str) -> str:
        """Retrieve the current stage of a registered model version.

        Args:
            model_name: Registered model name in MLflow.
            model_version: Version number string.

        Returns:
            Current stage string: "None", "Staging", "Production", or "Archived".
        """
        loop = asyncio.get_event_loop()
        stage: str = await loop.run_in_executor(
            None,
            partial(self._get_stage_sync, model_name=model_name, model_version=model_version),
        )
        return stage

    def _get_stage_sync(self, model_name: str, model_version: str) -> str:
        version_info = self._client.get_model_version(model_name, model_version)
        return version_info.current_stage

    async def promote(
        self,
        model_name: str,
        model_version: str,
        target_stage: str,
        promoted_by: str,
        reason: str,
        archive_existing: bool = True,
    ) -> dict[str, Any]:
        """Promote a model version to a new stage.

        Validates the stage transition is allowed, executes the transition via
        the MLflow API, records an audit log entry, and dispatches a notification.

        Args:
            model_name: Registered model name in MLflow.
            model_version: Version number string to promote.
            target_stage: Target stage. Must be one of "Staging", "Production",
                          "None", or "Archived".
            promoted_by: Username or service account triggering the promotion.
            reason: Human-readable justification for the promotion.
            archive_existing: If True and target_stage is "Production", archive
                              the currently active production version. Defaults to True.

        Returns:
            Promotion result dict with model_name, model_version, from_stage,
            to_stage, promoted_by, and promoted_at.

        Raises:
            ValueError: If the target_stage is not a valid transition from the
                        current stage.
        """
        current_stage = await self.get_model_stage(model_name, model_version)
        allowed_targets = _VALID_TRANSITIONS.get(current_stage, [])

        if target_stage not in allowed_targets:
            raise ValueError(
                f"Cannot transition '{model_name}' v{model_version} "
                f"from '{current_stage}' to '{target_stage}'. "
                f"Allowed: {allowed_targets}"
            )

        logger.info(
            "Promoting model",
            model_name=model_name,
            model_version=model_version,
            from_stage=current_stage,
            to_stage=target_stage,
            promoted_by=promoted_by,
        )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            partial(
                self._transition_stage_sync,
                model_name=model_name,
                model_version=model_version,
                stage=target_stage,
                archive_existing=archive_existing,
            ),
        )

        promoted_at = datetime.now(tz=timezone.utc).isoformat()
        audit_entry: dict[str, Any] = {
            "event_id": str(uuid.uuid4()),
            "model_name": model_name,
            "model_version": model_version,
            "from_stage": current_stage,
            "to_stage": target_stage,
            "promoted_by": promoted_by,
            "reason": reason,
            "promoted_at": promoted_at,
        }
        self._audit_log.append(audit_entry)

        if self._notification_callback is not None:
            await self._notification_callback(
                {
                    "event": "model_promoted",
                    "model_name": model_name,
                    "model_version": model_version,
                    "to_stage": target_stage,
                    "promoted_by": promoted_by,
                    "promoted_at": promoted_at,
                }
            )

        logger.info("Model promoted", **{k: v for k, v in audit_entry.items() if k != "event_id"})
        return audit_entry

    def _transition_stage_sync(
        self,
        model_name: str,
        model_version: str,
        stage: str,
        archive_existing: bool,
    ) -> None:
        self._client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )

    # ------------------------------------------------------------------ #
    # Promotion gate validation                                            #
    # ------------------------------------------------------------------ #

    async def validate_promotion_gates(
        self,
        model_name: str,
        model_version: str,
        required_metrics: dict[str, float],
        metric_comparison: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Verify that a model version meets all required metric thresholds.

        Retrieves the latest logged metrics for the model version's associated
        MLflow run and checks each required metric against its threshold.

        Args:
            model_name: Registered model name in MLflow.
            model_version: Version number string.
            required_metrics: Dict mapping metric name → threshold value. The
                              model must exceed (or meet, for inverse metrics)
                              each threshold.
            metric_comparison: Dict mapping metric name → "gte" | "lte".
                               Defaults to "gte" (greater-than-or-equal) for all
                               metrics not listed. Use "lte" for error rates.

        Returns:
            Gate validation result with gate_passed bool, passed_gates list,
            failed_gates list, and per-metric details.
        """
        loop = asyncio.get_event_loop()
        model_metrics: dict[str, float] = await loop.run_in_executor(
            None,
            partial(self._get_model_run_metrics_sync, model_name=model_name, model_version=model_version),
        )

        comparison = metric_comparison or {}
        passed_gates: list[str] = []
        failed_gates: list[dict[str, Any]] = []

        for metric_name, threshold in required_metrics.items():
            actual = model_metrics.get(metric_name)
            direction = comparison.get(metric_name, "gte")

            if actual is None:
                failed_gates.append(
                    {"metric": metric_name, "threshold": threshold, "actual": None, "reason": "metric_missing"}
                )
                continue

            if direction == "gte" and actual >= threshold:
                passed_gates.append(metric_name)
            elif direction == "lte" and actual <= threshold:
                passed_gates.append(metric_name)
            else:
                failed_gates.append(
                    {
                        "metric": metric_name,
                        "threshold": threshold,
                        "actual": round(actual, 4),
                        "direction": direction,
                        "reason": "threshold_not_met",
                    }
                )

        gate_passed = len(failed_gates) == 0
        logger.info(
            "Promotion gate validation",
            model_name=model_name,
            model_version=model_version,
            gate_passed=gate_passed,
            failed_count=len(failed_gates),
        )

        return {
            "model_name": model_name,
            "model_version": model_version,
            "gate_passed": gate_passed,
            "passed_gates": passed_gates,
            "failed_gates": failed_gates,
            "model_metrics": model_metrics,
        }

    def _get_model_run_metrics_sync(self, model_name: str, model_version: str) -> dict[str, float]:
        version_info = self._client.get_model_version(model_name, model_version)
        run_id = version_info.run_id
        if not run_id:
            return {}
        run = self._client.get_run(run_id)
        return dict(run.data.metrics)

    # ------------------------------------------------------------------ #
    # Rollback                                                             #
    # ------------------------------------------------------------------ #

    async def rollback_to_previous_production(
        self,
        model_name: str,
        rolled_back_by: str,
        reason: str,
    ) -> dict[str, Any] | None:
        """Roll back production to the most recent previously-active production version.

        Searches the model's version history for the most recent "Production"
        version that is not currently active and transitions it back.

        Args:
            model_name: Registered model name in MLflow.
            rolled_back_by: Username or service account requesting the rollback.
            reason: Human-readable rollback justification.

        Returns:
            Rollback result dict if a previous production version was found,
            None if no prior production version exists.
        """
        loop = asyncio.get_event_loop()
        previous_version: dict[str, Any] | None = await loop.run_in_executor(
            None,
            partial(self._find_previous_production_sync, model_name=model_name),
        )

        if previous_version is None:
            logger.warning("No previous production version found for rollback", model_name=model_name)
            return None

        return await self.promote(
            model_name=model_name,
            model_version=previous_version["version"],
            target_stage="Production",
            promoted_by=rolled_back_by,
            reason=f"Rollback: {reason}",
            archive_existing=True,
        )

    def _find_previous_production_sync(self, model_name: str) -> dict[str, Any] | None:
        versions = self._client.search_model_versions(f"name='{model_name}'")
        production_versions = [
            v for v in versions if v.current_stage == "Archived"
        ]
        # Sort by version number descending to get the most recent archived (previously prod)
        production_versions.sort(key=lambda v: int(v.version), reverse=True)

        if not production_versions:
            return None

        candidate = production_versions[0]
        return {"version": candidate.version, "stage": candidate.current_stage}

    # ------------------------------------------------------------------ #
    # Model comparison                                                     #
    # ------------------------------------------------------------------ #

    async def compare_with_production(
        self,
        model_name: str,
        candidate_version: str,
    ) -> dict[str, Any]:
        """Compare a candidate model version against the current production version.

        Retrieves logged metrics for both versions and computes per-metric
        deltas to help reviewers decide whether to promote the candidate.

        Args:
            model_name: Registered model name in MLflow.
            candidate_version: Version string of the model being considered.

        Returns:
            Comparison dict with production_version, candidate_version,
            metrics for each, deltas, and a recommendation.
        """
        loop = asyncio.get_event_loop()

        production_versions = await loop.run_in_executor(
            None,
            partial(self._find_current_production_sync, model_name=model_name),
        )

        if production_versions is None:
            return {
                "model_name": model_name,
                "candidate_version": candidate_version,
                "production_version": None,
                "message": "No production model found — first deployment.",
                "recommendation": "promote",
            }

        prod_metrics = await loop.run_in_executor(
            None,
            partial(
                self._get_model_run_metrics_sync,
                model_name=model_name,
                model_version=production_versions,
            ),
        )
        candidate_metrics = await loop.run_in_executor(
            None,
            partial(
                self._get_model_run_metrics_sync,
                model_name=model_name,
                model_version=candidate_version,
            ),
        )

        deltas: dict[str, float] = {}
        for metric_name, candidate_val in candidate_metrics.items():
            if metric_name in prod_metrics:
                deltas[metric_name] = round(candidate_val - prod_metrics[metric_name], 4)

        # Simple heuristic: if the majority of metrics improved, recommend promotion
        improvements = sum(1 for delta in deltas.values() if delta > 0)
        regressions = sum(1 for delta in deltas.values() if delta < 0)
        recommendation = "promote" if improvements >= regressions else "hold"

        return {
            "model_name": model_name,
            "production_version": production_versions,
            "candidate_version": candidate_version,
            "production_metrics": prod_metrics,
            "candidate_metrics": candidate_metrics,
            "deltas": deltas,
            "improvements": improvements,
            "regressions": regressions,
            "recommendation": recommendation,
        }

    def _find_current_production_sync(self, model_name: str) -> str | None:
        versions = self._client.search_model_versions(f"name='{model_name}'")
        production = [v for v in versions if v.current_stage == "Production"]
        if not production:
            return None
        return production[0].version

    # ------------------------------------------------------------------ #
    # Audit log                                                            #
    # ------------------------------------------------------------------ #

    async def get_audit_log(
        self,
        model_name: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retrieve the promotion audit trail.

        Args:
            model_name: Optional filter to return events for one model only.
            limit: Maximum number of events to return (most recent first).

        Returns:
            List of audit log dicts sorted by promoted_at descending.
        """
        entries = self._audit_log
        if model_name is not None:
            entries = [e for e in entries if e["model_name"] == model_name]
        sorted_entries = sorted(entries, key=lambda e: e["promoted_at"], reverse=True)
        return sorted_entries[:limit]

    # ------------------------------------------------------------------ #
    # Approval workflow integration                                        #
    # ------------------------------------------------------------------ #

    async def request_promotion_approval(
        self,
        model_name: str,
        model_version: str,
        target_stage: str,
        requested_by: str,
        justification: str,
    ) -> dict[str, Any]:
        """Create a promotion approval request for human review.

        In production this would persist the request to the approval workflow
        service (aumos-approval-workflow). For now it returns a structured
        request object with a pending status.

        Args:
            model_name: Registered model name.
            model_version: Version number string.
            target_stage: Desired target stage.
            requested_by: Username requesting the promotion.
            justification: Business justification for the promotion.

        Returns:
            Approval request dict with request_id and status "pending_approval".
        """
        request: dict[str, Any] = {
            "request_id": str(uuid.uuid4()),
            "model_name": model_name,
            "model_version": model_version,
            "target_stage": target_stage,
            "requested_by": requested_by,
            "justification": justification,
            "status": "pending_approval",
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        logger.info(
            "Promotion approval requested",
            request_id=request["request_id"],
            model_name=model_name,
            model_version=model_version,
            target_stage=target_stage,
        )
        return request
