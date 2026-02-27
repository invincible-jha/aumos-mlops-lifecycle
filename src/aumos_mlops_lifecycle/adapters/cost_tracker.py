"""ML compute cost tracker adapter for aumos-mlops-lifecycle.

Tracks per-experiment GPU-hour consumption, applies instance-type pricing,
enforces per-team budget caps, detects idle GPU waste, and generates cost
allocation reports. Integrates with aumos-ai-finops by emitting structured
cost events.

Configuration:
    AUMOS_MLOPS_DEFAULT_BUDGET_USD — Default monthly budget per team in USD
    AUMOS_MLOPS_GPU_IDLE_THRESHOLD — GPU utilisation % below which GPU is idle
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# USD per GPU-hour by instance type (cloud provider spot prices, approximate)
_GPU_INSTANCE_PRICES: dict[str, float] = {
    "a100_40gb": 2.21,
    "a100_80gb": 3.67,
    "v100_16gb": 0.74,
    "t4_16gb": 0.35,
    "a10g_24gb": 0.76,
    "h100_80gb": 8.50,
    "cpu_only": 0.048,
}


class MLCostTracker:
    """GPU-hour and compute cost tracker for ML workloads.

    Records training session costs per experiment, enforces budget limits
    per team or project, detects idle GPU resources, and generates detailed
    cost allocation reports for FinOps integration.

    Args:
        default_monthly_budget_usd: Default monthly budget cap per project.
        gpu_idle_utilisation_threshold: GPU utilisation % below which a GPU
                                        is considered idle and wasteful.
    """

    def __init__(
        self,
        default_monthly_budget_usd: float = 5000.0,
        gpu_idle_utilisation_threshold: float = 5.0,
    ) -> None:
        """Initialise the cost tracker.

        Args:
            default_monthly_budget_usd: Default monthly budget cap in USD.
            gpu_idle_utilisation_threshold: GPU utilisation % threshold for idle detection.
        """
        self._default_budget = default_monthly_budget_usd
        self._idle_threshold = gpu_idle_utilisation_threshold

        # experiment_id → list of usage records
        self._usage: dict[str, list[dict[str, Any]]] = defaultdict(list)
        # project_key → budget_usd
        self._budgets: dict[str, float] = {}
        # project_key → total_spend_usd (current period)
        self._spend: dict[str, float] = defaultdict(float)
        # gpu_id → last utilisation reading dict
        self._gpu_utilisation: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # GPU-hour tracking                                                    #
    # ------------------------------------------------------------------ #

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

        Calculates the cost from instance type pricing, records the usage event,
        and accumulates spend against the project budget.

        Args:
            experiment_id: MLflow experiment ID associated with the run.
            run_id: MLflow run ID for the training session.
            instance_type: GPU instance type key (e.g. "a100_40gb").
            gpu_count: Number of GPUs used.
            duration_seconds: Wall-clock duration of the training session.
            tenant_id: Tenant UUID string for namespace isolation.
            project_key: Budget allocation key (team or project identifier).

        Returns:
            Usage record dict with gpu_hours, cost_usd, and budget_remaining_usd.
        """
        price_per_gpu_hour = _GPU_INSTANCE_PRICES.get(instance_type, _GPU_INSTANCE_PRICES["cpu_only"])
        gpu_hours = (duration_seconds / 3600.0) * gpu_count
        cost_usd = gpu_hours * price_per_gpu_hour

        usage_record: dict[str, Any] = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "instance_type": instance_type,
            "gpu_count": gpu_count,
            "duration_seconds": duration_seconds,
            "gpu_hours": round(gpu_hours, 4),
            "cost_usd": round(cost_usd, 4),
            "tenant_id": tenant_id,
            "project_key": project_key,
            "recorded_at": datetime.now(tz=timezone.utc).isoformat(),
        }

        self._usage[experiment_id].append(usage_record)
        self._spend[project_key] += cost_usd

        budget_limit = self._budgets.get(project_key, self._default_budget)
        budget_remaining = budget_limit - self._spend[project_key]

        logger.info(
            "GPU usage recorded",
            experiment_id=experiment_id,
            run_id=run_id,
            gpu_hours=gpu_hours,
            cost_usd=cost_usd,
            budget_remaining_usd=budget_remaining,
            project_key=project_key,
        )

        if budget_remaining < 0:
            logger.warning(
                "Budget exceeded",
                project_key=project_key,
                overage_usd=abs(budget_remaining),
            )

        usage_record["budget_remaining_usd"] = round(budget_remaining, 2)
        return usage_record

    async def get_experiment_cost(self, experiment_id: str) -> dict[str, Any]:
        """Aggregate total cost for all runs in an experiment.

        Args:
            experiment_id: MLflow experiment ID.

        Returns:
            Summary dict with total_gpu_hours, total_cost_usd, run_count,
            and breakdown per run.
        """
        records = self._usage.get(experiment_id, [])
        total_gpu_hours = sum(r["gpu_hours"] for r in records)
        total_cost_usd = sum(r["cost_usd"] for r in records)

        return {
            "experiment_id": experiment_id,
            "run_count": len(records),
            "total_gpu_hours": round(total_gpu_hours, 4),
            "total_cost_usd": round(total_cost_usd, 4),
            "runs": [
                {
                    "run_id": r["run_id"],
                    "gpu_hours": r["gpu_hours"],
                    "cost_usd": r["cost_usd"],
                    "instance_type": r["instance_type"],
                }
                for r in records
            ],
        }

    # ------------------------------------------------------------------ #
    # Budget enforcement                                                   #
    # ------------------------------------------------------------------ #

    async def set_budget(self, project_key: str, monthly_budget_usd: float) -> None:
        """Set a monthly compute budget cap for a project or team.

        Args:
            project_key: Budget allocation key (e.g. "team_data_science").
            monthly_budget_usd: Monthly budget limit in USD.
        """
        self._budgets[project_key] = monthly_budget_usd
        logger.info("Budget set", project_key=project_key, budget_usd=monthly_budget_usd)

    async def check_budget(self, project_key: str) -> dict[str, Any]:
        """Check the current budget utilisation for a project.

        Args:
            project_key: Budget allocation key.

        Returns:
            Budget status dict with limit_usd, spent_usd, remaining_usd,
            utilisation_pct, and is_over_budget flag.
        """
        limit_usd = self._budgets.get(project_key, self._default_budget)
        spent_usd = self._spend.get(project_key, 0.0)
        remaining_usd = limit_usd - spent_usd
        utilisation_pct = (spent_usd / limit_usd * 100.0) if limit_usd > 0 else 0.0

        return {
            "project_key": project_key,
            "limit_usd": round(limit_usd, 2),
            "spent_usd": round(spent_usd, 4),
            "remaining_usd": round(remaining_usd, 2),
            "utilisation_pct": round(utilisation_pct, 2),
            "is_over_budget": remaining_usd < 0,
        }

    async def enforce_budget(self, project_key: str, estimated_cost_usd: float) -> bool:
        """Check if a proposed job would exceed the project's budget.

        Args:
            project_key: Budget allocation key.
            estimated_cost_usd: Estimated cost of the proposed training job.

        Returns:
            True if the job is within budget, False if it would exceed it.
        """
        status = await self.check_budget(project_key)
        fits_within_budget = estimated_cost_usd <= status["remaining_usd"]

        if not fits_within_budget:
            logger.warning(
                "Budget enforcement: job rejected",
                project_key=project_key,
                estimated_cost_usd=estimated_cost_usd,
                remaining_usd=status["remaining_usd"],
            )
        return fits_within_budget

    # ------------------------------------------------------------------ #
    # Cost trend analysis                                                  #
    # ------------------------------------------------------------------ #

    async def compute_cost_trend(
        self,
        project_key: str,
        tenant_id: str,
        window_days: int = 30,
    ) -> dict[str, Any]:
        """Compute a daily cost breakdown over the specified window.

        Aggregates usage records by day to surface spending trends.

        Args:
            project_key: Budget allocation key.
            tenant_id: Tenant UUID string for filtering.
            window_days: Number of past days to include in the trend.

        Returns:
            Trend dict with daily_costs list, total_cost_usd, and avg_daily_cost_usd.
        """
        all_records = [
            record
            for records in self._usage.values()
            for record in records
            if record["project_key"] == project_key and record["tenant_id"] == tenant_id
        ]

        daily_totals: dict[str, float] = defaultdict(float)
        for record in all_records:
            day = record["recorded_at"][:10]  # YYYY-MM-DD
            daily_totals[day] += record["cost_usd"]

        sorted_days = sorted(daily_totals.keys())[-window_days:]
        daily_costs = [{"date": day, "cost_usd": round(daily_totals[day], 4)} for day in sorted_days]

        total = sum(entry["cost_usd"] for entry in daily_costs)
        avg = total / len(daily_costs) if daily_costs else 0.0

        return {
            "project_key": project_key,
            "window_days": window_days,
            "daily_costs": daily_costs,
            "total_cost_usd": round(total, 4),
            "avg_daily_cost_usd": round(avg, 4),
        }

    # ------------------------------------------------------------------ #
    # Idle GPU detection                                                   #
    # ------------------------------------------------------------------ #

    async def record_gpu_utilisation(
        self,
        gpu_id: str,
        utilisation_pct: float,
        tenant_id: str,
        project_key: str,
    ) -> None:
        """Record a GPU utilisation sample for idle detection.

        Args:
            gpu_id: Unique identifier for the GPU instance.
            utilisation_pct: Current GPU compute utilisation percentage.
            tenant_id: Tenant UUID string.
            project_key: Budget allocation key for waste attribution.
        """
        self._gpu_utilisation[gpu_id] = {
            "gpu_id": gpu_id,
            "utilisation_pct": utilisation_pct,
            "tenant_id": tenant_id,
            "project_key": project_key,
            "sampled_at": datetime.now(tz=timezone.utc).isoformat(),
        }

    async def detect_idle_gpus(self, tenant_id: str) -> list[dict[str, Any]]:
        """Find GPUs with utilisation below the idle threshold.

        Args:
            tenant_id: Tenant UUID string to filter by.

        Returns:
            List of idle GPU dicts with gpu_id, utilisation_pct, and project_key.
        """
        idle = [
            reading
            for reading in self._gpu_utilisation.values()
            if reading["tenant_id"] == tenant_id
            and reading["utilisation_pct"] < self._idle_threshold
        ]

        if idle:
            logger.warning("Idle GPUs detected", tenant_id=tenant_id, idle_count=len(idle))

        return idle

    # ------------------------------------------------------------------ #
    # Cost allocation report                                               #
    # ------------------------------------------------------------------ #

    async def generate_cost_report(
        self,
        tenant_id: str,
        report_period_label: str = "current_month",
    ) -> dict[str, Any]:
        """Generate a comprehensive cost allocation report for a tenant.

        Breaks down spending by project, experiment, and instance type.
        Includes budget utilisation and idle GPU waste estimates.

        Args:
            tenant_id: Tenant UUID string for scoping.
            report_period_label: Human-readable label for the report period.

        Returns:
            Report dict with per-project breakdowns, totals, and idle waste.
        """
        all_records = [
            record
            for records in self._usage.values()
            for record in records
            if record["tenant_id"] == tenant_id
        ]

        by_project: dict[str, dict[str, Any]] = {}
        for record in all_records:
            project = record["project_key"]
            entry = by_project.setdefault(
                project,
                {
                    "project_key": project,
                    "total_cost_usd": 0.0,
                    "total_gpu_hours": 0.0,
                    "experiment_count": set(),
                    "by_instance_type": defaultdict(float),
                },
            )
            entry["total_cost_usd"] += record["cost_usd"]
            entry["total_gpu_hours"] += record["gpu_hours"]
            entry["experiment_count"].add(record["experiment_id"])
            entry["by_instance_type"][record["instance_type"]] += record["cost_usd"]

        # Serialise sets and defaultdicts
        serialised_projects = []
        for project, entry in by_project.items():
            budget = self._budgets.get(project, self._default_budget)
            serialised_projects.append(
                {
                    "project_key": project,
                    "total_cost_usd": round(entry["total_cost_usd"], 4),
                    "total_gpu_hours": round(entry["total_gpu_hours"], 4),
                    "experiment_count": len(entry["experiment_count"]),
                    "budget_limit_usd": budget,
                    "budget_utilisation_pct": round(
                        (entry["total_cost_usd"] / budget * 100.0) if budget > 0 else 0.0, 2
                    ),
                    "by_instance_type": dict(entry["by_instance_type"]),
                }
            )

        idle_gpus = await self.detect_idle_gpus(tenant_id)
        grand_total = sum(p["total_cost_usd"] for p in serialised_projects)

        return {
            "tenant_id": tenant_id,
            "report_period": report_period_label,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "grand_total_usd": round(grand_total, 4),
            "project_count": len(serialised_projects),
            "projects": serialised_projects,
            "idle_gpu_count": len(idle_gpus),
            "idle_gpus": idle_gpus,
        }
