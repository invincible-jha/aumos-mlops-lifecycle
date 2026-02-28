"""Model card generation service for aumos-mlops-lifecycle.

Assembles HuggingFace-compatible model cards from experiment metadata,
evaluation results, fairness assessments, and deployment history.

GAP-160: Model Card Generation
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from aumos_common.observability import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model card Pydantic models
# ---------------------------------------------------------------------------


class ModelCard(BaseModel):
    """Structured model card assembling metadata from multiple AumOS services.

    Follows the HuggingFace model card schema v2 format.
    """

    model_id: str = Field(description="Unique model identifier")
    model_name: str = Field(description="Human-readable model name")
    description: str | None = Field(default=None, description="Model purpose and scope")
    version: str = Field(default="0.0.1", description="Model version string")
    training_data: str = Field(default="Not specified", description="Training dataset description")
    framework: str = Field(default="Not specified", description="ML framework used")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Training metrics from experiment tags")
    evaluation_results: dict[str, Any] = Field(
        default_factory=dict,
        description="Evaluation scores from aumos-testing-harness",
    )
    deployment_environments: list[str] = Field(
        default_factory=list,
        description="Environments this model has been deployed to",
    )
    known_limitations: str = Field(
        default="None specified",
        description="Known model limitations and caveats",
    )
    license: str = Field(default="Apache-2.0", description="Model license")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when this card was generated",
    )


class ModelCardService:
    """Assembles model cards from experiment, evaluation, and deployment metadata.

    Args:
        experiment_repo: Repository for loading experiment records.
        deployment_repo: Repository for listing model deployments.
        testing_harness_client: Client for fetching evaluation results.
    """

    def __init__(
        self,
        experiment_repo: Any,
        deployment_repo: Any,
        testing_harness_client: Any,
    ) -> None:
        """Initialise the model card service with injected dependencies.

        Args:
            experiment_repo: IExperimentRepository implementation.
            deployment_repo: IDeploymentRepository implementation.
            testing_harness_client: TestingHarnessClient implementation.
        """
        self._experiment_repo = experiment_repo
        self._deployment_repo = deployment_repo
        self._testing_client = testing_harness_client

    async def generate(
        self,
        model_id: str,
        tenant_id: str,
        session: Any,
    ) -> ModelCard:
        """Generate a model card by assembling data from all relevant services.

        Args:
            model_id: UUID of the model (stored in experiment tags as model_id).
            tenant_id: Tenant UUID for scoped access.
            session: Async SQLAlchemy session.

        Returns:
            Populated ModelCard instance.
        """
        from aumos_common.auth import TenantContext  # type: ignore[import-untyped]

        tenant = TenantContext(tenant_id=uuid.UUID(tenant_id))

        experiment = await self._experiment_repo.get_by_model_id(model_id, tenant, session)
        deployments = await self._deployment_repo.list_by_model_id(model_id, tenant, page=1, page_size=100)
        eval_results = await self._testing_client.get_latest_results(
            model_id=model_id,
            tenant_id=tenant_id,
        )

        if experiment is None:
            logger.warning("model_card_experiment_not_found", model_id=model_id)
            return ModelCard(
                model_id=model_id,
                model_name=model_id,
                evaluation_results=eval_results,
            )

        tags: dict[str, Any] = experiment.tags or {}
        deployment_list, _ = deployments if isinstance(deployments, tuple) else (deployments, 0)
        environments = [
            str(getattr(d, "target_environment", "unknown"))
            for d in deployment_list
        ]
        latest_version = "0.0.1"
        if deployment_list:
            latest_version = str(getattr(deployment_list[-1], "model_version", "0.0.1"))

        return ModelCard(
            model_id=model_id,
            model_name=experiment.name,
            description=getattr(experiment, "description", None),
            version=latest_version,
            training_data=tags.get("dataset", "Not specified"),
            framework=tags.get("framework", "Not specified"),
            metrics=tags.get("metrics", {}),
            evaluation_results=eval_results,
            deployment_environments=environments,
            known_limitations=tags.get("limitations", "None specified"),
        )

    def to_markdown(self, card: ModelCard) -> str:
        """Render model card as HuggingFace-compatible Markdown.

        Args:
            card: Populated ModelCard instance.

        Returns:
            Markdown string following the HuggingFace model card format.
        """
        metrics_rows = "\n".join(
            f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |"
            for k, v in card.metrics.items()
        )
        eval_rows = "\n".join(
            f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |"
            for k, v in card.evaluation_results.items()
        )

        return f"""---
model_id: {card.model_id}
version: {card.version}
license: {card.license}
generated_at: {card.generated_at.isoformat()}
---

# Model Card: {card.model_name}

## Model Description
{card.description or "No description provided."}

## Training Data
{card.training_data}

## Framework
{card.framework}

## Training Metrics
| Metric | Score |
|--------|-------|
{metrics_rows or "| — | — |"}

## Evaluation Results
| Metric | Score |
|--------|-------|
{eval_rows or "| — | — |"}

## Deployment Environments
{", ".join(card.deployment_environments) or "Not deployed"}

## Known Limitations
{card.known_limitations}

---
*Auto-generated by AumOS MLOps Lifecycle on {card.generated_at.strftime("%Y-%m-%d %H:%M UTC")}*
"""
