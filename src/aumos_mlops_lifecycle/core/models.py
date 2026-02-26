"""SQLAlchemy ORM models for aumos-mlops-lifecycle.

All tenant-scoped tables extend AumOSModel which provides:
  - id: UUID primary key
  - tenant_id: UUID (RLS-enforced)
  - created_at: datetime
  - updated_at: datetime

Table naming convention: mlo_{table_name}
Examples: mlo_experiments, mlo_deployments, mlo_feature_sets, mlo_retraining_jobs
"""

from typing import Any

from sqlalchemy import String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from aumos_common.database import AumOSModel


class Experiment(AumOSModel):
    """MLflow-backed experiment with tenant isolation.

    Each experiment corresponds to an MLflow experiment whose ID is stored
    for cross-referencing. Experiments are namespaced per tenant in MLflow
    as: tenant_{tenant_id}/{name}

    Table: mlo_experiments
    """

    __tablename__ = "mlo_experiments"

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="active",
        comment="active | archived | deleted",
    )
    mlflow_experiment_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="Corresponding MLflow experiment ID for cross-referencing",
    )
    tags: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Key-value tags for filtering and grouping experiments",
    )


class Deployment(AumOSModel):
    """Model deployment record with strategy and traffic configuration.

    Tracks the lifecycle of a model deployment from creation through
    completion or rollback. Supports canary, A/B, shadow, and blue-green
    strategies via the traffic_split JSONB column.

    Table: mlo_deployments
    """

    __tablename__ = "mlo_deployments"

    model_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        comment="UUID of the model from aumos-model-registry (no FK constraint — cross-service)",
    )
    model_version: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Semantic version or commit SHA of the deployed model",
    )
    strategy: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="canary | ab | shadow | blue-green",
    )
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
        index=True,
        comment="pending | in_progress | completed | failed | rolled_back",
    )
    target_environment: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="development | staging | production",
    )
    traffic_split: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Traffic percentages, e.g. {'stable': 90, 'canary': 10}",
    )
    health_check_url: Mapped[str | None] = mapped_column(
        String(2048),
        nullable=True,
        comment="URL to poll for deployment health during canary progression",
    )


class FeatureSet(AumOSModel):
    """Feast-backed feature set definition with materialization schedule.

    Defines a named set of features associated with an entity (e.g. customer_id).
    The features column stores a list of FeatureDefinition dicts as JSONB for
    flexible schema evolution without ORM migrations.

    Table: mlo_feature_sets
    """

    __tablename__ = "mlo_feature_sets"

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Unique feature set name within the tenant",
    )
    entity_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Entity this feature set is associated with, e.g. customer_id",
    )
    features: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="List of feature definitions: [{name, dtype, description}]",
    )
    source_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="batch | stream | request",
    )
    schedule: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Cron expression for materialization schedule, e.g. '0 */6 * * *'",
    )


class RetrainingJob(AumOSModel):
    """Retraining job record tracking async model retraining.

    Created when a retraining is triggered (drift, scheduled, or manual).
    The actual retraining is performed asynchronously by downstream consumers
    of the MLO_RETRAINING_TRIGGERED Kafka event. This record tracks status
    and outcome metrics.

    Table: mlo_retraining_jobs
    """

    __tablename__ = "mlo_retraining_jobs"

    model_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        comment="UUID of the model being retrained (no FK constraint — cross-service)",
    )
    trigger_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="drift | scheduled | manual",
    )
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
        index=True,
        comment="pending | running | completed | failed",
    )
    started_at: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="ISO 8601 timestamp when retraining started",
    )
    completed_at: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="ISO 8601 timestamp when retraining completed or failed",
    )
    metrics: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Retraining outcome metrics, e.g. {accuracy, f1, training_rows}",
    )


__all__ = ["Experiment", "Deployment", "FeatureSet", "RetrainingJob"]
