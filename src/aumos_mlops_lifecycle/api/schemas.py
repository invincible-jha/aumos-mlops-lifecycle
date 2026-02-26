"""Pydantic request and response schemas for aumos-mlops-lifecycle API.

All API inputs and outputs use Pydantic models — never return raw dicts.
Schemas are grouped by resource: Experiment, Deployment, FeatureSet, RetrainingJob.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Experiment schemas
# ---------------------------------------------------------------------------


class ExperimentCreateRequest(BaseModel):
    """Request body for creating a new experiment."""

    name: str = Field(description="Unique experiment name within the tenant", min_length=1, max_length=255)
    description: str | None = Field(default=None, description="Human-readable description of the experiment goal")
    tags: dict[str, str] = Field(default_factory=dict, description="Key-value tags for filtering and grouping")


class ExperimentResponse(BaseModel):
    """Response schema for a single experiment."""

    id: uuid.UUID = Field(description="Unique experiment identifier")
    tenant_id: uuid.UUID = Field(description="Owning tenant identifier")
    name: str = Field(description="Experiment name")
    description: str | None = Field(description="Experiment description")
    status: str = Field(description="Experiment status: active, archived, deleted")
    mlflow_experiment_id: str | None = Field(description="Corresponding MLflow experiment ID")
    tags: dict[str, str] = Field(description="Key-value tags")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")

    model_config = {"from_attributes": True}


class ExperimentListResponse(BaseModel):
    """Paginated list of experiments."""

    items: list[ExperimentResponse] = Field(description="List of experiments for the current page")
    total: int = Field(description="Total number of experiments matching the query")
    page: int = Field(description="Current page number (1-indexed)")
    page_size: int = Field(description="Number of items per page")


# ---------------------------------------------------------------------------
# Experiment run schemas
# ---------------------------------------------------------------------------


class RunLogRequest(BaseModel):
    """Request body for logging a run to an experiment."""

    run_name: str | None = Field(default=None, description="Optional human-readable run name")
    metrics: dict[str, float] = Field(default_factory=dict, description="Numeric metrics (e.g. accuracy, f1_score)")
    params: dict[str, str] = Field(default_factory=dict, description="Hyperparameters and configuration values")
    tags: dict[str, str] = Field(default_factory=dict, description="Run-level tags")
    artifact_uri: str | None = Field(default=None, description="URI to model artifacts logged for this run")


class RunResponse(BaseModel):
    """Response schema for a logged experiment run."""

    run_id: str = Field(description="MLflow run ID")
    run_name: str | None = Field(description="Optional run name")
    experiment_id: uuid.UUID = Field(description="Parent experiment ID")
    metrics: dict[str, float] = Field(description="Logged metrics")
    params: dict[str, str] = Field(description="Logged hyperparameters")
    tags: dict[str, str] = Field(description="Run tags")
    artifact_uri: str | None = Field(description="Artifact storage URI")
    status: str = Field(description="Run status: RUNNING, FINISHED, FAILED")
    started_at: datetime = Field(description="Run start timestamp")
    ended_at: datetime | None = Field(description="Run end timestamp")


class RunListResponse(BaseModel):
    """Paginated list of experiment runs."""

    items: list[RunResponse] = Field(description="List of runs for the current page")
    total: int = Field(description="Total number of runs for this experiment")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Items per page")


# ---------------------------------------------------------------------------
# Deployment schemas
# ---------------------------------------------------------------------------


class DeploymentCreateRequest(BaseModel):
    """Request body for creating a model deployment."""

    model_id: uuid.UUID = Field(description="ID of the model to deploy (from aumos-model-registry)")
    model_version: str = Field(description="Model version to deploy", min_length=1)
    strategy: str = Field(
        description="Deployment strategy: canary, ab, shadow, or blue-green",
        pattern="^(canary|ab|shadow|blue-green)$",
    )
    target_environment: str = Field(
        description="Target environment: development, staging, or production",
        pattern="^(development|staging|production)$",
    )
    traffic_split: dict[str, int] = Field(
        default_factory=dict,
        description="Traffic split percentages, e.g. {'stable': 90, 'canary': 10}. Must sum to 100.",
    )
    health_check_url: str | None = Field(
        default=None,
        description="URL to poll for deployment health status",
    )


class DeploymentResponse(BaseModel):
    """Response schema for a single deployment."""

    id: uuid.UUID = Field(description="Unique deployment identifier")
    tenant_id: uuid.UUID = Field(description="Owning tenant identifier")
    model_id: uuid.UUID = Field(description="Deployed model ID")
    model_version: str = Field(description="Deployed model version")
    strategy: str = Field(description="Deployment strategy")
    status: str = Field(description="Deployment status: pending, in_progress, completed, failed, rolled_back")
    target_environment: str = Field(description="Target environment")
    traffic_split: dict[str, int] = Field(description="Current traffic split")
    health_check_url: str | None = Field(description="Health check endpoint URL")
    created_at: datetime = Field(description="Deployment creation timestamp")
    updated_at: datetime = Field(description="Last status update timestamp")

    model_config = {"from_attributes": True}


class DeploymentListResponse(BaseModel):
    """Paginated list of deployments."""

    items: list[DeploymentResponse] = Field(description="List of deployments for the current page")
    total: int = Field(description="Total number of deployments matching the query")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Items per page")


class RollbackRequest(BaseModel):
    """Request body for rolling back a deployment."""

    reason: str = Field(description="Reason for rollback, used in audit logs and Kafka events", min_length=1)


class RollbackResponse(BaseModel):
    """Response confirming a deployment rollback."""

    deployment_id: uuid.UUID = Field(description="ID of the rolled-back deployment")
    status: str = Field(description="New deployment status after rollback")
    reason: str = Field(description="Rollback reason")
    rolled_back_at: datetime = Field(description="Timestamp when rollback was initiated")


# ---------------------------------------------------------------------------
# Feature set schemas
# ---------------------------------------------------------------------------


class FeatureDefinition(BaseModel):
    """Definition of a single feature within a feature set."""

    name: str = Field(description="Feature name", min_length=1)
    dtype: str = Field(description="Feature data type: float32, float64, int32, int64, string, bool")
    description: str | None = Field(default=None, description="Optional feature description")


class FeatureSetCreateRequest(BaseModel):
    """Request body for creating a feature set."""

    name: str = Field(description="Unique feature set name within the tenant", min_length=1, max_length=255)
    entity_name: str = Field(description="Entity this feature set is associated with (e.g. customer_id, model_id)")
    features: list[FeatureDefinition] = Field(description="List of features in this set", min_length=1)
    source_type: str = Field(
        description="Data source type: batch, stream, or request",
        pattern="^(batch|stream|request)$",
    )
    schedule: str | None = Field(
        default=None,
        description="Cron schedule for materialization (e.g. '0 */6 * * *' for every 6 hours)",
    )


class FeatureSetResponse(BaseModel):
    """Response schema for a single feature set."""

    id: uuid.UUID = Field(description="Unique feature set identifier")
    tenant_id: uuid.UUID = Field(description="Owning tenant identifier")
    name: str = Field(description="Feature set name")
    entity_name: str = Field(description="Associated entity name")
    features: list[dict[str, Any]] = Field(description="Feature definitions as stored in Feast")
    source_type: str = Field(description="Data source type")
    schedule: str | None = Field(description="Materialization cron schedule")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")

    model_config = {"from_attributes": True}


class FeatureSetListResponse(BaseModel):
    """Paginated list of feature sets."""

    items: list[FeatureSetResponse] = Field(description="List of feature sets for the current page")
    total: int = Field(description="Total number of feature sets")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Items per page")


# ---------------------------------------------------------------------------
# Retraining job schemas
# ---------------------------------------------------------------------------


class RetrainingJobCreateRequest(BaseModel):
    """Request body for triggering a retraining job."""

    model_id: uuid.UUID = Field(description="ID of the model to retrain")
    trigger_type: str = Field(
        description="What initiated the retraining: drift, scheduled, or manual",
        pattern="^(drift|scheduled|manual)$",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional configuration overrides for the retraining run",
    )


class RetrainingJobResponse(BaseModel):
    """Response schema for a single retraining job."""

    id: uuid.UUID = Field(description="Unique retraining job identifier")
    tenant_id: uuid.UUID = Field(description="Owning tenant identifier")
    model_id: uuid.UUID = Field(description="Model being retrained")
    trigger_type: str = Field(description="What triggered retraining: drift, scheduled, manual")
    status: str = Field(description="Job status: pending, running, completed, failed")
    started_at: datetime | None = Field(description="Job start timestamp")
    completed_at: datetime | None = Field(description="Job completion timestamp")
    metrics: dict[str, Any] = Field(description="Retraining outcome metrics (e.g. new accuracy, data size)")
    created_at: datetime = Field(description="Job creation timestamp")
    updated_at: datetime = Field(description="Last status update timestamp")

    model_config = {"from_attributes": True}


class RetrainingJobListResponse(BaseModel):
    """Paginated list of retraining jobs."""

    items: list[RetrainingJobResponse] = Field(description="List of retraining jobs for the current page")
    total: int = Field(description="Total number of retraining jobs")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Items per page")
