"""Test fixtures for aumos-mlops-lifecycle.

Provides shared fixtures for tenant context, mock repositories, mock
MLflow/Feast clients, and mock Kafka publisher used across all test modules.
"""

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aumos_mlops_lifecycle.core.models import Deployment, Experiment, FeatureSet, RetrainingJob


# ---------------------------------------------------------------------------
# Tenant context fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tenant_id() -> uuid.UUID:
    """Return a fixed tenant UUID for test isolation."""
    return uuid.UUID("11111111-1111-1111-1111-111111111111")


@pytest.fixture
def tenant(tenant_id: uuid.UUID) -> MagicMock:
    """Return a mock TenantContext with a fixed tenant_id."""
    mock_tenant = MagicMock()
    mock_tenant.tenant_id = tenant_id
    mock_tenant.user_id = uuid.UUID("22222222-2222-2222-2222-222222222222")
    return mock_tenant


# ---------------------------------------------------------------------------
# ORM model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def experiment_id() -> uuid.UUID:
    """Return a fixed experiment UUID."""
    return uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")


@pytest.fixture
def sample_experiment(experiment_id: uuid.UUID, tenant_id: uuid.UUID) -> Experiment:
    """Return a populated Experiment ORM model instance."""
    experiment = Experiment(
        id=experiment_id,
        tenant_id=tenant_id,
        name="churn-prediction-v3",
        description="XGBoost churn model with RFM features",
        status="active",
        mlflow_experiment_id="42",
        tags={"team": "data-science"},
        created_at=datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc),
    )
    return experiment


@pytest.fixture
def deployment_id() -> uuid.UUID:
    """Return a fixed deployment UUID."""
    return uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


@pytest.fixture
def model_id() -> uuid.UUID:
    """Return a fixed model UUID."""
    return uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")


@pytest.fixture
def sample_deployment(deployment_id: uuid.UUID, tenant_id: uuid.UUID, model_id: uuid.UUID) -> Deployment:
    """Return a populated Deployment ORM model instance."""
    deployment = Deployment(
        id=deployment_id,
        tenant_id=tenant_id,
        model_id=str(model_id),
        model_version="3",
        strategy="canary",
        status="in_progress",
        target_environment="production",
        traffic_split={"stable": 90, "canary": 10},
        health_check_url="https://models.internal/churn/health",
        created_at=datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc),
    )
    return deployment


@pytest.fixture
def feature_set_id() -> uuid.UUID:
    """Return a fixed feature set UUID."""
    return uuid.UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")


@pytest.fixture
def sample_feature_set(feature_set_id: uuid.UUID, tenant_id: uuid.UUID) -> FeatureSet:
    """Return a populated FeatureSet ORM model instance."""
    feature_set = FeatureSet(
        id=feature_set_id,
        tenant_id=tenant_id,
        name="customer-rfm-features",
        entity_name="customer_id",
        features=[
            {"name": "recency_days", "dtype": "float32", "description": "Days since last purchase"},
            {"name": "frequency_count", "dtype": "int32", "description": "Total purchase count"},
            {"name": "monetary_value", "dtype": "float64", "description": "Total spend"},
        ],
        source_type="batch",
        schedule="0 */6 * * *",
        created_at=datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc),
    )
    return feature_set


@pytest.fixture
def job_id() -> uuid.UUID:
    """Return a fixed retraining job UUID."""
    return uuid.UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")


@pytest.fixture
def sample_retraining_job(job_id: uuid.UUID, tenant_id: uuid.UUID, model_id: uuid.UUID) -> RetrainingJob:
    """Return a populated RetrainingJob ORM model instance."""
    job = RetrainingJob(
        id=job_id,
        tenant_id=tenant_id,
        model_id=str(model_id),
        trigger_type="drift",
        status="pending",
        started_at=None,
        completed_at=None,
        metrics={},
        created_at=datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc),
    )
    return job


# ---------------------------------------------------------------------------
# Mock repository fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_experiment_repo() -> AsyncMock:
    """Return a mock ExperimentRepository (implements IExperimentRepository)."""
    return AsyncMock()


@pytest.fixture
def mock_deployment_repo() -> AsyncMock:
    """Return a mock DeploymentRepository (implements IDeploymentRepository)."""
    return AsyncMock()


@pytest.fixture
def mock_feature_set_repo() -> AsyncMock:
    """Return a mock FeatureSetRepository (implements IFeatureSetRepository)."""
    return AsyncMock()


@pytest.fixture
def mock_retraining_job_repo() -> AsyncMock:
    """Return a mock RetrainingJobRepository (implements IRetrainingJobRepository)."""
    return AsyncMock()


# ---------------------------------------------------------------------------
# Mock external client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mlflow_client() -> AsyncMock:
    """Return a mock MLflowClient (implements IMLflowClient)."""
    mock = AsyncMock()
    mock.create_experiment.return_value = "42"
    mock.log_run.return_value = {
        "run_id": "run-abc123",
        "status": "FINISHED",
        "start_time": datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc),
        "end_time": datetime(2026, 2, 26, 12, 5, 0, tzinfo=timezone.utc),
        "artifact_uri": "mlflow-artifacts:/42/run-abc123/artifacts",
    }
    mock.list_runs.return_value = ([], 0)
    return mock


@pytest.fixture
def mock_feast_client() -> AsyncMock:
    """Return a mock FeastClient (implements IFeastClient)."""
    mock = AsyncMock()
    mock.register_feature_view.return_value = True
    mock.materialize.return_value = True
    return mock


@pytest.fixture
def mock_publisher() -> AsyncMock:
    """Return a mock MLOpsEventPublisher (implements IMLOpsEventPublisher)."""
    return AsyncMock()
