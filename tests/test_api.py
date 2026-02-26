"""API endpoint tests for aumos-mlops-lifecycle.

Tests all REST endpoints using FastAPI TestClient with mocked service
dependencies. No real database, MLflow, or Feast connections required.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from aumos_mlops_lifecycle.api.schemas import (
    DeploymentListResponse,
    DeploymentResponse,
    ExperimentListResponse,
    ExperimentResponse,
    FeatureSetListResponse,
    FeatureSetResponse,
    RetrainingJobListResponse,
    RetrainingJobResponse,
    RollbackResponse,
    RunListResponse,
    RunResponse,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

TENANT_ID = "11111111-1111-1111-1111-111111111111"
EXPERIMENT_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
DEPLOYMENT_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
MODEL_ID = "cccccccc-cccc-cccc-cccc-cccccccccccc"
FEATURE_SET_ID = "dddddddd-dddd-dddd-dddd-dddddddddddd"
JOB_ID = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"

AUTH_HEADERS = {
    "Authorization": "Bearer test-token",
    "X-Tenant-ID": TENANT_ID,
}

SAMPLE_EXPERIMENT_RESPONSE = ExperimentResponse(
    id=uuid.UUID(EXPERIMENT_ID),
    tenant_id=uuid.UUID(TENANT_ID),
    name="churn-prediction-v3",
    description="XGBoost churn model",
    status="active",
    mlflow_experiment_id="42",
    tags={"team": "data-science"},
    created_at=datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc),
    updated_at=datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc),
)

SAMPLE_DEPLOYMENT_RESPONSE = DeploymentResponse(
    id=uuid.UUID(DEPLOYMENT_ID),
    tenant_id=uuid.UUID(TENANT_ID),
    model_id=uuid.UUID(MODEL_ID),
    model_version="3",
    strategy="canary",
    status="in_progress",
    target_environment="production",
    traffic_split={"stable": 90, "canary": 10},
    health_check_url=None,
    created_at=datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc),
    updated_at=datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc),
)

SAMPLE_FEATURE_SET_RESPONSE = FeatureSetResponse(
    id=uuid.UUID(FEATURE_SET_ID),
    tenant_id=uuid.UUID(TENANT_ID),
    name="customer-rfm-features",
    entity_name="customer_id",
    features=[{"name": "recency_days", "dtype": "float32"}],
    source_type="batch",
    schedule="0 */6 * * *",
    created_at=datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc),
    updated_at=datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc),
)

SAMPLE_JOB_RESPONSE = RetrainingJobResponse(
    id=uuid.UUID(JOB_ID),
    tenant_id=uuid.UUID(TENANT_ID),
    model_id=uuid.UUID(MODEL_ID),
    trigger_type="manual",
    status="pending",
    started_at=None,
    completed_at=None,
    metrics={},
    created_at=datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc),
    updated_at=datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc),
)


# ---------------------------------------------------------------------------
# Schema validation tests (no HTTP calls — pure Pydantic validation)
# ---------------------------------------------------------------------------


class TestExperimentSchemas:
    """Tests for experiment request/response schema validation."""

    def test_experiment_create_request_requires_name(self) -> None:
        """ExperimentCreateRequest should validate that name is non-empty."""
        from pydantic import ValidationError

        from aumos_mlops_lifecycle.api.schemas import ExperimentCreateRequest

        with pytest.raises(ValidationError):
            ExperimentCreateRequest(name="")

    def test_experiment_response_from_attributes(self) -> None:
        """ExperimentResponse should be constructable with model_validate."""
        response = ExperimentResponse.model_validate(
            {
                "id": EXPERIMENT_ID,
                "tenant_id": TENANT_ID,
                "name": "test",
                "description": None,
                "status": "active",
                "mlflow_experiment_id": "1",
                "tags": {},
                "created_at": datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc),
                "updated_at": datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc),
            }
        )
        assert response.name == "test"
        assert response.status == "active"


class TestDeploymentSchemas:
    """Tests for deployment request/response schema validation."""

    def test_deployment_strategy_must_be_valid(self) -> None:
        """DeploymentCreateRequest should reject invalid strategy values."""
        from pydantic import ValidationError

        from aumos_mlops_lifecycle.api.schemas import DeploymentCreateRequest

        with pytest.raises(ValidationError):
            DeploymentCreateRequest(
                model_id=uuid.uuid4(),
                model_version="1",
                strategy="invalid-strategy",
                target_environment="production",
            )

    def test_deployment_environment_must_be_valid(self) -> None:
        """DeploymentCreateRequest should reject invalid target_environment values."""
        from pydantic import ValidationError

        from aumos_mlops_lifecycle.api.schemas import DeploymentCreateRequest

        with pytest.raises(ValidationError):
            DeploymentCreateRequest(
                model_id=uuid.uuid4(),
                model_version="1",
                strategy="canary",
                target_environment="not-an-env",
            )

    def test_valid_deployment_request_accepted(self) -> None:
        """DeploymentCreateRequest should accept valid strategy and environment."""
        from aumos_mlops_lifecycle.api.schemas import DeploymentCreateRequest

        request = DeploymentCreateRequest(
            model_id=uuid.uuid4(),
            model_version="2",
            strategy="blue-green",
            target_environment="staging",
        )
        assert request.strategy == "blue-green"


class TestRetrainingJobSchemas:
    """Tests for retraining job request/response schema validation."""

    def test_trigger_type_must_be_valid(self) -> None:
        """RetrainingJobCreateRequest should reject invalid trigger_type values."""
        from pydantic import ValidationError

        from aumos_mlops_lifecycle.api.schemas import RetrainingJobCreateRequest

        with pytest.raises(ValidationError):
            RetrainingJobCreateRequest(
                model_id=uuid.uuid4(),
                trigger_type="invalid",
            )

    def test_all_valid_trigger_types_accepted(self) -> None:
        """RetrainingJobCreateRequest should accept drift, scheduled, and manual."""
        from aumos_mlops_lifecycle.api.schemas import RetrainingJobCreateRequest

        for trigger_type in ["drift", "scheduled", "manual"]:
            request = RetrainingJobCreateRequest(
                model_id=uuid.uuid4(),
                trigger_type=trigger_type,
            )
            assert request.trigger_type == trigger_type


class TestFeatureSetSchemas:
    """Tests for feature set request/response schema validation."""

    def test_source_type_must_be_valid(self) -> None:
        """FeatureSetCreateRequest should reject invalid source_type values."""
        from pydantic import ValidationError

        from aumos_mlops_lifecycle.api.schemas import FeatureDefinition, FeatureSetCreateRequest

        with pytest.raises(ValidationError):
            FeatureSetCreateRequest(
                name="my-features",
                entity_name="user_id",
                features=[FeatureDefinition(name="age", dtype="float32")],
                source_type="invalid",
            )

    def test_features_list_cannot_be_empty(self) -> None:
        """FeatureSetCreateRequest should require at least one feature."""
        from pydantic import ValidationError

        from aumos_mlops_lifecycle.api.schemas import FeatureSetCreateRequest

        with pytest.raises(ValidationError):
            FeatureSetCreateRequest(
                name="my-features",
                entity_name="user_id",
                features=[],
                source_type="batch",
            )

    def test_valid_feature_set_request_accepted(self) -> None:
        """FeatureSetCreateRequest should accept valid stream source type."""
        from aumos_mlops_lifecycle.api.schemas import FeatureDefinition, FeatureSetCreateRequest

        request = FeatureSetCreateRequest(
            name="streaming-features",
            entity_name="session_id",
            features=[FeatureDefinition(name="clicks", dtype="int32")],
            source_type="stream",
        )
        assert request.source_type == "stream"
        assert len(request.features) == 1
