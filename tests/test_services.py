"""Unit tests for business logic services in aumos-mlops-lifecycle.

Tests the ExperimentService, DeploymentService, FeatureStoreService, and
RetrainingService using mock repositories, MLflow clients, Feast clients,
and Kafka publishers. No real database or external services required.
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from aumos_common.errors import NotFoundError

from aumos_mlops_lifecycle.api.schemas import (
    DeploymentCreateRequest,
    FeatureDefinition,
    FeatureSetCreateRequest,
    RetrainingJobCreateRequest,
)
from aumos_mlops_lifecycle.core.services import (
    DeploymentService,
    ExperimentService,
    FeatureStoreService,
    RetrainingService,
)


# ---------------------------------------------------------------------------
# ExperimentService tests
# ---------------------------------------------------------------------------


class TestExperimentServiceCreate:
    """Tests for ExperimentService.create."""

    async def test_create_experiment_returns_experiment(
        self,
        mock_experiment_repo: AsyncMock,
        mock_mlflow_client: AsyncMock,
        mock_publisher: AsyncMock,
        sample_experiment: object,
        tenant: object,
    ) -> None:
        """create() returns the ORM model from the repository."""
        mock_mlflow_client.create_experiment.return_value = "42"
        mock_experiment_repo.create.return_value = sample_experiment

        service = ExperimentService(
            repository=mock_experiment_repo,
            mlflow_client=mock_mlflow_client,
            publisher=mock_publisher,
        )

        result = await service.create(
            name="churn-prediction-v3",
            description="XGBoost churn model",
            tags={"team": "data-science"},
            tenant=tenant,
        )

        assert result is sample_experiment
        mock_mlflow_client.create_experiment.assert_called_once()
        mock_experiment_repo.create.assert_called_once()

    async def test_create_experiment_publishes_kafka_event(
        self,
        mock_experiment_repo: AsyncMock,
        mock_mlflow_client: AsyncMock,
        mock_publisher: AsyncMock,
        sample_experiment: object,
        tenant: object,
    ) -> None:
        """create() publishes an experiment_created event to Kafka."""
        mock_mlflow_client.create_experiment.return_value = "42"
        mock_experiment_repo.create.return_value = sample_experiment

        service = ExperimentService(
            repository=mock_experiment_repo,
            mlflow_client=mock_mlflow_client,
            publisher=mock_publisher,
        )

        await service.create(
            name="test-experiment",
            description=None,
            tags={},
            tenant=tenant,
        )

        mock_publisher.publish_experiment_created.assert_called_once()

    async def test_create_experiment_uses_mlflow_namespace(
        self,
        mock_experiment_repo: AsyncMock,
        mock_mlflow_client: AsyncMock,
        mock_publisher: AsyncMock,
        sample_experiment: object,
        tenant: object,
    ) -> None:
        """create() calls MLflow client with tenant_id for namespace isolation."""
        mock_mlflow_client.create_experiment.return_value = "99"
        mock_experiment_repo.create.return_value = sample_experiment

        service = ExperimentService(
            repository=mock_experiment_repo,
            mlflow_client=mock_mlflow_client,
            publisher=mock_publisher,
        )

        await service.create(
            name="my-experiment",
            description=None,
            tags={},
            tenant=tenant,
        )

        call_kwargs = mock_mlflow_client.create_experiment.call_args
        assert str(tenant.tenant_id) in str(call_kwargs)


class TestExperimentServiceGetById:
    """Tests for ExperimentService.get_by_id."""

    async def test_get_by_id_returns_experiment(
        self,
        mock_experiment_repo: AsyncMock,
        mock_mlflow_client: AsyncMock,
        mock_publisher: AsyncMock,
        sample_experiment: object,
        experiment_id: uuid.UUID,
        tenant: object,
    ) -> None:
        """get_by_id() returns the experiment when found."""
        mock_experiment_repo.get_by_id.return_value = sample_experiment

        service = ExperimentService(
            repository=mock_experiment_repo,
            mlflow_client=mock_mlflow_client,
            publisher=mock_publisher,
        )

        result = await service.get_by_id(experiment_id=experiment_id, tenant=tenant)

        assert result is sample_experiment

    async def test_get_by_id_raises_not_found(
        self,
        mock_experiment_repo: AsyncMock,
        mock_mlflow_client: AsyncMock,
        mock_publisher: AsyncMock,
        tenant: object,
    ) -> None:
        """get_by_id() raises NotFoundError when experiment does not exist."""
        mock_experiment_repo.get_by_id.return_value = None

        service = ExperimentService(
            repository=mock_experiment_repo,
            mlflow_client=mock_mlflow_client,
            publisher=mock_publisher,
        )

        with pytest.raises(NotFoundError):
            await service.get_by_id(experiment_id=uuid.uuid4(), tenant=tenant)


class TestExperimentServiceListAll:
    """Tests for ExperimentService.list_all."""

    async def test_list_all_returns_paginated_response(
        self,
        mock_experiment_repo: AsyncMock,
        mock_mlflow_client: AsyncMock,
        mock_publisher: AsyncMock,
        sample_experiment: object,
        tenant: object,
    ) -> None:
        """list_all() returns an ExperimentListResponse with correct pagination fields."""
        mock_experiment_repo.list_all.return_value = ([sample_experiment], 1)

        service = ExperimentService(
            repository=mock_experiment_repo,
            mlflow_client=mock_mlflow_client,
            publisher=mock_publisher,
        )

        result = await service.list_all(tenant=tenant, page=1, page_size=20)

        assert result.total == 1
        assert result.page == 1
        assert result.page_size == 20
        assert len(result.items) == 1


# ---------------------------------------------------------------------------
# DeploymentService tests
# ---------------------------------------------------------------------------


class TestDeploymentServiceDeploy:
    """Tests for DeploymentService.deploy."""

    async def test_deploy_creates_deployment(
        self,
        mock_deployment_repo: AsyncMock,
        mock_publisher: AsyncMock,
        sample_deployment: object,
        model_id: uuid.UUID,
        tenant: object,
    ) -> None:
        """deploy() creates a deployment record and returns it."""
        mock_deployment_repo.create.return_value = sample_deployment

        service = DeploymentService(
            repository=mock_deployment_repo,
            publisher=mock_publisher,
        )

        request = DeploymentCreateRequest(
            model_id=model_id,
            model_version="3",
            strategy="canary",
            target_environment="production",
            traffic_split={"stable": 90, "canary": 10},
            health_check_url=None,
        )

        result = await service.deploy(request=request, tenant=tenant)

        assert result is sample_deployment
        mock_deployment_repo.create.assert_called_once()

    async def test_deploy_publishes_kafka_event(
        self,
        mock_deployment_repo: AsyncMock,
        mock_publisher: AsyncMock,
        sample_deployment: object,
        model_id: uuid.UUID,
        tenant: object,
    ) -> None:
        """deploy() publishes a deployment_created event to Kafka."""
        mock_deployment_repo.create.return_value = sample_deployment

        service = DeploymentService(
            repository=mock_deployment_repo,
            publisher=mock_publisher,
        )

        request = DeploymentCreateRequest(
            model_id=model_id,
            model_version="3",
            strategy="canary",
            target_environment="production",
            traffic_split={"stable": 90, "canary": 10},
        )

        await service.deploy(request=request, tenant=tenant)

        mock_publisher.publish_deployment_created.assert_called_once()


class TestDeploymentServiceRollback:
    """Tests for DeploymentService.rollback."""

    async def test_rollback_updates_status_to_rolled_back(
        self,
        mock_deployment_repo: AsyncMock,
        mock_publisher: AsyncMock,
        sample_deployment: object,
        deployment_id: uuid.UUID,
        tenant: object,
    ) -> None:
        """rollback() updates deployment status and returns RollbackResponse."""
        mock_deployment_repo.get_by_id.return_value = sample_deployment
        rolled_back_deployment = sample_deployment
        rolled_back_deployment.status = "rolled_back"
        mock_deployment_repo.update_status.return_value = rolled_back_deployment

        service = DeploymentService(
            repository=mock_deployment_repo,
            publisher=mock_publisher,
        )

        result = await service.rollback(
            deployment_id=deployment_id,
            reason="High error rate detected during canary",
            tenant=tenant,
        )

        assert result.deployment_id == deployment_id
        assert result.reason == "High error rate detected during canary"
        mock_publisher.publish_deployment_rolled_back.assert_called_once()

    async def test_rollback_raises_not_found_for_missing_deployment(
        self,
        mock_deployment_repo: AsyncMock,
        mock_publisher: AsyncMock,
        tenant: object,
    ) -> None:
        """rollback() raises NotFoundError when deployment does not exist."""
        mock_deployment_repo.get_by_id.return_value = None

        service = DeploymentService(
            repository=mock_deployment_repo,
            publisher=mock_publisher,
        )

        with pytest.raises(NotFoundError):
            await service.rollback(
                deployment_id=uuid.uuid4(),
                reason="Testing not found path",
                tenant=tenant,
            )


class TestDeploymentServiceCanaryProgress:
    """Tests for DeploymentService.canary_progress."""

    async def test_canary_progress_increments_traffic(
        self,
        mock_deployment_repo: AsyncMock,
        mock_publisher: AsyncMock,
        sample_deployment: object,
        deployment_id: uuid.UUID,
        tenant: object,
    ) -> None:
        """canary_progress() increments canary traffic by step_percent."""
        sample_deployment.traffic_split = {"stable": 90, "canary": 10}
        mock_deployment_repo.get_by_id.return_value = sample_deployment

        progressed_deployment = sample_deployment
        progressed_deployment.traffic_split = {"stable": 80, "canary": 20}
        progressed_deployment.status = "in_progress"
        mock_deployment_repo.update_status.return_value = progressed_deployment

        service = DeploymentService(
            repository=mock_deployment_repo,
            publisher=mock_publisher,
        )

        result = await service.canary_progress(
            deployment_id=deployment_id,
            step_percent=10,
            tenant=tenant,
        )

        call_kwargs = mock_deployment_repo.update_status.call_args[1]
        assert call_kwargs["traffic_split"]["canary"] == 20
        assert call_kwargs["traffic_split"]["stable"] == 80

    async def test_canary_progress_marks_completed_at_100(
        self,
        mock_deployment_repo: AsyncMock,
        mock_publisher: AsyncMock,
        sample_deployment: object,
        deployment_id: uuid.UUID,
        tenant: object,
    ) -> None:
        """canary_progress() sets status=completed when canary reaches 100%."""
        sample_deployment.traffic_split = {"stable": 10, "canary": 90}
        mock_deployment_repo.get_by_id.return_value = sample_deployment

        completed_deployment = sample_deployment
        completed_deployment.status = "completed"
        mock_deployment_repo.update_status.return_value = completed_deployment

        service = DeploymentService(
            repository=mock_deployment_repo,
            publisher=mock_publisher,
        )

        await service.canary_progress(
            deployment_id=deployment_id,
            step_percent=10,
            tenant=tenant,
        )

        call_kwargs = mock_deployment_repo.update_status.call_args[1]
        assert call_kwargs["status"] == "completed"
        assert call_kwargs["traffic_split"]["canary"] == 100


# ---------------------------------------------------------------------------
# FeatureStoreService tests
# ---------------------------------------------------------------------------


class TestFeatureStoreServiceCreate:
    """Tests for FeatureStoreService.create_feature_set."""

    async def test_create_feature_set_registers_in_feast(
        self,
        mock_feature_set_repo: AsyncMock,
        mock_feast_client: AsyncMock,
        mock_publisher: AsyncMock,
        sample_feature_set: object,
        tenant: object,
    ) -> None:
        """create_feature_set() calls feast_client.register_feature_view."""
        mock_feature_set_repo.create.return_value = sample_feature_set

        service = FeatureStoreService(
            repository=mock_feature_set_repo,
            feast_client=mock_feast_client,
            publisher=mock_publisher,
        )

        request = FeatureSetCreateRequest(
            name="customer-rfm-features",
            entity_name="customer_id",
            features=[
                FeatureDefinition(name="recency_days", dtype="float32"),
                FeatureDefinition(name="frequency_count", dtype="int32"),
            ],
            source_type="batch",
            schedule="0 */6 * * *",
        )

        result = await service.create_feature_set(request=request, tenant=tenant)

        assert result is sample_feature_set
        mock_feast_client.register_feature_view.assert_called_once()
        mock_feature_set_repo.create.assert_called_once()

    async def test_get_feature_set_raises_not_found(
        self,
        mock_feature_set_repo: AsyncMock,
        mock_feast_client: AsyncMock,
        mock_publisher: AsyncMock,
        tenant: object,
    ) -> None:
        """get_feature_set() raises NotFoundError when feature set does not exist."""
        mock_feature_set_repo.get_by_id.return_value = None

        service = FeatureStoreService(
            repository=mock_feature_set_repo,
            feast_client=mock_feast_client,
            publisher=mock_publisher,
        )

        with pytest.raises(NotFoundError):
            await service.get_feature_set(feature_set_id=uuid.uuid4(), tenant=tenant)


# ---------------------------------------------------------------------------
# RetrainingService tests
# ---------------------------------------------------------------------------


class TestRetrainingServiceTrigger:
    """Tests for RetrainingService.trigger."""

    async def test_trigger_creates_job_and_publishes_event(
        self,
        mock_retraining_job_repo: AsyncMock,
        mock_publisher: AsyncMock,
        sample_retraining_job: object,
        model_id: uuid.UUID,
        tenant: object,
    ) -> None:
        """trigger() creates a pending job and publishes retraining_triggered event."""
        mock_retraining_job_repo.create.return_value = sample_retraining_job

        service = RetrainingService(
            repository=mock_retraining_job_repo,
            publisher=mock_publisher,
        )

        request = RetrainingJobCreateRequest(
            model_id=model_id,
            trigger_type="manual",
        )

        result = await service.trigger(request=request, tenant=tenant)

        assert result is sample_retraining_job
        mock_retraining_job_repo.create.assert_called_once_with(
            model_id=str(model_id),
            trigger_type="manual",
            tenant=tenant,
        )
        mock_publisher.publish_retraining_triggered.assert_called_once()

    async def test_trigger_with_drift_trigger_type(
        self,
        mock_retraining_job_repo: AsyncMock,
        mock_publisher: AsyncMock,
        sample_retraining_job: object,
        model_id: uuid.UUID,
        tenant: object,
    ) -> None:
        """trigger() correctly passes drift trigger_type to the repository."""
        mock_retraining_job_repo.create.return_value = sample_retraining_job

        service = RetrainingService(
            repository=mock_retraining_job_repo,
            publisher=mock_publisher,
        )

        request = RetrainingJobCreateRequest(
            model_id=model_id,
            trigger_type="drift",
        )

        await service.trigger(request=request, tenant=tenant)

        call_kwargs = mock_retraining_job_repo.create.call_args[1]
        assert call_kwargs["trigger_type"] == "drift"

    async def test_get_status_raises_not_found(
        self,
        mock_retraining_job_repo: AsyncMock,
        mock_publisher: AsyncMock,
        tenant: object,
    ) -> None:
        """get_status() raises NotFoundError when job does not exist."""
        mock_retraining_job_repo.get_by_id.return_value = None

        service = RetrainingService(
            repository=mock_retraining_job_repo,
            publisher=mock_publisher,
        )

        with pytest.raises(NotFoundError):
            await service.get_status(job_id=uuid.uuid4(), tenant=tenant)

    async def test_list_jobs_returns_paginated_response(
        self,
        mock_retraining_job_repo: AsyncMock,
        mock_publisher: AsyncMock,
        sample_retraining_job: object,
        tenant: object,
    ) -> None:
        """list_jobs() returns a RetrainingJobListResponse with correct pagination."""
        mock_retraining_job_repo.list_all.return_value = ([sample_retraining_job], 1)

        service = RetrainingService(
            repository=mock_retraining_job_repo,
            publisher=mock_publisher,
        )

        result = await service.list_jobs(tenant=tenant, page=1, page_size=10)

        assert result.total == 1
        assert result.page == 1
        assert result.page_size == 10
        assert len(result.items) == 1
