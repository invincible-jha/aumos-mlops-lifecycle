"""Tests for repository implementations in aumos-mlops-lifecycle.

Tests verify repository query logic using mocked SQLAlchemy sessions.
Integration tests with real PostgreSQL use testcontainers (marked separately).
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aumos_mlops_lifecycle.adapters.repositories import (
    DeploymentRepository,
    ExperimentRepository,
    FeatureSetRepository,
    RetrainingJobRepository,
)
from aumos_mlops_lifecycle.core.models import Deployment, Experiment, FeatureSet, RetrainingJob


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock_session() -> MagicMock:
    """Create a mock async SQLAlchemy session."""
    session = MagicMock()
    session.execute = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.add = MagicMock()
    return session


def make_scalar_result(value: object) -> MagicMock:
    """Create a mock scalar result that returns the given value."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    result.scalar_one.return_value = value
    result.scalars.return_value.all.return_value = [value] if value is not None else []
    return result


# ---------------------------------------------------------------------------
# ExperimentRepository tests
# ---------------------------------------------------------------------------


class TestExperimentRepository:
    """Tests for ExperimentRepository."""

    async def test_get_by_id_returns_experiment_when_found(
        self,
        sample_experiment: Experiment,
        tenant: MagicMock,
        experiment_id: uuid.UUID,
    ) -> None:
        """get_by_id() returns the experiment when the database finds it."""
        session = make_mock_session()
        session.execute.return_value = make_scalar_result(sample_experiment)

        repo = ExperimentRepository(session=session)
        result = await repo.get_by_id(experiment_id=experiment_id, tenant=tenant)

        assert result is sample_experiment

    async def test_get_by_id_returns_none_when_not_found(
        self,
        tenant: MagicMock,
    ) -> None:
        """get_by_id() returns None when the experiment is not in the database."""
        session = make_mock_session()
        session.execute.return_value = make_scalar_result(None)

        repo = ExperimentRepository(session=session)
        result = await repo.get_by_id(experiment_id=uuid.uuid4(), tenant=tenant)

        assert result is None

    async def test_create_adds_experiment_to_session(
        self,
        sample_experiment: Experiment,
        tenant: MagicMock,
    ) -> None:
        """create() adds the Experiment to the session and flushes."""
        session = make_mock_session()
        session.refresh = AsyncMock(side_effect=lambda obj: None)

        repo = ExperimentRepository(session=session)

        with patch.object(Experiment, "__init__", return_value=None):
            experiment = Experiment.__new__(Experiment)
            experiment.id = sample_experiment.id
            experiment.tenant_id = tenant.tenant_id
            experiment.name = "test-experiment"
            experiment.description = None
            experiment.tags = {}
            experiment.mlflow_experiment_id = "42"
            experiment.status = "active"

            # Mock the session.add to capture the added object
            added_objects = []
            session.add.side_effect = lambda obj: added_objects.append(obj)

            # Patch the Experiment constructor call inside create
            with patch("aumos_mlops_lifecycle.adapters.repositories.Experiment") as mock_cls:
                mock_cls.return_value = experiment
                session.refresh.side_effect = lambda obj: None

                result = await repo.create(
                    name="test-experiment",
                    description=None,
                    tags={},
                    mlflow_experiment_id="42",
                    tenant=tenant,
                )

        session.flush.assert_called_once()


# ---------------------------------------------------------------------------
# DeploymentRepository tests
# ---------------------------------------------------------------------------


class TestDeploymentRepository:
    """Tests for DeploymentRepository."""

    async def test_get_by_id_returns_deployment_when_found(
        self,
        sample_deployment: Deployment,
        tenant: MagicMock,
        deployment_id: uuid.UUID,
    ) -> None:
        """get_by_id() returns the deployment when found in the database."""
        session = make_mock_session()
        session.execute.return_value = make_scalar_result(sample_deployment)

        repo = DeploymentRepository(session=session)
        result = await repo.get_by_id(deployment_id=deployment_id, tenant=tenant)

        assert result is sample_deployment

    async def test_get_by_id_returns_none_when_not_found(
        self,
        tenant: MagicMock,
    ) -> None:
        """get_by_id() returns None when deployment is not in the database."""
        session = make_mock_session()
        session.execute.return_value = make_scalar_result(None)

        repo = DeploymentRepository(session=session)
        result = await repo.get_by_id(deployment_id=uuid.uuid4(), tenant=tenant)

        assert result is None

    async def test_update_status_raises_value_error_when_not_found(
        self,
        tenant: MagicMock,
    ) -> None:
        """update_status() raises ValueError when deployment is not found."""
        session = make_mock_session()
        session.execute.return_value = make_scalar_result(None)

        repo = DeploymentRepository(session=session)

        with pytest.raises(ValueError, match="not found during status update"):
            await repo.update_status(
                deployment_id=uuid.uuid4(),
                status="rolled_back",
                tenant=tenant,
            )


# ---------------------------------------------------------------------------
# FeatureSetRepository tests
# ---------------------------------------------------------------------------


class TestFeatureSetRepository:
    """Tests for FeatureSetRepository."""

    async def test_get_by_id_returns_feature_set_when_found(
        self,
        sample_feature_set: FeatureSet,
        tenant: MagicMock,
        feature_set_id: uuid.UUID,
    ) -> None:
        """get_by_id() returns the feature set when found in the database."""
        session = make_mock_session()
        session.execute.return_value = make_scalar_result(sample_feature_set)

        repo = FeatureSetRepository(session=session)
        result = await repo.get_by_id(feature_set_id=feature_set_id, tenant=tenant)

        assert result is sample_feature_set


# ---------------------------------------------------------------------------
# RetrainingJobRepository tests
# ---------------------------------------------------------------------------


class TestRetrainingJobRepository:
    """Tests for RetrainingJobRepository."""

    async def test_get_by_id_returns_job_when_found(
        self,
        sample_retraining_job: RetrainingJob,
        tenant: MagicMock,
        job_id: uuid.UUID,
    ) -> None:
        """get_by_id() returns the retraining job when found in the database."""
        session = make_mock_session()
        session.execute.return_value = make_scalar_result(sample_retraining_job)

        repo = RetrainingJobRepository(session=session)
        result = await repo.get_by_id(job_id=job_id, tenant=tenant)

        assert result is sample_retraining_job

    async def test_count_running_returns_count(
        self,
        tenant: MagicMock,
    ) -> None:
        """count_running_for_tenant() returns the count from the database."""
        session = make_mock_session()
        count_result = MagicMock()
        count_result.scalar_one.return_value = 3
        session.execute.return_value = count_result

        repo = RetrainingJobRepository(session=session)
        count = await repo.count_running_for_tenant(tenant=tenant)

        assert count == 3
