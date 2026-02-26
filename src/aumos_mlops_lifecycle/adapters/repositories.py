"""SQLAlchemy repository implementations for aumos-mlops-lifecycle.

Repositories extend BaseRepository from aumos-common which provides:
  - Automatic RLS tenant isolation (set_tenant_context)
  - Standard CRUD operations (get, list, create, update, delete)
  - Pagination support via paginate()

All repositories implement their corresponding Protocol interfaces from core.interfaces,
enabling easy mocking in tests and clean dependency injection in services.
"""

import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import TenantContext
from aumos_common.database import BaseRepository

from aumos_mlops_lifecycle.core.interfaces import (
    IDeploymentRepository,
    IExperimentRepository,
    IFeatureSetRepository,
    IRetrainingJobRepository,
)
from aumos_mlops_lifecycle.core.models import Deployment, Experiment, FeatureSet, RetrainingJob


class ExperimentRepository(BaseRepository, IExperimentRepository):
    """PostgreSQL repository for Experiment records.

    Extends BaseRepository for standard CRUD operations with RLS tenant isolation.

    Args:
        session: The async SQLAlchemy session (injected by FastAPI dependency).
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        super().__init__(session)

    async def get_by_id(self, experiment_id: uuid.UUID, tenant: TenantContext) -> Experiment | None:
        """Retrieve an experiment by ID within tenant scope.

        Args:
            experiment_id: UUID of the experiment to look up.
            tenant: Tenant context — RLS enforces isolation automatically.

        Returns:
            The Experiment if found, None otherwise.
        """
        result = await self.session.execute(
            select(Experiment).where(
                Experiment.id == experiment_id,
                Experiment.tenant_id == tenant.tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_all(
        self, tenant: TenantContext, page: int, page_size: int
    ) -> tuple[list[Experiment], int]:
        """List experiments for a tenant with pagination.

        Args:
            tenant: Tenant context — RLS enforces isolation automatically.
            page: Page number (1-indexed).
            page_size: Items per page.

        Returns:
            Tuple of (experiments list, total count).
        """
        offset = (page - 1) * page_size
        count_result = await self.session.execute(
            select(func.count()).select_from(Experiment).where(Experiment.tenant_id == tenant.tenant_id)
        )
        total = count_result.scalar_one()

        result = await self.session.execute(
            select(Experiment)
            .where(Experiment.tenant_id == tenant.tenant_id)
            .order_by(Experiment.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        return list(result.scalars().all()), total

    async def create(
        self,
        name: str,
        description: str | None,
        tags: dict[str, str],
        mlflow_experiment_id: str | None,
        tenant: TenantContext,
    ) -> Experiment:
        """Create a new experiment record.

        Args:
            name: Experiment name.
            description: Optional description.
            tags: Key-value tags.
            mlflow_experiment_id: Corresponding MLflow experiment ID.
            tenant: Tenant context for tenant_id assignment.

        Returns:
            The created and persisted Experiment record.
        """
        experiment = Experiment(
            tenant_id=tenant.tenant_id,
            name=name,
            description=description,
            tags=tags,
            mlflow_experiment_id=mlflow_experiment_id,
            status="active",
        )
        self.session.add(experiment)
        await self.session.flush()
        await self.session.refresh(experiment)
        return experiment


class DeploymentRepository(BaseRepository, IDeploymentRepository):
    """PostgreSQL repository for Deployment records.

    Extends BaseRepository for standard CRUD operations with RLS tenant isolation.

    Args:
        session: The async SQLAlchemy session (injected by FastAPI dependency).
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        super().__init__(session)

    async def get_by_id(self, deployment_id: uuid.UUID, tenant: TenantContext) -> Deployment | None:
        """Retrieve a deployment by ID within tenant scope.

        Args:
            deployment_id: UUID of the deployment.
            tenant: Tenant context — RLS enforces isolation automatically.

        Returns:
            The Deployment if found, None otherwise.
        """
        result = await self.session.execute(
            select(Deployment).where(
                Deployment.id == deployment_id,
                Deployment.tenant_id == tenant.tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_all(
        self, tenant: TenantContext, page: int, page_size: int
    ) -> tuple[list[Deployment], int]:
        """List deployments for a tenant with pagination.

        Args:
            tenant: Tenant context — RLS enforces isolation automatically.
            page: Page number (1-indexed).
            page_size: Items per page.

        Returns:
            Tuple of (deployments list, total count).
        """
        offset = (page - 1) * page_size
        count_result = await self.session.execute(
            select(func.count()).select_from(Deployment).where(Deployment.tenant_id == tenant.tenant_id)
        )
        total = count_result.scalar_one()

        result = await self.session.execute(
            select(Deployment)
            .where(Deployment.tenant_id == tenant.tenant_id)
            .order_by(Deployment.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        return list(result.scalars().all()), total

    async def create(
        self,
        model_id: str,
        model_version: str,
        strategy: str,
        target_environment: str,
        traffic_split: dict[str, int],
        health_check_url: str | None,
        tenant: TenantContext,
    ) -> Deployment:
        """Create a new deployment record in pending status.

        Args:
            model_id: UUID string of the model from aumos-model-registry.
            model_version: Model version to deploy.
            strategy: Deployment strategy (canary, ab, shadow, blue-green).
            target_environment: Target environment.
            traffic_split: Initial traffic percentage allocation.
            health_check_url: Optional health check URL.
            tenant: Tenant context for tenant_id assignment.

        Returns:
            The created and persisted Deployment record.
        """
        deployment = Deployment(
            tenant_id=tenant.tenant_id,
            model_id=model_id,
            model_version=model_version,
            strategy=strategy,
            status="pending",
            target_environment=target_environment,
            traffic_split=traffic_split,
            health_check_url=health_check_url,
        )
        self.session.add(deployment)
        await self.session.flush()
        await self.session.refresh(deployment)
        return deployment

    async def update_status(
        self,
        deployment_id: uuid.UUID,
        status: str,
        tenant: TenantContext,
        traffic_split: dict[str, int] | None = None,
    ) -> Deployment:
        """Update a deployment's status and optionally its traffic split.

        Args:
            deployment_id: UUID of the deployment to update.
            status: New status value.
            tenant: Tenant context — RLS enforces isolation automatically.
            traffic_split: Optional new traffic split to apply.

        Returns:
            The updated Deployment record.

        Raises:
            ValueError: If the deployment is not found (caller should have pre-validated).
        """
        result = await self.session.execute(
            select(Deployment).where(
                Deployment.id == deployment_id,
                Deployment.tenant_id == tenant.tenant_id,
            )
        )
        deployment = result.scalar_one_or_none()
        if deployment is None:
            msg = f"Deployment {deployment_id} not found during status update"
            raise ValueError(msg)

        deployment.status = status
        if traffic_split is not None:
            deployment.traffic_split = traffic_split

        await self.session.flush()
        await self.session.refresh(deployment)
        return deployment


class FeatureSetRepository(BaseRepository, IFeatureSetRepository):
    """PostgreSQL repository for FeatureSet records.

    Extends BaseRepository for standard CRUD operations with RLS tenant isolation.

    Args:
        session: The async SQLAlchemy session (injected by FastAPI dependency).
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        super().__init__(session)

    async def get_by_id(self, feature_set_id: uuid.UUID, tenant: TenantContext) -> FeatureSet | None:
        """Retrieve a feature set by ID within tenant scope.

        Args:
            feature_set_id: UUID of the feature set.
            tenant: Tenant context — RLS enforces isolation automatically.

        Returns:
            The FeatureSet if found, None otherwise.
        """
        result = await self.session.execute(
            select(FeatureSet).where(
                FeatureSet.id == feature_set_id,
                FeatureSet.tenant_id == tenant.tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_all(
        self, tenant: TenantContext, page: int, page_size: int
    ) -> tuple[list[FeatureSet], int]:
        """List feature sets for a tenant with pagination.

        Args:
            tenant: Tenant context — RLS enforces isolation automatically.
            page: Page number (1-indexed).
            page_size: Items per page.

        Returns:
            Tuple of (feature sets list, total count).
        """
        offset = (page - 1) * page_size
        count_result = await self.session.execute(
            select(func.count()).select_from(FeatureSet).where(FeatureSet.tenant_id == tenant.tenant_id)
        )
        total = count_result.scalar_one()

        result = await self.session.execute(
            select(FeatureSet)
            .where(FeatureSet.tenant_id == tenant.tenant_id)
            .order_by(FeatureSet.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        return list(result.scalars().all()), total

    async def create(
        self,
        name: str,
        entity_name: str,
        features: list[dict[str, Any]],
        source_type: str,
        schedule: str | None,
        tenant: TenantContext,
    ) -> FeatureSet:
        """Create a new feature set record.

        Args:
            name: Feature set name.
            entity_name: Associated entity name.
            features: List of feature definitions as dicts.
            source_type: Data source type (batch, stream, request).
            schedule: Optional cron materialization schedule.
            tenant: Tenant context for tenant_id assignment.

        Returns:
            The created and persisted FeatureSet record.
        """
        feature_set = FeatureSet(
            tenant_id=tenant.tenant_id,
            name=name,
            entity_name=entity_name,
            features=features,
            source_type=source_type,
            schedule=schedule,
        )
        self.session.add(feature_set)
        await self.session.flush()
        await self.session.refresh(feature_set)
        return feature_set


class RetrainingJobRepository(BaseRepository, IRetrainingJobRepository):
    """PostgreSQL repository for RetrainingJob records.

    Extends BaseRepository for standard CRUD operations with RLS tenant isolation.

    Args:
        session: The async SQLAlchemy session (injected by FastAPI dependency).
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        super().__init__(session)

    async def get_by_id(self, job_id: uuid.UUID, tenant: TenantContext) -> RetrainingJob | None:
        """Retrieve a retraining job by ID within tenant scope.

        Args:
            job_id: UUID of the retraining job.
            tenant: Tenant context — RLS enforces isolation automatically.

        Returns:
            The RetrainingJob if found, None otherwise.
        """
        result = await self.session.execute(
            select(RetrainingJob).where(
                RetrainingJob.id == job_id,
                RetrainingJob.tenant_id == tenant.tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_all(
        self, tenant: TenantContext, page: int, page_size: int
    ) -> tuple[list[RetrainingJob], int]:
        """List retraining jobs for a tenant with pagination.

        Args:
            tenant: Tenant context — RLS enforces isolation automatically.
            page: Page number (1-indexed).
            page_size: Items per page.

        Returns:
            Tuple of (jobs list, total count).
        """
        offset = (page - 1) * page_size
        count_result = await self.session.execute(
            select(func.count()).select_from(RetrainingJob).where(RetrainingJob.tenant_id == tenant.tenant_id)
        )
        total = count_result.scalar_one()

        result = await self.session.execute(
            select(RetrainingJob)
            .where(RetrainingJob.tenant_id == tenant.tenant_id)
            .order_by(RetrainingJob.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        return list(result.scalars().all()), total

    async def create(
        self,
        model_id: str,
        trigger_type: str,
        tenant: TenantContext,
    ) -> RetrainingJob:
        """Create a new retraining job record in pending status.

        Args:
            model_id: UUID string of the model to retrain.
            trigger_type: What triggered retraining (drift, scheduled, manual).
            tenant: Tenant context for tenant_id assignment.

        Returns:
            The created and persisted RetrainingJob in pending status.
        """
        job = RetrainingJob(
            tenant_id=tenant.tenant_id,
            model_id=model_id,
            trigger_type=trigger_type,
            status="pending",
            metrics={},
        )
        self.session.add(job)
        await self.session.flush()
        await self.session.refresh(job)
        return job

    async def count_running_for_tenant(self, tenant: TenantContext) -> int:
        """Count active retraining jobs (pending + running) for a tenant.

        Used to enforce max_concurrent_retraining_jobs limit.

        Args:
            tenant: Tenant context — RLS enforces isolation automatically.

        Returns:
            Number of jobs with status 'pending' or 'running'.
        """
        result = await self.session.execute(
            select(func.count())
            .select_from(RetrainingJob)
            .where(
                RetrainingJob.tenant_id == tenant.tenant_id,
                RetrainingJob.status.in_(["pending", "running"]),
            )
        )
        return result.scalar_one()
