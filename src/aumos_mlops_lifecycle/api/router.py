"""API router for aumos-mlops-lifecycle.

All endpoints are registered here and included in main.py under /api/v1.
Routes delegate all logic to the service layer — no business logic in routes.

Endpoint groups:
  - /experiments — MLflow-backed experiment management
  - /deployments — Model deployment lifecycle (canary, A/B, shadow, blue-green)
  - /feature-sets — Feast feature store management
  - /retraining-jobs — Automated and manual model retraining
"""

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import TenantContext, get_current_user
from aumos_common.database import get_db_session

from aumos_mlops_lifecycle.adapters.feast_client import FeastClient
from aumos_mlops_lifecycle.adapters.kafka import MLOpsEventPublisher
from aumos_mlops_lifecycle.adapters.mlflow_client import MLflowClient
from aumos_mlops_lifecycle.adapters.repositories import (
    DeploymentRepository,
    ExperimentRepository,
    FeatureSetRepository,
    RetrainingJobRepository,
)
from aumos_mlops_lifecycle.api.schemas import (
    DeploymentCreateRequest,
    DeploymentListResponse,
    DeploymentResponse,
    ExperimentCreateRequest,
    ExperimentListResponse,
    ExperimentResponse,
    FeatureSetCreateRequest,
    FeatureSetListResponse,
    FeatureSetResponse,
    RetrainingJobCreateRequest,
    RetrainingJobListResponse,
    RetrainingJobResponse,
    RollbackRequest,
    RollbackResponse,
    RunListResponse,
    RunLogRequest,
    RunResponse,
)
from aumos_mlops_lifecycle.core.services import (
    DeploymentService,
    ExperimentService,
    FeatureStoreService,
    RetrainingService,
)

router = APIRouter(tags=["mlops-lifecycle"])


# ---------------------------------------------------------------------------
# Dependency factories
# ---------------------------------------------------------------------------


def get_experiment_service(
    session: AsyncSession = Depends(get_db_session),
) -> ExperimentService:
    """Construct ExperimentService with injected dependencies.

    Args:
        session: Async database session from FastAPI dependency injection.

    Returns:
        Configured ExperimentService instance.
    """
    return ExperimentService(
        repository=ExperimentRepository(session),
        mlflow_client=MLflowClient(),
        publisher=MLOpsEventPublisher(),
    )


def get_deployment_service(
    session: AsyncSession = Depends(get_db_session),
) -> DeploymentService:
    """Construct DeploymentService with injected dependencies.

    Args:
        session: Async database session from FastAPI dependency injection.

    Returns:
        Configured DeploymentService instance.
    """
    return DeploymentService(
        repository=DeploymentRepository(session),
        publisher=MLOpsEventPublisher(),
    )


def get_feature_store_service(
    session: AsyncSession = Depends(get_db_session),
) -> FeatureStoreService:
    """Construct FeatureStoreService with injected dependencies.

    Args:
        session: Async database session from FastAPI dependency injection.

    Returns:
        Configured FeatureStoreService instance.
    """
    return FeatureStoreService(
        repository=FeatureSetRepository(session),
        feast_client=FeastClient(),
        publisher=MLOpsEventPublisher(),
    )


def get_retraining_service(
    session: AsyncSession = Depends(get_db_session),
) -> RetrainingService:
    """Construct RetrainingService with injected dependencies.

    Args:
        session: Async database session from FastAPI dependency injection.

    Returns:
        Configured RetrainingService instance.
    """
    return RetrainingService(
        repository=RetrainingJobRepository(session),
        publisher=MLOpsEventPublisher(),
    )


# ---------------------------------------------------------------------------
# Experiment endpoints
# ---------------------------------------------------------------------------


@router.post("/experiments", response_model=ExperimentResponse, status_code=201)
async def create_experiment(
    request: ExperimentCreateRequest,
    tenant: TenantContext = Depends(get_current_user),
    service: ExperimentService = Depends(get_experiment_service),
) -> ExperimentResponse:
    """Create a new experiment with tenant-isolated MLflow tracking.

    Args:
        request: Experiment creation parameters including name, description, and tags.
        tenant: Authenticated tenant context from JWT middleware.
        service: Injected ExperimentService.

    Returns:
        The created experiment with its MLflow experiment ID.
    """
    experiment = await service.create(
        name=request.name,
        description=request.description,
        tags=request.tags,
        tenant=tenant,
    )
    return ExperimentResponse.model_validate(experiment)


@router.get("/experiments", response_model=ExperimentListResponse)
async def list_experiments(
    page: int = 1,
    page_size: int = 20,
    tenant: TenantContext = Depends(get_current_user),
    service: ExperimentService = Depends(get_experiment_service),
) -> ExperimentListResponse:
    """List all experiments for the current tenant.

    Args:
        page: Page number (1-indexed).
        page_size: Number of experiments per page.
        tenant: Authenticated tenant context from JWT middleware.
        service: Injected ExperimentService.

    Returns:
        Paginated list of experiments.
    """
    return await service.list_all(tenant=tenant, page=page, page_size=page_size)


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: uuid.UUID,
    tenant: TenantContext = Depends(get_current_user),
    service: ExperimentService = Depends(get_experiment_service),
) -> ExperimentResponse:
    """Get a single experiment by ID.

    Args:
        experiment_id: UUID of the experiment to retrieve.
        tenant: Authenticated tenant context from JWT middleware.
        service: Injected ExperimentService.

    Returns:
        The experiment if found.

    Raises:
        NotFoundError: If the experiment does not exist or belongs to a different tenant.
    """
    experiment = await service.get_by_id(experiment_id=experiment_id, tenant=tenant)
    return ExperimentResponse.model_validate(experiment)


@router.post("/experiments/{experiment_id}/runs", response_model=RunResponse, status_code=201)
async def log_run(
    experiment_id: uuid.UUID,
    request: RunLogRequest,
    tenant: TenantContext = Depends(get_current_user),
    service: ExperimentService = Depends(get_experiment_service),
) -> RunResponse:
    """Log a run (metrics, params, tags) to an existing experiment.

    Args:
        experiment_id: UUID of the parent experiment.
        request: Run data including metrics, hyperparameters, and tags.
        tenant: Authenticated tenant context from JWT middleware.
        service: Injected ExperimentService.

    Returns:
        The created run with its MLflow run ID.

    Raises:
        NotFoundError: If the experiment does not exist or belongs to a different tenant.
    """
    return await service.log_run(
        experiment_id=experiment_id,
        request=request,
        tenant=tenant,
    )


@router.get("/experiments/{experiment_id}/runs", response_model=RunListResponse)
async def list_runs(
    experiment_id: uuid.UUID,
    page: int = 1,
    page_size: int = 20,
    tenant: TenantContext = Depends(get_current_user),
    service: ExperimentService = Depends(get_experiment_service),
) -> RunListResponse:
    """List all runs for an experiment.

    Args:
        experiment_id: UUID of the parent experiment.
        page: Page number (1-indexed).
        page_size: Number of runs per page.
        tenant: Authenticated tenant context from JWT middleware.
        service: Injected ExperimentService.

    Returns:
        Paginated list of runs for the experiment.
    """
    return await service.list_runs(
        experiment_id=experiment_id,
        tenant=tenant,
        page=page,
        page_size=page_size,
    )


# ---------------------------------------------------------------------------
# Deployment endpoints
# ---------------------------------------------------------------------------


@router.post("/deployments", response_model=DeploymentResponse, status_code=201)
async def create_deployment(
    request: DeploymentCreateRequest,
    tenant: TenantContext = Depends(get_current_user),
    service: DeploymentService = Depends(get_deployment_service),
) -> DeploymentResponse:
    """Create and initiate a model deployment.

    Supports canary, A/B testing, shadow, and blue-green deployment strategies.
    Publishes a MLO_DEPLOYMENT_CREATED Kafka event on success.

    Args:
        request: Deployment parameters including model ID, version, strategy, and traffic split.
        tenant: Authenticated tenant context from JWT middleware.
        service: Injected DeploymentService.

    Returns:
        The created deployment record with its initial status.
    """
    deployment = await service.deploy(request=request, tenant=tenant)
    return DeploymentResponse.model_validate(deployment)


@router.get("/deployments", response_model=DeploymentListResponse)
async def list_deployments(
    page: int = 1,
    page_size: int = 20,
    tenant: TenantContext = Depends(get_current_user),
    service: DeploymentService = Depends(get_deployment_service),
) -> DeploymentListResponse:
    """List all deployments for the current tenant.

    Args:
        page: Page number (1-indexed).
        page_size: Number of deployments per page.
        tenant: Authenticated tenant context from JWT middleware.
        service: Injected DeploymentService.

    Returns:
        Paginated list of deployments.
    """
    return await service.list_all(tenant=tenant, page=page, page_size=page_size)


@router.get("/deployments/{deployment_id}", response_model=DeploymentResponse)
async def get_deployment(
    deployment_id: uuid.UUID,
    tenant: TenantContext = Depends(get_current_user),
    service: DeploymentService = Depends(get_deployment_service),
) -> DeploymentResponse:
    """Get the current status of a deployment.

    Args:
        deployment_id: UUID of the deployment to retrieve.
        tenant: Authenticated tenant context from JWT middleware.
        service: Injected DeploymentService.

    Returns:
        The deployment with its current status and traffic split.

    Raises:
        NotFoundError: If the deployment does not exist or belongs to a different tenant.
    """
    deployment = await service.get_status(deployment_id=deployment_id, tenant=tenant)
    return DeploymentResponse.model_validate(deployment)


@router.post("/deployments/{deployment_id}/rollback", response_model=RollbackResponse)
async def rollback_deployment(
    deployment_id: uuid.UUID,
    request: RollbackRequest,
    tenant: TenantContext = Depends(get_current_user),
    service: DeploymentService = Depends(get_deployment_service),
) -> RollbackResponse:
    """Roll back a deployment to the previous stable version.

    Immediately halts traffic to the new version and restores the stable version
    to 100% traffic. Publishes a MLO_DEPLOYMENT_ROLLED_BACK Kafka event.

    Args:
        deployment_id: UUID of the deployment to roll back.
        request: Rollback parameters including the reason for rollback.
        tenant: Authenticated tenant context from JWT middleware.
        service: Injected DeploymentService.

    Returns:
        Rollback confirmation with the new deployment status.

    Raises:
        NotFoundError: If the deployment does not exist or belongs to a different tenant.
    """
    return await service.rollback(deployment_id=deployment_id, reason=request.reason, tenant=tenant)


# ---------------------------------------------------------------------------
# Feature set endpoints
# ---------------------------------------------------------------------------


@router.post("/feature-sets", response_model=FeatureSetResponse, status_code=201)
async def create_feature_set(
    request: FeatureSetCreateRequest,
    tenant: TenantContext = Depends(get_current_user),
    service: FeatureStoreService = Depends(get_feature_store_service),
) -> FeatureSetResponse:
    """Create a new feature set and register it in Feast.

    Args:
        request: Feature set parameters including entity, features, and materialization schedule.
        tenant: Authenticated tenant context from JWT middleware.
        service: Injected FeatureStoreService.

    Returns:
        The created feature set with its Feast registration details.
    """
    feature_set = await service.create_feature_set(request=request, tenant=tenant)
    return FeatureSetResponse.model_validate(feature_set)


@router.get("/feature-sets", response_model=FeatureSetListResponse)
async def list_feature_sets(
    page: int = 1,
    page_size: int = 20,
    tenant: TenantContext = Depends(get_current_user),
    service: FeatureStoreService = Depends(get_feature_store_service),
) -> FeatureSetListResponse:
    """List all feature sets for the current tenant.

    Args:
        page: Page number (1-indexed).
        page_size: Number of feature sets per page.
        tenant: Authenticated tenant context from JWT middleware.
        service: Injected FeatureStoreService.

    Returns:
        Paginated list of feature sets.
    """
    return await service.list_feature_sets(tenant=tenant, page=page, page_size=page_size)


@router.get("/feature-sets/{feature_set_id}", response_model=FeatureSetResponse)
async def get_feature_set(
    feature_set_id: uuid.UUID,
    tenant: TenantContext = Depends(get_current_user),
    service: FeatureStoreService = Depends(get_feature_store_service),
) -> FeatureSetResponse:
    """Get a single feature set by ID.

    Args:
        feature_set_id: UUID of the feature set to retrieve.
        tenant: Authenticated tenant context from JWT middleware.
        service: Injected FeatureStoreService.

    Returns:
        The feature set if found.

    Raises:
        NotFoundError: If the feature set does not exist or belongs to a different tenant.
    """
    feature_set = await service.get_feature_set(feature_set_id=feature_set_id, tenant=tenant)
    return FeatureSetResponse.model_validate(feature_set)


# ---------------------------------------------------------------------------
# Retraining job endpoints
# ---------------------------------------------------------------------------


@router.post("/retraining-jobs", response_model=RetrainingJobResponse, status_code=201)
async def trigger_retraining(
    request: RetrainingJobCreateRequest,
    tenant: TenantContext = Depends(get_current_user),
    service: RetrainingService = Depends(get_retraining_service),
) -> RetrainingJobResponse:
    """Trigger a model retraining job.

    Creates a retraining job record and publishes a MLO_RETRAINING_TRIGGERED Kafka event.
    The actual retraining is performed asynchronously by a downstream consumer.

    Args:
        request: Retraining parameters including model ID and trigger type.
        tenant: Authenticated tenant context from JWT middleware.
        service: Injected RetrainingService.

    Returns:
        The created retraining job with its initial pending status.
    """
    job = await service.trigger(request=request, tenant=tenant)
    return RetrainingJobResponse.model_validate(job)


@router.get("/retraining-jobs", response_model=RetrainingJobListResponse)
async def list_retraining_jobs(
    page: int = 1,
    page_size: int = 20,
    tenant: TenantContext = Depends(get_current_user),
    service: RetrainingService = Depends(get_retraining_service),
) -> RetrainingJobListResponse:
    """List all retraining jobs for the current tenant.

    Args:
        page: Page number (1-indexed).
        page_size: Number of jobs per page.
        tenant: Authenticated tenant context from JWT middleware.
        service: Injected RetrainingService.

    Returns:
        Paginated list of retraining jobs ordered by creation time descending.
    """
    return await service.list_jobs(tenant=tenant, page=page, page_size=page_size)


@router.get("/retraining-jobs/{job_id}", response_model=RetrainingJobResponse)
async def get_retraining_job(
    job_id: uuid.UUID,
    tenant: TenantContext = Depends(get_current_user),
    service: RetrainingService = Depends(get_retraining_service),
) -> RetrainingJobResponse:
    """Get the status and outcome metrics of a retraining job.

    Args:
        job_id: UUID of the retraining job to retrieve.
        tenant: Authenticated tenant context from JWT middleware.
        service: Injected RetrainingService.

    Returns:
        The retraining job with its current status and any completed metrics.

    Raises:
        NotFoundError: If the job does not exist or belongs to a different tenant.
    """
    job = await service.get_status(job_id=job_id, tenant=tenant)
    return RetrainingJobResponse.model_validate(job)
