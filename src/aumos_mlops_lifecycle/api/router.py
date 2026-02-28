"""API router for aumos-mlops-lifecycle.

All endpoints are registered here and included in main.py under /api/v1.
Routes delegate all logic to the service layer — no business logic in routes.

Endpoint groups:
  - /experiments — MLflow-backed experiment management
  - /deployments — Model deployment lifecycle (canary, A/B, shadow, blue-green)
  - /feature-sets — Feast feature store management
  - /retraining-jobs — Automated and manual model retraining
  - /experiments/{id}/artifacts — MinIO/S3 artifact storage (GAP-158)
  - /model-cards — Model card generation (GAP-160)
  - /pipeline-dag — Pipeline DAG visualization (GAP-163)
"""

import uuid

from fastapi import APIRouter, Depends, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import TenantContext, get_current_user
from aumos_common.database import get_db_session

from aumos_mlops_lifecycle.adapters.artifact_store import ArtifactStore
from aumos_mlops_lifecycle.adapters.feast_client import FeastClient
from aumos_mlops_lifecycle.adapters.kafka import MLOpsEventPublisher
from aumos_mlops_lifecycle.adapters.mlflow_client import MLflowClient
from aumos_mlops_lifecycle.adapters.repositories import (
    DeploymentRepository,
    ExperimentRepository,
    FeatureSetRepository,
    RetrainingJobRepository,
)
from aumos_mlops_lifecycle.adapters.testing_harness_client import TestingHarnessClient
from aumos_mlops_lifecycle.api.schemas import (
    ArtifactDownloadUrlResponse,
    ArtifactListResponse,
    ArtifactResponse,
    DAGResponse,
    DeploymentCreateRequest,
    DeploymentListResponse,
    DeploymentResponse,
    ExperimentCreateRequest,
    ExperimentListResponse,
    ExperimentResponse,
    FeatureSetCreateRequest,
    FeatureSetListResponse,
    FeatureSetResponse,
    ModelCardResponse,
    RetrainingJobCreateRequest,
    RetrainingJobListResponse,
    RetrainingJobResponse,
    RollbackRequest,
    RollbackResponse,
    RunListResponse,
    RunLogRequest,
    RunResponse,
)
from aumos_mlops_lifecycle.core.dag_builder import PipelineDAGBuilder
from aumos_mlops_lifecycle.core.model_card import ModelCardService
from aumos_mlops_lifecycle.core.services import (
    DeploymentService,
    ExperimentService,
    FeatureStoreService,
    RetrainingService,
)
from aumos_mlops_lifecycle.settings import get_settings

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


# ---------------------------------------------------------------------------
# Artifact endpoints (GAP-158: Artifact Storage)
# ---------------------------------------------------------------------------


def get_artifact_store() -> ArtifactStore:
    """Construct ArtifactStore from settings.

    Returns:
        Configured ArtifactStore instance targeting MinIO/S3.
    """
    settings = get_settings()
    return ArtifactStore(
        bucket_name=getattr(settings, "artifact_bucket_name", "aumos-artifacts"),
        endpoint_url=getattr(settings, "artifact_endpoint_url", None),
        access_key=getattr(settings, "artifact_access_key", None),
        secret_key=getattr(settings, "artifact_secret_key", None),
    )


@router.post(
    "/experiments/{experiment_id}/artifacts",
    response_model=ArtifactResponse,
    status_code=201,
)
async def upload_artifact(
    experiment_id: uuid.UUID,
    file: UploadFile,
    tenant: TenantContext = Depends(get_current_user),
    store: ArtifactStore = Depends(get_artifact_store),
) -> ArtifactResponse:
    """Upload a file artifact for an experiment to MinIO/S3.

    Stores the artifact under artifacts/{tenant_id}/{experiment_id}/{filename}
    and returns the S3 URI for future reference.

    Args:
        experiment_id: UUID of the parent experiment.
        file: Multipart file upload.
        tenant: Authenticated tenant context from JWT middleware.
        store: Injected ArtifactStore.

    Returns:
        ArtifactResponse with the storage URI and metadata.
    """
    content = await file.read()
    content_type = file.content_type or "application/octet-stream"
    filename = file.filename or "artifact"
    artifact_uri = await store.upload(
        tenant_id=str(tenant.tenant_id),
        experiment_id=str(experiment_id),
        filename=filename,
        content=content,
        content_type=content_type,
    )
    return ArtifactResponse(
        experiment_id=experiment_id,
        artifact_name=filename,
        artifact_uri=artifact_uri,
        size_bytes=len(content),
        content_type=content_type,
    )


@router.get("/experiments/{experiment_id}/artifacts", response_model=ArtifactListResponse)
async def list_artifacts(
    experiment_id: uuid.UUID,
    tenant: TenantContext = Depends(get_current_user),
    store: ArtifactStore = Depends(get_artifact_store),
) -> ArtifactListResponse:
    """List all artifacts stored for an experiment.

    Args:
        experiment_id: UUID of the parent experiment.
        tenant: Authenticated tenant context from JWT middleware.
        store: Injected ArtifactStore.

    Returns:
        ArtifactListResponse with all artifacts for this experiment.
    """
    raw_artifacts = await store.list_artifacts(
        tenant_id=str(tenant.tenant_id),
        experiment_id=str(experiment_id),
    )
    artifacts = [
        ArtifactResponse(
            experiment_id=experiment_id,
            artifact_name=a.get("name", ""),
            artifact_uri=a.get("uri", ""),
            size_bytes=int(a.get("size_bytes", 0)),
            content_type=str(a.get("content_type", "application/octet-stream")),
        )
        for a in raw_artifacts
    ]
    return ArtifactListResponse(
        experiment_id=experiment_id,
        artifacts=artifacts,
        total=len(artifacts),
    )


@router.get(
    "/experiments/{experiment_id}/artifacts/{filename}/download-url",
    response_model=ArtifactDownloadUrlResponse,
)
async def get_artifact_download_url(
    experiment_id: uuid.UUID,
    filename: str,
    tenant: TenantContext = Depends(get_current_user),
    store: ArtifactStore = Depends(get_artifact_store),
) -> ArtifactDownloadUrlResponse:
    """Generate a presigned download URL for an artifact.

    The URL expires after 3600 seconds (1 hour). Clients should download
    the artifact immediately rather than caching the URL.

    Args:
        experiment_id: UUID of the parent experiment.
        filename: Name of the artifact file to download.
        tenant: Authenticated tenant context from JWT middleware.
        store: Injected ArtifactStore.

    Returns:
        ArtifactDownloadUrlResponse with the presigned URL and expiry.
    """
    download_url = await store.generate_presigned_url(
        tenant_id=str(tenant.tenant_id),
        experiment_id=str(experiment_id),
        filename=filename,
    )
    return ArtifactDownloadUrlResponse(
        artifact_name=filename,
        download_url=download_url,
        expires_in_seconds=3600,
    )


# ---------------------------------------------------------------------------
# Model card endpoints (GAP-160: Model Card Generation)
# ---------------------------------------------------------------------------


def get_model_card_service(
    session: AsyncSession = Depends(get_db_session),
) -> ModelCardService:
    """Construct ModelCardService with injected dependencies.

    Args:
        session: Async database session from FastAPI dependency injection.

    Returns:
        Configured ModelCardService instance.
    """
    settings = get_settings()
    testing_client = TestingHarnessClient(
        base_url=getattr(settings, "testing_harness_url", "http://aumos-testing-harness:8000"),
        api_key=getattr(settings, "internal_api_key", ""),
    )
    return ModelCardService(
        experiment_repo=ExperimentRepository(session),
        deployment_repo=DeploymentRepository(session),
        testing_harness_client=testing_client,
    )


@router.get("/model-cards/{model_id}", response_model=ModelCardResponse)
async def get_model_card(
    model_id: str,
    format: str = "json",
    tenant: TenantContext = Depends(get_current_user),
    service: ModelCardService = Depends(get_model_card_service),
    session: AsyncSession = Depends(get_db_session),
) -> ModelCardResponse:
    """Generate and return a model card for the specified model.

    Assembles metadata from experiments, evaluation results (via aumos-testing-harness),
    and deployment history to produce a HuggingFace-compatible model card.

    Args:
        model_id: UUID of the model to generate a card for.
        format: Response format — 'json' (default) or 'markdown'.
        tenant: Authenticated tenant context from JWT middleware.
        service: Injected ModelCardService.
        session: Async database session.

    Returns:
        ModelCardResponse with all model metadata and evaluation results.
    """
    card = await service.generate(
        model_id=model_id,
        tenant_id=str(tenant.tenant_id),
        session=session,
    )
    return ModelCardResponse(
        model_id=card.model_id,
        model_name=card.model_name,
        description=card.description,
        version=card.version,
        training_data=card.training_data,
        framework=card.framework,
        metrics=card.metrics,
        evaluation_results=card.evaluation_results,
        deployment_environments=card.deployment_environments,
        known_limitations=card.known_limitations,
        generated_at=card.generated_at,
    )


# ---------------------------------------------------------------------------
# Pipeline DAG endpoints (GAP-163: Pipeline DAG Visualization)
# ---------------------------------------------------------------------------


def get_dag_builder(
    session: AsyncSession = Depends(get_db_session),
) -> PipelineDAGBuilder:
    """Construct PipelineDAGBuilder with injected repository dependencies.

    Args:
        session: Async database session from FastAPI dependency injection.

    Returns:
        Configured PipelineDAGBuilder instance.
    """
    return PipelineDAGBuilder(
        experiment_repo=ExperimentRepository(session),
        deployment_repo=DeploymentRepository(session),
        retraining_repo=RetrainingJobRepository(session),
        feature_set_repo=FeatureSetRepository(session),
    )


@router.get("/pipeline-dag/{model_id}", response_model=DAGResponse)
async def get_pipeline_dag(
    model_id: str,
    tenant: TenantContext = Depends(get_current_user),
    builder: PipelineDAGBuilder = Depends(get_dag_builder),
    session: AsyncSession = Depends(get_db_session),
) -> DAGResponse:
    """Return a React Flow–compatible DAG for the ML pipeline lineage.

    Assembles all experiments, deployments, retraining jobs, and feature sets
    linked to the specified model and renders them as a directed acyclic graph
    for visualisation in aumos-platform-ui.

    Args:
        model_id: UUID of the model whose pipeline lineage to visualise.
        tenant: Authenticated tenant context from JWT middleware.
        builder: Injected PipelineDAGBuilder.
        session: Async database session.

    Returns:
        DAGResponse with React Flow node and edge objects.
    """
    dag = await builder.build(model_id=model_id, tenant=tenant, session=session)
    return DAGResponse(nodes=dag["nodes"], edges=dag["edges"])
