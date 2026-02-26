"""AumOS MLOps Lifecycle service entry point.

Creates the FastAPI application with lifespan management for database
initialization, Kafka publisher setup, and MLflow/Feast client configuration.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from aumos_common.app import create_app
from aumos_common.database import init_database
from aumos_common.observability import get_logger

from aumos_mlops_lifecycle.api.router import router
from aumos_mlops_lifecycle.settings import Settings

logger = get_logger(__name__)
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    Initializes the database connection pool, Kafka event publisher,
    MLflow tracking client, and Feast feature store client on startup,
    then closes all connections on shutdown.

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Control is yielded to the application during its runtime.
    """
    # Startup
    logger.info(
        "Starting aumos-mlops-lifecycle",
        version="0.1.0",
        environment=settings.environment,
        mlflow_uri=settings.mlflow_tracking_uri,
    )
    init_database(settings.database)
    # TODO: Initialize Kafka publisher via EventPublisher from aumos-common
    # TODO: Initialize MLflowClient with settings.mlflow_tracking_uri
    # TODO: Initialize FeastClient with settings.feast_registry_path
    yield
    # Shutdown
    logger.info("Shutting down aumos-mlops-lifecycle")
    # TODO: Close Kafka producer connections
    # TODO: Flush any pending MLflow runs


app: FastAPI = create_app(
    service_name="aumos-mlops-lifecycle",
    version="0.1.0",
    settings=settings,
    lifespan=lifespan,
    health_checks=[
        # HealthCheck(name="postgres", check_fn=check_db_health),
        # HealthCheck(name="kafka", check_fn=check_kafka_health),
        # HealthCheck(name="mlflow", check_fn=check_mlflow_health),
    ],
)

app.include_router(router, prefix="/api/v1")
