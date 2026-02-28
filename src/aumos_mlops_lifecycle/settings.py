"""Service-specific settings extending AumOS base config.

All standard AumOS configuration is inherited from AumOSSettings.
MLOps-specific settings use the AUMOS_MLOPS_ env prefix.
"""

from functools import lru_cache

from pydantic_settings import SettingsConfigDict

from aumos_common.config import AumOSSettings


class Settings(AumOSSettings):
    """Settings for aumos-mlops-lifecycle.

    Inherits all standard AumOS settings (database, kafka, keycloak, etc.)
    and adds MLOps-specific configuration for MLflow, Feast, and deployment
    strategy parameters.

    Environment variable prefix: AUMOS_MLOPS_
    """

    service_name: str = "aumos-mlops-lifecycle"

    # MLflow tracking server
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_artifact_root: str = "./mlruns"

    # Feast feature store
    feast_registry_path: str = "data/registry.db"
    feast_offline_store_type: str = "file"
    feast_materialization_batch_size: int = 10000

    # Canary deployment settings
    canary_error_threshold: float = 0.05
    canary_step_percent: int = 10
    canary_progression_interval_seconds: int = 300

    # Retraining job settings
    max_concurrent_retraining_jobs: int = 5
    retraining_job_timeout_seconds: int = 3600

    # Artifact storage (GAP-158)
    artifact_bucket_name: str = "aumos-artifacts"
    artifact_endpoint_url: str | None = None  # None = use AWS S3; set for MinIO
    artifact_access_key: str | None = None
    artifact_secret_key: str | None = None

    # Downstream service integration
    model_registry_url: str = "http://localhost:8001"
    model_registry_api_key: str = ""
    testing_harness_url: str = "http://aumos-testing-harness:8000"
    internal_api_key: str = ""

    # Cron scheduler (GAP-164)
    retraining_scheduler_interval_minutes: int = 1

    model_config = SettingsConfigDict(env_prefix="AUMOS_MLOPS_")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton.

    Returns:
        Singleton Settings instance loaded from environment variables.
    """
    return Settings()
