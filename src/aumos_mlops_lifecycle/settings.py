"""Service-specific settings extending AumOS base config.

All standard AumOS configuration is inherited from AumOSSettings.
MLOps-specific settings use the AUMOS_MLOPS_ env prefix.
"""

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

    # Downstream service integration
    model_registry_url: str = "http://localhost:8001"
    model_registry_api_key: str = ""

    model_config = SettingsConfigDict(env_prefix="AUMOS_MLOPS_")
