"""Experiment tracker adapter for aumos-mlops-lifecycle.

Provides hyperparameter logging, metric tracking per epoch, artifact
versioning, and run comparison against an MLflow backend. All blocking
SDK calls are offloaded to a thread-pool executor so the FastAPI event
loop is never blocked.

Configuration:
    AUMOS_MLOPS_MLFLOW_TRACKING_URI — MLflow server URL
"""

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from functools import partial
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_TENANT_NS = "tenant_{tenant_id}"


class ExperimentTracker:
    """MLflow-backed experiment tracker with per-tenant isolation.

    Handles hyperparameter logging, per-epoch metric streams, artifact
    versioning, run comparisons, and tag management. Every experiment and
    run is prefixed with the tenant ID so a shared MLflow server can serve
    multiple tenants without data leakage.

    Args:
        tracking_uri: MLflow tracking server URL. If None the MLFLOW_TRACKING_URI
                      environment variable is used.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        """Initialize the experiment tracker.

        Args:
            tracking_uri: MLflow tracking server URL. Passed directly to
                          mlflow.set_tracking_uri when provided.
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient()

    # ------------------------------------------------------------------ #
    # Experiment management                                                #
    # ------------------------------------------------------------------ #

    async def create_experiment(
        self,
        name: str,
        tenant_id: str,
        tags: dict[str, str] | None = None,
        artifact_location: str | None = None,
    ) -> str:
        """Create a new MLflow experiment namespaced to the tenant.

        If an experiment with the same namespaced name already exists the
        existing experiment ID is returned — creation is idempotent.

        Args:
            name: Human-readable experiment name.
            tenant_id: Tenant UUID string used to build the namespace prefix.
            tags: Optional key-value tags applied to the experiment on creation.
            artifact_location: Optional artifact storage URI (S3, GCS, ADLS).

        Returns:
            MLflow experiment ID string.
        """
        namespaced = f"{_TENANT_NS.format(tenant_id=tenant_id)}/{name}"
        logger.info("Creating experiment", namespaced_name=namespaced)

        loop = asyncio.get_event_loop()
        experiment_id: str = await loop.run_in_executor(
            None,
            partial(
                self._create_or_get_experiment_sync,
                namespaced_name=namespaced,
                tags=tags or {},
                artifact_location=artifact_location,
            ),
        )
        return experiment_id

    def _create_or_get_experiment_sync(
        self,
        namespaced_name: str,
        tags: dict[str, str],
        artifact_location: str | None,
    ) -> str:
        """Synchronously create or retrieve an MLflow experiment.

        Args:
            namespaced_name: Full tenant-namespaced experiment name.
            tags: Key-value tags to apply on creation.
            artifact_location: Optional artifact storage URI.

        Returns:
            MLflow experiment ID string.
        """
        existing = self._client.get_experiment_by_name(namespaced_name)
        if existing is not None:
            logger.info("Experiment already exists", experiment_id=existing.experiment_id)
            return existing.experiment_id

        experiment_id = self._client.create_experiment(
            name=namespaced_name,
            artifact_location=artifact_location,
            tags=tags,
        )
        logger.info("Experiment created", experiment_id=experiment_id)
        return experiment_id

    async def configure_experiment(
        self,
        mlflow_experiment_id: str,
        tags: dict[str, str],
    ) -> None:
        """Update the tags on an existing MLflow experiment.

        Args:
            mlflow_experiment_id: MLflow experiment ID to configure.
            tags: New key-value tags to set on the experiment.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            partial(self._set_experiment_tags_sync, mlflow_experiment_id=mlflow_experiment_id, tags=tags),
        )

    def _set_experiment_tags_sync(self, mlflow_experiment_id: str, tags: dict[str, str]) -> None:
        for key, value in tags.items():
            self._client.set_experiment_tag(mlflow_experiment_id, key, value)
        logger.info("Experiment tags updated", experiment_id=mlflow_experiment_id, tag_count=len(tags))

    # ------------------------------------------------------------------ #
    # Hyperparameter logging                                               #
    # ------------------------------------------------------------------ #

    async def log_hyperparameters(
        self,
        run_id: str,
        params: dict[str, str],
    ) -> None:
        """Log hyperparameters to an active MLflow run.

        Args:
            run_id: The MLflow run ID to log hyperparameters against.
            params: Hyperparameter key-value pairs (values must be strings).
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            partial(self._log_params_sync, run_id=run_id, params=params),
        )

    def _log_params_sync(self, run_id: str, params: dict[str, str]) -> None:
        for key, value in params.items():
            self._client.log_param(run_id, key, value)
        logger.info("Hyperparameters logged", run_id=run_id, param_count=len(params))

    # ------------------------------------------------------------------ #
    # Metric logging                                                       #
    # ------------------------------------------------------------------ #

    async def log_metrics(
        self,
        run_id: str,
        metrics: dict[str, float],
        step: int | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Log one or more metrics to an active MLflow run.

        Supports epoch-level logging via the step parameter. Metrics are
        batched into a single call to minimise round-trips.

        Args:
            run_id: The MLflow run ID to log metrics against.
            metrics: Metric name to float value mapping (e.g. loss, accuracy).
            step: Training epoch or global step index. If None, MLflow
                  auto-increments the step for each metric.
            timestamp: Optional UTC timestamp for the metric values.
                       Defaults to the current UTC time.
        """
        ts_ms = int((timestamp or datetime.now(tz=timezone.utc)).timestamp() * 1000)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            partial(
                self._log_metrics_sync,
                run_id=run_id,
                metrics=metrics,
                step=step,
                timestamp_ms=ts_ms,
            ),
        )

    def _log_metrics_sync(
        self,
        run_id: str,
        metrics: dict[str, float],
        step: int | None,
        timestamp_ms: int,
    ) -> None:
        for key, value in metrics.items():
            self._client.log_metric(run_id, key, value, timestamp=timestamp_ms, step=step)
        logger.info("Metrics logged", run_id=run_id, metric_count=len(metrics), step=step)

    # ------------------------------------------------------------------ #
    # Artifact versioning                                                  #
    # ------------------------------------------------------------------ #

    async def log_artifact(
        self,
        run_id: str,
        local_path: str,
        artifact_path: str | None = None,
    ) -> str:
        """Log a local file as a versioned MLflow artifact.

        The artifact is associated with the run identified by run_id and
        stored in the run's artifact URI (S3, GCS, or local filesystem).

        Args:
            run_id: MLflow run ID to attach the artifact to.
            local_path: Absolute path to the file to log.
            artifact_path: Optional subdirectory within the run's artifact root.

        Returns:
            Artifact URI string (e.g. s3://bucket/mlflow/…/artifact.pkl).
        """
        loop = asyncio.get_event_loop()
        artifact_uri: str = await loop.run_in_executor(
            None,
            partial(
                self._log_artifact_sync,
                run_id=run_id,
                local_path=local_path,
                artifact_path=artifact_path,
            ),
        )
        return artifact_uri

    def _log_artifact_sync(
        self,
        run_id: str,
        local_path: str,
        artifact_path: str | None,
    ) -> str:
        self._client.log_artifact(run_id, local_path, artifact_path=artifact_path)
        run_info = self._client.get_run(run_id)
        base_uri = run_info.info.artifact_uri
        import os

        filename = os.path.basename(local_path)
        artifact_uri = f"{base_uri}/{artifact_path}/{filename}" if artifact_path else f"{base_uri}/{filename}"
        logger.info("Artifact logged", run_id=run_id, artifact_uri=artifact_uri)
        return artifact_uri

    # ------------------------------------------------------------------ #
    # Run lifecycle                                                        #
    # ------------------------------------------------------------------ #

    async def start_run(
        self,
        mlflow_experiment_id: str,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Create and start a new MLflow run within an experiment.

        Args:
            mlflow_experiment_id: The MLflow experiment to associate the run with.
            run_name: Optional human-readable name for the run.
            tags: Optional key-value tags applied at run creation.

        Returns:
            MLflow run ID string for subsequent metric/param logging.
        """
        loop = asyncio.get_event_loop()
        run_id: str = await loop.run_in_executor(
            None,
            partial(
                self._start_run_sync,
                mlflow_experiment_id=mlflow_experiment_id,
                run_name=run_name,
                tags=tags or {},
            ),
        )
        return run_id

    def _start_run_sync(
        self,
        mlflow_experiment_id: str,
        run_name: str | None,
        tags: dict[str, str],
    ) -> str:
        run = self._client.create_run(
            experiment_id=mlflow_experiment_id,
            run_name=run_name,
            tags=tags,
        )
        logger.info("Run started", run_id=run.info.run_id, experiment_id=mlflow_experiment_id)
        return run.info.run_id

    async def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        """Terminate an active MLflow run.

        Args:
            run_id: MLflow run ID to terminate.
            status: Terminal status. One of FINISHED, FAILED, KILLED.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            partial(self._client.set_terminated, run_id, status),
        )
        logger.info("Run ended", run_id=run_id, status=status)

    # ------------------------------------------------------------------ #
    # Run comparison                                                       #
    # ------------------------------------------------------------------ #

    async def compare_runs(
        self,
        mlflow_experiment_id: str,
        metric_keys: list[str],
        max_runs: int = 20,
    ) -> list[dict[str, Any]]:
        """Compare runs within an experiment across specified metrics.

        Returns runs sorted by the first metric_key in descending order so
        the best-performing run appears first.

        Args:
            mlflow_experiment_id: MLflow experiment to search within.
            metric_keys: List of metric names to include in the comparison.
            max_runs: Maximum number of runs to return. Defaults to 20.

        Returns:
            List of dicts, each containing run_id, run_name, params,
            tags, and the requested metric values.
        """
        loop = asyncio.get_event_loop()
        comparison: list[dict[str, Any]] = await loop.run_in_executor(
            None,
            partial(
                self._compare_runs_sync,
                mlflow_experiment_id=mlflow_experiment_id,
                metric_keys=metric_keys,
                max_runs=max_runs,
            ),
        )
        return comparison

    def _compare_runs_sync(
        self,
        mlflow_experiment_id: str,
        metric_keys: list[str],
        max_runs: int,
    ) -> list[dict[str, Any]]:
        order_by = [f"metrics.{metric_keys[0]} DESC"] if metric_keys else ["start_time DESC"]
        runs = self._client.search_runs(
            experiment_ids=[mlflow_experiment_id],
            max_results=max_runs,
            order_by=order_by,
        )
        result: list[dict[str, Any]] = []
        for run in runs:
            entry: dict[str, Any] = {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "params": run.data.params,
                "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
            }
            for key in metric_keys:
                entry[key] = run.data.metrics.get(key)
            result.append(entry)
        return result

    # ------------------------------------------------------------------ #
    # Search and filtering                                                 #
    # ------------------------------------------------------------------ #

    async def search_runs(
        self,
        mlflow_experiment_id: str,
        filter_string: str = "",
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        """Search runs within an experiment using MLflow filter syntax.

        Args:
            mlflow_experiment_id: MLflow experiment to search within.
            filter_string: MLflow filter expression (e.g. "metrics.accuracy > 0.9").
            max_results: Maximum number of matching runs to return.

        Returns:
            List of matching run dicts with id, name, metrics, params, tags.
        """
        loop = asyncio.get_event_loop()
        results: list[dict[str, Any]] = await loop.run_in_executor(
            None,
            partial(
                self._search_runs_sync,
                mlflow_experiment_id=mlflow_experiment_id,
                filter_string=filter_string,
                max_results=max_results,
            ),
        )
        return results

    def _search_runs_sync(
        self,
        mlflow_experiment_id: str,
        filter_string: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        runs = self._client.search_runs(
            experiment_ids=[mlflow_experiment_id],
            filter_string=filter_string,
            max_results=max_results,
        )
        return [
            {
                "run_id": r.info.run_id,
                "run_name": r.info.run_name,
                "status": r.info.status,
                "metrics": r.data.metrics,
                "params": r.data.params,
                "tags": {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")},
                "artifact_uri": r.info.artifact_uri,
                "start_time": r.info.start_time,
                "end_time": r.info.end_time,
            }
            for r in runs
        ]

    # ------------------------------------------------------------------ #
    # Annotations                                                          #
    # ------------------------------------------------------------------ #

    async def annotate_run(self, run_id: str, tags: dict[str, str]) -> None:
        """Add or update tags on an existing MLflow run.

        Args:
            run_id: MLflow run ID to annotate.
            tags: Key-value annotations to apply to the run.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            partial(self._set_run_tags_sync, run_id=run_id, tags=tags),
        )

    def _set_run_tags_sync(self, run_id: str, tags: dict[str, str]) -> None:
        for key, value in tags.items():
            self._client.set_tag(run_id, key, value)
        logger.info("Run annotated", run_id=run_id, tag_count=len(tags))

    @staticmethod
    def compute_artifact_checksum(content: bytes) -> str:
        """Compute a SHA-256 checksum for an artifact for dedup detection.

        Args:
            content: Raw bytes of the artifact.

        Returns:
            Hex-encoded SHA-256 digest string.
        """
        return hashlib.sha256(content).hexdigest()
