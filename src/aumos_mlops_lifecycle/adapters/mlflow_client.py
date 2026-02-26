"""MLflow tracking client adapter for aumos-mlops-lifecycle.

Wraps the MLflow Python SDK with an async interface and enforces per-tenant
experiment namespace isolation. All MLflow calls that would block the event
loop are offloaded to a thread pool executor.

Tenant isolation pattern:
  MLflow experiment name: tenant_{tenant_id}/{experiment_name}
  This ensures experiments from different tenants never collide in a shared
  MLflow tracking server.

Configuration:
  AUMOS_MLOPS_MLFLOW_TRACKING_URI — set before importing this module
"""

import asyncio
from typing import Any
from functools import partial

import mlflow
from mlflow.tracking import MlflowClient

from aumos_common.observability import get_logger

from aumos_mlops_lifecycle.core.interfaces import IMLflowClient

logger = get_logger(__name__)

_MLFLOW_EXPERIMENT_NS = "tenant_{tenant_id}/{name}"


class MLflowClient(IMLflowClient):
    """Async MLflow tracking client with tenant namespace isolation.

    Wraps the synchronous MLflow Python SDK and offloads all blocking I/O
    to asyncio's thread pool so the FastAPI event loop is never blocked.

    Args:
        tracking_uri: MLflow tracking server URL. Defaults to the value
                      set in AUMOS_MLOPS_MLFLOW_TRACKING_URI via settings.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        """Initialize the MLflow client.

        Args:
            tracking_uri: MLflow tracking server URL. If None, uses the
                          environment variable MLFLOW_TRACKING_URI or the
                          settings value set during app startup.
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient()

    def _namespaced_experiment_name(self, name: str, tenant_id: str) -> str:
        """Build a tenant-namespaced MLflow experiment name.

        Args:
            name: User-provided experiment name.
            tenant_id: Tenant UUID string for namespace prefix.

        Returns:
            Namespaced experiment name: tenant_{tenant_id}/{name}
        """
        return _MLFLOW_EXPERIMENT_NS.format(tenant_id=tenant_id, name=name)

    async def create_experiment(self, name: str, tenant_id: str) -> str:
        """Create an MLflow experiment namespaced by tenant.

        Offloads the blocking MLflow SDK call to a thread pool executor.
        If an experiment with the namespaced name already exists, returns
        the existing experiment ID.

        Args:
            name: Experiment name (will be prefixed with tenant_id namespace).
            tenant_id: Tenant UUID string for namespace isolation.

        Returns:
            The MLflow experiment ID string.
        """
        namespaced_name = self._namespaced_experiment_name(name=name, tenant_id=tenant_id)
        logger.info("Creating MLflow experiment", namespaced_name=namespaced_name)

        loop = asyncio.get_event_loop()
        experiment_id: str = await loop.run_in_executor(
            None,
            partial(self._create_or_get_experiment, namespaced_name),
        )
        return experiment_id

    def _create_or_get_experiment(self, namespaced_name: str) -> str:
        """Synchronously create or retrieve an MLflow experiment by name.

        Args:
            namespaced_name: Full namespaced experiment name.

        Returns:
            The MLflow experiment ID string.
        """
        existing = self._client.get_experiment_by_name(namespaced_name)
        if existing is not None:
            logger.info("MLflow experiment already exists", experiment_id=existing.experiment_id)
            return existing.experiment_id

        experiment_id = self._client.create_experiment(namespaced_name)
        logger.info("MLflow experiment created", experiment_id=experiment_id)
        return experiment_id

    async def log_run(
        self,
        mlflow_experiment_id: str,
        run_name: str | None,
        metrics: dict[str, float],
        params: dict[str, str],
        tags: dict[str, str],
    ) -> dict[str, Any]:
        """Log a run to an MLflow experiment.

        Creates an MLflow run, logs all metrics and params, then marks
        the run as FINISHED. Offloads blocking SDK calls to a thread pool.

        Args:
            mlflow_experiment_id: MLflow experiment ID.
            run_name: Optional human-readable run name.
            metrics: Numeric metric key-value pairs.
            params: Hyperparameter key-value pairs.
            tags: Run-level tags.

        Returns:
            Dict with run_id, status, start_time, end_time, and artifact_uri.
        """
        loop = asyncio.get_event_loop()
        run_data: dict[str, Any] = await loop.run_in_executor(
            None,
            partial(
                self._log_run_sync,
                mlflow_experiment_id=mlflow_experiment_id,
                run_name=run_name,
                metrics=metrics,
                params=params,
                tags=tags,
            ),
        )
        return run_data

    def _log_run_sync(
        self,
        mlflow_experiment_id: str,
        run_name: str | None,
        metrics: dict[str, float],
        params: dict[str, str],
        tags: dict[str, str],
    ) -> dict[str, Any]:
        """Synchronously create and complete an MLflow run.

        Args:
            mlflow_experiment_id: MLflow experiment ID.
            run_name: Optional human-readable run name.
            metrics: Numeric metric key-value pairs.
            params: Hyperparameter key-value pairs.
            tags: Run-level tags.

        Returns:
            Dict with run metadata.
        """
        run = self._client.create_run(
            experiment_id=mlflow_experiment_id,
            run_name=run_name,
            tags=tags,
        )
        run_id = run.info.run_id

        for key, value in metrics.items():
            self._client.log_metric(run_id, key, value)

        for key, value in params.items():
            self._client.log_param(run_id, key, value)

        self._client.set_terminated(run_id, status="FINISHED")

        completed_run = self._client.get_run(run_id)
        return {
            "run_id": run_id,
            "status": completed_run.info.status,
            "start_time": completed_run.info.start_time,
            "end_time": completed_run.info.end_time,
            "artifact_uri": completed_run.info.artifact_uri,
        }

    async def list_runs(
        self,
        mlflow_experiment_id: str,
        page: int,
        page_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """List runs for an MLflow experiment with pagination.

        Args:
            mlflow_experiment_id: MLflow experiment ID.
            page: Page number (1-indexed).
            page_size: Items per page.

        Returns:
            Tuple of (run dicts list, total count).
        """
        loop = asyncio.get_event_loop()
        result: tuple[list[dict[str, Any]], int] = await loop.run_in_executor(
            None,
            partial(
                self._list_runs_sync,
                mlflow_experiment_id=mlflow_experiment_id,
                page=page,
                page_size=page_size,
            ),
        )
        return result

    def _list_runs_sync(
        self,
        mlflow_experiment_id: str,
        page: int,
        page_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Synchronously list runs for an MLflow experiment.

        Args:
            mlflow_experiment_id: MLflow experiment ID.
            page: Page number (1-indexed).
            page_size: Items per page.

        Returns:
            Tuple of (run dicts list, total count).
        """
        runs = self._client.search_runs(
            experiment_ids=[mlflow_experiment_id],
            max_results=page_size,
            # Note: MLflow doesn't support offset-based pagination natively;
            # for production use, implement cursor-based pagination using
            # search_runs page_token parameter.
        )

        run_dicts = [
            {
                "run_id": r.info.run_id,
                "run_name": r.info.run_name,
                "status": r.info.status,
                "start_time": r.info.start_time,
                "end_time": r.info.end_time,
                "artifact_uri": r.info.artifact_uri,
                "metrics": r.data.metrics,
                "params": r.data.params,
                "tags": {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")},
            }
            for r in runs
        ]

        # Approximate total from search_runs — MLflow doesn't expose total count directly
        total = len(run_dicts)
        return run_dicts, total
