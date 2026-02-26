"""Feast feature store client adapter for aumos-mlops-lifecycle.

Wraps the Feast Python SDK with an async interface and enforces per-tenant
feature view namespace isolation. All Feast calls that would block the event
loop are offloaded to a thread pool executor.

Tenant isolation pattern:
  Feast feature view name: tenant_{tenant_id}_{feature_set_name}
  This ensures feature views from different tenants never collide in a shared
  Feast registry.

Configuration:
  AUMOS_MLOPS_FEAST_REGISTRY_PATH — path or GCS URI for the Feast registry
"""

import asyncio
from functools import partial
from typing import Any

from aumos_common.observability import get_logger

from aumos_mlops_lifecycle.core.interfaces import IFeastClient

logger = get_logger(__name__)

_FEAST_FEATURE_VIEW_NS = "tenant_{tenant_id}_{name}"


class FeastClient(IFeastClient):
    """Async Feast feature store client with tenant namespace isolation.

    Wraps the synchronous Feast Python SDK and offloads all blocking I/O
    to asyncio's thread pool so the FastAPI event loop is never blocked.

    In production, this client initializes a Feast FeatureStore instance
    using the configured registry path. In tests, the IFeastClient protocol
    interface enables easy mocking without importing feast.

    Args:
        registry_path: Path to the Feast registry file or GCS URI.
                       Defaults to reading from settings at runtime.
    """

    def __init__(self, registry_path: str | None = None) -> None:
        """Initialize the Feast client.

        Args:
            registry_path: Feast registry path (file system or GCS URI).
                           If None, the client initializes lazily on first use.
        """
        self._registry_path = registry_path
        self._store: Any | None = None  # feast.FeatureStore, lazily initialized

    def _get_store(self) -> Any:
        """Lazily initialize and return the Feast FeatureStore instance.

        Returns:
            Initialized feast.FeatureStore instance.
        """
        if self._store is None:
            try:
                from feast import FeatureStore  # type: ignore[import-untyped]

                self._store = FeatureStore(repo_path=self._registry_path or ".")
            except ImportError as exc:
                msg = "feast package is required. Install with: pip install feast>=0.37.0"
                raise ImportError(msg) from exc
        return self._store

    def _namespaced_feature_view_name(self, name: str, tenant_id: str) -> str:
        """Build a tenant-namespaced Feast feature view name.

        Args:
            name: User-provided feature set name.
            tenant_id: Tenant UUID string for namespace prefix.

        Returns:
            Namespaced feature view name: tenant_{tenant_id}_{name}
        """
        return _FEAST_FEATURE_VIEW_NS.format(tenant_id=tenant_id.replace("-", "_"), name=name)

    async def register_feature_view(
        self,
        name: str,
        entity_name: str,
        features: list[dict[str, Any]],
        source_type: str,
        schedule: str | None,
        tenant_id: str,
    ) -> bool:
        """Register a feature view in the Feast registry.

        Constructs a Feast FeatureView from the provided definition and
        applies it to the registry. Offloads blocking SDK calls to a thread pool.

        Args:
            name: Feature view name (will be prefixed with tenant namespace).
            entity_name: Associated entity name (must be pre-registered in Feast).
            features: List of feature definitions: [{name, dtype, description}].
            source_type: Data source type (batch, stream, request).
            schedule: Optional cron materialization schedule.
            tenant_id: Tenant UUID string for namespace isolation.

        Returns:
            True if registration succeeded.
        """
        namespaced_name = self._namespaced_feature_view_name(name=name, tenant_id=tenant_id)
        logger.info(
            "Registering Feast feature view",
            namespaced_name=namespaced_name,
            entity_name=entity_name,
            feature_count=len(features),
        )

        loop = asyncio.get_event_loop()
        success: bool = await loop.run_in_executor(
            None,
            partial(
                self._register_feature_view_sync,
                namespaced_name=namespaced_name,
                entity_name=entity_name,
                features=features,
                source_type=source_type,
                schedule=schedule,
            ),
        )
        return success

    def _register_feature_view_sync(
        self,
        namespaced_name: str,
        entity_name: str,
        features: list[dict[str, Any]],
        source_type: str,
        schedule: str | None,
    ) -> bool:
        """Synchronously register a Feast feature view.

        Args:
            namespaced_name: Tenant-namespaced feature view name.
            entity_name: Associated entity name.
            features: Feature definitions.
            source_type: Data source type.
            schedule: Optional cron schedule.

        Returns:
            True if registration succeeded.
        """
        try:
            from feast import Entity, Feature, FeatureView, ValueType  # type: ignore[import-untyped]
            from feast.infra.offline_stores.file_source import FileSource  # type: ignore[import-untyped]
            from datetime import timedelta

            store = self._get_store()

            dtype_map: dict[str, Any] = {
                "float32": ValueType.FLOAT,
                "float64": ValueType.DOUBLE,
                "int32": ValueType.INT32,
                "int64": ValueType.INT64,
                "string": ValueType.STRING,
                "bool": ValueType.BOOL,
            }

            feast_features = [
                Feature(name=f["name"], dtype=dtype_map.get(f.get("dtype", "float64"), ValueType.DOUBLE))
                for f in features
            ]

            entity = Entity(name=entity_name, description=f"Entity for {namespaced_name}")

            # Use a placeholder file source for the feature view definition
            # In production, this would be replaced with the actual data source
            source = FileSource(path=f"data/{namespaced_name}.parquet", timestamp_field="event_timestamp")

            feature_view = FeatureView(
                name=namespaced_name,
                entities=[entity_name],
                ttl=timedelta(days=1),
                features=feast_features,
                online=True,
                source=source,
            )

            store.apply([entity, feature_view])
            logger.info("Feast feature view registered", namespaced_name=namespaced_name)
            return True

        except Exception:
            logger.exception("Failed to register Feast feature view", namespaced_name=namespaced_name)
            return False

    async def materialize(self, feature_view_name: str, tenant_id: str) -> bool:
        """Trigger materialization for a feature view.

        Offloads the blocking Feast materialization call to a thread pool executor.
        Materialization reads from the offline store and populates the online store.

        Args:
            feature_view_name: Name of the feature view to materialize (before namespacing).
            tenant_id: Tenant UUID string for namespace isolation.

        Returns:
            True if materialization was triggered successfully.
        """
        namespaced_name = self._namespaced_feature_view_name(name=feature_view_name, tenant_id=tenant_id)
        logger.info("Triggering Feast materialization", namespaced_name=namespaced_name)

        loop = asyncio.get_event_loop()
        success: bool = await loop.run_in_executor(
            None,
            partial(self._materialize_sync, namespaced_name=namespaced_name),
        )
        return success

    def _materialize_sync(self, namespaced_name: str) -> bool:
        """Synchronously trigger Feast materialization for a feature view.

        Args:
            namespaced_name: Tenant-namespaced feature view name.

        Returns:
            True if materialization succeeded.
        """
        try:
            from datetime import datetime, timezone, timedelta

            store = self._get_store()
            end_date = datetime.now(tz=timezone.utc)
            start_date = end_date - timedelta(days=1)

            store.materialize(
                start_date=start_date,
                end_date=end_date,
                feature_views=[namespaced_name],
            )
            logger.info("Feast materialization completed", namespaced_name=namespaced_name)
            return True

        except Exception:
            logger.exception("Feast materialization failed", namespaced_name=namespaced_name)
            return False

    async def get_online_features(
        self,
        feature_view_name: str,
        entity_rows: list[dict[str, Any]],
        feature_names: list[str],
        tenant_id: str,
    ) -> dict[str, Any]:
        """Retrieve online features from Feast for given entity rows.

        Args:
            feature_view_name: Feature view to retrieve from (before namespacing).
            entity_rows: List of entity key-value dicts for lookup.
            feature_names: List of feature names to retrieve.
            tenant_id: Tenant UUID string for namespace isolation.

        Returns:
            Dict mapping entity keys to feature values.
        """
        namespaced_name = self._namespaced_feature_view_name(name=feature_view_name, tenant_id=tenant_id)
        feature_refs = [f"{namespaced_name}:{fname}" for fname in feature_names]

        loop = asyncio.get_event_loop()
        result: dict[str, Any] = await loop.run_in_executor(
            None,
            partial(
                self._get_online_features_sync,
                feature_refs=feature_refs,
                entity_rows=entity_rows,
            ),
        )
        return result

    def _get_online_features_sync(
        self,
        feature_refs: list[str],
        entity_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Synchronously retrieve online features from Feast.

        Args:
            feature_refs: List of fully qualified feature references.
            entity_rows: Entity key-value dicts for lookup.

        Returns:
            Feature retrieval result as a dict.
        """
        try:
            store = self._get_store()
            feature_vector = store.get_online_features(
                features=feature_refs,
                entity_rows=entity_rows,
            )
            return feature_vector.to_dict()

        except Exception:
            logger.exception("Feast online feature retrieval failed", feature_refs=feature_refs)
            return {}
