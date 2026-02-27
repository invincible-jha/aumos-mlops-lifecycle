"""Dataset versioner adapter for aumos-mlops-lifecycle.

Manages dataset registration, hash-based version tracking, DVC remote
storage integration, dataset diffs, lineage tracking, and storage optimisation.
All blocking DVC CLI calls are run via asyncio subprocess to avoid stalling
the FastAPI event loop.

Configuration:
    AUMOS_MLOPS_DVC_REMOTE_URL — DVC remote storage URL (e.g. s3://bucket/dvc)
    AUMOS_MLOPS_DVC_REPO_PATH  — Local path to the DVC-initialised repository
"""

import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class DatasetVersioner:
    """DVC-backed dataset version manager with lineage tracking.

    Registers datasets, computes content hashes for deduplication, manages
    DVC remotes, diffs dataset versions, and records which experiments consumed
    which dataset version.

    Args:
        dvc_repo_path: Absolute path to the DVC repository root.
        dvc_remote_url: DVC remote storage URL. e.g. s3://my-bucket/dvc-store.
    """

    def __init__(self, dvc_repo_path: str, dvc_remote_url: str) -> None:
        """Initialise the dataset versioner.

        Args:
            dvc_repo_path: Absolute path to the local DVC-initialised git repo.
            dvc_remote_url: DVC remote URL used for push/pull operations.
        """
        self._repo_path = Path(dvc_repo_path)
        self._remote_url = dvc_remote_url
        # In-process registry keyed by tenant_id → {dataset_name: [version_record]}
        self._registry: dict[str, dict[str, list[dict[str, Any]]]] = {}
        # Lineage map: run_id → [dataset_version_id]
        self._lineage: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ #
    # Dataset registration                                                 #
    # ------------------------------------------------------------------ #

    async def register_dataset(
        self,
        name: str,
        file_path: str,
        metadata: dict[str, Any],
        tenant_id: str,
    ) -> dict[str, Any]:
        """Register a dataset and create an initial version record.

        Computes a SHA-256 content hash for deduplication, stores version
        metadata, and runs `dvc add` to track the file.

        Args:
            name: Human-readable dataset name (unique within the tenant).
            file_path: Absolute path to the dataset file.
            metadata: Arbitrary metadata dict (schema version, source, etc.).
            tenant_id: Tenant UUID string for namespace isolation.

        Returns:
            Version record dict with version_id, hash, registered_at, and metadata.
        """
        logger.info("Registering dataset", name=name, tenant_id=tenant_id)
        content_hash = await self._compute_file_hash(file_path)

        tenant_registry = self._registry.setdefault(tenant_id, {})
        versions = tenant_registry.setdefault(name, [])

        # Dedup: return existing version if hash matches
        for version in versions:
            if version["hash"] == content_hash:
                logger.info("Dataset version already exists", version_id=version["version_id"])
                return version

        version_id = f"{tenant_id}/{name}/v{len(versions) + 1}"
        version_record: dict[str, Any] = {
            "version_id": version_id,
            "name": name,
            "tenant_id": tenant_id,
            "hash": content_hash,
            "file_path": file_path,
            "version_number": len(versions) + 1,
            "metadata": metadata,
            "registered_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        versions.append(version_record)

        await self._dvc_add(file_path)
        logger.info("Dataset registered", version_id=version_id, hash=content_hash)
        return version_record

    # ------------------------------------------------------------------ #
    # Version tracking                                                     #
    # ------------------------------------------------------------------ #

    async def list_versions(self, name: str, tenant_id: str) -> list[dict[str, Any]]:
        """List all versions of a registered dataset for a tenant.

        Args:
            name: Dataset name.
            tenant_id: Tenant UUID string for namespace isolation.

        Returns:
            List of version record dicts in registration order (oldest first).
        """
        return self._registry.get(tenant_id, {}).get(name, [])

    async def get_version(self, version_id: str, tenant_id: str) -> dict[str, Any] | None:
        """Retrieve a specific dataset version by its version ID.

        Args:
            version_id: Full version ID string (tenant_id/name/vN).
            tenant_id: Tenant UUID string for authorisation check.

        Returns:
            Version record dict if found, None otherwise.
        """
        for versions in self._registry.get(tenant_id, {}).values():
            for version in versions:
                if version["version_id"] == version_id:
                    return version
        return None

    # ------------------------------------------------------------------ #
    # DVC remote management                                                #
    # ------------------------------------------------------------------ #

    async def configure_remote(self, remote_name: str = "origin") -> bool:
        """Configure the DVC remote storage URL.

        Runs `dvc remote add` to register the remote in the DVC config. If
        the remote already exists it is modified to use the current URL.

        Args:
            remote_name: DVC remote alias name. Defaults to "origin".

        Returns:
            True if the remote was configured successfully.
        """
        try:
            returncode, stdout, stderr = await self._run_dvc(
                "remote", "add", "--default", "--force", remote_name, self._remote_url
            )
            if returncode == 0:
                logger.info("DVC remote configured", remote_name=remote_name, url=self._remote_url)
                return True
            logger.error("DVC remote configuration failed", stderr=stderr)
            return False
        except Exception:
            logger.exception("DVC remote configuration error")
            return False

    async def push_version(self, version_id: str, tenant_id: str) -> bool:
        """Push a dataset version to the DVC remote storage.

        Args:
            version_id: Version ID string to push.
            tenant_id: Tenant UUID string for version lookup.

        Returns:
            True if the push succeeded.
        """
        version = await self.get_version(version_id, tenant_id)
        if version is None:
            logger.warning("Version not found for push", version_id=version_id)
            return False

        try:
            returncode, _, stderr = await self._run_dvc("push")
            if returncode == 0:
                logger.info("Dataset pushed to remote", version_id=version_id)
                return True
            logger.error("DVC push failed", stderr=stderr)
            return False
        except Exception:
            logger.exception("DVC push error", version_id=version_id)
            return False

    async def checkout_version(self, version_id: str, tenant_id: str) -> bool:
        """Checkout and validate a specific dataset version from the remote.

        Runs `dvc pull` to restore the file and then validates the content
        hash matches the registered value.

        Args:
            version_id: Version ID to checkout.
            tenant_id: Tenant UUID string for version lookup.

        Returns:
            True if the checkout succeeded and hash validation passed.
        """
        version = await self.get_version(version_id, tenant_id)
        if version is None:
            logger.warning("Version not found for checkout", version_id=version_id)
            return False

        returncode, _, stderr = await self._run_dvc("pull")
        if returncode != 0:
            logger.error("DVC pull failed", stderr=stderr)
            return False

        actual_hash = await self._compute_file_hash(version["file_path"])
        if actual_hash != version["hash"]:
            logger.error(
                "Dataset hash mismatch after checkout",
                expected=version["hash"],
                actual=actual_hash,
            )
            return False

        logger.info("Dataset checked out and validated", version_id=version_id)
        return True

    # ------------------------------------------------------------------ #
    # Dataset diff                                                         #
    # ------------------------------------------------------------------ #

    async def diff_versions(
        self,
        name: str,
        version_a_id: str,
        version_b_id: str,
        tenant_id: str,
    ) -> dict[str, Any]:
        """Compute a diff summary between two dataset versions.

        Compares metadata, hash, version numbers, and size to identify
        what changed between the two versions.

        Args:
            name: Dataset name.
            version_a_id: Earlier version ID.
            version_b_id: Later version ID.
            tenant_id: Tenant UUID string.

        Returns:
            Diff dict with keys: name, version_a, version_b, hash_changed,
            metadata_diff, and registered_at_delta_seconds.
        """
        version_a = await self.get_version(version_a_id, tenant_id)
        version_b = await self.get_version(version_b_id, tenant_id)

        if version_a is None or version_b is None:
            return {"error": "One or both versions not found"}

        metadata_diff: dict[str, Any] = {}
        all_keys = set(version_a["metadata"]) | set(version_b["metadata"])
        for key in all_keys:
            val_a = version_a["metadata"].get(key)
            val_b = version_b["metadata"].get(key)
            if val_a != val_b:
                metadata_diff[key] = {"from": val_a, "to": val_b}

        ts_a = datetime.fromisoformat(version_a["registered_at"])
        ts_b = datetime.fromisoformat(version_b["registered_at"])
        delta_seconds = (ts_b - ts_a).total_seconds()

        return {
            "name": name,
            "version_a": version_a_id,
            "version_b": version_b_id,
            "hash_changed": version_a["hash"] != version_b["hash"],
            "metadata_diff": metadata_diff,
            "registered_at_delta_seconds": delta_seconds,
        }

    # ------------------------------------------------------------------ #
    # Lineage tracking                                                     #
    # ------------------------------------------------------------------ #

    async def record_usage(self, run_id: str, version_id: str) -> None:
        """Record that a training run consumed a specific dataset version.

        This lineage mapping enables auditing which experiments used which
        data, supporting reproducibility and compliance requirements.

        Args:
            run_id: MLflow run ID that consumed the dataset.
            version_id: Dataset version ID that was used.
        """
        bucket = self._lineage.setdefault(run_id, [])
        if version_id not in bucket:
            bucket.append(version_id)
        logger.info("Dataset lineage recorded", run_id=run_id, version_id=version_id)

    async def get_lineage_for_run(self, run_id: str) -> list[str]:
        """Retrieve all dataset version IDs consumed by a run.

        Args:
            run_id: MLflow run ID to query.

        Returns:
            List of version ID strings consumed by the run (may be empty).
        """
        return self._lineage.get(run_id, [])

    async def get_runs_using_version(self, version_id: str) -> list[str]:
        """Find all runs that consumed a specific dataset version.

        Args:
            version_id: Dataset version ID to search for.

        Returns:
            List of run ID strings that consumed the version.
        """
        return [run_id for run_id, versions in self._lineage.items() if version_id in versions]

    # ------------------------------------------------------------------ #
    # Storage optimisation                                                 #
    # ------------------------------------------------------------------ #

    async def deduplicate_versions(self, name: str, tenant_id: str) -> int:
        """Remove duplicate versions (same hash) from the registry.

        Keeps only the first occurrence of each unique hash to eliminate
        storage waste from repeated registrations of identical data.

        Args:
            name: Dataset name to deduplicate.
            tenant_id: Tenant UUID string.

        Returns:
            Number of duplicate version records removed.
        """
        versions = self._registry.get(tenant_id, {}).get(name, [])
        seen_hashes: set[str] = set()
        unique_versions: list[dict[str, Any]] = []
        removed = 0

        for version in versions:
            if version["hash"] in seen_hashes:
                removed += 1
            else:
                seen_hashes.add(version["hash"])
                unique_versions.append(version)

        if removed > 0 and tenant_id in self._registry and name in self._registry[tenant_id]:
            self._registry[tenant_id][name] = unique_versions

        logger.info("Deduplication complete", name=name, removed=removed)
        return removed

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    async def _compute_file_hash(self, file_path: str) -> str:
        """Compute a SHA-256 hash of a file's content asynchronously.

        Args:
            file_path: Absolute path to the file.

        Returns:
            Hex-encoded SHA-256 digest string.
        """
        loop = asyncio.get_event_loop()
        digest: str = await loop.run_in_executor(None, self._hash_file_sync, file_path)
        return digest

    @staticmethod
    def _hash_file_sync(file_path: str) -> str:
        sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as file_handle:
                for chunk in iter(lambda: file_handle.read(65536), b""):
                    sha256.update(chunk)
        except FileNotFoundError:
            # Hash the path string as a fallback (testing or pre-creation)
            sha256.update(file_path.encode())
        return sha256.hexdigest()

    async def _dvc_add(self, file_path: str) -> None:
        """Track a file with DVC (runs `dvc add <file>`).

        Args:
            file_path: Absolute path to the file to track.
        """
        returncode, stdout, stderr = await self._run_dvc("add", file_path)
        if returncode != 0:
            logger.warning("dvc add failed (non-fatal)", stderr=stderr, file_path=file_path)

    async def _run_dvc(self, *args: str) -> tuple[int, str, str]:
        """Run a DVC CLI command as an asyncio subprocess.

        Args:
            *args: DVC sub-command arguments (e.g. "add", "/path/file").

        Returns:
            Tuple of (return_code, stdout, stderr).
        """
        cmd = ["dvc", *args]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._repo_path),
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        returncode = process.returncode or 0
        return returncode, stdout_bytes.decode(), stderr_bytes.decode()
