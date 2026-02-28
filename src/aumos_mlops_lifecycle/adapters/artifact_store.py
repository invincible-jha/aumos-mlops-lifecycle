"""MinIO/S3 artifact storage adapter for aumos-mlops-lifecycle.

Provides tenant-scoped artifact upload, listing, and presigned URL generation
using aioboto3 (async boto3 wrapper). Artifacts are stored at:
    s3://{bucket}/artifacts/{tenant_id}/{experiment_id}/{filename}

GAP-158: Artifact Storage
"""

from __future__ import annotations

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class ArtifactStore:
    """MinIO/S3 artifact storage adapter with tenant-scoped paths.

    Args:
        bucket_name: Name of the S3/MinIO bucket for artifact storage.
        endpoint_url: S3-compatible endpoint URL (None for AWS S3).
        access_key: S3 access key ID.
        secret_key: S3 secret access key.
    """

    def __init__(
        self,
        bucket_name: str,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        """Initialize artifact store with S3/MinIO connection parameters.

        Args:
            bucket_name: Target S3/MinIO bucket name.
            endpoint_url: Optional endpoint override for MinIO.
            access_key: AWS/MinIO access key ID.
            secret_key: AWS/MinIO secret access key.
        """
        self._bucket_name = bucket_name
        self._endpoint_url = endpoint_url
        self._access_key = access_key
        self._secret_key = secret_key

    def _key(self, tenant_id: str, experiment_id: str, filename: str) -> str:
        """Build the S3 object key for a tenant-scoped artifact.

        Args:
            tenant_id: Tenant UUID as string.
            experiment_id: Experiment UUID as string.
            filename: Artifact file name.

        Returns:
            S3 object key string.
        """
        return f"artifacts/{tenant_id}/{experiment_id}/{filename}"

    def _client_kwargs(self) -> dict[str, object]:
        """Build keyword arguments for the aioboto3 S3 client context manager."""
        kwargs: dict[str, object] = {}
        if self._endpoint_url:
            kwargs["endpoint_url"] = self._endpoint_url
        if self._access_key and self._secret_key:
            kwargs["aws_access_key_id"] = self._access_key
            kwargs["aws_secret_access_key"] = self._secret_key
        return kwargs

    async def upload(
        self,
        tenant_id: str,
        experiment_id: str,
        filename: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload artifact bytes to S3/MinIO and return the object URI.

        Args:
            tenant_id: Tenant UUID string for path scoping.
            experiment_id: Experiment UUID string.
            filename: Target filename within the experiment artifact directory.
            content: Raw file bytes to upload.
            content_type: MIME type of the artifact.

        Returns:
            S3 URI of the uploaded artifact (s3://bucket/key).
        """
        try:
            import aioboto3  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("aioboto3 is required for artifact storage. Install with: pip install aioboto3") from exc

        key = self._key(tenant_id, experiment_id, filename)
        session = aioboto3.Session()
        async with session.client("s3", **self._client_kwargs()) as s3:
            await s3.put_object(
                Bucket=self._bucket_name,
                Key=key,
                Body=content,
                ContentType=content_type,
            )
        uri = f"s3://{self._bucket_name}/{key}"
        logger.info("artifact_uploaded", uri=uri, size_bytes=len(content))
        return uri

    async def generate_presigned_url(
        self,
        tenant_id: str,
        experiment_id: str,
        filename: str,
        expiry_seconds: int = 3600,
    ) -> str:
        """Generate a presigned download URL for a stored artifact.

        Args:
            tenant_id: Tenant UUID string.
            experiment_id: Experiment UUID string.
            filename: Artifact filename.
            expiry_seconds: URL expiry in seconds (default 1 hour).

        Returns:
            Presigned HTTPS URL for direct artifact download.
        """
        try:
            import aioboto3  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("aioboto3 is required for artifact storage.") from exc

        key = self._key(tenant_id, experiment_id, filename)
        session = aioboto3.Session()
        async with session.client("s3", **self._client_kwargs()) as s3:
            url: str = await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket_name, "Key": key},
                ExpiresIn=expiry_seconds,
            )
        logger.info("presigned_url_generated", key=key, expiry_seconds=expiry_seconds)
        return url

    async def list_artifacts(
        self,
        tenant_id: str,
        experiment_id: str,
    ) -> list[dict[str, object]]:
        """List all artifacts stored for a given experiment.

        Args:
            tenant_id: Tenant UUID string.
            experiment_id: Experiment UUID string.

        Returns:
            List of dicts with keys: filename, uri, size_bytes.
        """
        try:
            import aioboto3  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("aioboto3 is required for artifact storage.") from exc

        prefix = f"artifacts/{tenant_id}/{experiment_id}/"
        session = aioboto3.Session()
        artifacts: list[dict[str, object]] = []
        async with session.client("s3", **self._client_kwargs()) as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self._bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key: str = obj["Key"]
                    filename = key[len(prefix):]
                    artifacts.append(
                        {
                            "filename": filename,
                            "uri": f"s3://{self._bucket_name}/{key}",
                            "size_bytes": obj.get("Size", 0),
                        }
                    )
        return artifacts
