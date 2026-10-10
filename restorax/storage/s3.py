"""
S3 / MinIO storage backend.

Uses boto3 under the hood. The endpoint_url setting makes it compatible
with self-hosted MinIO for local/development deployments.

Install: pip install boto3
"""

from __future__ import annotations

import logging

from restorax.core.exceptions import StorageError

logger = logging.getLogger(__name__)


class S3StorageBackend:
    def __init__(
        self,
        bucket: str,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        region: str = "us-east-1",
    ) -> None:
        try:
            import boto3
            from botocore.exceptions import ClientError as _CE

            self._ClientError = _CE
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for S3 storage. Install with: pip install boto3"
            ) from exc

        import boto3

        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        self._endpoint_url = endpoint_url.rstrip("/")
        self._ensure_bucket()

    def save(self, data: bytes, key: str) -> str:
        try:
            self._client.put_object(Bucket=self._bucket, Key=key, Body=data)
        except self._ClientError as exc:
            raise StorageError(f"S3 put failed for key '{key}': {exc}") from exc
        return key

    def load(self, key: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            data: bytes = response["Body"].read()
            return data
        except self._ClientError as exc:
            raise StorageError(f"S3 get failed for key '{key}': {exc}") from exc

    def url(self, key: str) -> str:
        # Pre-signed URL valid for 1 hour; works with MinIO and AWS S3
        try:
            url: str = self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=3600,
            )
            return url
        except Exception:
            return f"{self._endpoint_url}/{self._bucket}/{key}"

    def delete(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
        except self._ClientError as exc:
            logger.warning("S3 delete failed for key '%s': %s", key, exc)

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except self._ClientError:
            return False

    def _ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except self._ClientError:
            try:
                self._client.create_bucket(Bucket=self._bucket)
                logger.info("Created S3 bucket: %s", self._bucket)
            except self._ClientError as exc:
                raise StorageError(f"Cannot create bucket '{self._bucket}': {exc}") from exc


def get_storage_backend() -> object:
    """Factory: return the configured storage backend based on settings."""
    from restorax.config import settings

    if settings.storage_backend == "s3":
        return S3StorageBackend(
            bucket=settings.s3_bucket,
            endpoint_url=settings.s3_endpoint_url,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
        )
    from restorax.storage.local import LocalStorageBackend

    return LocalStorageBackend(root=settings.storage_local_root)
