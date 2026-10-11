"""StorageBackend protocol — implemented by local.py and s3.py."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Abstraction over local filesystem and S3-compatible object storage."""

    def save(self, data: bytes, key: str) -> str:
        """
        Persist data and return its storage key (path or object key).

        Args:
            data: Raw bytes to store.
            key:  Relative path/key within the storage root.

        Returns:
            The storage key that can later be passed to load() or url().
        """
        ...

    def load(self, key: str) -> bytes:
        """Retrieve bytes by storage key. Raises StorageError if not found."""
        ...

    def url(self, key: str) -> str:
        """Return a URL or absolute path string suitable for serving the resource."""
        ...

    def delete(self, key: str) -> None:
        """Remove a stored object. No-op if key does not exist."""
        ...

    def exists(self, key: str) -> bool:
        """Return True if the key exists in storage."""
        ...
