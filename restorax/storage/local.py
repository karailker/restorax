"""Local filesystem storage backend."""

from __future__ import annotations

from pathlib import Path

from restorax.core.exceptions import StorageError


class LocalStorageBackend:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        """Resolve a storage key, rejecting paths that escape the root."""
        path = (self._root / key).resolve()
        if not path.is_relative_to(self._root):
            raise StorageError(f"Storage key escapes root: {key!r}")
        return path

    def save(self, data: bytes, key: str) -> str:
        dest = self._resolve(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return key

    def load(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.exists():
            raise StorageError(f"Key not found in local storage: {key}")
        return path.read_bytes()

    def url(self, key: str) -> str:
        return str(self._resolve(key))

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        if path.exists():
            path.unlink()

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()
