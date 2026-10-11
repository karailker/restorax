"""Local filesystem storage backend."""

from __future__ import annotations

from pathlib import Path

from restorax.core.exceptions import StorageError


class LocalStorageBackend:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, data: bytes, key: str) -> str:
        dest = self._root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return key

    def load(self, key: str) -> bytes:
        path = self._root / key
        if not path.exists():
            raise StorageError(f"Key not found in local storage: {key}")
        return path.read_bytes()

    def url(self, key: str) -> str:
        return str((self._root / key).resolve())

    def delete(self, key: str) -> None:
        path = self._root / key
        if path.exists():
            path.unlink()

    def exists(self, key: str) -> bool:
        return (self._root / key).exists()
