"""Unit tests for the local filesystem storage backend."""

from __future__ import annotations

import pytest

from restorax.core.exceptions import StorageError
from restorax.storage.base import StorageBackend
from restorax.storage.local import LocalStorageBackend


class TestLocalStorageBackend:
    @pytest.fixture
    def store(self, tmp_path):
        return LocalStorageBackend(tmp_path / "storage")

    def test_implements_protocol(self, store):
        assert isinstance(store, StorageBackend)

    def test_save_and_load_roundtrip(self, store):
        data = b"hello restorax"
        store.save(data, "test/file.bin")
        assert store.load("test/file.bin") == data

    def test_save_returns_key(self, store):
        key = store.save(b"x", "mykey")
        assert key == "mykey"

    def test_exists_true_after_save(self, store):
        store.save(b"data", "exists_test.bin")
        assert store.exists("exists_test.bin") is True

    def test_exists_false_for_missing(self, store):
        assert store.exists("does_not_exist.bin") is False

    def test_load_missing_raises_storage_error(self, store):
        with pytest.raises(StorageError, match="not found"):
            store.load("nonexistent.bin")

    def test_delete_removes_file(self, store, tmp_path):
        store.save(b"bye", "todelete.bin")
        assert store.exists("todelete.bin")
        store.delete("todelete.bin")
        assert not store.exists("todelete.bin")

    def test_delete_nonexistent_is_noop(self, store):
        store.delete("ghost.bin")  # must not raise

    def test_url_returns_absolute_path(self, store):
        store.save(b"url_test", "subdir/file.mp4")
        url = store.url("subdir/file.mp4")
        assert "file.mp4" in url
        assert url.startswith("/")

    def test_save_nested_key_creates_directories(self, store):
        store.save(b"deep", "a/b/c/d.txt")
        assert store.exists("a/b/c/d.txt")

    def test_overwrite_existing_key(self, store):
        store.save(b"v1", "overwrite.txt")
        store.save(b"v2", "overwrite.txt")
        assert store.load("overwrite.txt") == b"v2"

    def test_save_empty_bytes(self, store):
        store.save(b"", "empty.bin")
        assert store.load("empty.bin") == b""

    def test_root_created_if_missing(self, tmp_path):
        new_root = tmp_path / "new" / "deep" / "path"
        s = LocalStorageBackend(new_root)
        s.save(b"x", "k")
        assert s.exists("k")
