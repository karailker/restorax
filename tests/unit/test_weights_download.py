"""Unit tests for the URL weight downloader (no network)."""

from __future__ import annotations

import hashlib
import io
import urllib.error
from pathlib import Path

import pytest

from restorax.core.exceptions import RestorerLoadError
from restorax.utils import weights as weights_mod
from restorax.utils.weights import download_file


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, content_length: int | None = None) -> None:
        super().__init__(payload)
        length = len(payload) if content_length is None else content_length
        self.headers = {"Content-Length": str(length)}

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse) -> None:
    monkeypatch.setattr(weights_mod.urllib.request, "urlopen", lambda url, timeout: response)


def test_downloads_atomically(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = b"x" * 4096
    _patch_urlopen(monkeypatch, _FakeResponse(payload))
    dest = tmp_path / "m" / "w.pth"
    out = download_file("https://example.com/w.pth", dest, min_bytes=1000)
    assert out == dest and dest.read_bytes() == payload
    assert list(dest.parent.glob("*.part")) == []


def test_rejects_non_https(tmp_path: Path) -> None:
    with pytest.raises(RestorerLoadError, match="non-HTTPS"):
        download_file("http://example.com/w.pth", tmp_path / "w.pth")


def test_truncated_download_leaves_no_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_urlopen(monkeypatch, _FakeResponse(b"abc", content_length=100))
    dest = tmp_path / "w.pth"
    with pytest.raises(RestorerLoadError, match="Truncated"):
        download_file("https://example.com/w.pth", dest)
    assert not dest.exists() and list(tmp_path.glob("*.part")) == []


def test_too_small_download_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_urlopen(monkeypatch, _FakeResponse(b"<html>404</html>"))
    with pytest.raises(RestorerLoadError, match="too small"):
        download_file("https://example.com/w.pth", tmp_path / "w.pth", min_bytes=1000)


def test_checksum(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = b"weights" * 100
    good = hashlib.sha256(payload).hexdigest()
    _patch_urlopen(monkeypatch, _FakeResponse(payload))
    assert download_file("https://example.com/w.pth", tmp_path / "ok.pth", sha256=good).exists()
    _patch_urlopen(monkeypatch, _FakeResponse(payload))
    with pytest.raises(RestorerLoadError, match="Checksum mismatch"):
        download_file("https://example.com/w.pth", tmp_path / "bad.pth", sha256="0" * 64)
    assert not (tmp_path / "bad.pth").exists()


def test_network_error_wrapped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom(url: str, timeout: float) -> None:
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(weights_mod.urllib.request, "urlopen", boom)
    with pytest.raises(RestorerLoadError, match="Cannot download"):
        download_file("https://example.com/w.pth", tmp_path / "w.pth")
