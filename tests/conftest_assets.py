# tests/conftest_assets.py
from __future__ import annotations

from pathlib import Path

import pytest

_ASSET_DIR = Path(__file__).parent / "assets"

_ASSETS = {
    "set5/butterfly.png": "https://huggingface.co/datasets/eugenesiow/Set5/resolve/main/data/butterfly.png",
    "set5/baby.png": "https://huggingface.co/datasets/eugenesiow/Set5/resolve/main/data/baby.png",
    "set5/bird.png": "https://huggingface.co/datasets/eugenesiow/Set5/resolve/main/data/bird.png",
    "set5/head.png": "https://huggingface.co/datasets/eugenesiow/Set5/resolve/main/data/head.png",
    "set5/woman.png": "https://huggingface.co/datasets/eugenesiow/Set5/resolve/main/data/woman.png",
    "big_buck_bunny_360p_10s.mp4": (
        "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4"
    ),
}


def _fetch(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)


@pytest.fixture(scope="session")
def test_assets() -> Path:
    """Download standard benchmark assets once per session. Skips if network unavailable."""
    _ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for rel, url in _ASSETS.items():
        dest = _ASSET_DIR / rel
        if not dest.exists():
            try:
                _fetch(url, dest)
            except Exception as exc:
                pytest.skip(f"Could not fetch asset {rel}: {exc}")
    return _ASSET_DIR
