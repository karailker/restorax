"""
Shared pytest fixtures.

All fixtures here are CPU-only and require no real model weights.
GPU fixtures are marked @pytest.mark.gpu and skipped in CI by default.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from restorax.core.restorer import (
    BaseRestorer,
    RestorerCapabilities,
    RestorerCategory,
    RestorerParams,
)
from restorax.core.registry import ModelRegistry

pytest_plugins = ["tests.conftest_assets"]

# ── Mock restorer ─────────────────────────────────────────────────────────────

class IdentityRestorer(BaseRestorer):
    """Pass-through restorer for pipeline/registry tests — no weights needed."""

    def __init__(self, scale: int = 1) -> None:
        self._scale = scale
        self._loaded = False
        self.load_call_count = 0
        self.unload_call_count = 0

    @property
    def name(self) -> str:
        return f"identity_x{self._scale}"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.SUPER_RESOLUTION,
            input_color_space="rgb",
            output_color_space="rgb",
            scale_factor=self._scale,
        )

    def load(self, device: torch.device) -> None:
        self.load_call_count += 1
        self._loaded = True

    def unload(self) -> None:
        self.unload_call_count += 1
        self._loaded = False

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        if self._scale == 1:
            return frame.copy()
        # Simple nearest-neighbour upscale for testing
        h, w = frame.shape[:2]
        return np.repeat(np.repeat(frame, self._scale, axis=0), self._scale, axis=1)


@pytest.fixture
def identity_restorer() -> IdentityRestorer:
    return IdentityRestorer(scale=1)


@pytest.fixture
def upscale_restorer() -> IdentityRestorer:
    return IdentityRestorer(scale=4)


@pytest.fixture
def mock_registry(upscale_restorer: IdentityRestorer) -> ModelRegistry:
    registry = ModelRegistry(max_loaded=2)

    # Patch register to accept instances directly for tests
    registry._catalog["identity_x4"] = lambda: upscale_restorer  # type: ignore[assignment]
    registry._catalog["identity_x1"] = lambda: IdentityRestorer(scale=1)  # type: ignore[assignment]
    return registry


# ── Synthetic video fixture ───────────────────────────────────────────────────

@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """
    Create a 5-frame synthetic RGB MP4 in a temp directory.
    Returns the path. No real video file needed — built with PyAV.
    """
    import av

    out_path = tmp_path / "sample.mp4"
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("libx264", rate=24)
    stream.width = 64
    stream.height = 64
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "28"}

    rng = np.random.default_rng(42)
    for i in range(5):
        frame_data = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        av_frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        av_frame.pts = i
        for packet in stream.encode(av_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return out_path


@pytest.fixture
def sample_frame() -> np.ndarray:
    """A 64×64 random RGB uint8 frame."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def default_params() -> RestorerParams:
    return RestorerParams(scale=4, half_precision=False)


def pytest_collection_modifyitems(config, items):
    from pathlib import Path
    try:
        from restorax.config import settings
        model_dir = Path(settings.model_dir)
    except Exception:
        model_dir = Path("models")

    asset_dir = Path(__file__).parent / "assets"

    skip_comfyui = pytest.mark.skip(reason="needs $COMFYUI_PATH set to a real ComfyUI checkout")

    for item in items:
        for marker in item.iter_markers("requires_weights"):
            model_name = marker.args[0] if marker.args else ""
            weight_dir = model_dir / model_name
            if not weight_dir.exists():
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"weights absent: {weight_dir}. Run: restorax download-models --model {model_name}"
                    )
                )
        if item.get_closest_marker("requires_assets"):
            if not asset_dir.exists() or not any(asset_dir.iterdir()):
                item.add_marker(pytest.mark.skip(reason="test assets not downloaded"))
        if "requires_comfyui" in item.keywords and not os.environ.get("COMFYUI_PATH"):
            item.add_marker(skip_comfyui)
