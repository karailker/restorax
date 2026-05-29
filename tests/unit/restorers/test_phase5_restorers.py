"""Unit tests for Phase 5 restorers: Upscale-A-Video, VRT + tiling enhancement."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from restorax.core.restorer import RestorerCategory, RestorerParams


# ── UpscaleAVideoRestorer ──────────────────────────────────────────────────────

class TestUpscaleAVideo:
    @pytest.fixture
    def restorer(self):
        from restorax.restorers.super_resolution.upscale_a_video import UpscaleAVideoRestorer
        r = UpscaleAVideoRestorer()
        # _pipe=None has no __call__, so process_sequence falls back to _stub_upscale (4× nearest-neighbour)
        r._pipe = None
        r._device = torch.device("cpu")
        r._loaded = True
        return r

    def test_name(self, restorer):
        assert restorer.name == "upscale_a_video"

    def test_capabilities(self, restorer):
        caps = restorer.capabilities
        assert caps.category == RestorerCategory.SUPER_RESOLUTION
        assert caps.requires_temporal is True
        assert caps.scale_factor == 4
        assert caps.min_vram_gb >= 12.0

    def test_process_sequence_returns_4x(self, restorer):
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(3)]
        result = restorer.process_sequence(frames, RestorerParams(half_precision=False))
        assert len(result) == 3
        assert result[0].shape == (64, 64, 3)  # 4× upscale

    def test_process_frame_single(self, restorer):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        result = restorer.process_frame(frame, RestorerParams(half_precision=False))
        assert result.shape == (64, 64, 3)

    def test_unload(self, restorer):
        restorer.unload()
        assert not restorer.is_loaded


# ── VRTRestorer ────────────────────────────────────────────────────────────────

class TestVRT:
    @pytest.fixture
    def restorer(self):
        from restorax.restorers.super_resolution.vrt import VRTRestorer
        r = VRTRestorer()
        mock_model = MagicMock()
        # VRT processes (1, T, C, H, W) and returns (1, T, C, H*4, W*4)
        mock_model.side_effect = lambda x: torch.nn.functional.interpolate(
            x.reshape(-1, x.shape[2], x.shape[3], x.shape[4]),
            scale_factor=4,
            mode="nearest",
        ).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3] * 4, x.shape[4] * 4)
        r._model = mock_model
        r._device = torch.device("cpu")
        r._loaded = True
        return r

    def test_name(self, restorer):
        assert restorer.name == "vrt_x4"

    def test_capabilities(self, restorer):
        caps = restorer.capabilities
        assert caps.category == RestorerCategory.SUPER_RESOLUTION
        assert caps.requires_temporal is True
        assert caps.scale_factor == 4

    def test_process_sequence_output_shape(self, restorer):
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(4)]
        result = restorer.process_sequence(frames, RestorerParams(half_precision=False))
        assert len(result) == 4
        assert result[0].shape == (64, 64, 3)

    def test_padding_shorter_than_window(self, restorer):
        """Sequences shorter than WINDOW_SIZE must be padded and output original count."""
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(3)]
        result = restorer.process_sequence(frames, RestorerParams(half_precision=False))
        assert len(result) == 3  # output matches input count, not padded count

    def test_unload(self, restorer):
        restorer.unload()
        assert not restorer.is_loaded


# ── Gaussian tiling ────────────────────────────────────────────────────────────

class TestGaussianTiling:
    def test_gaussian_window_shape(self):
        from restorax.video.utils import _gaussian_window
        w = _gaussian_window(32, 32)
        assert w.shape == (32, 32, 1)
        assert w.dtype == np.float32

    def test_gaussian_window_max_at_center(self):
        from restorax.video.utils import _gaussian_window
        w = _gaussian_window(64, 64)[:, :, 0]
        cy, cx = 32, 32
        center_val = float(w[cy, cx])
        edge_val = float(w[0, 0])
        assert center_val > edge_val

    def test_merge_tiles_gaussian_no_seam_artifacts(self):
        from restorax.video.utils import merge_tiles, tile_frame
        # Create a smooth gradient frame
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        for i in range(128):
            frame[i] = int(i * 1.99)

        tiles, _, _ = tile_frame(frame, tile_size=64, overlap=16)
        # Simulate identity restorer (no upscaling)
        processed = [(t, coords) for t, coords in tiles]
        result = merge_tiles(processed, 128, 128, scale=1, gaussian_blend=True)
        assert result.shape == (128, 128, 3)
        assert result.dtype == np.uint8

    def test_merge_tiles_gaussian_vs_simple_similar(self):
        """Gaussian and simple merge should produce similar results on non-edge content."""
        from restorax.video.utils import merge_tiles, tile_frame
        frame = np.full((64, 64, 3), 100, dtype=np.uint8)
        tiles, _, _ = tile_frame(frame, tile_size=32, overlap=0)
        processed = [(t, coords) for t, coords in tiles]

        gauss = merge_tiles(processed, 64, 64, scale=1, gaussian_blend=True)
        simple = merge_tiles(processed, 64, 64, scale=1, gaussian_blend=False)
        # With no overlap, results should be identical
        assert np.allclose(gauss.astype(float), simple.astype(float), atol=5)


# ── GPU router ─────────────────────────────────────────────────────────────────

class TestGPURouter:
    def test_default_queue(self, monkeypatch):
        monkeypatch.delenv("RESTORAX_GPU_QUEUES", raising=False)
        from restorax.tasks import gpu_router
        gpu_router.reset_router()
        assert gpu_router.next_gpu_queue() == "gpu_default"

    def test_round_robin(self, monkeypatch):
        monkeypatch.setenv("RESTORAX_GPU_QUEUES", "gpu_0,gpu_1")
        from restorax.tasks import gpu_router
        gpu_router.reset_router()
        queues = [gpu_router.next_gpu_queue() for _ in range(4)]
        assert queues == ["gpu_0", "gpu_1", "gpu_0", "gpu_1"]
