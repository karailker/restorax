"""Unit tests for Waifu2x, FlashVSR, EvTexture, SeedVR, DicFace restorers."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from restorax.core.restorer import RestorerCategory, RestorerParams


# ── Waifu2xRestorer ───────────────────────────────────────────────────────────

class TestWaifu2x:
    @pytest.fixture
    def restorer(self):
        from restorax.restorers.super_resolution.waifu2x import Waifu2xRestorer
        r = Waifu2xRestorer()
        mock_model = MagicMock()
        mock_model.side_effect = lambda x: torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        r._model = mock_model
        r._device = torch.device("cpu")
        r._loaded = True
        return r

    def test_name(self):
        from restorax.restorers.super_resolution.waifu2x import Waifu2xRestorer
        assert Waifu2xRestorer().name == "waifu2x_x2"

    def test_capabilities(self, restorer):
        caps = restorer.capabilities
        assert caps.category == RestorerCategory.SUPER_RESOLUTION
        assert caps.scale_factor == 2
        assert not caps.requires_temporal
        assert "anime" in caps.tags

    def test_process_frame_2x_output(self, restorer):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        result = restorer.process_frame(frame, RestorerParams(half_precision=False))
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8

    def test_unload(self, restorer):
        restorer.unload()
        assert not restorer.is_loaded


# ── FlashVSRRestorer ──────────────────────────────────────────────────────────

class TestFlashVSR:
    @pytest.fixture
    def restorer(self):
        from restorax.restorers.super_resolution.flashvsr import FlashVSRRestorer
        r = FlashVSRRestorer()
        mock_model = MagicMock()
        # video shape: (1, T, C, H, W) → output same shape with 4x spatial
        mock_model.side_effect = lambda v: torch.nn.functional.interpolate(
            v.flatten(0, 1), scale_factor=4, mode="nearest"
        ).unflatten(0, (v.shape[0], v.shape[1]))
        r._model = mock_model
        r._device = torch.device("cpu")
        r._loaded = True
        return r

    def test_name(self):
        from restorax.restorers.super_resolution.flashvsr import FlashVSRRestorer
        assert FlashVSRRestorer().name == "flashvsr_x4"

    def test_capabilities(self, restorer):
        caps = restorer.capabilities
        assert caps.scale_factor == 4
        assert caps.requires_temporal
        assert caps.min_vram_gb <= 4.0  # lightweight
        assert "fast" in caps.tags

    def test_process_sequence_4x(self, restorer):
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(3)]
        result = restorer.process_sequence(frames, RestorerParams(half_precision=False))
        assert len(result) == 3
        assert result[0].shape == (64, 64, 3)

    def test_unload(self, restorer):
        restorer.unload()
        assert not restorer.is_loaded


# ── EvTextureRestorer ─────────────────────────────────────────────────────────

class TestEvTexture:
    @pytest.fixture
    def restorer(self):
        from restorax.restorers.super_resolution.evtexture import EvTextureRestorer
        r = EvTextureRestorer()
        mock_model = MagicMock()
        # video shape: (1, T, C, H, W); events also passed but we ignore it
        mock_model.side_effect = lambda v, e: torch.nn.functional.interpolate(
            v.flatten(0, 1), scale_factor=4, mode="nearest"
        ).unflatten(0, (v.shape[0], v.shape[1]))
        r._model = mock_model
        r._device = torch.device("cpu")
        r._loaded = True
        return r

    def test_name(self):
        from restorax.restorers.super_resolution.evtexture import EvTextureRestorer
        assert EvTextureRestorer().name == "evtexture_x4"

    def test_capabilities(self, restorer):
        caps = restorer.capabilities
        assert caps.scale_factor == 4
        assert caps.requires_temporal
        assert "event_guided" in caps.tags

    def test_process_sequence_4x(self, restorer):
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(4)]
        result = restorer.process_sequence(frames, RestorerParams(half_precision=False))
        assert len(result) == 4
        assert result[0].shape == (64, 64, 3)

    def test_simulate_events_shape(self):
        from restorax.restorers.super_resolution.evtexture import EvTextureRestorer
        video = torch.zeros(1, 4, 3, 16, 16)
        events = EvTextureRestorer._simulate_events(video)
        assert events.shape == (1, 4, 2, 16, 16)  # b t 2(pos/neg) h w

    def test_unload(self, restorer):
        restorer.unload()
        assert not restorer.is_loaded


# ── SeedVRRestorer ────────────────────────────────────────────────────────────

class TestSeedVR:
    @pytest.fixture
    def restorer(self):
        from restorax.restorers.super_resolution.seedvr import SeedVRRestorer
        from PIL import Image
        r = SeedVRRestorer()
        mock_pipe = MagicMock()
        def _fake_pipe(image, num_inference_steps, guidance_scale):
            result = MagicMock()
            result.frames = [
                Image.fromarray(np.zeros((f.height * 4, f.width * 4, 3), dtype=np.uint8))
                for f in image
            ]
            return result
        mock_pipe.side_effect = _fake_pipe
        r._pipe = mock_pipe
        r._device = torch.device("cpu")
        r._loaded = True
        return r

    def test_name(self):
        from restorax.restorers.super_resolution.seedvr import SeedVRRestorer
        assert SeedVRRestorer().name == "seedvr"

    def test_capabilities(self, restorer):
        caps = restorer.capabilities
        assert caps.scale_factor == 4
        assert caps.requires_temporal
        assert caps.min_vram_gb >= 16.0
        assert "cvpr2025" in caps.tags

    def test_process_sequence_returns_4x(self, restorer):
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)]
        result = restorer.process_sequence(frames, RestorerParams(half_precision=False))
        assert len(result) == 2
        assert result[0].shape == (64, 64, 3)

    def test_unload(self, restorer):
        restorer.unload()
        assert not restorer.is_loaded


# ── DicFaceRestorer ───────────────────────────────────────────────────────────

class TestDicFace:
    @pytest.fixture
    def restorer(self):
        from restorax.restorers.face_restoration.dicface import DicFaceRestorer
        r = DicFaceRestorer()
        r._net = None
        mock_helper = MagicMock()
        mock_helper.cropped_faces = []  # no faces → early return with original frame
        r._face_helper = mock_helper
        r._device = torch.device("cpu")
        r._loaded = True
        return r

    def test_name(self):
        from restorax.restorers.face_restoration.dicface import DicFaceRestorer
        assert DicFaceRestorer().name == "dicface"

    def test_capabilities(self, restorer):
        caps = restorer.capabilities
        assert caps.category == RestorerCategory.FACE_RESTORATION
        assert caps.input_color_space == "bgr"
        assert caps.scale_factor == 1
        assert "iccv2023" in caps.tags

    def test_process_frame_passthrough_when_no_model(self, restorer):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = restorer.process_frame(frame, RestorerParams(extra={"fidelity": 0.5}))
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_unload(self, restorer):
        restorer.unload()
        assert not restorer.is_loaded


# ── BenchmarkSuites ───────────────────────────────────────────────────────────

class TestBenchmarkSuites:
    def test_sr_suite_runs(self):
        from restorax.benchmarks.suites import SuperResolutionSuite
        from tests.conftest import IdentityRestorer
        r = IdentityRestorer(scale=1)
        r.load(torch.device("cpu"))
        suite = SuperResolutionSuite(width=32, height=32, num_pairs=2)
        result = suite.run(r, device_str="cpu")
        assert len(result.results) == 4  # 4 degradation types

    def test_colorization_suite_runs(self):
        from restorax.benchmarks.suites import ColorizationSuite
        from tests.conftest import IdentityRestorer
        r = IdentityRestorer(scale=1)
        r.load(torch.device("cpu"))
        suite = ColorizationSuite(width=32, height=32, num_pairs=2)
        result = suite.run(r, device_str="cpu")
        assert len(result.results) == 3

    def test_face_restoration_suite_runs(self):
        from restorax.benchmarks.suites import FaceRestorationSuite
        from tests.conftest import IdentityRestorer
        r = IdentityRestorer(scale=1)
        r.load(torch.device("cpu"))
        suite = FaceRestorationSuite(width=32, height=32, num_pairs=2)
        result = suite.run(r, device_str="cpu")
        assert len(result.results) == 3  # light/medium/heavy
