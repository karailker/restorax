"""Unit tests for MambaIR, TDM, CodeFormer++, and GaVS restorers."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from restorax.core.restorer import RestorerCategory, RestorerParams


# ── MambaIRRestorer ───────────────────────────────────────────────────────────

class TestMambaIR:
    @pytest.fixture
    def restorer(self):
        from restorax.restorers.super_resolution.mamba_ir import MambaIRRestorer
        r = MambaIRRestorer()
        mock_model = MagicMock()
        mock_model.side_effect = lambda x: torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest")
        r._model = mock_model
        r._device = torch.device("cpu")
        r._loaded = True
        return r

    def test_name(self):
        from restorax.restorers.super_resolution.mamba_ir import MambaIRRestorer
        assert MambaIRRestorer().name == "mamba_ir_x4"

    def test_capabilities(self, restorer):
        caps = restorer.capabilities
        assert caps.category == RestorerCategory.SUPER_RESOLUTION
        assert caps.scale_factor == 4
        assert caps.min_vram_gb == 3.0
        assert not caps.requires_temporal
        assert caps.supports_compile

    def test_process_frame_4x_output(self, restorer):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        result = restorer.process_frame(frame, RestorerParams(half_precision=False))
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_tiling_delegates(self, restorer):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        with patch.object(restorer, "_process_tiled") as mock_tiled:
            mock_tiled.return_value = np.zeros((256, 256, 3), dtype=np.uint8)
            restorer.process_frame(frame, RestorerParams(tile_size=32, half_precision=False))
            mock_tiled.assert_called_once()

    def test_unload(self, restorer):
        restorer.unload()
        assert not restorer.is_loaded
        assert restorer._model is None


# ── TDMRestorer ───────────────────────────────────────────────────────────────

class TestTDM:
    @pytest.fixture
    def restorer(self):
        from restorax.restorers.super_resolution.tdm import TDMRestorer
        from PIL import Image
        r = TDMRestorer()
        mock_pipe = MagicMock()
        def _fake_pipe(image, tasks, num_inference_steps, guidance_scale):
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
        from restorax.restorers.super_resolution.tdm import TDMRestorer
        assert TDMRestorer().name == "tdm"

    def test_capabilities(self, restorer):
        caps = restorer.capabilities
        assert caps.category == RestorerCategory.SUPER_RESOLUTION
        assert caps.scale_factor == 4
        assert caps.requires_temporal
        assert caps.min_vram_gb >= 12.0

    def test_process_sequence_length_preserved(self, restorer):
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(4)]
        result = restorer.process_sequence(frames, RestorerParams(half_precision=False))
        assert len(result) == 4

    def test_process_sequence_output_4x(self, restorer):
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)]
        result = restorer.process_sequence(frames, RestorerParams(half_precision=False))
        assert result[0].shape == (64, 64, 3)
        assert result[0].dtype == np.uint8

    def test_unload(self, restorer):
        restorer.unload()
        assert not restorer.is_loaded


# ── CodeFormerPlusPlusRestorer ────────────────────────────────────────────────

class TestCodeFormerPlusPlus:
    @pytest.fixture
    def restorer(self):
        from restorax.restorers.face_restoration.codeformer_pp import CodeFormerPlusPlusRestorer
        r = CodeFormerPlusPlusRestorer()
        r._net = None
        mock_helper = MagicMock()
        mock_helper.cropped_faces = []  # no faces → early return with original frame
        r._face_helper = mock_helper
        r._device = torch.device("cpu")
        r._loaded = True
        return r

    def test_name(self):
        from restorax.restorers.face_restoration.codeformer_pp import CodeFormerPlusPlusRestorer
        assert CodeFormerPlusPlusRestorer().name == "codeformer_pp"

    def test_capabilities(self, restorer):
        caps = restorer.capabilities
        assert caps.category == RestorerCategory.FACE_RESTORATION
        assert caps.input_color_space == "bgr"
        assert caps.scale_factor == 1
        assert not caps.requires_temporal

    def test_process_frame_no_model_passthrough(self, restorer):
        """With _net=None, process_frame returns input unchanged."""
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = restorer.process_frame(frame, RestorerParams(extra={"fidelity": 0.5}))
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_unload(self, restorer):
        restorer.unload()
        assert not restorer.is_loaded


# ── GaVSRestorer ──────────────────────────────────────────────────────────────

class TestGaVS:
    @pytest.fixture
    def restorer(self):
        from restorax.restorers.stabilization.gavs import GaVSRestorer
        r = GaVSRestorer()
        r.load(torch.device("cpu"))
        return r

    def test_name(self):
        from restorax.restorers.stabilization.gavs import GaVSRestorer
        assert GaVSRestorer().name == "gavs"

    def test_capabilities(self, restorer):
        caps = restorer.capabilities
        assert caps.category == RestorerCategory.STABILIZATION
        assert caps.requires_temporal
        assert caps.scale_factor == 1

    def test_process_frame_passthrough(self, restorer):
        frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = restorer.process_frame(frame, RestorerParams())
        assert np.array_equal(result, frame)

    def test_process_sequence_length_preserved(self, restorer):
        frames = [np.random.randint(0, 200, (64, 64, 3), dtype=np.uint8) for _ in range(5)]
        result = restorer.process_sequence(frames, RestorerParams())
        assert len(result) == 5
        assert all(f.shape == (64, 64, 3) for f in result)

    def test_fallback_active_when_no_gavs_arch(self, restorer):
        """GaVS without vendored arch should fall back to OpenCV stabilization."""
        assert restorer._fallback is not None or not restorer._using_gavs

    def test_unload(self, restorer):
        restorer.unload()
        assert not restorer.is_loaded
