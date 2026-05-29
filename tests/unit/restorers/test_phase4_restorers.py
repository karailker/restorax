"""Unit tests for Phase 4 restorers: scratch removal, HDR, stabilization, deinterlacing."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from restorax.core.restorer import RestorerCategory, RestorerParams


# ── ScratchRemovalRestorer ─────────────────────────────────────────────────────

class TestScratchRemoval:
    @pytest.fixture
    def restorer(self):
        from restorax.restorers.artifact_removal.scratch_removal import ScratchRemovalRestorer
        r = ScratchRemovalRestorer()
        r._device = torch.device("cpu")
        r._loaded = True
        mock_model = MagicMock()
        mock_model.inpaint.side_effect = lambda frames, masks: frames
        r._model = mock_model
        return r

    def test_name(self, restorer):
        assert restorer.name == "scratch_removal"

    def test_capabilities(self, restorer):
        caps = restorer.capabilities
        assert caps.category == RestorerCategory.ARTIFACT_REMOVAL
        assert caps.requires_temporal is True
        assert caps.scale_factor == 1

    def test_process_frame_no_scratch(self, restorer):
        # Plain gray frame — no scratches
        frame = np.full((32, 32, 3), 128, dtype=np.uint8)
        result = restorer.process_frame(frame, RestorerParams())
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_process_sequence_length_preserved(self, restorer):
        frames = [np.full((32, 32, 3), 100, dtype=np.uint8) for _ in range(5)]
        result = restorer.process_sequence(frames, RestorerParams())
        assert len(result) == 5

    def test_process_frame_with_white_streak(self, restorer):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, 30:33, :] = 255  # bright vertical streak
        result = restorer.process_frame(frame, RestorerParams())
        assert result.shape == (64, 64, 3)

    def test_scratch_detection_temporal(self, restorer):
        from restorax.restorers.artifact_removal.scratch_removal import ScratchRemovalRestorer
        frames = [np.full((32, 32, 3), 50, dtype=np.uint8) for _ in range(4)]
        frames[2][:, 15:17] = 240  # scratch only in frame 2
        masks = ScratchRemovalRestorer._detect_scratches_temporal(frames)
        assert len(masks) == 4
        # Frame 2 should have more mask pixels than adjacent frames
        assert masks[2].sum() >= masks[0].sum()


# ── HDRTVDMRestorer ────────────────────────────────────────────────────────────

class TestHDRTVDM:
    @pytest.fixture
    def restorer(self):
        from restorax.restorers.hdr.hdrtvdm import HDRTVDMRestorer
        r = HDRTVDMRestorer()
        mock_model = MagicMock()
        mock_model.side_effect = lambda t: t  # passthrough: same shape tensor
        r._model = mock_model
        r._device = torch.device("cpu")
        r._loaded = True
        return r

    def test_name(self, restorer):
        assert restorer.name == "hdrtvdm"

    def test_capabilities(self, restorer):
        caps = restorer.capabilities
        assert caps.category == RestorerCategory.HDR_CONVERSION
        assert caps.scale_factor == 1
        assert not caps.requires_temporal

    def test_process_frame_output_shape(self, restorer):
        frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = restorer.process_frame(frame, RestorerParams())
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8

    def test_unload(self, restorer):
        restorer.unload()
        assert not restorer.is_loaded


# ── VideoStabilizationRestorer ─────────────────────────────────────────────────

class TestVideoStabilization:
    @pytest.fixture
    def restorer(self):
        from restorax.restorers.stabilization.deep_flow_stab import VideoStabilizationRestorer
        r = VideoStabilizationRestorer()
        r.load(torch.device("cpu"))
        return r

    def test_name(self, restorer):
        assert restorer.name == "video_stabilization"

    def test_capabilities(self, restorer):
        caps = restorer.capabilities
        assert caps.category == RestorerCategory.STABILIZATION
        assert caps.requires_temporal is True
        assert caps.min_vram_gb == 0.0

    def test_process_frame_passthrough(self, restorer):
        frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = restorer.process_frame(frame, RestorerParams())
        assert np.array_equal(result, frame)

    def test_process_sequence_length_preserved(self, restorer):
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(6)]
        result = restorer.process_sequence(frames, RestorerParams())
        assert len(result) == 6
        assert all(f.shape == (64, 64, 3) for f in result)

    def test_single_frame_sequence(self, restorer):
        frames = [np.zeros((32, 32, 3), dtype=np.uint8)]
        result = restorer.process_sequence(frames, RestorerParams())
        assert len(result) == 1

    def test_smooth_trajectory_shape(self):
        from restorax.restorers.stabilization.deep_flow_stab import VideoStabilizationRestorer
        traj = np.random.randn(20, 3)
        smoothed = VideoStabilizationRestorer._smooth_trajectory(traj)
        assert smoothed.shape == traj.shape


# ── AIDeinterlaceRestorer ──────────────────────────────────────────────────────

class TestAIDeinterlace:
    @pytest.fixture
    def restorer(self):
        from restorax.restorers.deinterlacing.ai_deinterlace import AIDeinterlaceRestorer
        r = AIDeinterlaceRestorer()
        mock_model = MagicMock()
        mock_model.side_effect = lambda t: t  # passthrough: same-shape tensor
        r._model = mock_model
        r._device = torch.device("cpu")
        r._loaded = True
        return r

    def test_name(self, restorer):
        assert restorer.name == "ai_deinterlace"

    def test_capabilities(self, restorer):
        caps = restorer.capabilities
        assert caps.category == RestorerCategory.DEINTERLACING
        assert not caps.requires_temporal
        assert caps.scale_factor == 1

    def test_process_frame_not_interlaced(self, restorer):
        frame = np.random.randint(0, 200, (64, 64, 3), dtype=np.uint8)
        result = restorer.process_frame(frame, RestorerParams())
        assert result.shape == (64, 64, 3)

    def test_process_sequence_length_preserved(self, restorer):
        frames = [np.random.randint(0, 200, (32, 32, 3), dtype=np.uint8) for _ in range(5)]
        result = restorer.process_sequence(frames, RestorerParams())
        assert len(result) == 5

    def test_is_interlaced_clean_frame_false(self, restorer):
        from restorax.restorers.deinterlacing.ai_deinterlace import AIDeinterlaceRestorer
        # Smooth gradient — not interlaced
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(64):
            frame[i] = i * 3
        assert not AIDeinterlaceRestorer._is_interlaced(frame)

    def test_unload(self, restorer):
        restorer.unload()
        assert not restorer.is_loaded
