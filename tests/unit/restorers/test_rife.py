"""Unit tests for RIFERestorer — no GPU, no real weights."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from restorax.core.restorer import RestorerCategory, RestorerParams
from restorax.restorers.frame_interpolation.rife import RIFERestorer


@pytest.fixture
def loaded_restorer() -> RIFERestorer:
    restorer = RIFERestorer()

    def fake_load(device: torch.device) -> None:
        mock_model = MagicMock()
        mock_model.inference.side_effect = lambda t0, t1, timestep=0.5: (t0 + t1) / 2
        restorer._model = mock_model
        restorer._device = device
        restorer._loaded = True

    with patch.object(restorer, "load", fake_load):
        restorer.load(torch.device("cpu"))
        yield restorer


def test_capabilities() -> None:
    caps = RIFERestorer().capabilities
    assert caps.category == RestorerCategory.FRAME_INTERPOLATION
    assert caps.requires_temporal is True
    assert caps.scale_factor == 1
    assert not caps.supports_compile


def test_name() -> None:
    assert RIFERestorer().name == "rife_v4"


def test_process_sequence_doubles_frame_count(loaded_restorer: RIFERestorer) -> None:
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(4)]
    params = RestorerParams()
    result = loaded_restorer.process_sequence(frames, params)
    # 4 input frames → 4*2-1 = 7 output frames
    assert len(result) == 7


def test_process_sequence_single_frame_passthrough(loaded_restorer: RIFERestorer) -> None:
    frames = [np.zeros((32, 32, 3), dtype=np.uint8)]
    params = RestorerParams()
    result = loaded_restorer.process_sequence(frames, params)
    assert len(result) == 1


def test_process_sequence_two_frames(loaded_restorer: RIFERestorer) -> None:
    f0 = np.full((16, 16, 3), 0, dtype=np.uint8)
    f1 = np.full((16, 16, 3), 200, dtype=np.uint8)
    params = RestorerParams()
    result = loaded_restorer.process_sequence([f0, f1], params)
    # 2 frames → 3 output (f0, mid, f1)
    assert len(result) == 3
    assert result[0].shape == (16, 16, 3)
    assert result[1].shape == (16, 16, 3)
    assert result[2].shape == (16, 16, 3)


def test_interpolated_frame_dtype(loaded_restorer: RIFERestorer) -> None:
    f0 = np.zeros((16, 16, 3), dtype=np.uint8)
    f1 = np.full((16, 16, 3), 100, dtype=np.uint8)
    result = loaded_restorer.process_sequence([f0, f1], RestorerParams())
    for frame in result:
        assert frame.dtype == np.uint8


def test_process_frame_is_passthrough(loaded_restorer: RIFERestorer) -> None:
    frame = np.full((16, 16, 3), 42, dtype=np.uint8)
    result = loaded_restorer.process_frame(frame, RestorerParams())
    assert np.array_equal(result, frame)


def test_unload(loaded_restorer: RIFERestorer) -> None:
    loaded_restorer.unload()
    assert not loaded_restorer.is_loaded
