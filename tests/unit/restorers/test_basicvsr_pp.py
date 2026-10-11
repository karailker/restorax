"""Unit tests for BasicVSRPlusPlusRestorer — no GPU, no real weights."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from restorax.core.restorer import RestorerCategory, RestorerParams
from restorax.restorers.super_resolution.basicvsr_pp import BasicVSRPlusPlusRestorer


@pytest.fixture
def mock_restorer() -> BasicVSRPlusPlusRestorer:
    restorer = BasicVSRPlusPlusRestorer()

    def fake_load(device: torch.device) -> None:
        mock_model = _make_mock_model()
        restorer._model = mock_model
        restorer._device = device
        restorer._loaded = True

    with patch.object(restorer, "load", fake_load):
        restorer.load(torch.device("cpu"))
        yield restorer


def _make_mock_model():
    """Returns a callable that simulates 4× upscaling on B T C H W tensors."""

    def forward(tensor: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = tensor.shape
        return tensor.repeat_interleave(4, dim=3).repeat_interleave(4, dim=4)

    class _FakeModel:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return forward(x)

        def eval(self):
            return self

        def to(self, d):
            return self

        def half(self):
            return self

    return _FakeModel()


def test_capabilities() -> None:
    caps = BasicVSRPlusPlusRestorer().capabilities
    assert caps.category == RestorerCategory.SUPER_RESOLUTION
    assert caps.requires_temporal is True
    assert caps.scale_factor == 4


def test_name() -> None:
    assert BasicVSRPlusPlusRestorer().name == "basicvsr_pp_x4"


def test_process_sequence_output_length(mock_restorer: BasicVSRPlusPlusRestorer) -> None:
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(5)]
    params = RestorerParams(scale=4, half_precision=False)
    with patch.object(mock_restorer, "_model") as m:
        m.side_effect = lambda t: t.repeat_interleave(4, dim=3).repeat_interleave(4, dim=4)
        result = mock_restorer.process_sequence(frames, params)
    assert len(result) == 5


def test_process_frame_delegates_to_sequence(mock_restorer: BasicVSRPlusPlusRestorer) -> None:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    params = RestorerParams(scale=4, half_precision=False)
    with patch.object(mock_restorer, "process_sequence") as mock_seq:
        mock_seq.return_value = [np.zeros((64, 64, 3), dtype=np.uint8)]
        mock_restorer.process_frame(frame, params)
        mock_seq.assert_called_once()


def test_is_loaded(mock_restorer: BasicVSRPlusPlusRestorer) -> None:
    assert mock_restorer.is_loaded


def test_unload_clears_model(mock_restorer: BasicVSRPlusPlusRestorer) -> None:
    mock_restorer.unload()
    assert not mock_restorer.is_loaded
    assert mock_restorer._model is None
