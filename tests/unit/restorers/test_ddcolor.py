"""Unit tests for DDColorRestorer — no GPU, no real weights."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from restorax.core.restorer import RestorerCategory, RestorerParams
from restorax.restorers.colorization.ddcolor import DDColorRestorer


@pytest.fixture
def loaded_restorer() -> DDColorRestorer:
    restorer = DDColorRestorer()

    def fake_load(device: torch.device) -> None:
        mock_model = MagicMock()
        mock_model.side_effect = lambda x: torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3])
        restorer._model = mock_model
        restorer._device = device
        restorer._loaded = True

    with patch.object(restorer, "load", fake_load):
        restorer.load(torch.device("cpu"))
        yield restorer


def test_capabilities() -> None:
    caps = DDColorRestorer().capabilities
    assert caps.category == RestorerCategory.COLORIZATION
    assert caps.input_color_space == "rgb"
    assert caps.output_color_space == "rgb"
    assert caps.scale_factor == 1
    assert not caps.requires_temporal


def test_name() -> None:
    assert DDColorRestorer().name == "ddcolor"


def test_process_frame_output_shape(loaded_restorer: DDColorRestorer) -> None:
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    params = RestorerParams()
    result = loaded_restorer.process_frame(frame, params)
    assert result.shape == (64, 64, 3)
    assert result.dtype == np.uint8


def test_process_frame_grayscale_input(loaded_restorer: DDColorRestorer) -> None:
    """Grayscale input (all channels equal) should produce valid output without crashing."""
    gray = np.full((32, 32, 3), 128, dtype=np.uint8)
    params = RestorerParams()
    result = loaded_restorer.process_frame(gray, params)
    assert result.shape == (32, 32, 3)


def test_is_loaded(loaded_restorer: DDColorRestorer) -> None:
    assert loaded_restorer.is_loaded


def test_unload(loaded_restorer: DDColorRestorer) -> None:
    loaded_restorer.unload()
    assert not loaded_restorer.is_loaded
    assert loaded_restorer._model is None
