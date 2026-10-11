"""
Unit tests for RealESRGANx4Restorer.

These tests do NOT load real weights or require a GPU.
They verify the restorer contract: capabilities, frame shape, dtype, tiling API.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from restorax.core.restorer import RestorerCategory, RestorerParams
from restorax.restorers.super_resolution.real_esrgan import RealESRGANx4Restorer


@pytest.fixture
def mock_restorer(tmp_path):
    """Return a RealESRGANx4Restorer with a mocked model that doubles pixel values."""
    restorer = RealESRGANx4Restorer()

    # Patch weight loading and model creation
    def fake_load(device: torch.device) -> None:
        mock_model = MagicMock()

        def fake_forward(tensor: torch.Tensor) -> torch.Tensor:
            # Simulate 4× upscale by repeating spatially
            b, c, h, w = tensor.shape
            return tensor.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)

        mock_model.side_effect = fake_forward
        mock_model.__call__ = fake_forward
        restorer._model = mock_model
        restorer._device = device
        restorer._loaded = True

    with patch.object(restorer, "load", fake_load):
        restorer.load(torch.device("cpu"))
        yield restorer


def test_capabilities() -> None:
    restorer = RealESRGANx4Restorer()
    caps = restorer.capabilities
    assert caps.category == RestorerCategory.SUPER_RESOLUTION
    assert caps.scale_factor == 4
    assert caps.input_color_space == "rgb"
    assert caps.output_color_space == "rgb"
    assert not caps.requires_temporal


def test_name() -> None:
    assert RealESRGANx4Restorer().name == "real_esrgan_x4plus"


def test_process_frame_output_shape(mock_restorer: RealESRGANx4Restorer) -> None:
    frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    params = RestorerParams(scale=4, half_precision=False)

    with patch.object(mock_restorer, "_process_full") as mock_full:
        expected = np.zeros((128, 128, 3), dtype=np.uint8)
        mock_full.return_value = expected
        result = mock_restorer.process_frame(frame, params)

    assert result.shape == (128, 128, 3)
    assert result.dtype == np.uint8


def test_is_loaded_after_load(mock_restorer: RealESRGANx4Restorer) -> None:
    assert mock_restorer.is_loaded


def test_is_not_loaded_after_unload(mock_restorer: RealESRGANx4Restorer) -> None:
    mock_restorer.unload()
    assert not mock_restorer.is_loaded


def test_tiling_mode_calls_process_tiled(mock_restorer: RealESRGANx4Restorer) -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    params = RestorerParams(scale=4, tile_size=32, half_precision=False)

    with patch.object(mock_restorer, "_process_tiled") as mock_tiled:
        mock_tiled.return_value = np.zeros((256, 256, 3), dtype=np.uint8)
        mock_restorer.process_frame(frame, params)
        mock_tiled.assert_called_once()


def test_no_tiling_when_tile_size_zero(mock_restorer: RealESRGANx4Restorer) -> None:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    params = RestorerParams(scale=4, tile_size=0, half_precision=False)

    with patch.object(mock_restorer, "_process_full") as mock_full:
        mock_full.return_value = np.zeros((128, 128, 3), dtype=np.uint8)
        mock_restorer.process_frame(frame, params)
        mock_full.assert_called_once()
