"""Unit tests for CodeFormerRestorer and GFPGANRestorer — no GPU, no real weights."""

from __future__ import annotations

import numpy as np
import torch

from restorax.core.restorer import RestorerCategory, RestorerParams
from restorax.restorers.face_restoration.codeformer import CodeFormerRestorer
from restorax.restorers.face_restoration.gfpgan import GFPGANRestorer

# ── CodeFormer ────────────────────────────────────────────────────────────────


def test_codeformer_capabilities() -> None:
    caps = CodeFormerRestorer().capabilities
    assert caps.category == RestorerCategory.FACE_RESTORATION
    assert caps.input_color_space == "bgr"
    assert caps.output_color_space == "bgr"
    assert caps.scale_factor == 1
    assert not caps.requires_temporal


def test_codeformer_name() -> None:
    assert CodeFormerRestorer().name == "codeformer"


def test_codeformer_process_frame_no_faces() -> None:
    """When face_helper detects no faces, the original frame must be returned unchanged."""
    from unittest.mock import MagicMock

    restorer = CodeFormerRestorer()
    restorer._loaded = True
    restorer._device = torch.device("cpu")

    mock_net = MagicMock()
    mock_helper = MagicMock()
    mock_helper.cropped_faces = []  # no faces detected

    restorer._net = mock_net
    restorer._face_helper = mock_helper

    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    params = RestorerParams(extra={"fidelity": 0.5})

    result = restorer.process_frame(frame, params)
    assert result.shape == frame.shape
    assert np.array_equal(result, frame)


def test_codeformer_unloads_model() -> None:
    restorer = CodeFormerRestorer()
    restorer._loaded = True
    restorer._net = object()
    restorer._device = torch.device("cpu")
    restorer.unload()
    assert not restorer.is_loaded
    assert restorer._net is None


# ── GFPGAN ────────────────────────────────────────────────────────────────────


def test_gfpgan_capabilities() -> None:
    caps = GFPGANRestorer().capabilities
    assert caps.category == RestorerCategory.FACE_RESTORATION
    assert caps.input_color_space == "bgr"
    assert caps.output_color_space == "bgr"
    assert caps.scale_factor == 1
    assert not caps.requires_temporal


def test_gfpgan_name() -> None:
    assert GFPGANRestorer().name == "gfpgan_v14"


def test_gfpgan_process_frame_returns_frame_on_no_faces() -> None:
    """When GFPGANer returns None for restored_img, the original frame is returned."""
    from unittest.mock import MagicMock

    restorer = GFPGANRestorer()
    restorer._loaded = True
    restorer._device = torch.device("cpu")

    mock_gfpgan = MagicMock()
    mock_gfpgan.enhance.return_value = ([], [], None)  # no output
    restorer._gfpgan = mock_gfpgan

    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    params = RestorerParams()
    result = restorer.process_frame(frame, params)
    assert np.array_equal(result, frame)


def test_gfpgan_process_frame_returns_restored_img() -> None:
    from unittest.mock import MagicMock

    restorer = GFPGANRestorer()
    restorer._loaded = True
    restorer._device = torch.device("cpu")

    restored = np.zeros((64, 64, 3), dtype=np.uint8)
    mock_gfpgan = MagicMock()
    mock_gfpgan.enhance.return_value = ([], [restored], restored)
    restorer._gfpgan = mock_gfpgan

    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    params = RestorerParams()
    result = restorer.process_frame(frame, params)
    assert result.shape == (64, 64, 3)


def test_gfpgan_unloads() -> None:
    restorer = GFPGANRestorer()
    restorer._loaded = True
    restorer._gfpgan = object()
    restorer._device = torch.device("cpu")
    restorer.unload()
    assert not restorer.is_loaded
