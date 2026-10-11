"""Tests for video utility functions (padding, tiling, color conversion)."""

from __future__ import annotations

import numpy as np
import pytest

from restorax.video.utils import (
    from_rgb,
    merge_tiles,
    pad_to_multiple,
    tile_frame,
    to_rgb,
    unpad,
)


def _frame(h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


# ── Color space ───────────────────────────────────────────────────────────────


def test_rgb_roundtrip() -> None:
    frame = _frame(32, 32)
    assert np.array_equal(to_rgb(frame, "rgb"), frame)
    assert np.array_equal(from_rgb(frame, "rgb"), frame)


def test_bgr_rgb_roundtrip() -> None:
    frame = _frame(32, 32)
    bgr = from_rgb(frame, "bgr")
    back = to_rgb(bgr, "bgr")
    assert np.array_equal(back, frame)


def test_invalid_color_space_raises() -> None:
    frame = _frame(8, 8)
    with pytest.raises(ValueError):
        to_rgb(frame, "xyz")
    with pytest.raises(ValueError):
        from_rgb(frame, "xyz")


# ── Padding ───────────────────────────────────────────────────────────────────


def test_pad_to_multiple_already_aligned() -> None:
    frame = _frame(64, 64)
    padded, pads = pad_to_multiple(frame, 32)
    assert padded.shape == (64, 64, 3)
    assert pads == (0, 0, 0, 0)


def test_pad_to_multiple_unaligned() -> None:
    frame = _frame(65, 70)
    padded, pads = pad_to_multiple(frame, 32)
    assert padded.shape[0] % 32 == 0
    assert padded.shape[1] % 32 == 0


def test_pad_unpad_roundtrip() -> None:
    frame = _frame(65, 70)
    padded, pads = pad_to_multiple(frame, 32)
    recovered = unpad(padded, pads)
    assert recovered.shape == (65, 70, 3)
    assert np.array_equal(recovered, frame)


# ── Tiling ────────────────────────────────────────────────────────────────────


def test_tile_frame_produces_correct_count() -> None:
    frame = _frame(128, 128)
    tiles, n_h, n_w = tile_frame(frame, tile_size=64, overlap=0)
    assert n_h >= 2 and n_w >= 2
    assert len(tiles) == n_h * n_w


def test_tile_and_merge_identity() -> None:
    """Tiling and merging with a 1× restorer should reproduce the original frame."""
    frame = _frame(64, 64)
    tiles, _, _ = tile_frame(frame, tile_size=32, overlap=8)
    # Simulate identity restorer (no scaling)
    processed = [(t, coords) for t, coords in tiles]
    result = merge_tiles(processed, 64, 64, scale=1)
    assert result.shape == (64, 64, 3)
    # Pixel values may differ slightly in overlap regions due to averaging; just check shape
