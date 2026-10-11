"""Tests for full-reference quality metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from restorax.metrics.full_reference import compute_all, psnr, ssim


def _identical_frames() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    f = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    return f, f.copy()


def _different_frames() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    f1 = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    f2 = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    return f1, f2


def test_psnr_identical_is_inf() -> None:
    f1, f2 = _identical_frames()
    result = psnr(f1, f2)
    assert result == float("inf")


def test_psnr_different_is_finite_positive() -> None:
    f1, f2 = _different_frames()
    result = psnr(f1, f2)
    assert math.isfinite(result)
    assert result > 0


def test_psnr_higher_for_similar_frames() -> None:
    rng = np.random.default_rng(7)
    ref = rng.integers(100, 150, (32, 32, 3), dtype=np.uint8)
    similar = np.clip(ref.astype(int) + 5, 0, 255).astype(np.uint8)
    different = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    assert psnr(similar, ref) > psnr(different, ref)


def test_ssim_identical_is_one() -> None:
    f1, f2 = _identical_frames()
    result = ssim(f1, f2)
    assert abs(result - 1.0) < 0.01


def test_ssim_range() -> None:
    f1, f2 = _different_frames()
    result = ssim(f1, f2)
    assert -1.0 <= result <= 1.0


def test_compute_all_returns_all_keys() -> None:
    f1, f2 = _identical_frames()
    result = compute_all([f1], [f2])
    assert "psnr" in result
    assert "ssim" in result
    assert "lpips" in result


def test_compute_all_length_mismatch_raises() -> None:
    f = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_all([f, f], [f])


def test_compute_all_identical_frames() -> None:
    f1, f2 = _identical_frames()
    result = compute_all([f1, f1], [f2, f2])
    assert result["psnr"] == float("inf")
    assert abs(result["ssim"] - 1.0) < 0.01


def test_pipeline_color_space_conversion() -> None:
    """Verify _convert_cs correctly round-trips RGB→BGR→RGB."""
    from restorax.core.pipeline import _convert_cs

    rng = np.random.default_rng(3)
    frame = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
    bgr = _convert_cs(frame, "rgb", "bgr")
    back = _convert_cs(bgr, "bgr", "rgb")
    assert np.array_equal(frame, back)
