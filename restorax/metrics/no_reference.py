"""
No-reference (blind) video quality metrics.

These metrics assess quality without needing a ground-truth reference,
making them suitable for evaluating real degraded footage where the
original clean version is unavailable.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def niqe(img: np.ndarray) -> float:
    """
    Natural Image Quality Evaluator (lower is better, ~3–5 for high quality).

    Uses OpenCV's NIQE implementation if available, otherwise falls back
    to a simplified statistical estimator.
    """
    try:
        import cv2

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        niqe_obj = cv2.quality.QualityNIQE_create()  # type: ignore[attr-defined]
        score, _ = niqe_obj.compute(gray)
        return float(score)
    except (ImportError, AttributeError):
        # Simplified NIQE proxy: use local variance statistics
        return _niqe_simple(img)


def brisque_score(img: np.ndarray) -> float:
    """
    Blind/Referenceless Image Spatial Quality Evaluator (lower is better).
    Returns 0.0 if the cv2.quality module is unavailable.
    """
    try:
        import cv2

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        brisque_obj = cv2.quality.QualityBRISQUE_create(  # type: ignore[attr-defined]
            model_file_path="",
            range_file_path="",
        )
        score, _ = brisque_obj.compute(gray)
        return float(score)
    except Exception:
        return 0.0


# ── Internal ──────────────────────────────────────────────────────────────────


def _niqe_simple(img: np.ndarray) -> float:
    """
    Rough NIQE proxy using local patch variance statistics.
    Not calibrated to the original NIQE model — use as a relative indicator only.
    """
    gray = img.mean(axis=2) if img.ndim == 3 else img.astype(np.float64)
    patch_size = 32
    h, w = gray.shape
    variances = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = gray[y : y + patch_size, x : x + patch_size]
            variances.append(float(patch.var()))
    if not variances:
        return 0.0
    # Lower variance = blurrier = worse quality → invert to match NIQE convention
    mean_var = float(np.mean(variances))
    return float(100.0 / (mean_var + 1e-6))
