"""
Full-reference video quality metrics.

All functions accept numpy uint8 arrays (HxWxC RGB) or lists thereof.
Returns float scalars (higher is better for PSNR/SSIM, lower for LPIPS).
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB. Higher is better. Max ~∞ for identical images."""
    mse = float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))


def ssim(img1: np.ndarray, img2: np.ndarray, device: torch.device | None = None) -> float:
    """
    Structural Similarity Index (0–1, higher is better).
    Uses piqa for a differentiable, GPU-accelerated implementation.
    Falls back to a pure-numpy implementation if piqa is unavailable.
    """
    try:
        return _ssim_piqa(img1, img2, device=device)
    except ImportError:
        return _ssim_numpy(img1, img2)


def lpips(
    img1: np.ndarray, img2: np.ndarray, net: str = "alex", device: torch.device | None = None
) -> float:
    """
    Learned Perceptual Image Patch Similarity (lower is better, 0=identical).
    Requires piqa. Falls back to 0.0 with a warning if unavailable.
    """
    try:
        import piqa

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metric = piqa.LPIPS(network=net).to(device)
        t1 = _to_tensor(img1, device)
        t2 = _to_tensor(img2, device)
        with torch.inference_mode():
            score = metric(t1, t2)
        return float(score.item())
    except ImportError:
        logger.warning("piqa not installed — LPIPS unavailable, returning 0.0")
        return 0.0


def compute_all(
    output_frames: list[np.ndarray],
    reference_frames: list[np.ndarray],
) -> dict[str, float]:
    """
    Compute average PSNR, SSIM, and LPIPS across all frame pairs.

    Args:
        output_frames:    Restored frames (RGB uint8).
        reference_frames: Ground-truth frames (RGB uint8), same length.

    Returns:
        Dict with keys "psnr", "ssim", "lpips".
    """
    if len(output_frames) != len(reference_frames):
        raise ValueError("output_frames and reference_frames must have the same length")

    psnr_vals, ssim_vals, lpips_vals = [], [], []
    for out, ref in zip(output_frames, reference_frames, strict=True):
        psnr_vals.append(psnr(out, ref))
        ssim_vals.append(ssim(out, ref))
        lpips_vals.append(lpips(out, ref))

    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "lpips": float(np.mean(lpips_vals)),
    }


# ── Internal helpers ──────────────────────────────────────────────────────────


def _ssim_piqa(img1: np.ndarray, img2: np.ndarray, device: torch.device | None = None) -> float:
    import piqa

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = piqa.SSIM().to(device)
    t1 = _to_tensor(img1, device)
    t2 = _to_tensor(img2, device)
    with torch.inference_mode():
        score = metric(t1, t2)
    return float(score.item())


def _ssim_numpy(img1: np.ndarray, img2: np.ndarray) -> float:
    """Simplified single-scale SSIM without piqa dependency."""
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    img1f = img1.astype(np.float64)
    img2f = img2.astype(np.float64)
    mu1 = img1f.mean()
    mu2 = img2f.mean()
    sigma1_sq = img1f.var()
    sigma2_sq = img2f.var()
    sigma12 = float(np.mean((img1f - mu1) * (img2f - mu2)))
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    return float(numerator / denominator)


def _to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)
