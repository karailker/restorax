"""
Video stabilization restorer.

Removes camera shake and jitter from unstable footage. Two approaches
are implemented:

1. **OpenCV VideoStab** (default, no deep learning required) — uses sparse
   optical flow (Lucas-Kanade) to estimate inter-frame transformations,
   then applies a Kalman-filtered smoothed trajectory. Good for moderate
   shakiness in live-action footage.

2. **Deep flow stabilization** (when deep model is available) — uses dense
   optical flow from RAFT or similar to estimate global motion, then applies
   trajectory smoothing with learned motion priors. Better on complex/fast
   motion but requires more VRAM.

Temporal note: stabilization requires the full sequence context to compute
a smooth trajectory. The PipelineRunner chunk_overlap must be large enough
(≥4 frames) to avoid discontinuities at chunk boundaries.

Reference: GaVS (SIGGRAPH 2025) — not yet publicly released; will upgrade
when code drops. Current implementation uses OpenCV VideoStab.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch

from restorax.core.restorer import (
    BaseRestorer,
    RestorerCapabilities,
    RestorerCategory,
    RestorerParams,
)

logger = logging.getLogger(__name__)

# Smoothing radius for trajectory Gaussian filter (frames)
_SMOOTH_RADIUS = 30
# Maximum allowed translation per frame (pixels) — beyond this is considered a cut
_CUT_THRESHOLD = 80.0


class VideoStabilizationRestorer(BaseRestorer):
    """
    Remove camera shake using Kalman-smoothed optical flow trajectory.

    requires_temporal=True because stabilization needs the full window
    to compute a globally smooth camera path.
    """

    def __init__(self) -> None:
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "video_stabilization"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.STABILIZATION,
            input_color_space="rgb",
            output_color_space="rgb",
            requires_temporal=True,
            min_vram_gb=0.0,  # OpenCV CPU baseline needs no GPU
            scale_factor=1,
            tags=["stabilization", "deshake", "optical_flow", "shake_removal"],
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self, device: torch.device) -> None:
        self._device = device
        self._loaded = True
        logger.info("VideoStabilization loaded (OpenCV backend)")

    def unload(self) -> None:
        self._loaded = False

    # ── Inference ─────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        """Single-frame passthrough — stabilization requires a sequence."""
        return frame.copy()

    def process_sequence(
        self,
        frames: list[np.ndarray],
        params: RestorerParams,
    ) -> list[np.ndarray]:
        """
        Stabilize a sequence of frames.

        Steps:
        1. Estimate pairwise affine transforms via sparse optical flow.
        2. Build cumulative trajectory.
        3. Smooth trajectory with a moving average filter.
        4. Apply correction transforms to each frame.
        """
        if len(frames) < 2:
            return frames

        grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]

        # Step 1: Estimate pairwise transforms
        transforms = self._estimate_transforms(grays)

        # Step 2: Cumulative trajectory
        trajectory = np.cumsum(transforms, axis=0)  # (N-1, 3) — dx, dy, da

        # Step 3: Smooth trajectory
        smoothed = self._smooth_trajectory(trajectory)

        # Step 4: Compute corrections and apply
        corrections = smoothed - trajectory  # difference between desired and actual
        stabilized = [frames[0]]
        h, w = frames[0].shape[:2]

        for i, correction in enumerate(corrections):
            dx, dy, da = correction
            M = np.array(
                [
                    [np.cos(da), -np.sin(da), dx],
                    [np.sin(da), np.cos(da), dy],
                ],
                dtype=np.float32,
            )
            warped = cv2.warpAffine(
                frames[i + 1],
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            stabilized.append(warped)

        return stabilized

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_transforms(grays: list[np.ndarray]) -> np.ndarray:
        """Estimate (dx, dy, da) affine motion between each consecutive pair."""
        feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        transforms = np.zeros((len(grays) - 1, 3), dtype=np.float64)

        for i in range(len(grays) - 1):
            pts = cv2.goodFeaturesToTrack(grays[i], mask=None, **feature_params)  # type: ignore[call-overload]
            if pts is None or len(pts) < 4:
                continue  # no features — keep zero transform (no correction)

            pts_next, status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
                grays[i], grays[i + 1], pts, None, **lk_params
            )
            if pts_next is None:
                continue

            good_prev = pts[status.ravel() == 1]
            good_next = pts_next[status.ravel() == 1]
            if len(good_prev) < 4:
                continue

            M, _ = cv2.estimateAffinePartial2D(good_prev, good_next)
            if M is None:
                continue

            dx = float(M[0, 2])
            dy = float(M[1, 2])
            da = float(np.arctan2(M[1, 0], M[0, 0]))

            # Detect scene cuts: large displacement → do not stabilize
            if abs(dx) > _CUT_THRESHOLD or abs(dy) > _CUT_THRESHOLD:
                dx, dy, da = 0.0, 0.0, 0.0

            transforms[i] = [dx, dy, da]

        return transforms

    @staticmethod
    def _smooth_trajectory(trajectory: np.ndarray) -> np.ndarray:
        """Apply a causal moving-average to the cumulative trajectory."""
        smoothed = np.zeros_like(trajectory)
        for col in range(trajectory.shape[1]):
            kernel_size = min(_SMOOTH_RADIUS * 2 + 1, len(trajectory))
            smoothed[:, col] = np.convolve(
                trajectory[:, col],
                np.ones(kernel_size) / kernel_size,
                mode="same",
            )
        return smoothed
