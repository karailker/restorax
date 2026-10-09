"""
GaVS — 3D-Grounded Video Stabilization.

GaVS (SIGGRAPH 2025) achieves state-of-the-art stabilization by reconstructing
the 3D scene structure from video and computing a globally-smooth camera
trajectory in 3D space (rather than 2D image-plane approximation). This
eliminates depth-ambiguous distortions that plague 2D optical-flow methods.

Model source: https://arxiv.org/abs/2506.23957
              Code pending release as of 2026-04.

Until the official code is available, this restorer delegates to the existing
VideoStabilizationRestorer (OpenCV optical flow) which provides solid results
for moderate shakiness. GaVS will automatically activate once the arch is
vendored into `stabilization/gavs_arch/`.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from restorax.core.restorer import (
    BaseRestorer,
    RestorerCapabilities,
    RestorerCategory,
    RestorerParams,
)

logger = logging.getLogger(__name__)


class GaVSRestorer(BaseRestorer):
    """
    3D-grounded video stabilization (GaVS, SIGGRAPH 2025).

    Uses 3D scene reconstruction for globally smooth camera paths.
    Falls back to 2D optical-flow stabilization until code is released.
    """

    def __init__(self) -> None:
        self._model: object | None = None
        self._fallback: object | None = None
        self._device: torch.device | None = None
        self._loaded = False
        self._using_gavs = False

    @property
    def name(self) -> str:
        return "gavs"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.STABILIZATION,
            input_color_space="rgb",
            output_color_space="rgb",
            requires_temporal=True,
            min_vram_gb=8.0,
            scale_factor=1,
            tags=["stabilization", "gavs", "3d_grounded", "siggraph2025", "deshake"],
        )

    def load(self, device: torch.device) -> None:
        self._using_gavs = self._try_load_gavs(device)
        if not self._using_gavs:
            self._load_fallback(device)
        self._device = device
        self._loaded = True
        if self._using_gavs:
            logger.info("GaVS loaded — mode: GaVS (3D) on %s", device)
        else:
            logger.warning(
                "GAVS arch not yet publicly released — using OpenCV stabilization fallback. "
                "See: https://arxiv.org/abs/2407.06009"
            )

    def unload(self) -> None:
        del self._model
        self._model = None
        self._fallback = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        return frame.copy()  # stabilization requires sequence context

    def process_sequence(
        self, frames: list[np.ndarray], params: RestorerParams
    ) -> list[np.ndarray]:
        if self._using_gavs and self._model is not None:
            try:
                return self._gavs_stabilize(frames, params)
            except Exception as exc:
                logger.warning("GaVS inference failed (%s) — falling back", exc)

        # Fallback to OpenCV stabilization
        if self._fallback is not None:
            return self._fallback.process_sequence(frames, params)  # type: ignore[attr-defined]

        return frames  # identity if nothing available

    def _gavs_stabilize(self, frames: list[np.ndarray], params: RestorerParams) -> list[np.ndarray]:
        """Placeholder for the GaVS inference call."""
        result = self._model.stabilize(frames)  # type: ignore[union-attr,attr-defined]
        return result

    def _try_load_gavs(self, device: torch.device) -> bool:
        try:
            from restorax.restorers.stabilization.gavs_arch import (
                GaVSPipeline,  # type: ignore[import]
            )

            self._model = GaVSPipeline(device=device)
            return True
        except ImportError:
            return False

    def _load_fallback(self, device: torch.device) -> None:
        from restorax.restorers.stabilization.deep_flow_stab import VideoStabilizationRestorer

        self._fallback = VideoStabilizationRestorer()
        self._fallback.load(device)
