"""
FlashVSR — Fast Lightweight Video Super-Resolution.

FlashVSR is designed for real-time or near-real-time video SR on consumer
GPUs, achieving a good quality/speed tradeoff by using a lightweight
recurrent architecture with efficient attention. It is competitive with
BasicVSR at 3–5× the inference speed.

Reference: "FlashVSR: Real-Time Video Super-Resolution with Flash Attention"
           (Technical report, 2024)

Note: Public weights and official code are not yet released.
      The vendored ``restorax.restorers.super_resolution.flashvsr_arch``
      module must be present for this restorer to load.  If that import
      fails, ``load()`` raises ``RestorerLoadError`` rather than silently
      degrading to a stub.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import (
    BaseRestorer,
    RestorerCapabilities,
    RestorerCategory,
    RestorerParams,
    HALF_PRECISION_SPEC,
)

logger = logging.getLogger(__name__)


class FlashVSRRestorer(BaseRestorer):
    """
    4× video SR optimised for real-time throughput.

    Designed for scenarios where BasicVSR++ quality is not required but
    speed matters: live-streaming restoration, real-time preview, edge devices.
    """

    PARAM_SCHEMA = [HALF_PRECISION_SPEC]

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "flashvsr_x4"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.SUPER_RESOLUTION,
            input_color_space="rgb",
            output_color_space="rgb",
            requires_temporal=True,  # uses lightweight recurrent context
            min_vram_gb=2.0,
            supports_compile=True,
            scale_factor=4,
            tags=["super_resolution", "flashvsr", "fast", "realtime", "x4", "lightweight"],
        )

    def load(self, device: torch.device) -> None:
        self._model = self._build_model(device)
        if self.capabilities.supports_compile and device.type == "cuda":
            try:
                self._model = torch.compile(self._model, mode="reduce-overhead")  # type: ignore[assignment]
            except Exception:
                pass
        self._device = device
        self._loaded = True
        logger.info("FlashVSR loaded on %s", device)

    def unload(self) -> None:
        del self._model
        self._model = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        return self.process_sequence([frame], params)[0]

    def process_sequence(self, frames: list[np.ndarray], params: RestorerParams) -> list[np.ndarray]:
        assert self._model is not None and self._device is not None
        tensors = [torch.from_numpy(f).float().div(255.0).permute(2, 0, 1) for f in frames]
        video = torch.stack(tensors).unsqueeze(0).to(self._device)  # 1 T C H W
        if params.half_precision and self._device.type == "cuda":
            video = video.half()
        with torch.inference_mode():
            out = self._model(video)
        out = out.squeeze(0).float().clamp(0, 1)
        return [out[t].permute(1, 2, 0).mul(255.0).byte().cpu().numpy() for t in range(len(frames))]

    @staticmethod
    def _build_model(device: torch.device) -> torch.nn.Module:
        try:
            from restorax.restorers.super_resolution.flashvsr_arch import FlashVSR  # type: ignore[import]
        except ImportError as exc:
            raise RestorerLoadError(
                "FlashVSR arch module is not available. "
                "The vendored 'flashvsr_arch' package must be installed "
                "before this restorer can be used."
            ) from exc
        logger.info("FlashVSR arch loaded from vendored module")
        return FlashVSR(scale=4).eval().to(device)
