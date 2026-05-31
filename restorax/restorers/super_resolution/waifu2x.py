"""
Waifu2x super-resolution restorer.

Waifu2x (2014) was the first CNN-based image SR model to gain widespread
community adoption, specifically targeting anime/illustration-style content.
It uses a noise-reduction step followed by 2x upscaling and was trained on
anime artwork, making it the reference baseline for anime SR.

Available implementations:
  - waifu2x-caffe (original, Windows-only)
  - waifu2x-ncnn-vulkan (cross-platform, GPU via Vulkan - recommended)
  - waifu2x-torch (PyTorch reimplementation)

This wrapper targets the PyTorch reimplementation. If the arch module or
weights are unavailable, load() raises RestorerLoadError - no silent fallback.

Model source: Original paper by nagadomi (2014)
Paper equivalent: ACM Multimedia 2016, "Real-Time Single Image and Video
                  Super-Resolution Using an Efficient Sub-Pixel CNN"
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import (
    BaseRestorer,
    RestorerCapabilities,
    RestorerCategory,
    RestorerParams,
    TILE_SIZE_SPEC,
    TILE_OVERLAP_SPEC,
    HALF_PRECISION_SPEC,
)

logger = logging.getLogger(__name__)


class Waifu2xRestorer(BaseRestorer):
    """
    2x super-resolution using Waifu2x (anime/illustration optimised).

    Supports noise levels 0-3 via params.extra["noise_level"] (default 1).
    Noise level 0 = upscale only; 1-3 = increasing denoising strength.
    """

    PARAM_SCHEMA = [TILE_SIZE_SPEC, TILE_OVERLAP_SPEC, HALF_PRECISION_SPEC]

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "waifu2x_x2"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.SUPER_RESOLUTION,
            input_color_space="rgb",
            output_color_space="rgb",
            requires_temporal=False,
            min_vram_gb=1.0,
            supports_compile=True,
            scale_factor=2,
            tags=["super_resolution", "waifu2x", "anime", "illustration", "x2"],
        )

    def load(self, device: torch.device) -> None:
        self._model = self._build_model(device)
        self._device = device
        self._loaded = True
        logger.info("Waifu2x loaded on %s", device)

    def unload(self) -> None:
        del self._model
        self._model = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        assert self._model is not None and self._device is not None
        if params.tile_size > 0:
            return self._process_tiled(frame, params)
        return self._process_full(frame, params)

    def _process_full(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        t = torch.from_numpy(frame).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(self._device)
        if params.half_precision and self._device.type == "cuda":
            t = t.half()
        with torch.inference_mode():
            out = self._model(t)
        return out.squeeze(0).permute(1, 2, 0).float().clamp(0, 1).mul(255.0).byte().cpu().numpy()

    def _process_tiled(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        from restorax.video.utils import merge_tiles, tile_frame
        tiles, _, _ = tile_frame(frame, params.tile_size, params.tile_overlap)
        processed = [(self._process_full(t, params), coords) for t, coords in tiles]
        h, w = frame.shape[:2]
        return merge_tiles(processed, h * 2, w * 2, scale=2)

    @staticmethod
    def _build_model(device: torch.device) -> torch.nn.Module:
        try:
            from restorax.restorers.super_resolution.waifu2x_arch import UpConvNet  # type: ignore[import]
        except ImportError as exc:
            raise RestorerLoadError(
                "waifu2x_arch module not found - install the waifu2x optional dependency group"
            ) from exc

        from restorax.config import settings
        weight_path = Path(settings.model_dir) / "waifu2x" / "waifu2x_x2.pth"

        if not weight_path.exists():
            try:
                from huggingface_hub import hf_hub_download
                weight_path.parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(
                    repo_id="deepghs/waifu2x",
                    filename="waifu2x_x2.pth",
                    local_dir=str(weight_path.parent),
                )
            except Exception as exc:
                raise RestorerLoadError(
                    f"Failed to download waifu2x weights to {weight_path}: {exc}"
                ) from exc

        try:
            model = UpConvNet(scale=2)
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt.get("params", ckpt), strict=False)
            model.to(device)
            return model
        except Exception as exc:
            raise RestorerLoadError(
                f"Failed to load waifu2x weights from {weight_path}: {exc}"
            ) from exc
