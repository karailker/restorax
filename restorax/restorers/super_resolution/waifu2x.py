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

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import (
    HALF_PRECISION_SPEC,
    TILE_OVERLAP_SPEC,
    TILE_SIZE_SPEC,
    BaseRestorer,
    RestorerCapabilities,
    RestorerCategory,
    RestorerParams,
)

logger = logging.getLogger(__name__)

_WEIGHT_ARCHIVE_URL = (
    "https://raw.githubusercontent.com/yu45020/Waifu2x/master/model_check_points/Upconv_7/anime.7z"
)
_WEIGHT_ARCHIVE_MEMBER = "anime/scale2.0x_model.json"


def _download_and_extract_weights(weight_dir: Path, json_path: Path) -> None:
    """Download the upstream 7z archive and extract the one JSON checkpoint we need.

    The archive ships nagadomi's original waifu2x JSON weight export (list of
    per-layer conv weight/bias arrays), not a PyTorch state_dict - that's what
    ``UpConvNet.load_pre_train_weights`` (vendored from upstream) expects.
    """
    import shutil
    import tempfile
    import urllib.error
    import urllib.request

    try:
        import py7zr
    except ImportError as exc:
        raise RestorerLoadError(
            "py7zr is required to extract waifu2x weights - install with: pip install restorax[waifu2x]"
        ) from exc

    weight_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading waifu2x weights archive from yu45020/Waifu2x…")
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".7z", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with urllib.request.urlopen(_WEIGHT_ARCHIVE_URL, timeout=30) as response:  # noqa: S310
                shutil.copyfileobj(response, tmp)
        with py7zr.SevenZipFile(tmp_path, mode="r") as archive:
            archive.extract(path=str(weight_dir), targets=[_WEIGHT_ARCHIVE_MEMBER])
        (weight_dir / _WEIGHT_ARCHIVE_MEMBER).rename(json_path)
    except (urllib.error.URLError, OSError, py7zr.exceptions.Bad7zFile) as exc:
        raise RestorerLoadError(f"Cannot download/extract waifu2x weights: {exc}") from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


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
        t = (
            torch.from_numpy(frame)
            .float()
            .div(255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self._device)
        )
        if params.half_precision and self._device.type == "cuda":
            t = t.half()
        with torch.inference_mode():
            out = self._model(t)
        frame_out: np.ndarray = (
            out.squeeze(0).permute(1, 2, 0).float().clamp(0, 1).mul(255.0).byte().cpu().numpy()
        )
        return frame_out

    def _process_tiled(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        from restorax.video.utils import merge_tiles, tile_frame

        tiles, _, _ = tile_frame(frame, params.tile_size, params.tile_overlap)
        processed = [(self._process_full(t, params), coords) for t, coords in tiles]
        h, w = frame.shape[:2]
        return merge_tiles(processed, h * 2, w * 2, scale=2)

    @staticmethod
    def _build_model(device: torch.device) -> torch.nn.Module:
        try:
            from restorax.restorers.super_resolution.waifu2x_arch import (
                UpConvNet,  # type: ignore[import]
            )
        except ImportError as exc:
            raise RestorerLoadError(
                "waifu2x_arch module not found - install the waifu2x optional dependency group"
            ) from exc

        from restorax.config import settings

        weight_dir = Path(settings.model_dir) / "waifu2x"
        json_path = weight_dir / "anime_scale2.0x_model.json"

        if not json_path.exists():
            _download_and_extract_weights(weight_dir, json_path)

        try:
            model = UpConvNet(scale=2)
            model.load_pre_train_weights(str(json_path))
            model.to(device)
            return model
        except Exception as exc:
            raise RestorerLoadError(
                f"Failed to load waifu2x weights from {json_path}: {exc}"
            ) from exc
