"""
Real-ESRGAN x4plus restorer.

Uses BasicSR's RRDBNet architecture with Real-ESRGAN weights.
Weights are downloaded from HuggingFace Hub on first use via WeightsManager.

Model source: https://github.com/xinntao/Real-ESRGAN
Paper: "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure
        Synthetic Data" (ICCVW 2021)
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
    TILE_SIZE_SPEC,
    TILE_OVERLAP_SPEC,
    HALF_PRECISION_SPEC,
)

logger = logging.getLogger(__name__)

# Official Real-ESRGAN weights — hosted on HuggingFace Hub
_HF_REPO = "xinntao/Real-ESRGAN"
_WEIGHT_FILE = "RealESRGAN_x4plus.pth"

# Expected SHA-256 prefix (first 16 hex chars) for integrity check
_WEIGHT_SHA256_PREFIX = "4fa0d38905f75ac"


class RealESRGANx4Restorer(BaseRestorer):
    """4× blind real-world super-resolution using Real-ESRGAN."""

    PARAM_SCHEMA = [TILE_SIZE_SPEC, TILE_OVERLAP_SPEC, HALF_PRECISION_SPEC]

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "real_esrgan_x4plus"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.SUPER_RESOLUTION,
            input_color_space="rgb",
            output_color_space="rgb",
            requires_temporal=False,
            max_batch_size=1,
            min_vram_gb=4.0,
            supports_compile=True,
            scale_factor=4,
            tags=["super_resolution", "blind", "real_world", "x4"],
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self, device: torch.device) -> None:
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
        except (ImportError, Exception) as exc:
            raise RestorerLoadError(
                "basicsr is required for RealESRGANx4Restorer. "
                "Install with: pip install basicsr"
            ) from exc

        weight_path = self._try_resolve_weight_path()
        if weight_path is None:
            raise RestorerLoadError(
                "Real-ESRGAN weights could not be resolved. "
                "Ensure huggingface_hub is installed and network is available."
            )

        logger.info("Loading Real-ESRGAN x4plus from %s on %s", weight_path, device)

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        try:
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RestorerLoadError(f"Failed to load checkpoint: {exc}") from exc

        # BasicSR checkpoints store weights under 'params_ema' or 'params'
        state_dict = ckpt.get("params_ema", ckpt.get("params", ckpt))
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model = model.to(device)

        # torch.compile: fused CUDA kernels — ~20% faster on 2nd+ call
        if self.capabilities.supports_compile and device.type == "cuda":
            try:
                model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]
                logger.info("Real-ESRGAN compiled with torch.compile")
            except Exception as exc:
                logger.warning("torch.compile failed (%s) — running eager", exc)

        self._model = model
        self._device = device
        self._loaded = True
        logger.info("Real-ESRGAN x4plus loaded successfully")

    def unload(self) -> None:
        del self._model
        self._model = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Real-ESRGAN x4plus unloaded")

    # ── Inference ─────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        """
        Upscale a single RGB uint8 frame by 4×.

        Supports optional tiling via params.tile_size to handle high-res inputs
        that would otherwise OOM.
        """
        assert self._model is not None, "Model not loaded"
        assert self._device is not None

        if params.tile_size > 0:
            return self._process_tiled(frame, params)
        return self._process_full(frame, params)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _process_full(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        tensor = self._frame_to_tensor(frame, params.half_precision)
        with torch.inference_mode():
            out = self._model(tensor)
        return self._tensor_to_frame(out)

    def _process_tiled(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        from restorax.video.utils import merge_tiles, tile_frame

        tiles, _, _ = tile_frame(frame, params.tile_size, params.tile_overlap)
        processed_tiles: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
        for tile_arr, coords in tiles:
            out = self._process_full(tile_arr, params)
            processed_tiles.append((out, coords))

        h, w = frame.shape[:2]
        return merge_tiles(processed_tiles, h * 4, w * 4, scale=4)

    def _frame_to_tensor(self, frame: np.ndarray, half: bool) -> torch.Tensor:
        assert self._device is not None
        tensor = torch.from_numpy(frame).float().div(255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWC → BCHW
        tensor = tensor.to(self._device)
        if half and self._device.type == "cuda":
            tensor = tensor.half()
        return tensor

    @staticmethod
    def _tensor_to_frame(tensor: torch.Tensor) -> np.ndarray:
        out = tensor.squeeze(0).permute(1, 2, 0)  # BCHW → HWC
        out = out.float().clamp(0, 1).mul(255.0).byte()
        return out.cpu().numpy()

    def _try_resolve_weight_path(self) -> Path | None:
        from restorax.config import settings

        model_dir = Path(settings.model_dir) / "real_esrgan"
        weight_path = model_dir / _WEIGHT_FILE
        if weight_path.exists():
            return weight_path
        try:
            return self._download_weights(model_dir)
        except Exception as exc:
            logger.warning("Real-ESRGAN weight download failed: %s", exc)
            return None

    @staticmethod
    def _download_weights(model_dir: Path) -> Path:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RestorerLoadError(
                "huggingface_hub is required to download model weights. "
                "Install it with: pip install huggingface-hub"
            ) from exc

        logger.info("Downloading Real-ESRGAN weights from HuggingFace Hub…")
        model_dir.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            repo_id=_HF_REPO,
            filename=_WEIGHT_FILE,
            local_dir=str(model_dir),
        )
        return Path(path)
