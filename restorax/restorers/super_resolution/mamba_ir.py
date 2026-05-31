"""
MambaIR — State Space Model-based image / video super-resolution.

MambaIR replaces the self-attention in SwinIR with a Mamba (SSM) block,
giving linear complexity instead of quadratic. Achieves comparable PSNR
to SwinIR at significantly lower VRAM and ~50% faster inference.

Model source: https://github.com/csguoh/MambaIR
Paper: "MambaIR: A Simple Baseline for Image Restoration with
        State-Space Model" (ECCV 2024)

Vendoring: copy arch from csguoh/MambaIR into
    restorers/super_resolution/mamba_ir_arch.py
Weights:   HuggingFace Hub "csguoh/MambaIR" → MambaIR_SR_x4.pth
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

_HF_REPO = "csguoh/MambaIR"
_WEIGHT_FILE = "MambaIR_SR_x4.pth"


class MambaIRRestorer(BaseRestorer):
    """
    4× super-resolution using MambaIR (SSM-based architecture).

    More efficient than transformer-based VRT while achieving similar quality.
    Suitable for lower-VRAM GPUs (3 GB vs 8 GB for VRT).
    """

    PARAM_SCHEMA = [TILE_SIZE_SPEC, TILE_OVERLAP_SPEC, HALF_PRECISION_SPEC]

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "mamba_ir_x4"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.SUPER_RESOLUTION,
            input_color_space="rgb",
            output_color_space="rgb",
            requires_temporal=False,
            min_vram_gb=3.0,
            supports_compile=True,
            scale_factor=4,
            tags=["super_resolution", "mamba", "ssm", "efficient", "x4"],
        )

    def load(self, device: torch.device) -> None:
        model = self._build_model(device)
        if self.capabilities.supports_compile and device.type == "cuda":
            try:
                model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]
            except Exception:
                pass
        self._model = model
        self._device = device
        self._loaded = True
        logger.info("MambaIR x4 loaded on %s", device)

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
        return merge_tiles(processed, h * 4, w * 4, scale=4)

    @staticmethod
    def _build_model(device: torch.device) -> torch.nn.Module:
        try:
            from restorax.restorers.super_resolution.mamba_ir_arch import MambaIR  # type: ignore[import]
            from restorax.config import settings

            weight_path = Path(settings.model_dir) / "mamba_ir" / _WEIGHT_FILE
            if not weight_path.exists():
                from huggingface_hub import hf_hub_download
                weight_path.parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(repo_id=_HF_REPO, filename=_WEIGHT_FILE,
                                local_dir=str(weight_path.parent))

            model = MambaIR(upscale=4, img_size=64, embed_dim=180)
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt.get("params", ckpt), strict=False)
            model.eval().to(device)
            logger.info("MambaIR arch loaded from vendored module")
            return model
        except ImportError as exc:
            raise RestorerLoadError(
                f"MambaIR unavailable: {exc}. Install: pip install mamba-ssm"
            ) from exc
        except Exception as exc:
            raise RestorerLoadError(f"MambaIR failed to load: {exc}") from exc
