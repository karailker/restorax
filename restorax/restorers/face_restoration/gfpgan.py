"""
GFPGAN face restorer.

Uses generative facial priors from a pretrained StyleGAN2 generator for
blind face restoration. Excellent at recovering high-frequency facial
texture from heavily degraded inputs.

Model source: https://github.com/TencentARC/GFPGAN
Paper: "GFP-GAN: Towards Real-World Blind Face Restoration with
        Generative Facial Prior" (CVPR 2021)

Uses the `gfpgan` PyPI package which bundles the architecture and
facexlib for detection/alignment.
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
)

logger = logging.getLogger(__name__)

_HF_REPO = "nlightcho/gfpgan_v14"
_WEIGHT_FILE = "GFPGANv1.4.pth"


class GFPGANRestorer(BaseRestorer):
    """
    Blind face restoration using GFPGANv1.4.

    Detects faces, restores them with StyleGAN2 priors, pastes back.
    The `upscale` extra param (default 1) controls face upscaling inside
    the face region only; the overall frame scale is still 1×.
    """

    def __init__(self) -> None:
        self._gfpgan: object | None = None
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "gfpgan_v14"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.FACE_RESTORATION,
            input_color_space="bgr",
            output_color_space="bgr",
            requires_temporal=False,
            min_vram_gb=4.0,
            scale_factor=1,
            tags=["face_restoration", "blind", "gfpgan", "stylegan2"],
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self, device: torch.device) -> None:
        try:
            from gfpgan import GFPGANer
        except ImportError as exc:
            raise RestorerLoadError(
                "gfpgan is required. Install with: pip install gfpgan"
            ) from exc

        weight_path = self._resolve_weight_path()
        logger.info("Loading GFPGAN v1.4 from %s on %s", weight_path, device)

        # GFPGANer handles detection + restoration internally
        self._gfpgan = GFPGANer(
            model_path=str(weight_path),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,  # no background upsampling; handled by SR stage
        )
        self._device = device
        self._loaded = True
        logger.info("GFPGAN v1.4 loaded successfully")

    def unload(self) -> None:
        del self._gfpgan
        self._gfpgan = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Inference ─────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        """
        Detect and restore faces in a single BGR frame.

        Returns frame with enhanced faces. If no faces detected, returns
        the original frame unchanged.
        """
        assert self._gfpgan is not None

        has_aligned = False  # input is full frame, not pre-aligned face
        only_center_face = False

        try:
            _, restored_faces, restored_img = self._gfpgan.enhance(  # type: ignore[union-attr]
                frame,
                has_aligned=has_aligned,
                only_center_face=only_center_face,
                paste_back=True,
            )
        except Exception as exc:
            logger.warning("GFPGAN failed on frame: %s — returning unchanged", exc)
            return frame

        if restored_img is None:
            return frame

        return restored_img

    # ── Internal ──────────────────────────────────────────────────────────────

    def _resolve_weight_path(self) -> Path:
        from restorax.config import settings

        model_dir = Path(settings.model_dir) / "gfpgan"
        weight_path = model_dir / _WEIGHT_FILE
        if not weight_path.exists():
            weight_path = self._download_weights(model_dir)
        return weight_path

    @staticmethod
    def _download_weights(model_dir: Path) -> Path:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RestorerLoadError("huggingface_hub required.") from exc

        logger.info("Downloading GFPGAN v1.4 weights…")
        model_dir.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            repo_id=_HF_REPO,
            filename=_WEIGHT_FILE,
            local_dir=str(model_dir),
        )
        return Path(path)
