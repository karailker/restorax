"""
SeedVR — Seeding the Future of Video Restoration.

SeedVR (2025) is a large-scale diffusion-based video restoration model
trained on diverse degradation types. It achieves state-of-the-art
perceptual quality on real-world video restoration benchmarks including
REDS, YouHQ, and the NTIRE 2025 Video Restoration Challenge.

Architecture: DiT (Diffusion Transformer) backbone with temporal attention
and flow-guided consistency — similar to Sora but specialised for restoration.

Model source: https://github.com/IceClear/SeedVR
Paper: "SeedVR: Seeding the Future of Video Restoration with Latent Diffusion"
       (CVPR 2025)
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
    ParamSpec,
)

logger = logging.getLogger(__name__)

_HF_REPO = "IceClear/SeedVR"
_DEFAULT_STEPS = 25


class SeedVRRestorer(BaseRestorer):
    """
    All-in-one video restoration using SeedVR (DiT diffusion, CVPR 2025).

    Handles: super-resolution (4×), denoising, deblurring, artifact removal,
    stabilization in a unified diffusion pass. Highest quality among all
    included restorers but very compute-intensive.

    extra params:
      num_inference_steps: int (default 25)
      guidance_scale: float (default 7.5)
      tasks: list[str] — subset of ["sr", "denoising", "deblurring"]
    """

    PARAM_SCHEMA = [
        ParamSpec("num_inference_steps", "int", _DEFAULT_STEPS, "Inference steps",
                  minimum=1, maximum=100, step=1),
        ParamSpec("guidance_scale", "float", 7.5, "Guidance scale",
                  minimum=1.0, maximum=20.0, step=0.5),
    ]

    def __init__(self) -> None:
        self._pipe: object | None = None
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "seedvr"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.SUPER_RESOLUTION,
            input_color_space="rgb",
            output_color_space="rgb",
            requires_temporal=True,
            min_vram_gb=16.0,
            supports_compile=False,
            scale_factor=4,
            tags=["super_resolution", "diffusion", "seedvr", "dit", "all_in_one", "cvpr2025", "sota"],
        )

    def load(self, device: torch.device) -> None:
        self._pipe = self._build_pipeline(device)
        self._device = device
        self._loaded = True
        logger.info("SeedVR loaded on %s", device)

    def unload(self) -> None:
        del self._pipe
        self._pipe = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        return self.process_sequence([frame], params)[0]

    def process_sequence(self, frames: list[np.ndarray], params: RestorerParams) -> list[np.ndarray]:
        assert self._device is not None
        steps = int(params.extra.get("num_inference_steps", _DEFAULT_STEPS))
        guidance = float(params.extra.get("guidance_scale", 7.5))

        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in frames]
        result = self._pipe(image=pil_frames, num_inference_steps=steps,  # type: ignore[operator]
                            guidance_scale=guidance)
        return [np.array(img) for img in result.frames]

    @staticmethod
    def _build_pipeline(device: torch.device) -> object:
        try:
            from restorax.restorers.super_resolution.seedvr_arch import SeedVRPipeline  # type: ignore[import]
        except ImportError as exc:
            raise RestorerLoadError(
                f"SeedVR requires diffusers: pip install 'restorax[diffusion]'"
            ) from exc
        try:
            from restorax.config import settings
            weight_dir = Path(settings.model_dir) / "seedvr"
            if not weight_dir.exists():
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=_HF_REPO, local_dir=str(weight_dir))
            pipe = SeedVRPipeline.from_pretrained(str(weight_dir)).to(device)
            logger.info("SeedVR pipeline loaded")
            return pipe
        except Exception as exc:
            raise RestorerLoadError(f"SeedVR failed to load: {exc}") from exc
