"""
TDM — Temporally-Consistent Diffusion Model for All-in-One Video Restoration.

TDM (2025) is a single diffusion-based model that handles multiple restoration
tasks simultaneously: super-resolution, denoising, deblurring, colorization,
and artifact removal. It uses a pretrained Stable Diffusion backbone with a
fine-tuned ControlNet and Task Prompt Guidance (TPG) to select tasks.

Model source: https://arxiv.org/abs/2501.02269
Weights:      ChenyangSi/TDM on HuggingFace Hub (pending public release)

Temporal consistency is achieved via Sliding Window Cross-Frame Attention
(SW-CFA) — the restorer requires a temporal sequence (requires_temporal=True).
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

_HF_REPO = "ChenyangSi/TDM"
_DEFAULT_TASKS = ["sr", "denoising"]
_DEFAULT_STEPS = 20


class TDMRestorer(BaseRestorer):
    """
    All-in-one video restoration using TDM (diffusion-based).

    Slower than Real-ESRGAN but handles multiple degradation types
    simultaneously. Best for severely degraded sources where single-task
    models are insufficient.

    extra params:
      tasks: list of tasks, e.g. ["sr", "denoising", "colorization"]
             Supported: "sr", "denoising", "deblurring", "colorization"
      num_inference_steps: int (default 20, lower = faster but lower quality)
      guidance_scale: float (default 7.5)
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
        return "tdm"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.SUPER_RESOLUTION,
            input_color_space="rgb",
            output_color_space="rgb",
            requires_temporal=True,
            min_vram_gb=12.0,
            supports_compile=False,
            scale_factor=4,
            tags=["super_resolution", "diffusion", "all_in_one", "temporal", "slow"],
        )

    def load(self, device: torch.device) -> None:
        self._pipe = self._build_pipeline(device)
        self._device = device
        self._loaded = True
        logger.info("TDM loaded on %s", device)

    def unload(self) -> None:
        del self._pipe
        self._pipe = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        return self.process_sequence([frame], params)[0]

    def process_sequence(
        self, frames: list[np.ndarray], params: RestorerParams
    ) -> list[np.ndarray]:
        assert self._device is not None
        tasks = list(params.extra.get("tasks", _DEFAULT_TASKS))
        steps = int(params.extra.get("num_inference_steps", _DEFAULT_STEPS))
        guidance = float(params.extra.get("guidance_scale", 7.5))
        return self._diffusion_inference(frames, tasks, steps, guidance)

    def _diffusion_inference(
        self, frames: list[np.ndarray], tasks: list[str], steps: int, guidance: float
    ) -> list[np.ndarray]:
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in frames]
        result = self._pipe(  # type: ignore[operator]
            image=pil_frames, tasks=tasks,
            num_inference_steps=steps, guidance_scale=guidance,
        )
        return [np.array(img) for img in result.frames]

    @staticmethod
    def _build_pipeline(device: torch.device) -> object:
        try:
            from restorax.restorers.super_resolution.tdm_arch import TDMPipeline  # type: ignore[import]
        except ImportError as exc:
            raise RestorerLoadError(
                f"TDM requires diffusers: pip install 'restorax[diffusion]'"
            ) from exc
        try:
            from restorax.config import settings
            weight_dir = Path(settings.model_dir) / "tdm"
            if not weight_dir.exists():
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=_HF_REPO, local_dir=str(weight_dir))
            pipe = TDMPipeline.from_pretrained(str(weight_dir)).to(device)
            logger.info("TDM pipeline loaded from vendored module")
            return pipe
        except Exception as exc:
            raise RestorerLoadError(f"TDM failed to load: {exc}") from exc
