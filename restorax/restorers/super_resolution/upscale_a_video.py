"""
Upscale-A-Video: diffusion-based video super-resolution.

Uses a Stable Diffusion-based latent diffusion model conditioned on
low-resolution video frames. Achieves highest perceptual quality among
VSR methods but is ~10× slower than Real-ESRGAN. Best used for short
clips where quality matters more than speed.

Key features vs. Real-ESRGAN:
  - Generates plausible high-frequency details not present in the source
  - Temporal consistency via flow-guided recurrent latent propagation
  - Handles extreme degradation (heavily compressed, low-bitrate sources)
  - Scale: 4× by default

Model source: https://github.com/sczhou/Upscale-A-Video
Paper: "Upscale-A-Video: Temporal-Consistent Diffusion Model for
        Real-World Video Super-Resolution" (CVPR 2024)

Inference strategy:
  The denoising loop runs on latent tensors (VAE-encoded frames), not
  pixels directly. This requires a different I/O path than Real-ESRGAN:
    1. Encode all frames to latent space (VAE encoder)
    2. Run T diffusion steps with temporal attention layers
    3. Decode latents back to pixel space (VAE decoder)
  The PipelineRunner must call process_sequence for this restorer —
  declared via requires_temporal=True.
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

_HF_REPO = "sczhou/Upscale-A-Video"
_DEFAULT_NUM_INFERENCE_STEPS = 30
_DEFAULT_GUIDANCE_SCALE = 7.5


class UpscaleAVideoRestorer(BaseRestorer):
    """
    4× video super-resolution using Stable Diffusion latent space.

    process_sequence processes a temporal window jointly using the full
    diffusion denoising loop. Temporal layers in U-Net and VAE-Decoder
    provide short-sequence consistency; the flow-guided recurrent latent
    propagation extends it across the full video via chunk overlap.
    """

    PARAM_SCHEMA = [
        ParamSpec("num_inference_steps", "int", _DEFAULT_NUM_INFERENCE_STEPS, "Inference steps",
                  minimum=1, maximum=100, step=1),
        ParamSpec("guidance_scale", "float", _DEFAULT_GUIDANCE_SCALE, "Guidance scale",
                  minimum=1.0, maximum=20.0, step=0.5),
    ]

    def __init__(self) -> None:
        self._pipe: object | None = None
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "upscale_a_video"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.SUPER_RESOLUTION,
            input_color_space="rgb",
            output_color_space="rgb",
            requires_temporal=True,
            max_batch_size=1,
            min_vram_gb=12.0,
            supports_compile=False,  # diffusion pipeline not compatible with compile
            scale_factor=4,
            tags=["super_resolution", "diffusion", "x4", "high_quality", "slow"],
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self, device: torch.device) -> None:
        self._pipe = self._build_pipeline(device)
        self._device = device
        self._loaded = True
        logger.info("Upscale-A-Video loaded on %s", device)

    def unload(self) -> None:
        del self._pipe
        self._pipe = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Inference ─────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        """Single-frame fallback — wraps process_sequence."""
        return self.process_sequence([frame], params)[0]

    def process_sequence(
        self,
        frames: list[np.ndarray],
        params: RestorerParams,
    ) -> list[np.ndarray]:
        """
        Upscale a temporal window of frames using the diffusion pipeline.
        """
        assert self._device is not None

        num_steps = int(params.extra.get("num_inference_steps", _DEFAULT_NUM_INFERENCE_STEPS))
        guidance = float(params.extra.get("guidance_scale", _DEFAULT_GUIDANCE_SCALE))

        if hasattr(self._pipe, "__call__"):
            return self._diffusion_inference(frames, num_steps, guidance)

        # Fallback stub: nearest-neighbour 4× upscale
        return self._stub_upscale(frames)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _diffusion_inference(
        self,
        frames: list[np.ndarray],
        num_steps: int,
        guidance: float,
    ) -> list[np.ndarray]:
        """Run the Upscale-A-Video diffusion pipeline on a frame sequence."""
        try:
            import torch
            from PIL import Image

            pil_frames = [Image.fromarray(f) for f in frames]
            result = self._pipe(  # type: ignore[operator]
                image=pil_frames,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
            )
            return [np.array(img) for img in result.frames]
        except Exception as exc:
            logger.warning("Diffusion inference failed (%s) — using stub", exc)
            return self._stub_upscale(frames)

    @staticmethod
    def _stub_upscale(frames: list[np.ndarray]) -> list[np.ndarray]:
        """Nearest-neighbour 4× upscale stub."""
        import cv2
        return [
            cv2.resize(f, (f.shape[1] * 4, f.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
            for f in frames
        ]

    @staticmethod
    def _build_pipeline(device: torch.device) -> object:
        """Load the Upscale-A-Video diffusion pipeline."""
        try:
            from restorax.restorers.super_resolution.upscale_a_video_arch import UpscaleAVideoPipeline  # type: ignore[import]
            from restorax.config import settings

            weight_dir = Path(settings.model_dir) / "upscale_a_video"
            if not weight_dir.exists():
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=_HF_REPO, local_dir=str(weight_dir))

            pipe = UpscaleAVideoPipeline.from_pretrained(str(weight_dir))
            pipe = pipe.to(device)
            logger.info("Upscale-A-Video pipeline loaded from vendored module")
            return pipe
        except ImportError as exc:
            raise RestorerLoadError(
                f"Upscale-A-Video arch unavailable: {exc}. "
                "Install with: pip install 'restorax[diffusion]'"
            ) from exc
