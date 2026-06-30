"""
VRT — Video Restoration Transformer.

Transformer-based video restoration that achieves state-of-the-art results
on multiple tasks: super-resolution, denoising, deblurring, and artifact
removal. Unlike BasicVSR++ (recurrent), VRT processes the full temporal
window with mutual attention between frames in a sliding window scheme.

Model source: BasicSR (XPixelGroup/BasicSR)
Paper: "VRT: A Video Restoration Transformer" (IEEE TIP 2024)

Integration: VRT architecture is already in BasicSR. We load it directly
from basicsr.archs and download the official checkpoint.

Task: video super-resolution (VSR) — 4× upscaling.
Checkpoint: VRT_videosr_bi_Vimeo_7frames.pth (bicubic degradation)
         or VRT_videosr_bd_Vimeo_7frames.pth (blur-downscale degradation)
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

_WEIGHT_FILE = "VRT_videosr_bi_Vimeo_7frames.pth"

# VRT optimal window size (frames). Must be ≤ chunk_size in pipeline preset.
_WINDOW_SIZE = 7


class VRTRestorer(BaseRestorer):
    """
    4× video super-resolution using the Video Restoration Transformer.

    VRT processes temporal windows of 7 frames with mutual attention,
    so requires_temporal=True and chunk_size in presets should be ≥7.
    """

    PARAM_SCHEMA = [HALF_PRECISION_SPEC]

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "vrt_x4"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.SUPER_RESOLUTION,
            input_color_space="rgb",
            output_color_space="rgb",
            requires_temporal=True,
            min_vram_gb=8.0,
            scale_factor=4,
            tags=["super_resolution", "transformer", "temporal", "x4", "vrt"],
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self, device: torch.device) -> None:
        model = self._build_model(device)
        self._model = model
        self._device = device
        self._loaded = True
        logger.info("VRT loaded on %s", device)

    def unload(self) -> None:
        del self._model
        self._model = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Inference ─────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        return self.process_sequence([frame], params)[0]

    def process_sequence(
        self,
        frames: list[np.ndarray],
        params: RestorerParams,
    ) -> list[np.ndarray]:
        """Process a temporal window with VRT. Pads to _WINDOW_SIZE if shorter."""
        assert self._model is not None and self._device is not None
        return self._vrt_inference(frames, params)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _vrt_inference(self, frames: list[np.ndarray], params: RestorerParams) -> list[np.ndarray]:
        """Run VRT on a temporal window."""
        # Pad to window size if too short
        padded = frames[:]
        while len(padded) < _WINDOW_SIZE:
            padded.append(padded[-1])

        # Stack to B T C H W
        tensors = []
        for f in padded:
            t = torch.from_numpy(f).float().div(255.0).permute(2, 0, 1)
            tensors.append(t)
        video = torch.stack(tensors).unsqueeze(0).to(self._device)  # 1 T C H W

        if params.half_precision and self._device.type == "cuda":
            video = video.half()

        with torch.inference_mode():
            out = self._model(video)  # 1 T C H W

        out = out.squeeze(0).float().clamp(0, 1)  # T C H W
        result = []
        for t in range(len(frames)):  # only return non-padded frames
            frame_t = out[t].permute(1, 2, 0).mul(255.0).byte().cpu().numpy()
            result.append(frame_t)
        return result

    @staticmethod
    def _build_model(device: torch.device) -> torch.nn.Module:
        """Load VRT from BasicSR or fall back to a bicubic upsampler stub."""
        try:
            from restorax.restorers.super_resolution.vrt_arch import VRT
            from restorax.config import settings

            weight_path = Path(settings.model_dir) / "vrt" / _WEIGHT_FILE
            if not weight_path.exists():
                weight_path = _download_weights(weight_path.parent)

            model = VRT(
                upscale=4,
                img_size=[6, 64, 64],
                window_size=[6, 8, 8],
                depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4],
                indep_reconsts=[11, 12],
                embed_dims=[120, 120, 120, 120, 120, 120, 120, 180, 180, 180, 180, 180, 180],
                num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                pa_frames=2,
                deformable_groups=12,
            )
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt.get("params", ckpt), strict=True)
            model.eval().to(device)
            logger.info("VRT loaded from BasicSR arch")
            return model
        except (ImportError, Exception) as exc:
            raise RestorerLoadError(f"VRT load failed: {exc}") from exc


def _download_weights(model_dir: Path) -> Path:
    raise RestorerLoadError(
        f"VRT weights ({_WEIGHT_FILE}) have no public mirror. "
        "Download from https://github.com/JingyunLiang/VRT/releases "
        f"and place at {model_dir / _WEIGHT_FILE}"
    )


