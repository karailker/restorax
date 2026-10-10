"""
VRT — Video Restoration Transformer.

Transformer-based video restoration that achieves state-of-the-art results
on multiple tasks: super-resolution, denoising, deblurring, and artifact
removal. Unlike BasicVSR++ (recurrent), VRT processes the full temporal
window with mutual attention between frames in a sliding window scheme.

Model source: JingyunLiang/VRT (Apache-2.0). The architecture is vendored in
``vrt_arch.py`` and the official checkpoint is downloaded from the project's
GitHub release (v0.0) on first use.

Task: video super-resolution (VSR) — 4× upscaling, bicubic degradation.
Checkpoint: 001_VRT_videosr_bi_REDS_6frames.pth (166 MB), trained on REDS with
6-frame windows; the network configuration below matches it exactly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import (
    HALF_PRECISION_SPEC,
    BaseRestorer,
    RestorerCapabilities,
    RestorerCategory,
    RestorerParams,
)

logger = logging.getLogger(__name__)

_WEIGHT_FILE = "001_VRT_videosr_bi_REDS_6frames.pth"
_WEIGHT_URL = (
    "https://github.com/JingyunLiang/VRT/releases/download/v0.0/001_VRT_videosr_bi_REDS_6frames.pth"
)
# The release asset is 165,665,559 bytes; anything much smaller is a failed download.
_WEIGHT_MIN_BYTES = 160_000_000

# VRT optimal window size (frames). Must be ≤ chunk_size in pipeline preset.
_WINDOW_SIZE = 6

# The network's optical-flow pyramid needs frame sides that are multiples of 32
# and at least 64 px; other sizes fail with opaque tensor-shape errors, so input
# is padded to this grid and the output cropped back.
_SPATIAL_MULTIPLE = 32
_MIN_SPATIAL = 64


class VRTRestorer(BaseRestorer):
    """
    4× video super-resolution using the Video Restoration Transformer.

    VRT processes temporal windows of 6 frames with mutual attention,
    so requires_temporal=True and chunk_size in presets should be ≥6.
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
        """Run VRT on a sequence, padding it to a multiple of the 6-frame window.

        Like the reference test script, the sequence is padded at the end with
        mirrored frames (clamped for very short clips) and the padding is dropped
        from the output.
        """
        d = len(frames)
        d_pad = (-d) % _WINDOW_SIZE
        padded = frames + [frames[max(d - 1 - k, 0)] for k in range(d_pad)]

        # Stack to B T C H W
        tensors = []
        for f in padded:
            t = torch.from_numpy(f).float().div(255.0).permute(2, 0, 1)
            tensors.append(t)
        video = torch.stack(tensors).unsqueeze(0).to(self._device)  # 1 T C H W

        height, width = video.shape[-2:]
        pad_h = max(_MIN_SPATIAL, -(-height // _SPATIAL_MULTIPLE) * _SPATIAL_MULTIPLE) - height
        pad_w = max(_MIN_SPATIAL, -(-width // _SPATIAL_MULTIPLE) * _SPATIAL_MULTIPLE) - width
        if pad_h or pad_w:
            video = _pad_spatial(video, pad_h, pad_w)

        if params.half_precision and self._device.type == "cuda":
            video = video.half()

        with torch.inference_mode():
            out = self._model(video)  # 1 T C H W

        scale = self.capabilities.scale_factor
        out = out[..., : height * scale, : width * scale]
        out = out.squeeze(0).float().clamp(0, 1)  # T C H W
        result = []
        for i in range(len(frames)):  # only return non-padded frames
            frame_t = out[i].permute(1, 2, 0).mul(255.0).byte().cpu().numpy()
            result.append(frame_t)
        return result

    @staticmethod
    def _build_model(device: torch.device) -> torch.nn.Module:
        """Build VRT with the REDS 6-frame configuration and load the official weights."""
        try:
            from restorax.config import settings
            from restorax.restorers.super_resolution.vrt_arch import VRT

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
            logger.info("VRT loaded (%s)", _WEIGHT_FILE)
            return model
        except (ImportError, Exception) as exc:
            raise RestorerLoadError(f"VRT load failed: {exc}") from exc


def _pad_spatial(video: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    """Pad the bottom/right of a (1, T, C, H, W) clip, reflecting when possible."""
    import torch.nn.functional as F

    _, t, c, h, w = video.shape
    frames = video.reshape(t, c, h, w)
    if pad_w:
        frames = F.pad(frames, (0, pad_w, 0, 0), mode="reflect" if pad_w < w else "replicate")
    if pad_h:
        frames = F.pad(frames, (0, 0, 0, pad_h), mode="reflect" if pad_h < h else "replicate")
    return frames.reshape(1, t, c, h + pad_h, w + pad_w)


def _download_weights(model_dir: Path) -> Path:
    from restorax.utils.weights import download_file

    return download_file(_WEIGHT_URL, model_dir / _WEIGHT_FILE, min_bytes=_WEIGHT_MIN_BYTES)
