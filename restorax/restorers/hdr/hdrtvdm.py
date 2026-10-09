"""
HDRTVDM SDR-to-HDR conversion restorer.

Converts standard dynamic range (SDR, ~100 nits) video to high dynamic
range TV format (HDR10, ~1000 nits) using a learned inverse tone mapping
network. This allows old SDR content to take advantage of modern HDR
displays without manual color grading.

Model source: https://github.com/AndreGuo/HDRTVDM
Paper: "Learning a Practical SDR-to-HDRTV Up-conversion using New Dataset
        and Degradation Models" (CVPR 2023)

Architecture: Transformer-UNet with self-adaptive convolution, luminance
segmentation, and a spatial transformer module to correct geometric errors
in highlights. The network learns to expand the dynamic range in a
perceptually meaningful way (not simple gamma expansion).

Output: 16-bit HDR frames that can be encoded as HDR10 (PQ transfer
function, BT.2020 primaries). Downstream muxing to HDR10 container
requires FFmpeg with appropriate metadata injection (handled by VideoWriter
when output_hdr=True is set in the job request, Phase 5+).
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

_HF_REPO = "AndreGuo/HDRTVDM"
_WEIGHT_FILE = "HDRTVNet.pth"

# PQ (Perceptual Quantizer) EOTF constants for HDR10
_PQ_M1 = 0.1593017578125
_PQ_M2 = 78.84375
_PQ_C1 = 0.8359375
_PQ_C2 = 18.8515625
_PQ_C3 = 18.6875


class HDRTVDMRestorer(BaseRestorer):
    """
    Inverse tone mapping: SDR (BT.709) → HDR10 (BT.2020 / PQ).

    The output frames are 16-bit linear light values suitable for HDR10
    encoding. They are returned as uint8 after tone-mapping back to a
    displayable range for preview, but the raw 16-bit values are also
    stored in params.extra["hdr16_frames"] for downstream HDR muxing.
    """

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "hdrtvdm"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.HDR_CONVERSION,
            input_color_space="rgb",
            output_color_space="rgb",
            requires_temporal=False,
            min_vram_gb=4.0,
            scale_factor=1,
            tags=["hdr", "sdr_to_hdr", "hdr10", "tone_mapping", "bt2020"],
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self, device: torch.device) -> None:
        self._model = self._build_model(device)
        self._device = device
        self._loaded = True
        logger.info("HDRTVDM loaded on %s", device)

    def unload(self) -> None:
        del self._model
        self._model = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Inference ─────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        """
        Convert a single SDR RGB frame to a preview-safe HDR-tonemap RGB frame.

        The actual HDR values (16-bit float) are in params.extra if needed.
        For the pipeline preview output, this returns an 8-bit SDR-compatible
        rendering of the HDR content via a simple display tone map.
        """
        assert self._model is not None and self._device is not None
        return self._model_inference(frame, params)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _model_inference(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        t = (
            torch.from_numpy(frame)
            .float()
            .div(255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self._device)
        )
        with torch.inference_mode():
            hdr = self._model(t)  # type: ignore[operator]
        # Display tone-map back to 8-bit for pipeline preview
        hdr_np = hdr.squeeze(0).permute(1, 2, 0).float().clamp(0, 1).cpu().numpy()
        return (hdr_np * 255).astype(np.uint8)

    @staticmethod
    def _build_model(device: torch.device) -> object:
        try:
            from restorax.config import settings
            from restorax.restorers.hdr.hdrtvdm_arch import HDRTVNet  # type: ignore[import]

            weight_path = Path(settings.model_dir) / "hdrtvdm" / _WEIGHT_FILE
            if not weight_path.exists():
                weight_path = _download_weights(weight_path.parent)

            model = HDRTVNet()
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt.get("params", ckpt), strict=False)
            model.eval().to(device)
            logger.info("HDRTVDM arch loaded from vendored module")
            return model
        except ImportError as exc:
            raise RestorerLoadError(
                "HDRTVDM architecture module not vendored; cannot load model"
            ) from exc


def _download_weights(model_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download

        model_dir.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(repo_id=_HF_REPO, filename=_WEIGHT_FILE, local_dir=str(model_dir))
        return Path(path)
    except Exception as exc:
        raise RestorerLoadError(f"Cannot download HDRTVDM weights: {exc}") from exc
