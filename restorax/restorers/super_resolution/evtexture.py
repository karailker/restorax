"""
EvTexture — Event-guided Texture Enhancement for Video SR.

EvTexture (ICML 2024) uses event camera data (or its simulation from frame
differences) to guide high-frequency texture recovery in video SR. It achieves
significantly better fine detail recovery than flow-based methods on scenes
with fast motion and complex textures.

Without real event data, the restorer simulates events from inter-frame
temporal differences — this is close to the paper's evaluation protocol
on standard video benchmarks.

Model source: https://github.com/DachunKai/EvTexture
Paper: "EvTexture: Event-driven Texture Enhancement for Video Super-Resolution"
       (ICML 2024)
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

_HF_REPO = "DachunKai/EvTexture"
_WEIGHT_FILE = "evtexture_x4.pth"


class EvTextureRestorer(BaseRestorer):
    """
    Event-guided 4× video super-resolution.

    Uses temporal frame differences to simulate event camera data, which
    guides fine texture recovery. Superior to BasicVSR++ on scenes with
    rapid motion or intricate repeating textures (fabric, foliage, text).
    """

    PARAM_SCHEMA = [HALF_PRECISION_SPEC]

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "evtexture_x4"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.SUPER_RESOLUTION,
            input_color_space="rgb",
            output_color_space="rgb",
            requires_temporal=True,
            min_vram_gb=6.0,
            scale_factor=4,
            tags=["super_resolution", "evtexture", "event_guided", "texture", "x4", "icml2024"],
        )

    def load(self, device: torch.device) -> None:
        self._model = self._build_model(device)
        self._device = device
        self._loaded = True
        logger.info("EvTexture loaded on %s", device)

    def unload(self) -> None:
        del self._model
        self._model = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        return self.process_sequence([frame], params)[0]

    def process_sequence(self, frames: list[np.ndarray], params: RestorerParams) -> list[np.ndarray]:
        assert self._model is not None and self._device is not None
        tensors = [torch.from_numpy(f).float().div(255.0).permute(2, 0, 1) for f in frames]
        video = torch.stack(tensors).unsqueeze(0).to(self._device)  # 1 T C H W

        # Simulate event data from inter-frame differences
        events = self._simulate_events(video)

        if params.half_precision and self._device.type == "cuda":
            video = video.half()
            events = events.half()

        with torch.inference_mode():
            out = self._model(video, events)  # type: ignore[operator]

        out = out.squeeze(0).float().clamp(0, 1)
        return [out[t].permute(1, 2, 0).mul(255.0).byte().cpu().numpy() for t in range(len(frames))]

    @staticmethod
    def _simulate_events(video: torch.Tensor) -> torch.Tensor:
        """Simulate event stream via temporal frame differences (polarity: ±1)."""
        b, t, c, h, w = video.shape
        if t < 2:
            return torch.zeros(b, t, 2, h, w, device=video.device)
        gray = video.mean(dim=2, keepdim=True)  # b t 1 h w
        diff = torch.zeros_like(gray)
        diff[:, 1:] = gray[:, 1:] - gray[:, :-1]
        pos = (diff > 0.02).float()
        neg = (diff < -0.02).float()
        return torch.cat([pos, neg], dim=2)  # b t 2 h w

    @staticmethod
    def _build_model(device: torch.device) -> torch.nn.Module:
        try:
            from restorax.restorers.super_resolution.evtexture_arch import EvTexture  # type: ignore[import]
            from restorax.config import settings
            weight_path = Path(settings.model_dir) / "evtexture" / _WEIGHT_FILE
            if not weight_path.exists():
                from huggingface_hub import hf_hub_download
                weight_path.parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(repo_id=_HF_REPO, filename=_WEIGHT_FILE,
                                local_dir=str(weight_path.parent))
            model = EvTexture(scale=4)
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt.get("params", ckpt), strict=False)
            return model.eval().to(device)
        except (ImportError, Exception) as exc:
            raise RestorerLoadError(f"EvTexture arch unavailable: {exc}") from exc
