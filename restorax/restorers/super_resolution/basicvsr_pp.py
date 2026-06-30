"""
BasicVSR++ restorer — temporally-aware video super-resolution.

Processes a sequence (window) of frames jointly using bidirectional
propagation and flow-guided deformable alignment. Significantly better
than frame-by-frame SR on real video (reduces flickering, recovers
temporal details lost in per-frame processing).

Model source: https://github.com/ckkelvinchan/BasicVSR_PlusPlus
Paper: "BasicVSR++: Improving Video Super-Resolution with Enhanced
        Propagation and Alignment" (CVPR 2022)

Weights: loaded from BasicSR's official HuggingFace checkpoint.
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

_WEIGHT_FILE = "BasicVSR_PlusPlus_REDS4.pth"


class BasicVSRPlusPlusRestorer(BaseRestorer):
    """
    4× video super-resolution using BasicVSR++.

    Requires a temporal sequence — override process_sequence instead of
    process_frame to exploit inter-frame information.
    """

    PARAM_SCHEMA = [HALF_PRECISION_SPEC]

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "basicvsr_pp_x4"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.SUPER_RESOLUTION,
            input_color_space="rgb",
            output_color_space="rgb",
            requires_temporal=True,   # uses process_sequence, not process_frame
            max_batch_size=1,
            min_vram_gb=8.0,
            supports_compile=False,   # deformable conv not compatible with compile
            scale_factor=4,
            tags=["super_resolution", "temporal", "x4", "video"],
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self, device: torch.device) -> None:
        try:
            from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus
        except ImportError as exc:
            raise RestorerLoadError(
                "basicsr is required for BasicVSRPlusPlusRestorer."
            ) from exc

        weight_path = self._resolve_weight_path()
        logger.info("Loading BasicVSR++ from %s on %s", weight_path, device)

        model = BasicVSRPlusPlus(num_feat=64, num_block=7, spynet_path=None)

        try:
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RestorerLoadError(f"Failed to load checkpoint: {exc}") from exc

        state_dict = ckpt.get("params", ckpt.get("params_ema", ckpt))
        model.load_state_dict(state_dict, strict=True)
        model.eval().to(device)

        self._model = model
        self._device = device
        self._loaded = True
        logger.info("BasicVSR++ loaded successfully")

    def unload(self) -> None:
        del self._model
        self._model = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("BasicVSR++ unloaded")

    # ── Inference ─────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        """Single-frame fallback — wraps process_sequence for one frame."""
        return self.process_sequence([frame], params)[0]

    def process_sequence(
        self,
        frames: list[np.ndarray],
        params: RestorerParams,
    ) -> list[np.ndarray]:
        """
        Process a temporal window of frames jointly.

        Args:
            frames: List of HxWxC uint8 RGB frames.
            params: RestorerParams (scale, half_precision etc.)

        Returns:
            List of upscaled HxWxC uint8 RGB frames (same length as input).
        """
        assert self._model is not None, "Model not loaded"
        assert self._device is not None

        # Stack frames into BTCHW tensor
        tensor = self._frames_to_tensor(frames, params.half_precision)

        with torch.inference_mode():
            out = self._model(tensor)  # B T C H W

        return self._tensor_to_frames(out)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _frames_to_tensor(self, frames: list[np.ndarray], half: bool) -> torch.Tensor:
        assert self._device is not None
        # Each frame: HWC uint8 → CHW float [0,1]
        tensors = []
        for f in frames:
            t = torch.from_numpy(f).float().div(255.0).permute(2, 0, 1)
            tensors.append(t)
        # Stack: T C H W → unsqueeze batch → 1 T C H W
        tensor = torch.stack(tensors, dim=0).unsqueeze(0).to(self._device)
        if half and self._device.type == "cuda":
            tensor = tensor.half()
        return tensor

    @staticmethod
    def _tensor_to_frames(tensor: torch.Tensor) -> list[np.ndarray]:
        # tensor: B T C H W → iterate T
        tensor = tensor.squeeze(0).float().clamp(0, 1)  # T C H W
        frames = []
        for t in range(tensor.shape[0]):
            frame = tensor[t].permute(1, 2, 0).mul(255.0).byte().cpu().numpy()
            frames.append(frame)
        return frames

    def _resolve_weight_path(self) -> Path:
        from restorax.config import settings

        model_dir = Path(settings.model_dir) / "basicvsr_pp"
        weight_path = model_dir / _WEIGHT_FILE
        if not weight_path.exists():
            weight_path = self._download_weights(model_dir)
        return weight_path

    @staticmethod
    def _download_weights(model_dir: Path) -> Path:
        raise RestorerLoadError(
            f"BasicVSR++ weights ({_WEIGHT_FILE}) have no public mirror. "
            "Download manually from https://github.com/ckkelvinchan/BasicVSR_PlusPlus "
            f"and place at {model_dir / _WEIGHT_FILE}"
        )
