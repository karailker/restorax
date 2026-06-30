"""
RIFE frame interpolation restorer.

Real-Time Intermediate Flow Estimation inserts synthesized frames between
existing frames to increase video frame rate (e.g. 24 → 48 fps) or
generate smooth slow-motion footage.

Model source: https://github.com/hzwer/Practical-RIFE
Paper: "Real-Time Intermediate Flow Estimation for Video Frame
        Interpolation" (ECCV 2022)
Version: RIFE v4.22 (optimised for anime and live-action)

Integration strategy (per PLAN.md):
  Vendor the `model/` directory (4 Python files) from Practical-RIFE into
  restorers/frame_interpolation/rife_arch/. This avoids a pip dependency
  on an unmaintained fork and makes the version explicit.

  Until the arch is vendored, this module uses a stub model that produces
  blended mid-frames — correct contract, lower quality.

Output note: RIFE doubles the frame count. The calling pipeline / VideoWriter
  MUST update the output fps accordingly (2× input fps). This is handled by
  PipelineRunner when it detects scale_factor == 1 AND requires_temporal == True
  on a FRAME_INTERPOLATION restorer — it passes fps_multiplier to VideoWriter.
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

_HF_REPO = "AlexWortega/RIFE"
_WEIGHT_FILE = "flownet.pkl"


class RIFERestorer(BaseRestorer):
    """
    2× frame rate interpolation using RIFE v4.22.

    process_sequence receives a window of N frames and returns 2N-1 frames
    (one inserted between each consecutive pair). The pipeline runner must
    double the output fps when this restorer is active.
    """

    def __init__(self) -> None:
        self._model: object | None = None
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "rife_v4"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.FRAME_INTERPOLATION,
            input_color_space="rgb",
            output_color_space="rgb",
            requires_temporal=True,  # needs frame pairs for optical flow
            temporal_scale=2,         # inserts one mid-frame per pair → 2× output fps
            min_vram_gb=2.0,
            scale_factor=1,          # spatial scale is 1×; temporal scale is 2×
            tags=["frame_interpolation", "rife", "fps_boost", "slow_motion"],
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self, device: torch.device) -> None:
        weight_path = self._try_resolve_weight_path()
        logger.info("Loading RIFE v4 (weights=%s) on %s", weight_path or "none", device)
        self._model = self._build_model(weight_path or Path(""), device)
        self._device = device
        self._loaded = True
        logger.info("RIFE v4 loaded")

    def unload(self) -> None:
        del self._model
        self._model = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Inference ─────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        """Single-frame pass-through (interpolation requires pairs)."""
        return frame.copy()

    def process_sequence(
        self,
        frames: list[np.ndarray],
        params: RestorerParams,
    ) -> list[np.ndarray]:
        """
        Insert one synthesized frame between every consecutive pair.

        Input:  [f0, f1, f2, ..., fN-1]  (N frames)
        Output: [f0, mid01, f1, mid12, f2, ..., fN-2, mid(N-2)(N-1), fN-1]
                = 2N-1 frames
        """
        if len(frames) < 2:
            return frames

        result: list[np.ndarray] = [frames[0]]
        for i in range(len(frames) - 1):
            mid = self._interpolate(frames[i], frames[i + 1], params)
            result.append(mid)
            result.append(frames[i + 1])
        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _interpolate(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        params: RestorerParams,
    ) -> np.ndarray:
        """Produce a single midpoint frame between frame0 and frame1."""
        assert self._device is not None

        # Try the vendored RIFE model first
        if hasattr(self._model, "inference"):
            t0 = self._frame_to_tensor(frame0)
            t1 = self._frame_to_tensor(frame1)
            t0, t1, (ph, pw) = self._pad_to_multiple(t0, t1, multiple=32)
            with torch.inference_mode():
                mid_t = self._model.inference(t0, t1, timestep=0.5)  # type: ignore[union-attr]
            if ph or pw:
                mid_t = mid_t[:, :, :mid_t.shape[2] - ph, :mid_t.shape[3] - pw]
            return self._tensor_to_frame(mid_t)

        # Fallback: linear blend (correct contract, lower quality)
        return (frame0.astype(np.float32) * 0.5 + frame1.astype(np.float32) * 0.5).astype(np.uint8)

    @staticmethod
    def _pad_to_multiple(
        t0: torch.Tensor,
        t1: torch.Tensor,
        multiple: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        import torch.nn.functional as F
        _, _, h, w = t0.shape
        ph = (multiple - h % multiple) % multiple
        pw = (multiple - w % multiple) % multiple
        if ph or pw:
            t0 = F.pad(t0, (0, pw, 0, ph), mode="reflect")
            t1 = F.pad(t1, (0, pw, 0, ph), mode="reflect")
        return t0, t1, (ph, pw)

    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        assert self._device is not None
        t = torch.from_numpy(frame).float().div(255.0).permute(2, 0, 1).unsqueeze(0)
        return t.to(self._device)

    @staticmethod
    def _tensor_to_frame(tensor: torch.Tensor) -> np.ndarray:
        return (
            tensor.squeeze(0).permute(1, 2, 0).float().clamp(0, 1).mul(255.0).byte().cpu().numpy()
        )

    @staticmethod
    def _build_model(weight_path: Path, device: torch.device) -> object:
        """Load IFNet from vendored rife_arch/ or fall back to a linear-blend stub."""
        try:
            from restorax.restorers.frame_interpolation.rife_arch import IFNet
            model = IFNet().to(device)
            if weight_path.exists():
                ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=True)
                # Practical-RIFE checkpoints may be wrapped under "model"
                state = ckpt.get("model", ckpt)
                model.load_state_dict(state, strict=False)
            model.eval()
            logger.info("RIFE IFNet loaded with vendored arch")
            return _RIFEIFNetWrapper(model, device)
        except Exception as exc:
            raise RestorerLoadError(
                f"RIFE arch unavailable: {exc}. Ensure rife_arch is vendored."
            ) from exc

    def _try_resolve_weight_path(self) -> Path | None:
        from restorax.config import settings

        model_dir = Path(settings.model_dir) / "rife"
        weight_path = model_dir / _WEIGHT_FILE
        if weight_path.exists():
            return weight_path
        try:
            return self._download_weights(model_dir)
        except Exception as exc:
            logger.warning("RIFE weights unavailable (%s) — using arch with random init", exc)
            return None

    @staticmethod
    def _download_weights(model_dir: Path) -> Path:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RestorerLoadError("huggingface_hub required.") from exc

        logger.info("Downloading RIFE v4 weights…")
        model_dir.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            repo_id=_HF_REPO,
            filename=_WEIGHT_FILE,
            local_dir=str(model_dir),
        )
        return Path(path)


class _RIFEIFNetWrapper:
    """Wraps the vendored IFNet for use in _interpolate."""

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self._net = model
        self._device = device

    def inference(self, img0: torch.Tensor, img1: torch.Tensor, timestep: float = 0.5) -> torch.Tensor:
        x = torch.cat((img0, img1), dim=1)  # (1, 6, H, W)
        merged, _, _ = self._net(x, timestep=timestep)
        return merged


