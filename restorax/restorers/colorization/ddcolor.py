"""
DDColor colorization restorer.

Converts grayscale / desaturated frames to vivid color using a
dual-decoder architecture that conditions the color decoder on
both low-level image features and high-level semantic queries.
Significantly better temporal stability than DeOldify for video.

Model source: https://github.com/piddnad/DDColor
Paper: "DDColor: Towards Photo-Realistic Image Colorization via
        Dual Decoders" (ICCV 2023)

Integration: vendor the core arch from the repo; skip the modelscope
dependency which is optional and pulls in heavy cloud SDK deps.

Weights: DDColor-artistic or DDColor (modelscope) from HuggingFace Hub.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import (
    BaseRestorer,
    RestorerCapabilities,
    RestorerCategory,
    RestorerParams,
)

logger = logging.getLogger(__name__)

_HF_REPO = "piddnad/ddcolor_models"
_WEIGHT_FILE = "ddcolor_artistic.pth"
# Input size the model was trained at — resize to this, then restore original size
_MODEL_SIZE = 512


class DDColorRestorer(BaseRestorer):
    """
    Colorize grayscale or desaturated video frames using DDColor.

    Operates in LAB color space:
      1. Extract L channel from input frame.
      2. Run DDColor to predict AB channels.
      3. Merge L + AB and convert back to RGB.

    The input_color_space is "rgb" — the LAB conversion is internal.
    """

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "ddcolor"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.COLORIZATION,
            input_color_space="rgb",
            output_color_space="rgb",
            requires_temporal=False,
            min_vram_gb=4.0,
            scale_factor=1,
            supports_compile=True,
            tags=["colorization", "ddcolor", "grayscale", "lab"],
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self, device: torch.device) -> None:
        weight_path = self._resolve_weight_path()
        logger.info("Loading DDColor from %s on %s", weight_path, device)

        model = self._build_model(weight_path, device)
        model.eval()

        self._model = model
        self._device = device
        self._loaded = True
        logger.info("DDColor loaded successfully")

    def unload(self) -> None:
        del self._model
        self._model = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Inference ─────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        """
        Colorize a single RGB frame.

        Steps:
          1. Convert RGB → LAB.
          2. Resize L to _MODEL_SIZE for the network.
          3. Run DDColor to predict AB channels.
          4. Resize AB back to original resolution.
          5. Merge with original L and convert LAB → RGB.
        """
        assert self._model is not None and self._device is not None

        h, w = frame.shape[:2]

        # 1. RGB → LAB (float32, L in [0,100], AB in [-128,127])
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_chan = lab[:, :, 0:1]  # H W 1

        # 2. Prepare model input: resize L to MODEL_SIZE, normalize to [-1,1]
        l_resized = cv2.resize(
            l_chan[:, :, 0], (_MODEL_SIZE, _MODEL_SIZE), interpolation=cv2.INTER_LINEAR
        )
        tensor = torch.from_numpy(l_resized / 50.0 - 1.0).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self._device)  # 1 1 H W
        # DDColor expects 3-channel input (repeat L into RGB channels)
        tensor = tensor.repeat(1, 3, 1, 1)

        with torch.inference_mode():
            ab_pred = self._model(tensor)  # 1 2 H W, values in [-1, 1]

        # 3. Resize predicted AB back to original resolution
        ab_pred = F.interpolate(ab_pred, size=(h, w), mode="bilinear", align_corners=False)
        ab_np = ab_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()  # H W 2
        ab_np = ab_np * 110.0  # rescale to LAB AB range

        # 4. Merge original L + predicted AB
        colorized_lab = (
            np.concatenate([l_chan, ab_np], axis=2)
            .clip([-0.0, -127.0, -127.0], [100.0, 127.0, 127.0])
            .astype(np.float32)
        )
        rgb = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2RGB)
        return (rgb * 255.0).clip(0, 255).astype(np.uint8)

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_model(weight_path: Path, device: torch.device) -> torch.nn.Module:
        """
        Load the DDColor encoder-decoder.

        DDColor uses a ConvNeXt/Swin encoder + colorization decoder.
        Requires the vendored arch module (see PLAN.md: vendor from piddnad/DDColor).
        """
        try:
            from restorax.restorers.colorization.ddcolor_arch import (
                DDColorArch,  # type: ignore[import]
            )

            model = DDColorArch(encoder_name="convnext-l")
        except ImportError as exc:
            raise RestorerLoadError(f"DDColor unavailable: {exc}") from exc

        try:
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RestorerLoadError(f"Failed to load DDColor checkpoint: {exc}") from exc

        state_dict = ckpt.get("params", ckpt.get("state_dict", ckpt))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("DDColor checkpoint: %d missing keys", len(missing))

        ready: torch.nn.Module = model.to(device)
        return ready

    def _resolve_weight_path(self) -> Path:
        from restorax.config import settings

        model_dir = Path(settings.model_dir) / "ddcolor"
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

        logger.info("Downloading DDColor weights…")
        model_dir.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            repo_id=_HF_REPO,
            filename=_WEIGHT_FILE,
            local_dir=str(model_dir),
        )
        return Path(path)
