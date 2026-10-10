"""
DDColor colorization restorer.

Converts grayscale / desaturated frames to color with the DDColor dual-decoder
network (Kang et al., ICCV 2023, Apache-2.0). The architecture is vendored in
``ddcolor_arch/`` (from github.com/piddnad/DDColor, no ``basicsr`` needed) and the
official weights are downloaded from Hugging Face (``piddnad/ddcolor_*``).

Inference follows the reference pipeline: the luminance of the frame is turned
into a gray RGB image, the network predicts the a/b chroma channels at 512x512,
and those are upsampled and combined with the *original* full-resolution L channel,
so detail and brightness of the source are preserved exactly.

The model works frame by frame (no temporal modelling), so colors can flicker
slightly between frames of a video.

Variants (``DDColorRestorer(variant=...)``): ``modelscope`` (default, DDColor-L,
best for photos outside ImageNet), ``artistic`` (DDColor-L, more varied colors),
``paper`` (DDColor-L, paper model) and ``paper_tiny`` (DDColor-T, fastest).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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

_HF_ORG = "piddnad"
_VARIANTS = ("modelscope", "artistic", "paper", "paper_tiny")
_DEFAULT_VARIANT = "modelscope"
# The network was trained at this size; chroma is predicted at 512x512 and upsampled.
_MODEL_SIZE = 512
# Files a ``piddnad/ddcolor_*`` repo may contain, in order of preference.
_WEIGHT_CANDIDATES = ("model.safetensors", "pytorch_model.bin", "pytorch_model.pt")
_HF_REPO = f"{_HF_ORG}/ddcolor_{_DEFAULT_VARIANT}"
_WEIGHT_FILE = _WEIGHT_CANDIDATES[0]


class DDColorRestorer(BaseRestorer):
    """Colorize grayscale or desaturated frames with DDColor (frame by frame)."""

    def __init__(self, variant: str = _DEFAULT_VARIANT) -> None:
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}, got {variant!r}")
        self._variant = variant
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
        model = self._build_model(self._variant, device)
        model.eval()
        self._model = model
        self._device = device
        self._loaded = True
        logger.info("DDColor (%s) loaded on %s", self._variant, device)

    def unload(self) -> None:
        del self._model
        self._model = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Inference ─────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        """Colorize one RGB uint8 frame; the output has the same size and L channel."""
        assert self._model is not None and self._device is not None
        height, width = frame.shape[:2]

        img = frame.astype(np.float32) / 255.0
        orig_l = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, :, :1]

        # Gray RGB image at the model resolution, built from the luminance only.
        resized = cv2.resize(img, (_MODEL_SIZE, _MODEL_SIZE))
        l_small = cv2.cvtColor(resized, cv2.COLOR_RGB2Lab)[:, :, :1]
        gray_lab = np.concatenate(
            (l_small, np.zeros_like(l_small), np.zeros_like(l_small)), axis=-1
        )
        gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)
        tensor = torch.from_numpy(gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0)

        with torch.inference_mode():
            ab = self._model(tensor.to(self._device)).float().cpu()  # 1 2 S S, Lab a/b units

        ab = F.interpolate(ab, size=(height, width), mode="bilinear", align_corners=False)
        ab_np = ab[0].numpy().transpose(1, 2, 0)
        lab = np.concatenate((orig_l, ab_np), axis=-1).astype(np.float32)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        out: np.ndarray = (rgb * 255.0).round().clip(0, 255).astype(np.uint8)
        return out

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_model(variant: str, device: torch.device) -> torch.nn.Module:
        """Build DDColor from the repo's config and load its official weights."""
        from restorax.restorers.colorization.ddcolor_arch import DDColor

        weight_path, config = _download_weights(variant)
        try:
            model = DDColor(**_model_kwargs(variant, config))
            state_dict = _load_state_dict(weight_path)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
        except Exception as exc:
            raise RestorerLoadError(f"Failed to load DDColor weights ({variant}): {exc}") from exc
        # Normalisation buffers are not in every checkpoint; anything else missing is an error.
        real_missing = [k for k in missing if k not in ("mean", "std")]
        if real_missing or unexpected:
            raise RestorerLoadError(
                f"DDColor checkpoint does not match the architecture: "
                f"{len(real_missing)} missing and {len(unexpected)} unexpected keys "
                f"(e.g. {(real_missing or list(unexpected))[:3]})"
            )
        ready: torch.nn.Module = model.to(device)
        return ready


def _model_kwargs(variant: str, config: dict[str, Any] | None) -> dict[str, Any]:
    """Constructor arguments: the repo's config.json when present, else the paper defaults."""
    kwargs: dict[str, Any] = {
        "encoder_name": "convnext-t" if variant == "paper_tiny" else "convnext-l",
        "decoder_name": "MultiScaleColorDecoder",
        "num_input_channels": 3,
        "input_size": [_MODEL_SIZE, _MODEL_SIZE],
        "nf": 512,
        "num_output_channels": 2,
        "last_norm": "Spectral",
        "do_normalize": False,
        "num_queries": 100,
        "num_scales": 3,
        "dec_layers": 9,
    }
    if config:
        allowed = set(kwargs)
        kwargs.update({k: v for k, v in config.items() if k in allowed})
    return kwargs


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor]
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
        return state
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("params", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            ckpt = ckpt[key]
            break
    state = ckpt
    return state


def _download_weights(variant: str) -> tuple[Path, dict[str, Any] | None]:
    """Fetch the weights (and config.json when available) of a ``piddnad/ddcolor_*`` repo."""
    from restorax.config import settings

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError as exc:
        raise RestorerLoadError("huggingface_hub is required to download DDColor weights.") from exc

    repo = f"{_HF_ORG}/ddcolor_{variant}"
    # The default variant lives directly in ``ddcolor/`` (where ``restorax download-models``
    # puts it); the others get their own sub-directory.
    model_dir = Path(settings.model_dir) / "ddcolor"
    if variant != _DEFAULT_VARIANT:
        model_dir = model_dir / variant
    model_dir.mkdir(parents=True, exist_ok=True)

    config: dict[str, Any] | None = None
    try:
        config_path = hf_hub_download(
            repo_id=repo, filename="config.json", local_dir=str(model_dir)
        )
        config = json.loads(Path(config_path).read_text())
    except Exception:  # noqa: BLE001 - config is optional, defaults match the paper
        logger.debug("No config.json for %s, using defaults", repo, exc_info=True)

    for filename in _WEIGHT_CANDIDATES:
        existing = model_dir / filename
        if existing.exists():
            return existing, config
        try:
            logger.info("Downloading DDColor weights %s/%s", repo, filename)
            return Path(
                hf_hub_download(repo_id=repo, filename=filename, local_dir=str(model_dir))
            ), config
        except EntryNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001
            raise RestorerLoadError(f"Cannot download DDColor weights from {repo}: {exc}") from exc
    raise RestorerLoadError(f"No weight file ({', '.join(_WEIGHT_CANDIDATES)}) found in {repo}")
