"""
ComfyUI custom-node pack for RestoraX restorers.

ComfyUI is not a pip-installable library — custom nodes are plain classes
matching a structural contract (INPUT_TYPES/RETURN_TYPES/FUNCTION/CATEGORY).
This module never imports anything from the ComfyUI app itself.
"""
from __future__ import annotations

import numpy as np
import torch

from restorax.core.registry import ModelRegistry
from restorax.core.restorer import ParamSpec

_registry: ModelRegistry | None = None


def comfy_image_to_frames(image: torch.Tensor) -> list[np.ndarray]:
    """ComfyUI IMAGE (B,H,W,C) float32 [0,1] RGB -> list of (H,W,3) uint8 RGB frames."""
    arr = (image.clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return [arr[i] for i in range(arr.shape[0])]


def frames_to_comfy_image(frames: list[np.ndarray]) -> torch.Tensor:
    """List of (H,W,3) uint8 RGB frames -> ComfyUI IMAGE (B,H,W,C) float32 [0,1] RGB."""
    stacked = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(stacked)


def get_registry() -> ModelRegistry:
    """Module-level registry singleton, one per ComfyUI process (mirrors
    restorax.tasks.job_tasks._get_registry's per-worker-process pattern)."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def param_spec_to_input(spec: ParamSpec) -> tuple:
    """Map a RestoraX ParamSpec to a ComfyUI INPUT_TYPES (type, options) pair."""
    if spec.kind == "int":
        opts: dict = {"default": int(spec.default)}
        if spec.minimum is not None:
            opts["min"] = int(spec.minimum)
        if spec.maximum is not None:
            opts["max"] = int(spec.maximum)
        if spec.step is not None:
            opts["step"] = int(spec.step)
        return ("INT", opts)
    if spec.kind == "float":
        opts = {"default": float(spec.default)}
        if spec.minimum is not None:
            opts["min"] = float(spec.minimum)
        if spec.maximum is not None:
            opts["max"] = float(spec.maximum)
        if spec.step is not None:
            opts["step"] = float(spec.step)
        return ("FLOAT", opts)
    if spec.kind == "bool":
        return ("BOOLEAN", {"default": bool(spec.default)})
    if spec.kind == "enum":
        return (list(spec.choices or ()), {"default": spec.default})
    if spec.kind == "multiselect":
        # ComfyUI has no native multiselect widget — expose as a comma-separated STRING.
        default_str = ",".join(str(v) for v in (spec.default or []))
        return ("STRING", {"default": default_str, "multiline": False})
    raise ValueError(f"Unsupported ParamSpec.kind: {spec.kind!r}")
