"""
ComfyUI custom-node pack for RestoraX restorers.

ComfyUI is not a pip-installable library — custom nodes are plain classes
matching a structural contract (INPUT_TYPES/RETURN_TYPES/FUNCTION/CATEGORY).
This module never imports anything from the ComfyUI app itself.
"""
from __future__ import annotations

import numpy as np
import torch


def comfy_image_to_frames(image: torch.Tensor) -> list[np.ndarray]:
    """ComfyUI IMAGE (B,H,W,C) float32 [0,1] RGB -> list of (H,W,3) uint8 RGB frames."""
    arr = (image.clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return [arr[i] for i in range(arr.shape[0])]


def frames_to_comfy_image(frames: list[np.ndarray]) -> torch.Tensor:
    """List of (H,W,3) uint8 RGB frames -> ComfyUI IMAGE (B,H,W,C) float32 [0,1] RGB."""
    stacked = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(stacked)
