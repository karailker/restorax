"""
RestoraX ComfyUI custom-node pack.

ComfyUI's custom-node loader imports NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS
from this file.
"""
from . import (
    artifact_removal,
    audio,
    colorization,
    deinterlacing,
    face_restoration,
    frame_interpolation,
    hdr,
    stabilization,
    super_resolution,
)

_MODULES = [
    artifact_removal, audio, colorization, deinterlacing, face_restoration,
    frame_interpolation, hdr, stabilization, super_resolution,
]

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}
for _module in _MODULES:
    NODE_CLASS_MAPPINGS.update(_module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_module.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
