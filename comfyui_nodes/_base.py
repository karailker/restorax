"""
ComfyUI custom-node pack for RestoraX restorers.

ComfyUI is not a pip-installable library — custom nodes are plain classes
matching a structural contract (INPUT_TYPES/RETURN_TYPES/FUNCTION/CATEGORY).
This module never imports anything from the ComfyUI app itself.
"""
from __future__ import annotations

import numpy as np
import torch

from restorax.audio.pipeline import AudioModelRegistry
from restorax.audio.restorer import AudioRestorer, AudioRestorerParams
from restorax.core.registry import ModelRegistry
from restorax.core.restorer import BaseRestorer, ParamSpec, RestorerParams
from restorax.video.utils import from_rgb, to_rgb

_registry: ModelRegistry | None = None
_audio_registry: AudioModelRegistry | None = None


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


def _instance_attr(restorer_cls: type[BaseRestorer], attr_name: str):
    """Read a no-init-dependency property (name/capabilities) the same way
    restorax.core.registry.ModelRegistry does, without calling __init__."""
    instance = object.__new__(restorer_cls)
    return getattr(restorer_cls, attr_name).fget(instance)


def comfy_audio_to_array(audio: dict) -> tuple[np.ndarray, int]:
    """ComfyUI AUDIO {"waveform": (B,C,T) float32, "sample_rate": int}
    -> (num_samples, num_channels) float32 [-1,1], sample_rate."""
    waveform = audio["waveform"]  # (B, C, T)
    sample_rate = int(audio["sample_rate"])
    # take first batch item, transpose (C,T) -> (T,C) = (samples, channels)
    arr = waveform[0].cpu().numpy().T  # (samples, channels)
    return arr.astype(np.float32), sample_rate


def array_to_comfy_audio(arr: np.ndarray, sample_rate: int) -> dict:
    """(num_samples, num_channels) float32 [-1,1], sample_rate
    -> ComfyUI AUDIO {"waveform": (1,C,T) float32, "sample_rate": int}."""
    # transpose (samples, channels) -> (channels, samples) = (C, T)
    waveform = torch.from_numpy(arr.T).unsqueeze(0).float()  # (1, C, T)
    return {"waveform": waveform, "sample_rate": sample_rate}


def get_audio_registry() -> AudioModelRegistry:
    """Module-level AudioModelRegistry singleton."""
    global _audio_registry
    if _audio_registry is None:
        _audio_registry = AudioModelRegistry()
    return _audio_registry


def make_audio_restorer_node(restorer_cls: type, category_label: str) -> type:
    """Build a ComfyUI node class for a single RestoraX audio restorer."""
    name = _instance_attr(restorer_cls, "name")
    param_specs = restorer_cls.PARAM_SCHEMA  # list[ParamSpec]

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        required: dict = {"audio": ("AUDIO",)}
        for spec in param_specs:
            required[spec.name] = param_spec_to_input(spec)
        return {"required": required}

    def restore(self, audio, **kwargs):
        restorer = get_audio_registry().get(name, get_device())
        params = AudioRestorerParams()
        for spec in param_specs:
            value = kwargs.get(spec.name, spec.default)
            if spec.kind == "multiselect" and isinstance(value, str):
                value = [v for v in value.split(",") if v]
            if spec.target == "param":
                setattr(params, spec.name, value)
            else:
                params.extra[spec.name] = value

        arr, sample_rate = comfy_audio_to_array(audio)
        params.sample_rate = sample_rate
        out_arr = restorer.process_audio(arr, params)
        return (array_to_comfy_audio(out_arr, sample_rate),)

    return type(
        f"{restorer_cls.__name__}Node",
        (object,),
        {
            "INPUT_TYPES": INPUT_TYPES,
            "RETURN_TYPES": ("AUDIO",),
            "FUNCTION": "restore",
            "CATEGORY": f"RestoraX/{category_label}",
            "restore": restore,
        },
    )


def make_restorer_node(restorer_cls: type[BaseRestorer], category_label: str) -> type:
    """Build a ComfyUI node class for a single RestoraX video restorer."""
    name = _instance_attr(restorer_cls, "name")
    caps = _instance_attr(restorer_cls, "capabilities")
    param_specs = restorer_cls.PARAM_SCHEMA

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 — ComfyUI's required casing
        required: dict = {"image": ("IMAGE",)}
        if caps.scale_factor != 1:
            required["scale"] = ("INT", {"default": caps.scale_factor, "min": 1, "max": 8})
        for spec in param_specs:
            required[spec.name] = param_spec_to_input(spec)
        return {"required": required}

    def restore(self, image, scale=None, **kwargs):
        restorer = get_registry().get(name, get_device())
        params = RestorerParams(scale=scale if scale is not None else caps.scale_factor)
        for spec in param_specs:
            value = kwargs.get(spec.name, spec.default)
            if spec.kind == "multiselect" and isinstance(value, str):
                value = [v for v in value.split(",") if v]
            if spec.target == "param":
                setattr(params, spec.name, value)
            else:
                params.extra[spec.name] = value

        frames = comfy_image_to_frames(image)
        out_frames = []
        for frame in frames:
            converted_in = from_rgb(frame, caps.input_color_space) if caps.input_color_space != "rgb" else frame
            processed = restorer.process_frame(converted_in, params)
            out_frames.append(
                to_rgb(processed, caps.output_color_space) if caps.output_color_space != "rgb" else processed
            )
        return (frames_to_comfy_image(out_frames),)

    return type(
        f"{restorer_cls.__name__}Node",
        (object,),
        {
            "INPUT_TYPES": INPUT_TYPES,
            "RETURN_TYPES": ("IMAGE",),
            "FUNCTION": "restore",
            "CATEGORY": f"RestoraX/{category_label}",
            "restore": restore,
        },
    )
