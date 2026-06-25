import numpy as np
import torch

from comfyui_nodes._base import comfy_image_to_frames, frames_to_comfy_image, param_spec_to_input
from restorax.core.registry import ModelRegistry
from restorax.core.restorer import ParamSpec


def test_comfy_image_to_frames_converts_batch_to_uint8_frames():
    image = torch.tensor(
        [[[[0.0, 0.5, 1.0], [1.0, 0.0, 0.5]]]], dtype=torch.float32
    )  # shape (1, 1, 2, 3)
    frames = comfy_image_to_frames(image)
    assert len(frames) == 1
    assert frames[0].dtype == np.uint8
    assert frames[0].shape == (1, 2, 3)
    np.testing.assert_array_equal(frames[0][0, 0], [0, 128, 255])


def test_comfy_image_to_frames_clamps_out_of_range_values():
    image = torch.tensor([[[[1.5, -0.5, 0.5]]]], dtype=torch.float32)
    frames = comfy_image_to_frames(image)
    np.testing.assert_array_equal(frames[0][0, 0], [255, 0, 128])


def test_frames_to_comfy_image_converts_uint8_frames_to_batch():
    frame = np.array([[[0, 128, 255]]], dtype=np.uint8)
    image = frames_to_comfy_image([frame, frame])
    assert image.shape == (2, 1, 1, 3)
    assert image.dtype == torch.float32
    assert torch.allclose(image[0, 0, 0], torch.tensor([0.0, 128 / 255, 1.0]), atol=1e-6)


def test_round_trip_preserves_pixel_values():
    frame = np.array([[[10, 20, 30], [200, 210, 220]]], dtype=np.uint8)
    image = frames_to_comfy_image([frame])
    restored = comfy_image_to_frames(image)
    np.testing.assert_array_equal(restored[0], frame)


def test_get_registry_returns_singleton():
    import comfyui_nodes._base as base
    base._registry = None  # reset module state for test isolation
    first = base.get_registry()
    second = base.get_registry()
    assert first is second
    assert isinstance(first, ModelRegistry)


def test_get_device_returns_cpu_when_cuda_unavailable(monkeypatch):
    import comfyui_nodes._base as base
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert base.get_device() == torch.device("cpu")


def test_int_spec_maps_to_int_widget():
    spec = ParamSpec("tile_size", "int", 0, "Tile size", target="param", minimum=0, maximum=2048, step=32)
    result = param_spec_to_input(spec)
    assert result == ("INT", {"default": 0, "min": 0, "max": 2048, "step": 32})


def test_float_spec_maps_to_float_widget():
    spec = ParamSpec("strength", "float", 0.5, "Strength", minimum=0.0, maximum=1.0, step=0.05)
    result = param_spec_to_input(spec)
    assert result == ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05})


def test_bool_spec_maps_to_boolean_widget():
    spec = ParamSpec("half_precision", "bool", True, "Half precision (fp16)", target="param")
    assert param_spec_to_input(spec) == ("BOOLEAN", {"default": True})


def test_enum_spec_maps_to_choice_list():
    spec = ParamSpec("mode", "enum", "fast", "Mode", choices=("fast", "quality"))
    result = param_spec_to_input(spec)
    assert result == (["fast", "quality"], {"default": "fast"})


def test_multiselect_spec_maps_to_comma_separated_string():
    spec = ParamSpec("tags", "multiselect", ["a", "b"], "Tags")
    assert param_spec_to_input(spec) == ("STRING", {"default": "a,b", "multiline": False})


def test_multiselect_spec_handles_empty_default():
    spec = ParamSpec("tags", "multiselect", None, "Tags")
    assert param_spec_to_input(spec) == ("STRING", {"default": "", "multiline": False})
