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


from restorax.core.restorer import (
    BaseRestorer, RestorerCapabilities, RestorerCategory, RestorerParams, ParamSpec,
)
from comfyui_nodes._base import make_restorer_node


class _FakeUpscaler(BaseRestorer):
    """Doubles every pixel value and reports the resulting params for assertions."""
    PARAM_SCHEMA = [ParamSpec("boost", "float", 1.0, "Boost", target="extra", minimum=0.0, maximum=4.0)]
    last_params: RestorerParams | None = None

    @property
    def name(self) -> str:
        return "fake_upscaler"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.SUPER_RESOLUTION,
            input_color_space="rgb", output_color_space="rgb", scale_factor=2,
        )

    def load(self, device):
        pass

    def unload(self):
        pass

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        _FakeUpscaler.last_params = params
        boosted = np.clip(frame.astype(np.float32) * params.extra.get("boost", 1.0), 0, 255)
        return boosted.astype(np.uint8)


class _FakeBGRFaceRestorer(BaseRestorer):
    """BGR in/out — exercises the color-space conversion path."""

    @property
    def name(self) -> str:
        return "fake_bgr_face"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.FACE_RESTORATION,
            input_color_space="bgr", output_color_space="bgr",
        )

    def load(self, device):
        pass

    def unload(self):
        pass

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        # Identity — just proves the node round-trips BGR<->RGB correctly.
        return frame


def test_make_restorer_node_builds_input_types_from_param_schema():
    node_cls = make_restorer_node(_FakeUpscaler, "Super Resolution")
    input_types = node_cls.INPUT_TYPES()
    assert input_types["required"]["image"] == ("IMAGE",)
    assert input_types["required"]["boost"] == ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0})
    assert input_types["required"]["scale"] == ("INT", {"default": 2, "min": 1, "max": 8})
    assert node_cls.RETURN_TYPES == ("IMAGE",)
    assert node_cls.FUNCTION == "restore"
    assert node_cls.CATEGORY == "RestoraX/Super Resolution"


def test_node_restore_doubles_pixel_values_via_extra_param(monkeypatch):
    import comfyui_nodes._base as base
    fresh_registry = ModelRegistry()
    fresh_registry.register(_FakeUpscaler)
    monkeypatch.setattr(base, "get_registry", lambda: fresh_registry)

    node_cls = make_restorer_node(_FakeUpscaler, "Super Resolution")
    node = node_cls()
    image = torch.tensor([[[[0.2, 0.2, 0.2]]]], dtype=torch.float32)  # (1,1,1,3)

    (result,) = node.restore(image, scale=2, boost=2.0)

    assert _FakeUpscaler.last_params.extra["boost"] == 2.0
    assert _FakeUpscaler.last_params.scale == 2
    expected_value = min(round(0.2 * 255) * 2, 255) / 255
    assert torch.allclose(result[0, 0, 0], torch.tensor([expected_value] * 3), atol=1e-3)


def test_node_restore_round_trips_bgr_restorer_without_color_shift(monkeypatch):
    import comfyui_nodes._base as base
    fresh_registry = ModelRegistry()
    fresh_registry.register(_FakeBGRFaceRestorer)
    monkeypatch.setattr(base, "get_registry", lambda: fresh_registry)

    node_cls = make_restorer_node(_FakeBGRFaceRestorer, "Face Restoration")
    node = node_cls()
    image = torch.tensor([[[[0.1, 0.4, 0.8]]]], dtype=torch.float32)

    (result,) = node.restore(image)

    assert torch.allclose(result, image, atol=1 / 255)


# Audio conversion helpers
from comfyui_nodes._base import comfy_audio_to_array, array_to_comfy_audio


def test_comfy_audio_to_array_converts_waveform_to_numpy():
    # (1, 2, 4) waveform = 1 batch, 2 channels, 4 samples
    waveform = torch.tensor([[[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, -0.3, -0.4]]])
    audio = {"waveform": waveform, "sample_rate": 44100}
    arr, sr = comfy_audio_to_array(audio)
    assert sr == 44100
    assert arr.dtype == np.float32
    assert arr.shape == (4, 2)  # (samples, channels)
    np.testing.assert_allclose(arr[:, 0], [0.1, 0.2, 0.3, 0.4], atol=1e-6)


def test_array_to_comfy_audio_converts_numpy_to_waveform():
    arr = np.array([[0.1, -0.1], [0.2, -0.2]], dtype=np.float32)  # (2 samples, 2 channels)
    result = array_to_comfy_audio(arr, 44100)
    assert result["sample_rate"] == 44100
    assert result["waveform"].shape == (1, 2, 2)  # (B=1, C=2, T=2)
    assert result["waveform"].dtype == torch.float32


def test_audio_round_trip_preserves_values():
    arr = np.array([[0.5, -0.5], [0.3, -0.3]], dtype=np.float32)
    result = array_to_comfy_audio(arr, 22050)
    restored_arr, sr = comfy_audio_to_array(result)
    assert sr == 22050
    np.testing.assert_allclose(restored_arr, arr, atol=1e-6)
