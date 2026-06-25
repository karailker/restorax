import numpy as np
import torch

from comfyui_nodes._base import comfy_image_to_frames, frames_to_comfy_image
from restorax.core.registry import ModelRegistry


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
