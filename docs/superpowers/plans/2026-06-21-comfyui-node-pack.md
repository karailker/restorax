# ComfyUI Node Pack (SP4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a `comfyui_nodes/` package that exposes every RestoraX **video** restorer (22 of the 25 in the catalog) as a ComfyUI custom node, generated from a single generic factory driven by each restorer's existing `RestorerCapabilities`/`PARAM_SCHEMA` — no per-node hand-written `INPUT_TYPES`.

**Architecture:** ComfyUI custom nodes are plain Python classes matching a structural contract (`INPUT_TYPES` classmethod, `RETURN_TYPES`, `FUNCTION`, `CATEGORY`) registered in a module-level `NODE_CLASS_MAPPINGS` dict — they need not subclass anything from ComfyUI itself, and ComfyUI itself is not a pip-installable library (it's a git-cloned app). So: (1) a `make_restorer_node(restorer_cls, category_label)` factory in `comfyui_nodes/_base.py` introspects `restorer_cls.PARAM_SCHEMA` + `restorer_cls.capabilities` (read via the same `object.__new__` trick `ModelRegistry`/`models.py` already use, since both are pure properties with no instance state) to build `INPUT_TYPES` and a `restore()` method that converts ComfyUI's `IMAGE` tensor to/from the numpy frames `BaseRestorer.process_frame` expects, respecting each restorer's declared `input_color_space`/`output_color_space` (face-restoration restorers are BGR; most others are RGB) via the existing `restorax.video.utils.to_rgb`/`from_rgb` helpers. (2) One file per `RestorerCategory` builds nodes for that category's restorer classes and exports its slice of the mappings. (3) `comfyui_nodes/__init__.py` aggregates all category mappings — this is the file ComfyUI's custom-node loader actually imports. Audio restorers (Demucs/VoiceFixer/RNNoise) are explicitly **out of scope**: they use `AudioRestorerParams`/`process()`, not `RestorerParams`/`process_frame()` — a structurally different contract already flagged as a DAG-incompatibility gap in `PLAN.md` §5.

**Tech Stack:** Python 3.11, PyTorch + NumPy (already core deps, no new runtime dependency), pytest, ComfyUI installed as a **dev-only** tool via `comfy-cli` (PyPI package) for local node-loading verification — never imported at runtime by `comfyui_nodes/` code.

## Global Constraints

- No new runtime dependency added to `restorax`'s core install — `comfyui_nodes/` only imports `torch`, `numpy`, and `restorax.*` (all already present).
- `comfyui_nodes/` code must never `import comfy` or anything from the ComfyUI app itself — the contract is purely structural (duck-typed), matching how real ComfyUI custom-node packs are written.
- Audio restorers (`restorax/restorers/audio/*`) are excluded from this plan — do not build nodes for them.
- Every node factory output must be unit-testable without a real ComfyUI installation or real model weights (use a fake/mock `BaseRestorer` subclass, mirroring the existing `IdentityRestorer` pattern in `tests/conftest.py`).
- Follow the existing skip-marker convention (`requires_weights`, `requires_assets` in `tests/conftest.py` / `pyproject.toml`) by adding a new `requires_comfyui` marker for the one integration test that needs a real ComfyUI checkout.
- `comfyui_manifest.json` / ComfyUI-Manager community-list PR is explicitly **out of scope for this plan** (deferred — its manifest schema needs to be verified against ComfyUI-Manager's current docs at execution time, not assumed).

---

## File Structure

```text
comfyui_nodes/
  __init__.py              ← aggregates NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS
  _base.py                 ← tensor<->numpy conversion, registry/device singletons, make_restorer_node factory
  super_resolution.py       ← 10 nodes (Real-ESRGAN, BasicVSR++, Upscale-A-Video, VRT, MambaIR, TDM, SeedVR, Waifu2x, FlashVSR, EvTexture)
  face_restoration.py       ← 4 nodes (CodeFormer, CodeFormer++, GFPGAN, DicFace)
  colorization.py           ← 1 node (DDColor)
  frame_interpolation.py    ← 1 node (RIFE)
  deinterlacing.py          ← 2 nodes (AI deinterlace, YADIF)
  artifact_removal.py       ← 1 node (ScratchRemoval)
  hdr.py                    ← 1 node (HDRTVDM)
  stabilization.py          ← 2 nodes (VideoStabilization, GaVS)
  requirements.txt          ← pins restorax itself (ComfyUI-Manager installs this when a user adds the pack)
tests/unit/test_comfyui_nodes.py  ← conversion + factory unit tests (no real ComfyUI/weights needed)
tests/integration/test_comfyui_load.py  ← requires_comfyui: loads the pack inside a real ComfyUI checkout
pyproject.toml              ← add `comfyui` dev-extra + `requires_comfyui` marker
```

---

## Task 1: Tensor↔numpy conversion helpers

**Files:**

- Create: `comfyui_nodes/__init__.py` (empty placeholder for now — package marker)
- Create: `comfyui_nodes/_base.py`
- Test: `tests/unit/test_comfyui_nodes.py`

**Interfaces:**

- Produces: `comfy_image_to_frames(image: torch.Tensor) -> list[np.ndarray]`, `frames_to_comfy_image(frames: list[np.ndarray]) -> torch.Tensor`. ComfyUI `IMAGE` = `torch.Tensor` shape `(B, H, W, C)`, `float32`, range `[0, 1]`, RGB channel order. RestoraX frames = `np.ndarray` shape `(H, W, 3)`, `uint8`, channel order per-restorer (see Task 3).

- [ ] **Step 1: Create the package marker**

```python
# comfyui_nodes/__init__.py
```

(Left empty for now — Task 8 fills in the real aggregation.)

- [ ] **Step 2: Write the failing tests for conversion round-trip**

```python
# tests/unit/test_comfyui_nodes.py
import numpy as np
import torch

from comfyui_nodes._base import comfy_image_to_frames, frames_to_comfy_image


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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/test_comfyui_nodes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'comfyui_nodes._base'`

- [ ] **Step 4: Implement the conversion functions**

```python
# comfyui_nodes/_base.py
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_comfyui_nodes.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add comfyui_nodes/__init__.py comfyui_nodes/_base.py tests/unit/test_comfyui_nodes.py
git commit -m "feat(comfyui): add IMAGE tensor <-> numpy frame conversion helpers"
```

---

## Task 2: Registry + device singletons

**Files:**

- Modify: `comfyui_nodes/_base.py`
- Test: `tests/unit/test_comfyui_nodes.py`

**Interfaces:**

- Consumes: `restorax.core.registry.ModelRegistry` (`__init__(max_loaded=2)`, `.register(cls)`, `.get(name, device) -> BaseRestorer`).
- Produces: `get_registry() -> ModelRegistry` (module-level singleton, mirrors the per-worker-process pattern in `restorax/tasks/job_tasks.py::_get_registry`), `get_device() -> torch.device`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_comfyui_nodes.py
from restorax.core.registry import ModelRegistry


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_comfyui_nodes.py -k "registry or device" -v`
Expected: FAIL with `AttributeError: module 'comfyui_nodes._base' has no attribute 'get_registry'`

- [ ] **Step 3: Implement the singletons**

```python
# add to comfyui_nodes/_base.py, after the imports
from restorax.core.registry import ModelRegistry

_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Module-level registry singleton, one per ComfyUI process (mirrors
    restorax.tasks.job_tasks._get_registry's per-worker-process pattern)."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_comfyui_nodes.py -k "registry or device" -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add comfyui_nodes/_base.py tests/unit/test_comfyui_nodes.py
git commit -m "feat(comfyui): add per-process registry and device singletons"
```

---

## Task 3: ParamSpec -> ComfyUI INPUT_TYPES mapping

**Files:**

- Modify: `comfyui_nodes/_base.py`
- Test: `tests/unit/test_comfyui_nodes.py`

**Interfaces:**

- Consumes: `restorax.core.restorer.ParamSpec` (fields: `name`, `kind: Literal["int","float","bool","enum","multiselect"]`, `default`, `target: Literal["param","extra"]`, `minimum`, `maximum`, `step`, `choices`).
- Produces: `param_spec_to_input(spec: ParamSpec) -> tuple` — a ComfyUI `INPUT_TYPES`-shaped `(type_or_choices, options_dict)` pair.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_comfyui_nodes.py
from restorax.core.restorer import ParamSpec
from comfyui_nodes._base import param_spec_to_input


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_comfyui_nodes.py -k "spec" -v`
Expected: FAIL with `ImportError: cannot import name 'param_spec_to_input'`

- [ ] **Step 3: Implement the mapping**

```python
# add to comfyui_nodes/_base.py
from restorax.core.restorer import ParamSpec


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_comfyui_nodes.py -k "spec" -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add comfyui_nodes/_base.py tests/unit/test_comfyui_nodes.py
git commit -m "feat(comfyui): map ParamSpec kinds to ComfyUI INPUT_TYPES widgets"
```

---

## Task 4: `make_restorer_node` factory

**Files:**

- Modify: `comfyui_nodes/_base.py`
- Test: `tests/unit/test_comfyui_nodes.py`

**Interfaces:**

- Consumes: `restorax.core.restorer.BaseRestorer` subclasses, `restorax.core.restorer.RestorerParams`, `restorax.video.utils.to_rgb`/`from_rgb` (signatures: `to_rgb(frame: np.ndarray, src: str) -> np.ndarray`, `from_rgb(frame: np.ndarray, dst: str) -> np.ndarray`), `get_registry()`/`get_device()`/`param_spec_to_input()` from Tasks 1-3.
- Produces: `make_restorer_node(restorer_cls: type[BaseRestorer], category_label: str) -> type` — returns a node class with `INPUT_TYPES` classmethod, `RETURN_TYPES = ("IMAGE",)`, `FUNCTION = "restore"`, `CATEGORY = f"RestoraX/{category_label}"`, and a bound `restore(self, image, **kwargs) -> tuple[torch.Tensor]` method. Later category-file tasks rely on calling this with a restorer class + a human label string.

- [ ] **Step 1: Write the failing tests using a fake restorer**

```python
# append to tests/unit/test_comfyui_nodes.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_comfyui_nodes.py -k "make_restorer_node or node_restore" -v`
Expected: FAIL with `ImportError: cannot import name 'make_restorer_node'`

- [ ] **Step 3: Implement the factory**

```python
# add to comfyui_nodes/_base.py
from restorax.core.restorer import BaseRestorer, RestorerParams
from restorax.video.utils import from_rgb, to_rgb


def _instance_attr(restorer_cls: type[BaseRestorer], attr_name: str):
    """Read a no-init-dependency property (name/capabilities) the same way
    restorax.core.registry.ModelRegistry does, without calling __init__."""
    instance = object.__new__(restorer_cls)
    return getattr(restorer_cls, attr_name).fget(instance)


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_comfyui_nodes.py -v`
Expected: all tests in the file pass (16 total across Tasks 1-4)

- [ ] **Step 5: Commit**

```bash
git add comfyui_nodes/_base.py tests/unit/test_comfyui_nodes.py
git commit -m "feat(comfyui): add generic make_restorer_node factory with color-space handling"
```

---

## Task 5: Super-resolution + face-restoration node files

**Files:**

- Create: `comfyui_nodes/super_resolution.py`
- Create: `comfyui_nodes/face_restoration.py`
- Test: `tests/unit/test_comfyui_node_files.py`

**Interfaces:**

- Consumes: `make_restorer_node` (Task 4); restorer classes `RealESRGANx4Restorer`, `BasicVSRPlusPlusRestorer`, `UpscaleAVideoRestorer`, `VRTRestorer`, `MambaIRRestorer`, `TDMRestorer`, `SeedVRRestorer`, `Waifu2xRestorer`, `FlashVSRRestorer`, `EvTextureRestorer` (all in `restorax/restorers/super_resolution/`); `CodeFormerRestorer`, `CodeFormerPlusPlusRestorer`, `GFPGANRestorer`, `DicFaceRestorer` (all in `restorax/restorers/face_restoration/`).
- Produces: each file exports `NODE_CLASS_MAPPINGS: dict[str, type]` and `NODE_DISPLAY_NAME_MAPPINGS: dict[str, str]`, keyed by a `RestoraX_<RestorerName>` string — the convention `comfyui_nodes/__init__.py` (Task 8) merges across files.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_comfyui_node_files.py
def test_super_resolution_module_exports_ten_nodes():
    from comfyui_nodes import super_resolution as sr
    assert len(sr.NODE_CLASS_MAPPINGS) == 10
    assert set(sr.NODE_CLASS_MAPPINGS) == set(sr.NODE_DISPLAY_NAME_MAPPINGS)
    for node_cls in sr.NODE_CLASS_MAPPINGS.values():
        assert node_cls.CATEGORY == "RestoraX/Super Resolution"


def test_face_restoration_module_exports_four_nodes():
    from comfyui_nodes import face_restoration as fr
    assert len(fr.NODE_CLASS_MAPPINGS) == 4
    for node_cls in fr.NODE_CLASS_MAPPINGS.values():
        assert node_cls.CATEGORY == "RestoraX/Face Restoration"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_comfyui_node_files.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'comfyui_nodes.super_resolution'`

- [ ] **Step 3: Implement `comfyui_nodes/super_resolution.py`**

```python
from restorax.restorers.super_resolution.basicvsr_pp import BasicVSRPlusPlusRestorer
from restorax.restorers.super_resolution.evtexture import EvTextureRestorer
from restorax.restorers.super_resolution.flashvsr import FlashVSRRestorer
from restorax.restorers.super_resolution.mamba_ir import MambaIRRestorer
from restorax.restorers.super_resolution.real_esrgan import RealESRGANx4Restorer
from restorax.restorers.super_resolution.seedvr import SeedVRRestorer
from restorax.restorers.super_resolution.tdm import TDMRestorer
from restorax.restorers.super_resolution.upscale_a_video import UpscaleAVideoRestorer
from restorax.restorers.super_resolution.vrt import VRTRestorer
from restorax.restorers.super_resolution.waifu2x import Waifu2xRestorer

from ._base import make_restorer_node

_CATEGORY = "Super Resolution"
_RESTORERS = [
    ("RealESRGAN", RealESRGANx4Restorer, "RestoraX Real-ESRGAN x4"),
    ("BasicVSRPP", BasicVSRPlusPlusRestorer, "RestoraX BasicVSR++"),
    ("UpscaleAVideo", UpscaleAVideoRestorer, "RestoraX Upscale-A-Video"),
    ("VRT", VRTRestorer, "RestoraX VRT"),
    ("MambaIR", MambaIRRestorer, "RestoraX MambaIR"),
    ("TDM", TDMRestorer, "RestoraX TDM"),
    ("SeedVR", SeedVRRestorer, "RestoraX SeedVR"),
    ("Waifu2x", Waifu2xRestorer, "RestoraX Waifu2x"),
    ("FlashVSR", FlashVSRRestorer, "RestoraX FlashVSR"),
    ("EvTexture", EvTextureRestorer, "RestoraX EvTexture"),
]

NODE_CLASS_MAPPINGS = {
    f"RestoraX_{key}": make_restorer_node(cls, _CATEGORY) for key, cls, _ in _RESTORERS
}
NODE_DISPLAY_NAME_MAPPINGS = {f"RestoraX_{key}": label for key, _, label in _RESTORERS}
```

- [ ] **Step 4: Implement `comfyui_nodes/face_restoration.py`**

```python
from restorax.restorers.face_restoration.codeformer import CodeFormerRestorer
from restorax.restorers.face_restoration.codeformer_pp import CodeFormerPlusPlusRestorer
from restorax.restorers.face_restoration.dicface import DicFaceRestorer
from restorax.restorers.face_restoration.gfpgan import GFPGANRestorer

from ._base import make_restorer_node

_CATEGORY = "Face Restoration"
_RESTORERS = [
    ("CodeFormer", CodeFormerRestorer, "RestoraX CodeFormer"),
    ("CodeFormerPP", CodeFormerPlusPlusRestorer, "RestoraX CodeFormer++"),
    ("GFPGAN", GFPGANRestorer, "RestoraX GFPGAN"),
    ("DicFace", DicFaceRestorer, "RestoraX DicFace"),
]

NODE_CLASS_MAPPINGS = {
    f"RestoraX_{key}": make_restorer_node(cls, _CATEGORY) for key, cls, _ in _RESTORERS
}
NODE_DISPLAY_NAME_MAPPINGS = {f"RestoraX_{key}": label for key, _, label in _RESTORERS}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_comfyui_node_files.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add comfyui_nodes/super_resolution.py comfyui_nodes/face_restoration.py tests/unit/test_comfyui_node_files.py
git commit -m "feat(comfyui): add super-resolution and face-restoration node files"
```

---

## Task 6: Colorization, frame-interpolation, deinterlacing node files

**Files:**

- Create: `comfyui_nodes/colorization.py`
- Create: `comfyui_nodes/frame_interpolation.py`
- Create: `comfyui_nodes/deinterlacing.py`
- Modify: `tests/unit/test_comfyui_node_files.py`

**Interfaces:**

- Consumes: `make_restorer_node`; `DDColorRestorer` (`restorax/restorers/colorization/ddcolor.py`); `RIFERestorer` (`restorax/restorers/frame_interpolation/rife.py`); `AIDeinterlaceRestorer` (`restorax/restorers/deinterlacing/ai_deinterlace.py`), `YadifDeinterlaceRestorer` (`restorax/restorers/deinterlacing/yadif_deinterlace.py`).
- Produces: same `NODE_CLASS_MAPPINGS`/`NODE_DISPLAY_NAME_MAPPINGS` convention as Task 5.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_comfyui_node_files.py
def test_colorization_module_exports_one_node():
    from comfyui_nodes import colorization as col
    assert len(col.NODE_CLASS_MAPPINGS) == 1
    assert next(iter(col.NODE_CLASS_MAPPINGS.values())).CATEGORY == "RestoraX/Colorization"


def test_frame_interpolation_module_exports_one_node():
    from comfyui_nodes import frame_interpolation as fi
    assert len(fi.NODE_CLASS_MAPPINGS) == 1
    assert next(iter(fi.NODE_CLASS_MAPPINGS.values())).CATEGORY == "RestoraX/Frame Interpolation"


def test_deinterlacing_module_exports_two_nodes():
    from comfyui_nodes import deinterlacing as dei
    assert len(dei.NODE_CLASS_MAPPINGS) == 2
    for node_cls in dei.NODE_CLASS_MAPPINGS.values():
        assert node_cls.CATEGORY == "RestoraX/Deinterlacing"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_comfyui_node_files.py -k "colorization or interpolation or deinterlacing" -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `comfyui_nodes/colorization.py`**

```python
from restorax.restorers.colorization.ddcolor import DDColorRestorer

from ._base import make_restorer_node

NODE_CLASS_MAPPINGS = {"RestoraX_DDColor": make_restorer_node(DDColorRestorer, "Colorization")}
NODE_DISPLAY_NAME_MAPPINGS = {"RestoraX_DDColor": "RestoraX DDColor"}
```

- [ ] **Step 4: Implement `comfyui_nodes/frame_interpolation.py`**

```python
from restorax.restorers.frame_interpolation.rife import RIFERestorer

from ._base import make_restorer_node

NODE_CLASS_MAPPINGS = {"RestoraX_RIFE": make_restorer_node(RIFERestorer, "Frame Interpolation")}
NODE_DISPLAY_NAME_MAPPINGS = {"RestoraX_RIFE": "RestoraX RIFE"}
```

- [ ] **Step 5: Implement `comfyui_nodes/deinterlacing.py`**

```python
from restorax.restorers.deinterlacing.ai_deinterlace import AIDeinterlaceRestorer
from restorax.restorers.deinterlacing.yadif_deinterlace import YadifDeinterlaceRestorer

from ._base import make_restorer_node

_CATEGORY = "Deinterlacing"
NODE_CLASS_MAPPINGS = {
    "RestoraX_AIDeinterlace": make_restorer_node(AIDeinterlaceRestorer, _CATEGORY),
    "RestoraX_Yadif": make_restorer_node(YadifDeinterlaceRestorer, _CATEGORY),
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RestoraX_AIDeinterlace": "RestoraX AI Deinterlace",
    "RestoraX_Yadif": "RestoraX YADIF",
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_comfyui_node_files.py -v`
Expected: 5 passed

- [ ] **Step 7: Commit**

```bash
git add comfyui_nodes/colorization.py comfyui_nodes/frame_interpolation.py comfyui_nodes/deinterlacing.py tests/unit/test_comfyui_node_files.py
git commit -m "feat(comfyui): add colorization, frame-interpolation, and deinterlacing node files"
```

---

## Task 7: Artifact-removal, HDR, stabilization node files

**Files:**

- Create: `comfyui_nodes/artifact_removal.py`
- Create: `comfyui_nodes/hdr.py`
- Create: `comfyui_nodes/stabilization.py`
- Modify: `tests/unit/test_comfyui_node_files.py`

**Interfaces:**

- Consumes: `make_restorer_node`; `ScratchRemovalRestorer` (`restorax/restorers/artifact_removal/scratch_removal.py`); `HDRTVDMRestorer` (`restorax/restorers/hdr/hdrtvdm.py`); `VideoStabilizationRestorer` (`restorax/restorers/stabilization/deep_flow_stab.py`), `GaVSRestorer` (`restorax/restorers/stabilization/gavs.py`).
- Produces: same convention as Tasks 5-6.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_comfyui_node_files.py
def test_artifact_removal_module_exports_one_node():
    from comfyui_nodes import artifact_removal as ar
    assert len(ar.NODE_CLASS_MAPPINGS) == 1
    assert next(iter(ar.NODE_CLASS_MAPPINGS.values())).CATEGORY == "RestoraX/Artifact Removal"


def test_hdr_module_exports_one_node():
    from comfyui_nodes import hdr
    assert len(hdr.NODE_CLASS_MAPPINGS) == 1
    assert next(iter(hdr.NODE_CLASS_MAPPINGS.values())).CATEGORY == "RestoraX/HDR Conversion"


def test_stabilization_module_exports_two_nodes():
    from comfyui_nodes import stabilization as stab
    assert len(stab.NODE_CLASS_MAPPINGS) == 2
    for node_cls in stab.NODE_CLASS_MAPPINGS.values():
        assert node_cls.CATEGORY == "RestoraX/Stabilization"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_comfyui_node_files.py -k "artifact_removal or hdr or stabilization" -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `comfyui_nodes/artifact_removal.py`**

```python
from restorax.restorers.artifact_removal.scratch_removal import ScratchRemovalRestorer

from ._base import make_restorer_node

NODE_CLASS_MAPPINGS = {
    "RestoraX_ScratchRemoval": make_restorer_node(ScratchRemovalRestorer, "Artifact Removal"),
}
NODE_DISPLAY_NAME_MAPPINGS = {"RestoraX_ScratchRemoval": "RestoraX Scratch Removal"}
```

- [ ] **Step 4: Implement `comfyui_nodes/hdr.py`**

```python
from restorax.restorers.hdr.hdrtvdm import HDRTVDMRestorer

from ._base import make_restorer_node

NODE_CLASS_MAPPINGS = {"RestoraX_HDRTVDM": make_restorer_node(HDRTVDMRestorer, "HDR Conversion")}
NODE_DISPLAY_NAME_MAPPINGS = {"RestoraX_HDRTVDM": "RestoraX HDRTVDM"}
```

- [ ] **Step 5: Implement `comfyui_nodes/stabilization.py`**

```python
from restorax.restorers.stabilization.deep_flow_stab import VideoStabilizationRestorer
from restorax.restorers.stabilization.gavs import GaVSRestorer

from ._base import make_restorer_node

_CATEGORY = "Stabilization"
NODE_CLASS_MAPPINGS = {
    "RestoraX_DeepFlowStab": make_restorer_node(VideoStabilizationRestorer, _CATEGORY),
    "RestoraX_GaVS": make_restorer_node(GaVSRestorer, _CATEGORY),
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RestoraX_DeepFlowStab": "RestoraX Optical-Flow Stabilization",
    "RestoraX_GaVS": "RestoraX GaVS",
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_comfyui_node_files.py -v`
Expected: 8 passed (cumulative across Tasks 5-7)

- [ ] **Step 7: Commit**

```bash
git add comfyui_nodes/artifact_removal.py comfyui_nodes/hdr.py comfyui_nodes/stabilization.py tests/unit/test_comfyui_node_files.py
git commit -m "feat(comfyui): add artifact-removal, HDR, and stabilization node files"
```

---

## Task 8: Package aggregation (`__init__.py`) + `requirements.txt`

**Files:**

- Modify: `comfyui_nodes/__init__.py`
- Create: `comfyui_nodes/requirements.txt`
- Test: `tests/unit/test_comfyui_node_files.py`

**Interfaces:**

- Consumes: `NODE_CLASS_MAPPINGS`/`NODE_DISPLAY_NAME_MAPPINGS` from all 8 category modules (Tasks 5-7).
- Produces: `comfyui_nodes/__init__.py` module-level `NODE_CLASS_MAPPINGS` (22 entries total) and `NODE_DISPLAY_NAME_MAPPINGS` — this is the exact dict pair ComfyUI's custom-node loader imports from a pack's `__init__.py`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_comfyui_node_files.py
def test_package_init_aggregates_all_22_video_restorer_nodes():
    import comfyui_nodes
    assert len(comfyui_nodes.NODE_CLASS_MAPPINGS) == 22
    assert set(comfyui_nodes.NODE_CLASS_MAPPINGS) == set(comfyui_nodes.NODE_DISPLAY_NAME_MAPPINGS)
    assert all(key.startswith("RestoraX_") for key in comfyui_nodes.NODE_CLASS_MAPPINGS)


def test_package_init_has_no_duplicate_node_keys_across_categories():
    import comfyui_nodes
    from comfyui_nodes import (
        artifact_removal, colorization, deinterlacing, face_restoration,
        frame_interpolation, hdr, stabilization, super_resolution,
    )
    modules = [
        artifact_removal, colorization, deinterlacing, face_restoration,
        frame_interpolation, hdr, stabilization, super_resolution,
    ]
    total_individual = sum(len(m.NODE_CLASS_MAPPINGS) for m in modules)
    assert total_individual == len(comfyui_nodes.NODE_CLASS_MAPPINGS) == 22
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_comfyui_node_files.py -k "package_init" -v`
Expected: FAIL with `AttributeError: module 'comfyui_nodes' has no attribute 'NODE_CLASS_MAPPINGS'`

- [ ] **Step 3: Implement `comfyui_nodes/__init__.py`**

```python
"""
RestoraX ComfyUI custom-node pack.

ComfyUI's custom-node loader imports NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS
from this file. Audio restorers (Demucs/VoiceFixer/RNNoise) are intentionally excluded —
they use AudioRestorerParams/process(), not RestorerParams/process_frame().
"""
from . import (
    artifact_removal,
    colorization,
    deinterlacing,
    face_restoration,
    frame_interpolation,
    hdr,
    stabilization,
    super_resolution,
)

_MODULES = [
    artifact_removal, colorization, deinterlacing, face_restoration,
    frame_interpolation, hdr, stabilization, super_resolution,
]

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}
for _module in _MODULES:
    NODE_CLASS_MAPPINGS.update(_module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_module.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

- [ ] **Step 4: Create `comfyui_nodes/requirements.txt`**

```text
restorax>=0.1.0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_comfyui_node_files.py -v`
Expected: 10 passed (cumulative across Tasks 5-8)

- [ ] **Step 6: Commit**

```bash
git add comfyui_nodes/__init__.py comfyui_nodes/requirements.txt tests/unit/test_comfyui_node_files.py
git commit -m "feat(comfyui): aggregate all category node mappings in package __init__"
```

---

## Task 9: Dev ComfyUI install + `requires_comfyui` integration test

**Files:**

- Modify: `pyproject.toml`
- Modify: `tests/conftest.py`
- Create: `tests/integration/test_comfyui_load.py`

**Interfaces:**

- Consumes: existing `pytest_collection_modifyitems` skip-marker pattern in `tests/conftest.py` (currently handles `requires_weights`/`requires_assets`).
- Produces: a `requires_comfyui` pytest marker that skips unless a real ComfyUI checkout is present at `$COMFYUI_PATH` (env var, no default — this test never auto-installs ComfyUI).

- [ ] **Step 1: Add the `comfyui` dev extra and the marker to `pyproject.toml`**

Find the `[project.optional-dependencies]` table's `dev` entry and the `[tool.pytest.ini_options] markers` list (both confirmed present at lines ~58-98 and ~150-152). Add:

```toml
[project.optional-dependencies]
dev = [
    # ...existing dev deps unchanged...
    "comfy-cli>=1.2.0",
]
```

```toml
[tool.pytest.ini_options]
markers = [
    # ...existing markers unchanged...
    "requires_comfyui: needs a real ComfyUI checkout at $COMFYUI_PATH",
]
```

- [ ] **Step 2: Read the current skip-marker block in `tests/conftest.py` (lines ~130-152) and add the new marker alongside it**

```python
# in tests/conftest.py, inside pytest_collection_modifyitems, alongside the
# existing requires_weights/requires_assets handling:
import os

skip_comfyui = pytest.mark.skip(reason="needs $COMFYUI_PATH set to a real ComfyUI checkout")
for item in items:
    if "requires_comfyui" in item.keywords and not os.environ.get("COMFYUI_PATH"):
        item.add_marker(skip_comfyui)
```

- [ ] **Step 3: Write the integration test**

```python
# tests/integration/test_comfyui_load.py
"""
Verifies the comfyui_nodes pack loads cleanly inside a real ComfyUI checkout.

Setup (one-time, manual — not run by CI):
    pip install comfy-cli
    comfy --skip-prompt install --fast-deps  # clones ComfyUI to ./ComfyUI by default
    export COMFYUI_PATH=$(pwd)/ComfyUI
    ln -s $(pwd)/comfyui_nodes $COMFYUI_PATH/custom_nodes/restorax
    pytest tests/integration/test_comfyui_load.py -v
"""
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.requires_comfyui


def test_comfyui_nodes_pack_importable_from_custom_nodes_dir():
    comfyui_path = os.environ["COMFYUI_PATH"]
    custom_nodes_link = os.path.join(comfyui_path, "custom_nodes", "restorax")
    assert os.path.islink(custom_nodes_link) or os.path.isdir(custom_nodes_link), (
        f"Expected comfyui_nodes/ symlinked into {custom_nodes_link} — see module docstring for setup."
    )
    result = subprocess.run(
        [sys.executable, "-c", "import comfyui_nodes; print(len(comfyui_nodes.NODE_CLASS_MAPPINGS))"],
        cwd=custom_nodes_link,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "22"
```

- [ ] **Step 4: Run the unit suite to confirm the marker doesn't break collection when `$COMFYUI_PATH` is unset**

Run: `pytest tests/integration/test_comfyui_load.py -v`
Expected: 1 skipped, reason "needs $COMFYUI_PATH set to a real ComfyUI checkout"

- [ ] **Step 5: Run the full unit suite to confirm no regressions**

Run: `pytest tests/unit -v`
Expected: all previously-passing tests still pass, plus the new `test_comfyui_nodes.py` / `test_comfyui_node_files.py` tests (16 + 10 = 26 new) green.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tests/conftest.py tests/integration/test_comfyui_load.py
git commit -m "feat(comfyui): add comfy-cli dev extra and requires_comfyui integration marker"
```

---

## Self-Review

**1. Spec coverage** — `PLAN.md` §4.2 requirements checked against tasks:

- "Tensor↔numpy conversion shared in `_base.py`" → Task 1.
- "Temporal restorers accept batched `IMAGE` (B>1 = sequence)" → handled implicitly: `comfy_image_to_frames` already returns one numpy frame per batch element, and `make_restorer_node`'s `restore()` loops `process_frame` over every frame in the batch (Task 4) — a batch of >1 is exactly "a sequence" from the node's point of view. Not calling `process_sequence` is a deliberate scope cut for this skeleton (noted below as an Open Gap, not silently dropped).
- "Audio nodes use ComfyUI `AUDIO` type" → explicitly out of scope per user's "full skeleton across all categories" answer being about *video* categories; audio's structural incompatibility (`process()` vs `process_frame()`) is called out in Global Constraints and the Architecture section, consistent with the already-documented gap in `PLAN.md` §5.
- "Lazy weight download on first node execution" → already true for free: `ModelRegistry.get()` calls `restorer.load(device)` on first access, and every restorer's `load()` already triggers `huggingface_hub` auto-download (Track 3.3) — no new code needed, but flagged here so it's not mistaken for an omission.
- "Phases: base conversion layer → SR nodes → face/color/interpolation nodes → stabilization/deinterlace/HDR/artifact nodes → audio nodes → manifest + PR" → Tasks 1-4 (base), Task 5 (SR + face), Task 6 (color/interpolation/deinterlace), Task 7 (stabilization/HDR/artifact), audio and manifest/PR explicitly deferred (see below).

**2. Placeholder scan** — every step has real, complete code; no "TBD"/"add error handling" patterns found.

**3. Type consistency** — `make_restorer_node(restorer_cls, category_label)` signature is identical across Task 4's definition and every call site in Tasks 5-7; `param_spec_to_input` / `get_registry` / `get_device` / `comfy_image_to_frames` / `frames_to_comfy_image` names match between definition (Tasks 1-3) and usage (Task 4).

**Deferred / explicitly out of scope (not silent gaps):**

- Audio nodes (Demucs/VoiceFixer/RNNoise) — structural mismatch, needs its own design pass on a `process()`-based factory variant.
- `comfyui_manifest.json` + ComfyUI-Manager community-list PR — needs verification against ComfyUI-Manager's current manifest schema before writing it; guessing the schema would violate CLAUDE.md §2.
- Calling `process_sequence` instead of per-frame `process_frame` for temporal restorers (BasicVSR++, RIFE) inside a ComfyUI batch — works correctly today (per-frame fallback is the documented default behavior on `BaseRestorer`), but loses the temporal-context benefit those restorers' overridden `process_sequence` provides. Worth a follow-up task once the skeleton is verified end-to-end in a real ComfyUI workflow.
- `comfy-cli`'s exact CLI flags (`--skip-prompt`, `--fast-deps`) in Task 9's docstring are written from best available knowledge but **must be verified against `comfy --help` at execution time** — flagged per CLAUDE.md §2 rather than asserted as fact.

---

**Plan complete and saved to `docs/superpowers/plans/2026-06-21-comfyui-node-pack.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
