from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np
import torch


class RestorerCategory(str, Enum):
    SUPER_RESOLUTION = "super_resolution"
    COLORIZATION = "colorization"
    FACE_RESTORATION = "face_restoration"
    FRAME_INTERPOLATION = "frame_interpolation"
    DEINTERLACING = "deinterlacing"
    ARTIFACT_REMOVAL = "artifact_removal"
    HDR_CONVERSION = "hdr_conversion"
    STABILIZATION = "stabilization"
    AUDIO_RESTORATION = "audio_restoration"  # reserved for future video+audio hybrid restorers


@dataclass
class RestorerCapabilities:
    category: RestorerCategory
    input_color_space: str  # "rgb" | "bgr" | "yuv420"
    output_color_space: str
    requires_temporal: bool = False  # True for BasicVSR++, RIFE, etc.
    max_batch_size: int = 1
    min_vram_gb: float = 4.0
    supports_compile: bool = False  # eligible for torch.compile()
    scale_factor: int = 1      # output/input spatial ratio (e.g. 4 for 4× SR)
    temporal_scale: int = 1    # output/input temporal ratio (2 for RIFE 2×FPS, 1 for all others)
    tags: list[str] = field(default_factory=list)


@dataclass
class RestorerParams:
    scale: int = 1
    tile_size: int = 0  # 0 = no tiling; use for high-res inputs to avoid OOM
    tile_overlap: int = 32
    half_precision: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParamSpec:
    """
    Declares one tunable parameter a restorer actually reads, so UIs can render
    a typed control instead of a raw JSON blob.

    `target` says where the value lives in the serialised params dict:
      - "param": a top-level RestorerParams field (e.g. tile_size)
      - "extra": a key inside RestorerParams.extra (restorer-specific)
    """
    name: str
    kind: Literal["int", "float", "bool", "enum", "multiselect"]
    default: Any
    label: str
    target: Literal["param", "extra"] = "extra"
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[Any, ...] | None = None
    help: str | None = None


# Reusable specs for the common RestorerParams fields, shared by tiling/fp16 restorers.
TILE_SIZE_SPEC = ParamSpec(
    "tile_size", "int", 0, "Tile size", target="param",
    minimum=0, maximum=2048, step=32, help="0 = no tiling; raise for high-res inputs to avoid OOM",
)
TILE_OVERLAP_SPEC = ParamSpec(
    "tile_overlap", "int", 32, "Tile overlap", target="param", minimum=0, maximum=256, step=8,
)
HALF_PRECISION_SPEC = ParamSpec(
    "half_precision", "bool", True, "Half precision (fp16)", target="param",
    help="Faster, lower VRAM; CUDA only",
)


class BaseRestorer(ABC):
    """
    Contract every restorer must satisfy.

    Lifecycle per worker: __init__ → load() → process_* (many calls) → unload()
    """

    # Tunable parameters this restorer actually reads. Empty = no user-facing knobs.
    # Override on subclasses that read params.tile_size / params.extra[...] etc.
    PARAM_SCHEMA: list[ParamSpec] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique slug, e.g. 'real_esrgan_x4plus'. Used as registry key."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> RestorerCapabilities:
        ...

    @abstractmethod
    def load(self, device: torch.device) -> None:
        """Load weights into VRAM/RAM. Called once per worker process lifetime."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release VRAM. Called by ModelRegistry when evicting from LRU cache."""
        ...

    @abstractmethod
    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        """
        Process a single frame.

        Args:
            frame: HxWxC uint8 array in the restorer's input color space.
            params: Per-job parameters.

        Returns:
            Processed frame as HxWxC uint8 in the restorer's output color space.
        """
        ...

    def process_sequence(
        self,
        frames: list[np.ndarray],
        params: RestorerParams,
    ) -> list[np.ndarray]:
        """
        Process a temporal window of frames.

        Override in temporally-aware restorers (BasicVSR++, RIFE, etc.).
        Default: apply process_frame independently to each frame.
        """
        return [self.process_frame(f, params) for f in frames]

    @property
    def is_loaded(self) -> bool:
        return bool(getattr(self, "_loaded", False))
