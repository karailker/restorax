"""
AudioRestorer — abstract base class for all audio restoration models.

Deliberately separate from BaseRestorer (video frames) because:
  - I/O contract: float32 arrays (samples × channels) instead of uint8 HxWxC frames
  - Pipeline: full-clip processing (no chunk overlap needed — source separation
    requires global context), not chunk-based frame iteration
  - Capabilities: sample_rates, supports_stereo (vs. scale_factor, color_space)
  - VRAM budget: audio models share no registry with video models
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

from restorax.core.restorer import ParamSpec


class AudioRestorerCategory(str, Enum):
    SOURCE_SEPARATION = "source_separation"     # Demucs: separate vocals/instruments
    SPEECH_ENHANCEMENT = "speech_enhancement"   # Voicefixer: restore degraded speech
    NOISE_SUPPRESSION = "noise_suppression"     # RNNoise: suppress background noise


@dataclass
class AudioRestorerCapabilities:
    category: AudioRestorerCategory
    sample_rates: list[int]      # supported input sample rates, e.g. [44100, 48000]
    supports_stereo: bool = True
    min_ram_gb: float = 1.0
    tags: list[str] = field(default_factory=list)


@dataclass
class AudioRestorerParams:
    sample_rate: int = 44100
    extra: dict[str, Any] = field(default_factory=dict)


class AudioRestorer(ABC):
    """
    Contract for all audio restoration models.

    Lifecycle: __init__ → load() → process_audio() (many calls) → unload()

    I/O contract:
        - Input:  np.ndarray float32 (num_samples, num_channels), values in [-1.0, 1.0]
        - Output: np.ndarray float32 (num_samples, num_channels), values in [-1.0, 1.0]
        - Shape is preserved: output.shape == input.shape
    """

    # Tunable parameters this restorer reads from params.extra. Empty = no knobs.
    PARAM_SCHEMA: list[ParamSpec] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique registry slug, e.g. 'demucs_htdemucs'."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> AudioRestorerCapabilities:
        ...

    @abstractmethod
    def load(self, device: torch.device) -> None:
        """Load model weights. Called once per worker process lifetime."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release RAM/VRAM. Called by AudioModelRegistry when evicting."""
        ...

    @abstractmethod
    def process_audio(
        self,
        audio: np.ndarray,            # (num_samples, num_channels) float32 [-1,1]
        params: AudioRestorerParams,
    ) -> np.ndarray:                  # same shape and dtype
        """Process the full audio array. Must preserve shape and dtype."""
        ...

    @property
    def is_loaded(self) -> bool:
        return bool(getattr(self, "_loaded", False))
