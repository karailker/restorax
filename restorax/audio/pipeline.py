"""
Audio pipeline: AudioStage, AudioPipeline, AudioPipelineRunner, AudioModelRegistry.

Mirrors the video pipeline pattern but operates on full audio clips
(no chunking needed — source separation models require global context).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from restorax.audio.restorer import AudioRestorer, AudioRestorerParams
from restorax.core.exceptions import RestorerNotFoundError

logger = logging.getLogger(__name__)


# ── Pipeline ──────────────────────────────────────────────────────────────────


@dataclass
class AudioStage:
    restorer: AudioRestorer
    params: AudioRestorerParams
    enabled: bool = True


@dataclass
class AudioPipeline:
    name: str
    stages: list[AudioStage]


class AudioPipelineRunner:
    """Apply AudioStages sequentially on the full audio clip."""

    def run(
        self,
        pipeline: AudioPipeline,
        audio: np.ndarray,  # (num_samples, num_channels) float32
        sample_rate: int,
    ) -> np.ndarray:
        current = audio
        for stage in pipeline.stages:
            if not stage.enabled:
                continue
            stage.params.sample_rate = sample_rate
            current = stage.restorer.process_audio(current, stage.params)
        return current


# ── Registry ──────────────────────────────────────────────────────────────────


class AudioModelRegistry:
    """
    LRU cache for AudioRestorer instances.

    Same eviction strategy as ModelRegistry but typed to AudioRestorer.
    Kept separate so audio and video model budgets don't interfere.
    """

    def __init__(self, max_loaded: int = 2) -> None:
        self._catalog: dict[str, type[AudioRestorer]] = {}
        self._loaded: OrderedDict[str, AudioRestorer] = OrderedDict()
        self._max_loaded = max_loaded

    def register(self, cls: type[AudioRestorer]) -> None:
        instance = object.__new__(cls)
        name = cls.name.fget(instance)  # type: ignore[attr-defined]
        self._catalog[name] = cls
        logger.debug("Registered audio restorer: %s", name)

    def get(self, name: str, device: torch.device) -> AudioRestorer:
        if name in self._loaded:
            self._loaded.move_to_end(name)
            return self._loaded[name]
        if name not in self._catalog:
            raise RestorerNotFoundError(
                f"Audio restorer '{name}' not registered. Available: {sorted(self._catalog)}"
            )
        if len(self._loaded) >= self._max_loaded:
            evicted_name, evicted = self._loaded.popitem(last=False)
            evicted.unload()
            logger.info("Evicted audio restorer '%s' (LRU)", evicted_name)
        instance = self._catalog[name]()
        instance.load(device)
        self._loaded[name] = instance
        return instance

    def list_available(self) -> list[str]:
        return sorted(self._catalog)

    def unload_all(self) -> None:
        for r in self._loaded.values():
            r.unload()
        self._loaded.clear()


# ── YAML loader ───────────────────────────────────────────────────────────────


def load_audio_pipeline_from_config(
    config: dict[str, Any],
    registry: AudioModelRegistry,
    device: torch.device,
) -> AudioPipeline | None:
    """
    Parse the 'audio_stages' key from an already-loaded preset config dict.
    Returns None when the key is absent (audio restoration disabled).
    """
    stages_cfg = config.get("audio_stages", [])
    if not stages_cfg:
        return None

    stages: list[AudioStage] = []
    for cfg in stages_cfg:
        restorer = registry.get(cfg["restorer"], device)
        params = AudioRestorerParams(
            sample_rate=cfg.get("sample_rate", 44100),
            extra=cfg.get("extra", {}),
        )
        stages.append(
            AudioStage(
                restorer=restorer,
                params=params,
                enabled=cfg.get("enabled", True),
            )
        )

    return AudioPipeline(name=config.get("name", "audio"), stages=stages)
