"""
Demucs audio source separation / denoising restorer.

Demucs (Hybrid Transformer Demucs — HTDemucs) separates audio into stems
(vocals, drums, bass, other) and recombines the desired stems. For film
restoration, keeping vocals + other (removing isolated noise/drum bleed)
gives cleaner dialogue while preserving background ambience.

Model source: https://github.com/facebookresearch/demucs
Paper: "Hybrid Transformers for Music Source Separation" (ICASSP 2023)

Requires: pip install demucs
Weights: downloaded via demucs.pretrained.get_model() on first use
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from restorax.audio.restorer import (
    AudioRestorer,
    AudioRestorerCapabilities,
    AudioRestorerCategory,
    AudioRestorerParams,
)
from restorax.core.restorer import ParamSpec

logger = logging.getLogger(__name__)

_DEFAULT_STEMS = ("vocals", "other")


class DemucsRestorer(AudioRestorer):
    """
    Source separation using Facebook HTDemucs.

    Separates audio into up to 4 stems and recombines selected ones.
    Default: keeps vocals + other (removes noise from isolated drums/bass).

    extra params:
      stems: list of stems to keep, e.g. ["vocals", "other"], ["vocals"],
             or None to return the full mix (identity for testing).
             Supported: "vocals", "drums", "bass", "other"
    """

    PARAM_SCHEMA = [
        ParamSpec("stems", "multiselect", list(_DEFAULT_STEMS), "Stems to keep",
                  choices=("vocals", "drums", "bass", "other")),
    ]

    def __init__(self) -> None:
        self._model: object | None = None
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "demucs_htdemucs"

    @property
    def capabilities(self) -> AudioRestorerCapabilities:
        return AudioRestorerCapabilities(
            category=AudioRestorerCategory.SOURCE_SEPARATION,
            sample_rates=[44100],
            supports_stereo=True,
            min_ram_gb=3.0,
            tags=["demucs", "source_separation", "vocals", "noise_removal", "speech"],
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self, device: torch.device) -> None:
        self._model = self._build_model(device)
        self._device = device
        self._loaded = True
        logger.info("Demucs HTDemucs loaded on %s", device)

    def unload(self) -> None:
        del self._model
        self._model = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Inference ─────────────────────────────────────────────────────────────

    def process_audio(
        self,
        audio: np.ndarray,          # (num_samples, num_channels) float32
        params: AudioRestorerParams,
    ) -> np.ndarray:
        assert self._device is not None

        stems_to_keep = list(params.extra.get("stems", list(_DEFAULT_STEMS)))

        if isinstance(self._model, _DemucsStub):
            return audio.copy()

        return self._demucs_separate(audio, stems_to_keep)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _demucs_separate(self, audio: np.ndarray, stems_to_keep: list[str]) -> np.ndarray:
        """Run Demucs HTDemucs separation and recombine selected stems."""
        try:
            from demucs.apply import apply_model

            # (num_samples, channels) → (channels, num_samples) → (1, channels, samples)
            tensor = torch.from_numpy(audio.T).float().unsqueeze(0).to(self._device)

            with torch.inference_mode():
                # apply_model returns (batch, stems, channels, samples)
                separated = apply_model(self._model, tensor, device=self._device, progress=False)

            stem_names: list[str] = getattr(self._model, "sources", ["drums", "bass", "other", "vocals"])
            result = torch.zeros_like(tensor.squeeze(0))  # (channels, samples)
            for i, stem_name in enumerate(stem_names):
                if stem_name in stems_to_keep:
                    result += separated[0, i]  # (channels, samples)

            return result.cpu().numpy().T.astype(np.float32)
        except Exception as exc:
            logger.warning("Demucs separation failed (%s) — returning original audio", exc)
            return audio.copy()

    @staticmethod
    def _build_model(device: torch.device) -> object:
        try:
            from demucs.pretrained import get_model
            model = get_model("htdemucs")
            model = model.to(device)
            model.eval()
            logger.info("Demucs HTDemucs weights loaded from pretrained")
            return model
        except ImportError:
            logger.info("demucs not installed — using passthrough stub")
            return _DemucsStub()
        except Exception as exc:
            logger.warning("Demucs load failed (%s) — using stub", exc)
            return _DemucsStub()


class _DemucsStub:
    """Identity stub: returns input unchanged. Used when demucs is not installed."""
    pass
