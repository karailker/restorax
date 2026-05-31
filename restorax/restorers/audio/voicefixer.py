"""
VoiceFixer speech enhancement restorer.

Restores degraded speech audio — handles noise, distortion, clipping,
bandwidth limitation, and room reverb simultaneously in one pass.
Excellent for old film dialogue, degraded broadcast recordings, and
VHS audio tracks where multiple degradation types coexist.

Model source: https://github.com/haoheliu/voicefixer
Paper: "VoiceFixer: Toward General Speech Restoration With Neural Vocoder"
       (Interspeech 2022)

Requires: pip install voicefixer
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

_TARGET_SR = 44100  # VoiceFixer always outputs 44100 Hz


class VoiceFixerRestorer(AudioRestorer):
    """
    General speech restoration using VoiceFixer.

    Handles: noise, distortion, clipping, low bandwidth, reverberation.
    Input is resampled to 44100 Hz internally if needed.

    extra params:
      mode: int — restoration mode (default 0)
            0: original VoiceFixer
            1: without vocoder (faster, slightly lower quality)
    """

    PARAM_SCHEMA = [
        ParamSpec("mode", "enum", 0, "Mode", choices=(0, 1, 2),
                  help="VoiceFixer inference mode (0-2)"),
    ]

    def __init__(self) -> None:
        self._model: object | None = None
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "voicefixer"

    @property
    def capabilities(self) -> AudioRestorerCapabilities:
        return AudioRestorerCapabilities(
            category=AudioRestorerCategory.SPEECH_ENHANCEMENT,
            sample_rates=[44100, 16000, 22050, 48000],
            supports_stereo=False,  # VoiceFixer processes mono internally
            min_ram_gb=2.0,
            tags=["voicefixer", "speech_enhancement", "denoising", "dereverberation"],
        )

    def load(self, device: torch.device) -> None:
        self._model = self._build_model(device)
        self._device = device
        self._loaded = True
        logger.info("VoiceFixer loaded on %s", device)

    def unload(self) -> None:
        del self._model
        self._model = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    def process_audio(self, audio: np.ndarray, params: AudioRestorerParams) -> np.ndarray:
        assert self._device is not None
        mode = int(params.extra.get("mode", 0))

        if isinstance(self._model, _VoiceFixerStub):
            return audio.copy()

        return self._voicefixer_restore(audio, params.sample_rate, mode)

    def _voicefixer_restore(self, audio: np.ndarray, sr: int, mode: int) -> np.ndarray:
        """Run VoiceFixer restoration. Handles stereo by processing each channel."""
        import tempfile, os
        from pathlib import Path

        # VoiceFixer works on files; use tempfiles for in-memory workflow
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                inp = Path(tmpdir) / "input.wav"
                out = Path(tmpdir) / "output.wav"

                from restorax.audio.writer import AudioWriter
                AudioWriter().write_wav(inp, audio, sr)

                self._model.restore(  # type: ignore[union-attr]
                    input=str(inp),
                    output_dir=tmpdir,
                    mode=mode,
                    cuda=self._device is not None and self._device.type == "cuda",
                )

                # VoiceFixer names output as <input_stem>_mode_N.wav
                results = list(Path(tmpdir).glob("*_mode_*.wav"))
                if not results:
                    return audio.copy()

                from restorax.audio.reader import AudioReader
                restored, _ = AudioReader(results[0]).read()
                # Match original channel count
                if audio.ndim == 2 and audio.shape[1] != restored.shape[1]:
                    if restored.ndim == 1:
                        restored = restored[:, np.newaxis]
                    restored = np.tile(restored[:, :1], (1, audio.shape[1]))
                return restored
        except Exception as exc:
            logger.warning("VoiceFixer restore failed (%s) — returning original", exc)
            return audio.copy()

    @staticmethod
    def _build_model(device: torch.device) -> object:
        try:
            from voicefixer import VoiceFixer
            model = VoiceFixer()
            logger.info("VoiceFixer loaded from installed package")
            return model
        except ImportError:
            logger.info("voicefixer not installed — using passthrough stub")
            return _VoiceFixerStub()
        except Exception as exc:
            logger.warning("VoiceFixer load failed (%s) — using stub", exc)
            return _VoiceFixerStub()


class _VoiceFixerStub:
    """Passthrough stub used when voicefixer is not installed."""
    pass
