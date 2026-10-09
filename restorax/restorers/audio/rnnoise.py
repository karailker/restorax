"""
RNNoise real-time noise suppression restorer.

RNNoise is a noise suppression library using a recurrent neural network
trained on a large dataset of noise types. It operates at 48 kHz, mono,
in 480-sample frames — extremely fast (~10× realtime on CPU).

For audio not at 48 kHz: the restorer resamples to 48 kHz before processing
and back to the original sample rate after, using scipy.signal.resample_poly.

Model source: https://jmvalin.ca/demo/rnnoise/
Python binding: pip install rnnoise-python (or pip install noisereduce as fallback)
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

logger = logging.getLogger(__name__)

_RNNOISE_SR = 48000
_FRAME_SIZE = 480


class RNNoiseRestorer(AudioRestorer):
    """
    Real-time noise suppression using RNNoise.

    Processes audio at 48 kHz mono in 480-sample frames. Input is
    resampled if necessary. Stereo inputs are mixed to mono before
    processing and duplicated back to stereo on output.
    """

    def __init__(self) -> None:
        self._denoiser: object | None = None
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "rnnoise"

    @property
    def capabilities(self) -> AudioRestorerCapabilities:
        return AudioRestorerCapabilities(
            category=AudioRestorerCategory.NOISE_SUPPRESSION,
            sample_rates=[48000, 44100, 16000, 22050],
            supports_stereo=True,  # we handle the mono conversion internally
            min_ram_gb=0.1,
            tags=["rnnoise", "noise_suppression", "realtime", "lightweight"],
        )

    def load(self, device: torch.device) -> None:
        self._denoiser = self._build_denoiser()
        self._device = device
        self._loaded = True
        logger.info("RNNoise loaded (CPU)")

    def unload(self) -> None:
        del self._denoiser
        self._denoiser = None
        self._loaded = False

    def process_audio(self, audio: np.ndarray, params: AudioRestorerParams) -> np.ndarray:
        sr = params.sample_rate

        if hasattr(self._denoiser, "process_frame") or hasattr(self._denoiser, "reduce_noise"):
            return self._denoise(audio, sr)

        # Stub: return input unchanged
        return audio.copy()

    def _denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Resample → denoise → resample back."""
        original_sr = sr
        mono = audio.mean(axis=1) if audio.ndim == 2 and audio.shape[1] > 1 else audio.squeeze()

        # Resample to 48 kHz if needed
        if sr != _RNNOISE_SR:
            mono = _resample(mono, sr, _RNNOISE_SR)

        # Try rnnoise-python first, then noisereduce fallback
        try:
            denoised = self._run_rnnoise(mono)
        except Exception as exc:
            logger.warning("RNNoise processing failed (%s) — returning original", exc)
            denoised = mono

        # Resample back
        if original_sr != _RNNOISE_SR:
            denoised = _resample(denoised, _RNNOISE_SR, original_sr)

        # Restore channel count
        if audio.ndim == 2 and audio.shape[1] > 1:
            return np.stack([denoised] * audio.shape[1], axis=1)
        return denoised[:, np.newaxis] if audio.ndim == 2 else denoised

    def _run_rnnoise(self, mono: np.ndarray) -> np.ndarray:
        """Process 480-sample frames with rnnoise or noisereduce."""
        # Attempt rnnoise-python binding
        if hasattr(self._denoiser, "process_frame"):
            # Pad to multiple of frame size
            pad_len = (-len(mono)) % _FRAME_SIZE
            padded = np.pad(mono, (0, pad_len))
            pcm = (np.clip(padded, -1.0, 1.0) * 32767).astype(np.int16)
            out_frames = []
            for i in range(0, len(pcm), _FRAME_SIZE):
                frame = pcm[i : i + _FRAME_SIZE].tobytes()
                processed = self._denoiser.process_frame(frame)  # type: ignore[union-attr]
                out_frames.append(np.frombuffer(processed, dtype=np.int16))
            result_pcm = np.concatenate(out_frames)[: len(mono)]
            return result_pcm.astype(np.float32) / 32768.0

        # noisereduce fallback
        if hasattr(self._denoiser, "reduce_noise"):
            return self._denoiser.reduce_noise(  # type: ignore[union-attr]
                y=mono.astype(np.float32), sr=_RNNOISE_SR
            )

        return mono

    @staticmethod
    def _build_denoiser() -> object:
        # Try rnnoise-python binding
        try:
            import rnnoise

            dn = rnnoise.RNNoise()
            logger.info("RNNoise loaded via rnnoise-python")
            return dn
        except ImportError:
            pass

        # Try noisereduce as fallback
        try:
            import noisereduce as nr

            logger.info("RNNoise using noisereduce fallback")
            return _NoisereduceAdapter(nr)
        except ImportError:
            pass

        logger.info("No noise suppression library found — using passthrough stub")
        return _RNNoiseStub()


def _resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(from_sr, to_sr)
        return resample_poly(audio, to_sr // g, from_sr // g).astype(np.float32)
    except ImportError:
        # Naive linear interpolation if scipy unavailable
        n_samples = int(len(audio) * to_sr / from_sr)
        return np.interp(
            np.linspace(0, len(audio) - 1, n_samples),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)


class _NoisereduceAdapter:
    def __init__(self, nr_module: object) -> None:
        self._nr = nr_module

    def reduce_noise(self, y: np.ndarray, sr: int) -> np.ndarray:
        return self._nr.reduce_noise(y=y, sr=sr)  # type: ignore[attr-defined]


class _RNNoiseStub:
    """Passthrough stub."""

    pass
