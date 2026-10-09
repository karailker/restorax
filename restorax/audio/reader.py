"""
AudioReader — extract audio from a video container as a float32 numpy array.

Uses PyAV for direct container access (same as VideoReader) so no FFmpeg
subprocess is spawned per file. The extracted array is normalized to [-1.0, 1.0].
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np

from restorax.core.exceptions import AudioReadError


class AudioReader:
    """
    Extract all audio from a video (or audio-only) container.

    Returns the full audio as a single (num_samples, num_channels) float32
    array normalized to [-1.0, 1.0], plus the sample rate.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise AudioReadError(f"File not found: {self._path}")

    def read(self) -> tuple[np.ndarray, int]:
        """
        Decode the first audio stream and return (audio_array, sample_rate).

        Raises AudioReadError if no audio stream is present.
        """
        try:
            container = av.open(str(self._path))
        except Exception as exc:
            raise AudioReadError(f"Cannot open {self._path}: {exc}") from exc

        if not container.streams.audio:
            container.close()
            raise AudioReadError(f"No audio stream in {self._path}")

        audio_stream = container.streams.audio[0]
        sample_rate = audio_stream.rate

        chunks: list[np.ndarray] = []
        try:
            for frame in container.decode(audio_stream):
                # PyAV returns frames as (channels, samples) in various formats
                arr = frame.to_ndarray()  # shape: (channels, samples) or (samples,)
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]  # ensure (channels, samples)
                chunks.append(arr)
        except Exception as exc:
            raise AudioReadError(f"Audio decode error in {self._path}: {exc}") from exc
        finally:
            container.close()

        if not chunks:
            raise AudioReadError(f"No audio frames decoded from {self._path}")

        # Concatenate along the samples axis → (channels, total_samples)
        audio_concat = np.concatenate(chunks, axis=1)
        # Transpose to (total_samples, channels)
        audio_transposed = audio_concat.T  # (total_samples, channels)

        # Normalize to float32 [-1.0, 1.0]
        return _normalize_to_float32(audio_transposed), sample_rate


def _normalize_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert any integer or float dtype to float32 normalized to [-1.0, 1.0]."""
    if audio.dtype == np.float32:
        return np.clip(audio, -1.0, 1.0)
    if audio.dtype == np.float64:
        return np.clip(audio.astype(np.float32), -1.0, 1.0)
    if audio.dtype == np.int16:
        return (audio.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    if audio.dtype == np.int32:
        return (audio.astype(np.float32) / 2147483648.0).clip(-1.0, 1.0)
    if audio.dtype == np.uint8:
        return ((audio.astype(np.float32) - 128.0) / 128.0).clip(-1.0, 1.0)
    # Fallback: try float cast
    return audio.astype(np.float32).clip(-1.0, 1.0)
