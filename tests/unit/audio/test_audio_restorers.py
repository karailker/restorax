"""Unit tests for audio restorers — no real models, no GPU required."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from restorax.audio.restorer import (
    AudioRestorerCategory,
    AudioRestorerParams,
)


def _stereo(seconds: float = 1.0, sr: int = 44100) -> np.ndarray:
    """Synthetic stereo float32 audio, values in [-1, 1]."""
    samples = int(seconds * sr)
    rng = np.random.default_rng(0)
    return rng.uniform(-0.5, 0.5, (samples, 2)).astype(np.float32)


def _mono(seconds: float = 1.0, sr: int = 48000) -> np.ndarray:
    samples = int(seconds * sr)
    rng = np.random.default_rng(1)
    return rng.uniform(-0.5, 0.5, (samples, 1)).astype(np.float32)


# ── DemucsRestorer ────────────────────────────────────────────────────────────


class TestDemucsRestorer:
    @pytest.fixture
    def restorer(self):
        from restorax.restorers.audio.demucs import DemucsRestorer, _DemucsStub

        r = DemucsRestorer()
        r._model = _DemucsStub()
        r._device = torch.device("cpu")
        r._loaded = True
        return r

    def test_name(self, restorer):
        from restorax.restorers.audio.demucs import DemucsRestorer

        assert DemucsRestorer().name == "demucs_htdemucs"

    def test_category(self, restorer):
        assert restorer.capabilities.category == AudioRestorerCategory.SOURCE_SEPARATION

    def test_sample_rates(self, restorer):
        assert 44100 in restorer.capabilities.sample_rates

    def test_process_audio_preserves_shape(self, restorer):
        audio = _stereo()
        result = restorer.process_audio(audio, AudioRestorerParams(sample_rate=44100))
        assert result.shape == audio.shape

    def test_process_audio_dtype_float32(self, restorer):
        audio = _stereo()
        result = restorer.process_audio(audio, AudioRestorerParams(sample_rate=44100))
        assert result.dtype == np.float32

    def test_process_audio_values_bounded(self, restorer):
        audio = _stereo()
        result = restorer.process_audio(audio, AudioRestorerParams(sample_rate=44100))
        assert result.max() <= 1.0 and result.min() >= -1.0

    def test_is_loaded(self, restorer):
        assert restorer.is_loaded

    def test_unload(self, restorer):
        restorer.unload()
        assert not restorer.is_loaded


# ── VoiceFixerRestorer ────────────────────────────────────────────────────────


class TestVoiceFixerRestorer:
    @pytest.fixture
    def restorer(self):
        from restorax.restorers.audio.voicefixer import VoiceFixerRestorer, _VoiceFixerStub

        r = VoiceFixerRestorer()
        r._model = _VoiceFixerStub()
        r._device = torch.device("cpu")
        r._loaded = True
        return r

    def test_name(self):
        from restorax.restorers.audio.voicefixer import VoiceFixerRestorer

        assert VoiceFixerRestorer().name == "voicefixer"

    def test_category(self, restorer):
        assert restorer.capabilities.category == AudioRestorerCategory.SPEECH_ENHANCEMENT

    def test_multiple_sample_rates_supported(self, restorer):
        assert len(restorer.capabilities.sample_rates) >= 2

    def test_process_audio_preserves_shape(self, restorer):
        audio = _stereo()
        result = restorer.process_audio(audio, AudioRestorerParams(sample_rate=44100))
        assert result.shape == audio.shape

    def test_process_audio_dtype(self, restorer):
        audio = _stereo()
        result = restorer.process_audio(audio, AudioRestorerParams(sample_rate=44100))
        assert result.dtype == np.float32

    def test_unload(self, restorer):
        restorer.unload()
        assert not restorer.is_loaded


# ── RNNoiseRestorer ───────────────────────────────────────────────────────────


class TestRNNoiseRestorer:
    @pytest.fixture
    def restorer(self):
        from restorax.restorers.audio.rnnoise import RNNoiseRestorer, _RNNoiseStub

        r = RNNoiseRestorer()
        r._denoiser = _RNNoiseStub()
        r._device = torch.device("cpu")
        r._loaded = True
        return r

    def test_name(self):
        from restorax.restorers.audio.rnnoise import RNNoiseRestorer

        assert RNNoiseRestorer().name == "rnnoise"

    def test_category(self, restorer):
        assert restorer.capabilities.category == AudioRestorerCategory.NOISE_SUPPRESSION

    def test_process_audio_preserves_shape_mono(self, restorer):
        audio = _mono()
        result = restorer.process_audio(audio, AudioRestorerParams(sample_rate=48000))
        assert result.shape == audio.shape

    def test_process_audio_preserves_shape_stereo(self, restorer):
        audio = _stereo()
        result = restorer.process_audio(audio, AudioRestorerParams(sample_rate=44100))
        assert result.shape == audio.shape

    def test_process_audio_dtype(self, restorer):
        audio = _mono()
        result = restorer.process_audio(audio, AudioRestorerParams(sample_rate=48000))
        assert result.dtype == np.float32

    def test_unload(self, restorer):
        restorer.unload()
        assert not restorer.is_loaded
