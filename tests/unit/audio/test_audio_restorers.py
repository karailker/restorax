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
        from restorax.restorers.audio.demucs import DemucsRestorer

        r = DemucsRestorer()
        r._device = torch.device("cpu")
        r._loaded = True
        return r

    def test_name(self):
        from restorax.restorers.audio.demucs import DemucsRestorer

        assert DemucsRestorer().name == "demucs_htdemucs"

    def test_category(self, restorer):
        assert restorer.capabilities.category == AudioRestorerCategory.SOURCE_SEPARATION

    def test_sample_rates(self, restorer):
        assert 44100 in restorer.capabilities.sample_rates

    def test_process_audio_without_model_raises(self, restorer):
        audio = _stereo()
        with pytest.raises((RuntimeError, Exception)):
            restorer.process_audio(audio, AudioRestorerParams(sample_rate=44100))

    def test_is_loaded(self, restorer):
        assert restorer.is_loaded

    def test_unload(self, restorer):
        restorer.unload()
        assert not restorer.is_loaded

# ── VoiceFixerRestorer ────────────────────────────────────────────────────────

class TestVoiceFixerRestorer:
    @pytest.fixture
    def restorer(self):
        from restorax.restorers.audio.voicefixer import VoiceFixerRestorer

        r = VoiceFixerRestorer()
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

    def test_process_audio_without_model_raises(self, restorer):
        audio = _stereo()
        with pytest.raises((RuntimeError, Exception)):
            restorer.process_audio(audio, AudioRestorerParams(sample_rate=44100))

    def test_unload(self, restorer):
        restorer.unload()
        assert not restorer.is_loaded

# ── RNNoiseRestorer ───────────────────────────────────────────────────────────

class TestRNNoiseRestorer:
    @pytest.fixture
    def restorer(self):
        from restorax.restorers.audio.rnnoise import RNNoiseRestorer

        r = RNNoiseRestorer()
        r._device = torch.device("cpu")
        r._loaded = True
        return r

    def test_name(self):
        from restorax.restorers.audio.rnnoise import RNNoiseRestorer

        assert RNNoiseRestorer().name == "rnnoise"

    def test_category(self, restorer):
        assert restorer.capabilities.category == AudioRestorerCategory.NOISE_SUPPRESSION

    def test_build_denoiser_without_library_raises(self):
        """With no backend installed, _build_denoiser must raise, not return a stub."""
        import unittest.mock as mock

        from restorax.restorers.audio.rnnoise import RNNoiseRestorer

        with mock.patch.dict("sys.modules", {"rnnoise": None, "noisereduce": None}):
            with pytest.raises(RuntimeError):
                RNNoiseRestorer()._build_denoiser()

    def test_unload(self, restorer):
        restorer.unload()
        assert not restorer.is_loaded
