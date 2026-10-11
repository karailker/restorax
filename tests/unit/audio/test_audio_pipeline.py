"""Tests for AudioPipelineRunner, AudioModelRegistry, AudioReader, AudioWriter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from restorax.audio.restorer import AudioRestorerParams


def _make_audio(seconds: float = 0.5, sr: int = 44100, channels: int = 2) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.uniform(-0.3, 0.3, (int(seconds * sr), channels)).astype(np.float32)


# ── AudioPipelineRunner ───────────────────────────────────────────────────────


class TestAudioPipelineRunner:
    def _make_identity_restorer(self, name="identity_audio"):
        from restorax.audio.restorer import (
            AudioRestorer,
            AudioRestorerCapabilities,
            AudioRestorerCategory,
        )

        class _Identity(AudioRestorer):
            @property
            def name(self):
                return name

            @property
            def capabilities(self):
                return AudioRestorerCapabilities(
                    category=AudioRestorerCategory.NOISE_SUPPRESSION,
                    sample_rates=[44100],
                )

            def load(self, device):
                self._loaded = True

            def unload(self):
                self._loaded = False

            def process_audio(self, audio, params):
                return audio * 0.9  # multiply for traceability

        r = _Identity()
        r._loaded = True
        return r

    def test_single_stage_applied(self):
        from restorax.audio.pipeline import AudioPipeline, AudioPipelineRunner, AudioStage

        r = self._make_identity_restorer()
        pipeline = AudioPipeline(
            name="test",
            stages=[AudioStage(restorer=r, params=AudioRestorerParams(), enabled=True)],
        )
        audio = _make_audio()
        result = AudioPipelineRunner().run(pipeline, audio, 44100)
        assert result.shape == audio.shape
        # After one multiply-by-0.9 stage, values should differ from input
        assert not np.allclose(result, audio)

    def test_disabled_stage_skipped(self):
        from restorax.audio.pipeline import AudioPipeline, AudioPipelineRunner, AudioStage

        r = self._make_identity_restorer()
        pipeline = AudioPipeline(
            name="test",
            stages=[AudioStage(restorer=r, params=AudioRestorerParams(), enabled=False)],
        )
        audio = _make_audio()
        result = AudioPipelineRunner().run(pipeline, audio, 44100)
        assert np.array_equal(result, audio)

    def test_empty_pipeline_returns_input(self):
        from restorax.audio.pipeline import AudioPipeline, AudioPipelineRunner

        pipeline = AudioPipeline(name="empty", stages=[])
        audio = _make_audio()
        result = AudioPipelineRunner().run(pipeline, audio, 44100)
        assert np.array_equal(result, audio)

    def test_two_stages_applied_in_order(self):
        from restorax.audio.pipeline import AudioPipeline, AudioPipelineRunner, AudioStage

        calls = []

        class _TracingRestorer:
            _loaded = True

            @property
            def name(self):
                return "tracer"

            @property
            def capabilities(self):
                from restorax.audio.restorer import AudioRestorerCapabilities, AudioRestorerCategory

                return AudioRestorerCapabilities(AudioRestorerCategory.NOISE_SUPPRESSION, [44100])

            def load(self, d):
                pass

            def unload(self):
                pass

            def process_audio(self, audio, params):
                calls.append(id(self))
                return audio * 0.5

        r1, r2 = _TracingRestorer(), _TracingRestorer()
        pipeline = AudioPipeline(
            name="two_stage",
            stages=[
                AudioStage(r1, AudioRestorerParams()),
                AudioStage(r2, AudioRestorerParams()),
            ],
        )
        audio = _make_audio()
        result = AudioPipelineRunner().run(pipeline, audio, 44100)
        assert calls == [id(r1), id(r2)]
        # Two 0.5× stages = 0.25× final
        assert np.allclose(result, audio * 0.25, atol=1e-5)


# ── AudioModelRegistry ────────────────────────────────────────────────────────


class TestAudioModelRegistry:
    def _make_cls(self, slug):
        from restorax.audio.restorer import (
            AudioRestorer,
            AudioRestorerCapabilities,
            AudioRestorerCategory,
        )

        _slug = slug

        class _R(AudioRestorer):
            @property
            def name(self):
                return _slug

            @property
            def capabilities(self):
                return AudioRestorerCapabilities(AudioRestorerCategory.NOISE_SUPPRESSION, [44100])

            def load(self, d):
                self._loaded = True

            def unload(self):
                self._loaded = False

            def process_audio(self, a, p):
                return a

        return _R

    def test_register_and_get(self):
        from restorax.audio.pipeline import AudioModelRegistry

        reg = AudioModelRegistry(max_loaded=2)
        cls = self._make_cls("test_r1")
        reg.register(cls)
        r = reg.get("test_r1", torch.device("cpu"))
        assert r.is_loaded
        assert r.name == "test_r1"

    def test_lru_eviction(self):
        from restorax.audio.pipeline import AudioModelRegistry

        reg = AudioModelRegistry(max_loaded=1)
        c1, c2 = self._make_cls("ar1"), self._make_cls("ar2")
        reg.register(c1)
        reg.register(c2)
        r1 = reg.get("ar1", torch.device("cpu"))
        assert r1.is_loaded
        reg.get("ar2", torch.device("cpu"))
        # ar1 should have been evicted
        assert not r1.is_loaded

    def test_unknown_name_raises(self):
        from restorax.audio.pipeline import AudioModelRegistry
        from restorax.core.exceptions import RestorerNotFoundError

        reg = AudioModelRegistry()
        with pytest.raises(RestorerNotFoundError):
            reg.get("nonexistent_audio", torch.device("cpu"))


# ── AudioReader ───────────────────────────────────────────────────────────────


class TestAudioReader:
    def test_raises_on_missing_file(self):
        from restorax.audio.reader import AudioReader
        from restorax.core.exceptions import AudioReadError

        with pytest.raises(AudioReadError):
            AudioReader("/nonexistent/audio.wav").read()

    def test_read_from_synthetic_video(self, synthetic_video: Path):
        """synthetic_video fixture has no audio — should raise AudioReadError."""
        from restorax.audio.reader import AudioReader
        from restorax.core.exceptions import AudioReadError

        with pytest.raises(AudioReadError):
            AudioReader(synthetic_video).read()


# ── AudioWriter ───────────────────────────────────────────────────────────────


class TestAudioWriter:
    def test_write_wav_creates_file(self, tmp_path: Path):
        from restorax.audio.writer import AudioWriter

        audio = _make_audio(0.1, 44100, 2)
        out = tmp_path / "test.wav"
        AudioWriter().write_wav(out, audio, 44100)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_write_wav_readable(self, tmp_path: Path):
        from restorax.audio.reader import AudioReader
        from restorax.audio.writer import AudioWriter

        audio = _make_audio(0.2, 44100, 1)
        out = tmp_path / "roundtrip.wav"
        AudioWriter().write_wav(out, audio, 44100)
        restored, sr = AudioReader(out).read()
        assert sr == 44100
        assert restored.shape[0] > 0
        assert restored.dtype == np.float32
