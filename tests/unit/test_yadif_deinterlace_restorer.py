"""Unit tests for YadifDeinterlaceRestorer (classical, weight-free deinterlacer)."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from restorax.core.registry import ModelRegistry
from restorax.core.restorer import RestorerCategory, RestorerParams
from restorax.restorers.deinterlacing.yadif_deinterlace import YadifDeinterlaceRestorer

torch = pytest.importorskip("torch")


def _interlaced_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Vertical [0,255,255,0] pattern → strong combing (detector ratio ≈ 1.97)."""
    pattern = np.array([0, 255, 255, 0], dtype=np.uint8)
    col = pattern[np.arange(h) % 4]
    plane = np.repeat(col[:, None], w, axis=1)
    return np.stack([plane] * 3, axis=-1)


def _progressive_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Smooth vertical gradient → no combing."""
    col = np.linspace(0, 255, h).astype(np.uint8)
    plane = np.repeat(col[:, None], w, axis=1)
    return np.stack([plane] * 3, axis=-1)


class TestYadifMeta:
    def test_name(self):
        assert YadifDeinterlaceRestorer().name == "yadif_deinterlace"

    def test_category(self):
        assert YadifDeinterlaceRestorer().capabilities.category == RestorerCategory.DEINTERLACING

    def test_scale_factor(self):
        assert YadifDeinterlaceRestorer().capabilities.scale_factor == 1

    def test_color_spaces(self):
        caps = YadifDeinterlaceRestorer().capabilities
        assert caps.input_color_space == "rgb"
        assert caps.output_color_space == "rgb"

    def test_no_params(self):
        assert YadifDeinterlaceRestorer.PARAM_SCHEMA == []

    def test_min_vram_zero(self):
        assert YadifDeinterlaceRestorer().capabilities.min_vram_gb == 0.0


class TestYadifLifecycle:
    def test_load_sets_loaded_without_weights(self):
        r = YadifDeinterlaceRestorer()
        assert not r.is_loaded
        r.load(torch.device("cpu"))
        assert r.is_loaded

    def test_unload_clears_loaded(self):
        r = YadifDeinterlaceRestorer()
        r.load(torch.device("cpu"))
        r.unload()
        assert not r.is_loaded


class TestYadifDetection:
    def test_detects_interlaced(self):
        assert YadifDeinterlaceRestorer._is_interlaced(_interlaced_frame()) is True

    def test_detects_progressive(self):
        assert YadifDeinterlaceRestorer._is_interlaced(_progressive_frame()) is False


class TestYadifProcessing:
    def _restorer(self) -> YadifDeinterlaceRestorer:
        r = YadifDeinterlaceRestorer()
        r.load(torch.device("cpu"))
        return r

    def test_progressive_sequence_passthrough(self):
        r = self._restorer()
        frames = [_progressive_frame() for _ in range(4)]
        out = r.process_sequence(frames, RestorerParams())
        assert out is frames  # untouched

    def test_progressive_frame_passthrough(self):
        r = self._restorer()
        f = _progressive_frame()
        assert r.process_frame(f, RestorerParams()) is f

    def test_empty_sequence(self):
        r = self._restorer()
        assert r.process_sequence([], RestorerParams()) == []

    def test_interlaced_sequence_preserves_shape_and_count(self):
        r = self._restorer()
        frames = [_interlaced_frame() for _ in range(4)]
        out = r.process_sequence(frames, RestorerParams())
        assert len(out) == 4
        for o in out:
            assert o.shape == (64, 64, 3)
            assert o.dtype == np.uint8

    def test_interlaced_frame_preserves_shape(self):
        r = self._restorer()
        out = r.process_frame(_interlaced_frame(), RestorerParams())
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8


class TestYadifBobFallback:
    def test_bob_preserves_shape_and_count(self):
        frames = [_interlaced_frame() for _ in range(3)]
        out = YadifDeinterlaceRestorer._bob(frames)
        assert len(out) == 3
        for o in out:
            assert o.shape == (64, 64, 3)
            assert o.dtype == np.uint8

    def test_falls_back_when_ffmpeg_missing(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _: None)
        r = YadifDeinterlaceRestorer()
        r.load(torch.device("cpu"))
        frames = [_interlaced_frame() for _ in range(2)]
        out = r.process_sequence(frames, RestorerParams())
        assert len(out) == 2
        assert out[0].shape == (64, 64, 3)


class TestYadifRegistration:
    def test_registers_and_resolves_by_name(self):
        reg = ModelRegistry(max_loaded=2)
        reg.register(YadifDeinterlaceRestorer)
        assert "yadif_deinterlace" in reg.list_available()
