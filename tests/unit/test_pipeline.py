"""Tests for PipelineRunner chunking, overlap trimming, and progress callbacks."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from restorax.core.pipeline import Pipeline, PipelineRunner, Stage, compute_output_fps
from restorax.core.restorer import RestorerParams
from tests.conftest import IdentityRestorer


def _make_frames(n: int, h: int = 8, w: int = 8) -> list[np.ndarray]:
    return [np.full((h, w, 3), i, dtype=np.uint8) for i in range(n)]


class _FakeReader:
    """Mimics VideoReader for pipeline tests without opening a real file."""
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self.meta = type("Meta", (), {"frame_count": len(frames)})()  # type: ignore

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self._frames)


class _CapturingWriter:
    """Collects written frames instead of encoding them."""
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def write_frame(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())


def _run(frames: list[np.ndarray], stages: list[Stage], chunk_size: int = 4, overlap: int = 0) -> list[np.ndarray]:
    pipeline = Pipeline(name="test", stages=stages, chunk_size=chunk_size, chunk_overlap=overlap)
    reader = _FakeReader(frames)
    writer = _CapturingWriter()
    PipelineRunner().run(pipeline, reader, writer)  # type: ignore[arg-type]
    return writer.frames


def test_identity_pipeline_preserves_all_frames() -> None:
    frames = _make_frames(10)
    restorer = IdentityRestorer(scale=1)
    restorer.load(None)  # type: ignore[arg-type]
    stages = [Stage(restorer=restorer, params=RestorerParams())]
    result = _run(frames, stages, chunk_size=4, overlap=0)
    assert len(result) == 10
    for i, frame in enumerate(result):
        assert np.array_equal(frame, frames[i])


def test_pipeline_with_overlap_no_duplicate_frames() -> None:
    frames = _make_frames(12)
    restorer = IdentityRestorer(scale=1)
    restorer.load(None)  # type: ignore[arg-type]
    stages = [Stage(restorer=restorer, params=RestorerParams())]
    result = _run(frames, stages, chunk_size=4, overlap=2)
    assert len(result) == 12


def test_disabled_stage_is_skipped() -> None:
    frames = _make_frames(6)
    restorer = IdentityRestorer(scale=4)
    restorer.load(None)  # type: ignore[arg-type]
    stages = [Stage(restorer=restorer, params=RestorerParams(scale=4), enabled=False)]
    result = _run(frames, stages, chunk_size=4, overlap=0)
    # Disabled stage → frames should pass through unchanged (8×8 not 32×32)
    assert len(result) == 6
    assert result[0].shape == (8, 8, 3)


def test_progress_callback_called() -> None:
    frames = _make_frames(8)
    restorer = IdentityRestorer(scale=1)
    restorer.load(None)  # type: ignore[arg-type]
    stages = [Stage(restorer=restorer, params=RestorerParams())]
    pipeline = Pipeline(name="test", stages=stages, chunk_size=4, chunk_overlap=0)
    reader = _FakeReader(frames)
    writer = _CapturingWriter()
    progress_values: list[float] = []
    PipelineRunner().run(pipeline, reader, writer, progress_cb=progress_values.append)  # type: ignore[arg-type]
    assert len(progress_values) > 0
    assert progress_values[-1] == 1.0
    assert all(0.0 <= p <= 1.0 for p in progress_values)


def test_single_frame_video() -> None:
    frames = _make_frames(1)
    restorer = IdentityRestorer(scale=1)
    restorer.load(None)  # type: ignore[arg-type]
    stages = [Stage(restorer=restorer, params=RestorerParams())]
    result = _run(frames, stages, chunk_size=4, overlap=0)
    assert len(result) == 1


# ── compute_output_fps ─────────────────────────────────────────────────────────

def test_compute_output_fps_identity_restorer() -> None:
    """Default temporal_scale=1 → fps unchanged."""
    restorer = IdentityRestorer(scale=1)
    restorer.load(None)  # type: ignore[arg-type]
    pipeline = Pipeline("test", [Stage(restorer=restorer, params=RestorerParams())])
    assert compute_output_fps(pipeline, 24.0) == 24.0


def test_compute_output_fps_rife_doubles() -> None:
    """RIFE has temporal_scale=2 → fps doubles."""
    from unittest.mock import MagicMock
    import torch
    from restorax.restorers.frame_interpolation.rife import RIFERestorer
    restorer = RIFERestorer()
    mock_model = MagicMock()
    mock_model.inference.side_effect = lambda t0, t1, timestep=0.5: (t0 + t1) / 2
    restorer._model = mock_model
    restorer._device = torch.device("cpu")
    restorer._loaded = True
    pipeline = Pipeline("rife_test", [Stage(restorer=restorer, params=RestorerParams())])
    assert compute_output_fps(pipeline, 24.0) == 48.0


def test_compute_output_fps_disabled_rife_no_change() -> None:
    """Disabled RIFE stage should not multiply fps."""
    from unittest.mock import MagicMock
    import torch
    from restorax.restorers.frame_interpolation.rife import RIFERestorer
    restorer = RIFERestorer()
    mock_model = MagicMock()
    mock_model.inference.side_effect = lambda t0, t1, timestep=0.5: (t0 + t1) / 2
    restorer._model = mock_model
    restorer._device = torch.device("cpu")
    restorer._loaded = True
    pipeline = Pipeline("disabled", [Stage(restorer=restorer, params=RestorerParams(), enabled=False)])
    assert compute_output_fps(pipeline, 24.0) == 24.0


def test_compute_output_fps_chained() -> None:
    """Two enabled stages with temporal_scale=1 each → fps unchanged."""
    r1 = IdentityRestorer(scale=1)
    r2 = IdentityRestorer(scale=4)
    r1.load(None)  # type: ignore[arg-type]
    r2.load(None)  # type: ignore[arg-type]
    pipeline = Pipeline("chain", [
        Stage(restorer=r1, params=RestorerParams()),
        Stage(restorer=r2, params=RestorerParams()),
    ])
    assert compute_output_fps(pipeline, 30.0) == 30.0


def test_fewer_frames_than_chunk_size() -> None:
    frames = _make_frames(3)
    restorer = IdentityRestorer(scale=1)
    restorer.load(None)  # type: ignore[arg-type]
    stages = [Stage(restorer=restorer, params=RestorerParams())]
    result = _run(frames, stages, chunk_size=16, overlap=2)
    assert len(result) == 3
