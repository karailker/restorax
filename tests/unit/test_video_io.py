"""Tests for VideoReader and VideoWriter using the synthetic_video fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from restorax.video.reader import VideoReader
from restorax.video.writer import VideoWriter


def test_reader_opens_and_reads_frames(synthetic_video: Path) -> None:
    with VideoReader(synthetic_video) as reader:
        meta = reader.meta
        assert meta.width == 64
        assert meta.height == 64
        assert meta.fps > 0

        frames = list(reader)
    assert len(frames) >= 1
    assert frames[0].shape == (64, 64, 3)
    assert frames[0].dtype == np.uint8


def test_reader_meta_accessible_before_iteration(synthetic_video: Path) -> None:
    with VideoReader(synthetic_video) as reader:
        meta = reader.meta
        assert meta.width > 0
        assert meta.height > 0


def test_reader_raises_on_missing_file(tmp_path: Path) -> None:
    from restorax.core.exceptions import VideoReadError

    with pytest.raises(VideoReadError):
        VideoReader(tmp_path / "nonexistent.mp4").open()


def test_writer_produces_output_file(synthetic_video: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "out.mp4"
    with VideoReader(synthetic_video) as reader:
        meta = reader.meta
        frames = list(reader)

    with VideoWriter(out_path, meta=meta, out_width=64, out_height=64) as writer:
        for f in frames:
            writer.write_frame(f)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_writer_output_readable(synthetic_video: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "out.mp4"
    with VideoReader(synthetic_video) as reader:
        meta = reader.meta
        in_frames = list(reader)

    with VideoWriter(out_path, meta=meta, out_width=64, out_height=64) as writer:
        for f in in_frames:
            writer.write_frame(f)

    with VideoReader(out_path) as reader2:
        out_frames = list(reader2)

    assert len(out_frames) == len(in_frames)
    assert out_frames[0].shape == (64, 64, 3)
