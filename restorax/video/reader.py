from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from restorax.core.exceptions import VideoReadError


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float  # seconds
    codec_name: str
    has_audio: bool
    color_range: str  # "tv" | "pc" | "unknown"
    color_primaries: str
    color_trc: str  # transfer characteristics
    color_space: str  # colorspace / matrix


class VideoReader:
    """
    Iterate over decoded RGB frames from a video file.

    Uses PyAV for direct container access, preserving PTS/metadata needed
    for audio muxing and HDR handling.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise VideoReadError(f"Input file not found: {self._path}")
        self._container: av.container.InputContainer | None = None
        self._meta: VideoMeta | None = None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> VideoReader:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        try:
            self._container = av.open(str(self._path))
        except Exception as exc:
            raise VideoReadError(f"Cannot open {self._path}: {exc}") from exc
        self._meta = self._read_meta()

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def meta(self) -> VideoMeta:
        if self._meta is None:
            raise VideoReadError("VideoReader is not open. Call open() first.")
        return self._meta

    @property
    def path(self) -> Path:
        return self._path

    # ── Frame iteration ───────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._container is None:
            raise VideoReadError("VideoReader is not open.")
        video_stream = self._container.streams.video[0]
        self._container.seek(0)
        for packet in self._container.demux(video_stream):
            for frame in packet.decode():
                yield frame.to_ndarray(format="rgb24")

    def frames(self) -> Iterator[np.ndarray]:
        """Alias for __iter__ for explicit use."""
        return iter(self)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _read_meta(self) -> VideoMeta:
        assert self._container is not None
        vs = self._container.streams.video[0]
        fps_frac = vs.average_rate or vs.base_rate
        fps = float(fps_frac) if fps_frac else 0.0
        frame_count = vs.frames
        if not frame_count:
            # Estimate from duration when exact count is unavailable
            dur = float(self._container.duration or 0) / av.time_base
            frame_count = int(dur * fps) if fps else 0
        duration = float(self._container.duration or 0) / av.time_base
        cc = vs.codec_context
        has_audio = bool(self._container.streams.audio)
        return VideoMeta(
            width=vs.width,
            height=vs.height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            codec_name=vs.codec_context.name,
            has_audio=has_audio,
            color_range=str(cc.color_range) if cc.color_range else "unknown",
            color_primaries=str(cc.color_primaries) if cc.color_primaries else "unknown",
            color_trc=str(cc.color_trc) if cc.color_trc else "unknown",
            color_space=str(cc.colorspace) if cc.colorspace else "unknown",
        )
