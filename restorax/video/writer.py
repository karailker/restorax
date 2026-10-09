from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from restorax.core.exceptions import VideoWriteError
from restorax.video.reader import VideoMeta


class VideoWriter:
    """
    Write RGB frames to a video file, optionally muxing the audio stream
    from the original source container.

    Uses PyAV for direct container access to preserve PTS accuracy.
    """

    def __init__(
        self,
        path: str | Path,
        meta: VideoMeta,
        out_width: int,
        out_height: int,
        fps: float | None = None,
        codec: str = "libx264",
        crf: int = 18,
        source_path: str | Path | None = None,  # for audio passthrough
    ) -> None:
        self._path = Path(path)
        self._meta = meta
        self._out_width = out_width
        self._out_height = out_height
        self._fps = fps or meta.fps
        self._codec = codec
        self._crf = crf
        self._source_path = Path(source_path) if source_path else None
        self._container: av.container.OutputContainer | None = None
        self._video_stream: av.video.stream.VideoStream | None = None
        self._pts = 0

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> VideoWriter:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._container = av.open(str(self._path), mode="w")
            self._video_stream = self._container.add_stream(
                self._codec, rate=Fraction(self._fps).limit_denominator(1001)
            )
            self._video_stream.width = self._out_width
            self._video_stream.height = self._out_height
            self._video_stream.pix_fmt = "yuv420p"
            self._video_stream.options = {"crf": str(self._crf)}
        except Exception as exc:
            raise VideoWriteError(f"Cannot open output {self._path}: {exc}") from exc

    def close(self) -> None:
        if self._container is None:
            return
        try:
            # Flush encoder
            for packet in self._video_stream.encode():  # type: ignore[union-attr]
                self._container.mux(packet)
            # Mux audio from source if available
            if self._source_path and self._meta.has_audio:
                self._mux_audio()
            self._container.close()
        except Exception as exc:
            raise VideoWriteError(f"Error closing {self._path}: {exc}") from exc
        finally:
            self._container = None
            self._video_stream = None
            self._pts = 0

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single RGB uint8 HxWxC frame."""
        if self._container is None or self._video_stream is None:
            raise VideoWriteError("VideoWriter is not open.")
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        av_frame.pts = self._pts
        self._pts += 1
        for packet in self._video_stream.encode(av_frame):
            self._container.mux(packet)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _mux_audio(self) -> None:
        """Re-mux audio from the source file without re-encoding."""
        assert self._container is not None
        assert self._source_path is not None
        src = av.open(str(self._source_path))
        if not src.streams.audio:
            src.close()
            return
        # Re-open output in append mode is not possible; use a temp approach:
        # Write audio packets from source directly into output container.
        # Note: This works when container is still open (called before close flush).
        audio_stream = self._container.add_stream(template=src.streams.audio[0])
        for packet in src.demux(src.streams.audio[0]):
            if packet.dts is None:
                continue
            packet.stream = audio_stream
            self._container.mux(packet)
        src.close()
