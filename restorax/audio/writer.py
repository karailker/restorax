"""
AudioWriter — encode float32 audio and optionally mux into a video container.

Two operations:
  write_wav()       — write a standalone .wav file
  mux_into_video()  — replace the audio track of an existing video file
                      (video-only output of the video pipeline) with
                      the processed audio.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import av
import numpy as np

from restorax.core.exceptions import AudioWriteError


class AudioWriter:
    """Encode and write processed float32 audio arrays."""

    def write_wav(
        self,
        path: str | Path,
        audio: np.ndarray,  # (num_samples, num_channels) float32 [-1.0, 1.0]
        sample_rate: int,
    ) -> None:
        """Write audio to a WAV file using PCM 16-bit encoding."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        num_channels = audio.shape[1] if audio.ndim == 2 else 1
        audio_2d = audio if audio.ndim == 2 else audio[:, np.newaxis]

        try:
            layout = "stereo" if num_channels == 2 else "mono"
            container = av.open(str(out_path), mode="w", format="wav")
            stream = container.add_stream("pcm_s16le", rate=sample_rate, layout=layout)
            stream.codec_context.format = "s16"

            # Convert float32 → int16 planar: shape (channels, samples), must be C-contiguous
            pcm = np.ascontiguousarray((np.clip(audio_2d, -1.0, 1.0) * 32767).astype(np.int16).T)
            # s16p = planar signed 16-bit; from_ndarray accepts (channels, samples)
            frame = av.AudioFrame.from_ndarray(pcm, format="s16p", layout=layout)
            frame.sample_rate = sample_rate
            frame.pts = 0

            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        except Exception as exc:
            raise AudioWriteError(f"Failed to write WAV to {out_path}: {exc}") from exc

    def mux_into_video(
        self,
        video_path: str | Path,  # video-only file (output of video pipeline)
        audio: np.ndarray,  # (num_samples, num_channels) float32 [-1.0, 1.0]
        sample_rate: int,
        output_path: str | Path,  # may equal video_path (in-place replacement)
    ) -> None:
        """
        Combine a video-only file with processed audio into a single container.

        Strategy:
        1. Write processed audio to a temp WAV file.
        2. Open the video-only file (input) + temp WAV (audio source).
        3. Write video stream + new audio stream into a temp output.
        4. Replace output_path atomically.
        """
        video_path = Path(video_path)
        output_path = Path(output_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_wav = Path(tmpdir) / "audio.wav"
            tmp_out = Path(tmpdir) / "output.mp4"

            # Step 1: write processed audio to WAV
            self.write_wav(tmp_wav, audio, sample_rate)

            # Step 2 & 3: mux video + new audio
            try:
                self._mux(video_path, tmp_wav, tmp_out)
            except Exception as exc:
                raise AudioWriteError(f"Mux failed: {exc}") from exc

            # Step 4: atomic replace
            shutil.move(str(tmp_out), str(output_path))

    @staticmethod
    def _mux(video_only_path: Path, audio_path: Path, output_path: Path) -> None:
        """Copy video stream + encode audio into output container using PyAV."""
        in_video = av.open(str(video_only_path))
        in_audio = av.open(str(audio_path))
        out = av.open(str(output_path), mode="w")

        # Copy video stream without re-encoding
        in_vs = in_video.streams.video[0]
        out_vs = out.add_stream(template=in_vs)  # type: ignore[call-overload]

        # Add audio stream from temp WAV
        in_as = in_audio.streams.audio[0]
        out_as = out.add_stream(template=in_as)  # type: ignore[call-overload]

        # Mux video packets
        for packet in in_video.demux(in_vs):
            if packet.dts is None:
                continue
            packet.stream = out_vs
            out.mux(packet)

        # Mux audio packets
        for packet in in_audio.demux(in_as):  # type: ignore[assignment]
            if packet.dts is None:
                continue
            packet.stream = out_as
            out.mux(packet)

        in_video.close()
        in_audio.close()
        out.close()
