"""
Classical YADIF video deinterlacing restorer.

YADIF ("Yet Another DeInterlacing Filter") is the motion-adaptive deinterlacer
from FFmpeg/MPlayer. Unlike :class:`AIDeinterlaceRestorer` it needs no neural
weights — it runs entirely through the system ``ffmpeg`` binary — so it works
out of the box on any machine with FFmpeg installed. This makes it the default,
always-available deinterlacing stage for interlaced broadcast/VHS/DVD sources.

If ``ffmpeg`` is unavailable the restorer degrades gracefully to a pure-numpy
"bob" deinterlace (drop one field, line-double the other).

Like the AI deinterlacer, frames detected as already-progressive are returned
unchanged so progressive footage is never softened.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from restorax.core.restorer import (
    BaseRestorer,
    RestorerCapabilities,
    RestorerCategory,
    RestorerParams,
)

logger = logging.getLogger(__name__)


class YadifDeinterlaceRestorer(BaseRestorer):
    """Deinterlace frames with FFmpeg's YADIF filter (no model weights)."""

    # No tunable knobs: motion-adaptive YADIF with auto field parity (mode=send_frame,
    # one output frame per input frame) is the sensible universal default.
    PARAM_SCHEMA: list = []

    def __init__(self) -> None:
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "yadif_deinterlace"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.DEINTERLACING,
            input_color_space="rgb",
            output_color_space="rgb",
            requires_temporal=False,
            min_vram_gb=0.0,
            scale_factor=1,
            tags=["deinterlacing", "interlaced", "combing", "yadif", "ffmpeg", "broadcast", "vhs"],
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self, device: torch.device) -> None:
        """No weights to load. Records the device and warns if FFmpeg is missing."""
        self._device = device
        if shutil.which("ffmpeg") is None:
            logger.warning(
                "ffmpeg not found on PATH — yadif_deinterlace will use the numpy bob fallback"
            )
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False
        self._device = None

    # ── Inference ─────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        if not self._is_interlaced(frame):
            return frame
        return self._deinterlace([frame])[0]

    def process_sequence(
        self,
        frames: list[np.ndarray],
        params: RestorerParams,
    ) -> list[np.ndarray]:
        if not frames:
            return frames
        if not any(self._is_interlaced(f) for f in frames[:3]):
            return frames  # progressive — leave untouched
        return self._deinterlace(frames)

    # ── Detection ────────────────────────────────────────────────────────────

    @staticmethod
    def _is_interlaced(frame: np.ndarray) -> bool:
        """Detect combing via alternating-line energy vs. natural vertical gradient."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        even = gray[0::2, :]
        odd = gray[1::2, :]
        min_rows = min(even.shape[0], odd.shape[0])
        diff = np.abs(even[:min_rows] - odd[:min_rows])
        vert_diff = np.abs(gray[1:] - gray[:-1])
        combing_ratio = float(diff.mean()) / (float(vert_diff.mean()) + 1e-6)
        return combing_ratio > 1.8

    # ── Deinterlacing ──────────────────────────────────────────────────────────

    def _deinterlace(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Run YADIF via FFmpeg; fall back to numpy bob on any failure."""
        if shutil.which("ffmpeg") is None:
            return self._bob(frames)
        try:
            return self._yadif_ffmpeg(frames)
        except Exception as exc:  # noqa: BLE001 — any ffmpeg failure → safe fallback
            logger.warning("YADIF ffmpeg failed (%s) — using numpy bob fallback", exc)
            return self._bob(frames)

    @staticmethod
    def _yadif_ffmpeg(frames: list[np.ndarray]) -> list[np.ndarray]:
        """Motion-adaptive YADIF through the ffmpeg binary (one frame in → one out)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for i, f in enumerate(frames):
                cv2.imwrite(str(tmp / f"in_{i:04d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-framerate", "25",
                    "-i", str(tmp / "in_%04d.png"),
                    "-vf", "yadif=mode=0:parity=-1:deint=0",
                    str(tmp / "out_%04d.png"),
                ],
                capture_output=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode(errors="replace"))
            out: list[np.ndarray] = []
            for i in range(len(frames)):
                p = tmp / f"out_{i + 1:04d}.png"
                if p.exists():
                    out.append(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
                else:
                    out.append(frames[i])
            return out

    @staticmethod
    def _bob(frames: list[np.ndarray]) -> list[np.ndarray]:
        """Numpy fallback: keep even field, line-double to full height."""
        h, w = frames[0].shape[:2]
        return [cv2.resize(f[0::2], (w, h), interpolation=cv2.INTER_LINEAR) for f in frames]
