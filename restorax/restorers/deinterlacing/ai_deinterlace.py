"""
AI-assisted video deinterlacing restorer.

Converts interlaced video (fields interleaved, e.g. 1080i/576i/480i from
broadcast TV, VHS, DVD) to progressive frames. Interlaced content shows
horizontal combing artifacts on fast motion — alternating lines come from
different moments in time.

Uses a deformable convolution + self-attention architecture trained on
synthetic interlacing. Reference: "A new multi-picture architecture for
learned video deinterlacing and demosaicing" (Image and Vision Computing,
2024). Superior on damaged or noisy interlaced sources.

Requires the vendored ``deinterlace_arch`` module (``DeinterlaceNet``).
If the arch is not present, ``load()`` raises ``RestorerLoadError``.

Autodetect interlacing: the restorer checks whether the source material
is actually interlaced before processing. If not interlaced, frames are
returned unchanged.
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

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import (
    BaseRestorer,
    RestorerCapabilities,
    RestorerCategory,
    RestorerParams,
)

logger = logging.getLogger(__name__)


class AIDeinterlaceRestorer(BaseRestorer):
    """
    Deinterlace video frames using a deep AI model (DeinterlaceNet).

    Single-frame and sequence deinterlacing both use the vendored arch.
    Raises ``RestorerLoadError`` at ``load()`` time if the vendored
    ``deinterlace_arch`` module is not present.
    """

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "ai_deinterlace"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.DEINTERLACING,
            input_color_space="rgb",
            output_color_space="rgb",
            requires_temporal=False,
            min_vram_gb=0.0,
            scale_factor=1,
            tags=["deinterlacing", "interlaced", "combing", "ai", "broadcast", "vhs"],
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self, device: torch.device) -> None:
        model = self._load_arch(device)
        model.train(False)
        self._model = model
        self._device = device
        self._loaded = True
        logger.info("AIDeinterlace loaded on %s", device)

    def unload(self) -> None:
        del self._model
        self._model = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Inference ─────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        """
        Deinterlace a single frame.

        Checks for combing artifacts first — if not interlaced, returns as-is.
        """
        if not self._is_interlaced(frame):
            return frame
        return self._run_model(frame)

    def process_sequence(
        self,
        frames: list[np.ndarray],
        params: RestorerParams,
    ) -> list[np.ndarray]:
        """Deinterlace a sequence of frames using the AI model."""
        if not any(self._is_interlaced(f) for f in frames[:3]):
            return frames  # Not interlaced — skip
        return [self._run_model(f) for f in frames]

    # ── Detection ────────────────────────────────────────────────────────────

    @staticmethod
    def _is_interlaced(frame: np.ndarray) -> bool:
        """
        Detect combing artifacts via inter-line difference analysis.

        Interlaced frames have high alternating-line variance because
        odd and even lines come from different time instants.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        # Measure alternating-line energy: difference between even and odd rows
        even = gray[0::2, :]
        odd = gray[1::2, :]
        min_rows = min(even.shape[0], odd.shape[0])
        diff = np.abs(even[:min_rows] - odd[:min_rows])
        # Also compare to vertical gradient (natural edge energy)
        vert_diff = np.abs(gray[1:] - gray[:-1])
        combing_ratio = float(diff.mean()) / (float(vert_diff.mean()) + 1e-6)
        return combing_ratio > 1.8

    # ── YADIF (sequence fallback only) ────────────────────────────────────────

    @staticmethod
    def _yadif_sequence(frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        Full YADIF via FFmpeg subprocess — motion-adaptive, best quality.
        Falls back to per-frame bob if FFmpeg is unavailable.
        """
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_p = Path(tmpdir)
                # Write frames to disk as PNG
                for i, f in enumerate(frames):
                    bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(tmpdir_p / f"in_{i:04d}.png"), bgr)

                # Run FFmpeg YADIF
                ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
                result = subprocess.run(  # noqa: S603 - fixed argv, no shell
                    [
                        ffmpeg_bin,
                        "-y",
                        "-framerate",
                        "25",
                        "-i",
                        str(tmpdir_p / "in_%04d.png"),
                        "-vf",
                        "yadif=mode=0:parity=-1:deint=0",
                        str(tmpdir_p / "out_%04d.png"),
                    ],
                    capture_output=True,
                    timeout=60,
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.decode())

                output: list[np.ndarray] = []
                for i in range(len(frames)):
                    p = tmpdir_p / f"out_{i + 1:04d}.png"
                    if p.exists():
                        bgr = cv2.imread(str(p))
                        output.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    else:
                        output.append(frames[i])
                return output
        except Exception as exc:
            logger.warning("YADIF subprocess failed (%s) — using bob fallback", exc)
            h, w = frames[0].shape[:2]
            return [cv2.resize(f[0::2], (w, h), interpolation=cv2.INTER_LINEAR) for f in frames]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_model(self, frame: np.ndarray) -> np.ndarray:
        """Run DeinterlaceNet inference on a single frame."""
        assert self._model is not None and self._device is not None
        h, w = frame.shape[:2]
        tensor = (
            torch.from_numpy(frame.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self._device)
        )
        with torch.inference_mode():
            out = self._model(tensor)  # 1 3 H W
        result = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (result * 255.0).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _load_arch(device: torch.device) -> torch.nn.Module:
        """Import DeinterlaceNet from the vendored arch module.

        Raises RestorerLoadError if the arch is not present.
        """
        try:
            from restorax.restorers.deinterlacing.deinterlace_arch import (
                DeinterlaceNet,  # type: ignore[import]
            )
        except ImportError as exc:
            raise RestorerLoadError(
                f"AIDeinterlaceRestorer requires the vendored deinterlace_arch module "
                f"(DeinterlaceNet) which is not installed: {exc}"
            ) from exc
        return DeinterlaceNet().to(device)
