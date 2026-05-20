"""
ProPainter-based scratch and dust removal restorer.

Uses temporal video inpainting to fill detected scratch/dust regions
by propagating information from neighbouring frames. Significantly better
than single-frame inpainting because film scratches are temporally
incoherent (they appear in different positions each frame) while the
underlying image content is stable.

Scratch detection: pixel-wise thresholding on the temporal difference
between consecutive frames — scratches appear as sudden bright vertical
streaks with no temporal consistency.

Model source: https://github.com/sczhou/ProPainter (NeurIPS 2023)

Integration: ProPainter wraps its own flow estimator (RAFT) and
recurrent inpainting network. The class below loads the full ProPainter
pipeline; raises RestorerLoadError if the arch or weights are unavailable.
"""
from __future__ import annotations

import logging
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

_HF_REPO = "sczhou/ProPainter"
_WEIGHT_DIR = "weights"

# Temporal difference threshold for scratch detection (0–255 scale)
_SCRATCH_THRESHOLD = 40
# Minimum scratch width in pixels to be considered a scratch (not noise)
_MIN_SCRATCH_WIDTH = 2


class ScratchRemovalRestorer(BaseRestorer):
    """
    Detect and remove film scratches and dust using ProPainter.

    requires_temporal=True: needs a window of frames to propagate clean
    content across the scratch regions temporally.
    """

    def __init__(self) -> None:
        self._model: object | None = None
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "scratch_removal"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.ARTIFACT_REMOVAL,
            input_color_space="rgb",
            output_color_space="rgb",
            requires_temporal=True,
            min_vram_gb=6.0,
            scale_factor=1,
            tags=["scratch_removal", "dust", "inpainting", "propainter", "film_restoration"],
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self, device: torch.device) -> None:
        self._model = self._build_model(device)
        self._device = device
        self._loaded = True
        logger.info("ScratchRemoval (ProPainter) loaded on %s", device)

    def unload(self) -> None:
        del self._model
        self._model = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Inference ─────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        """Single-frame fallback: median filter on detected scratch regions."""
        mask = self._detect_scratches_single(frame)
        if not mask.any():
            return frame
        return self._inpaint_single(frame, mask)

    def process_sequence(
        self,
        frames: list[np.ndarray],
        params: RestorerParams,
    ) -> list[np.ndarray]:
        """
        Temporally-aware scratch removal using ProPainter inpainting.

        Detects scratches via inter-frame temporal difference, then delegates
        to the ProPainter recurrent network.
        """
        if len(frames) < 2:
            return [self.process_frame(f, params) for f in frames]

        # Detect scratch masks via temporal incoherence
        masks = self._detect_scratches_temporal(frames)

        return self._propainter_inpaint(frames, masks)

    # ── Detection ─────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_scratches_single(frame: np.ndarray) -> np.ndarray:
        """Heuristic single-frame scratch detection via vertical edge analysis."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Detect sharp vertical gradients (scratch edges are predominantly vertical)
        sobel_x = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        bright_mask = (gray > 200).astype(np.uint8)
        edge_mask = (np.abs(sobel_x) > 80).astype(np.uint8)
        raw_mask = cv2.bitwise_and(bright_mask, edge_mask)
        # Morphological closing to fill thin scratch bodies
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        return cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel).astype(bool)

    @staticmethod
    def _detect_scratches_temporal(frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        Detect scratches by their temporal incoherence.

        A scratch pixel has high intensity AND appears suddenly (large diff
        from neighbouring frames) AND disappears the next frame.
        """
        masks: list[np.ndarray] = []
        n = len(frames)
        grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames]

        for i in range(n):
            prev_g = grays[i - 1] if i > 0 else grays[i]
            next_g = grays[i + 1] if i < n - 1 else grays[i]
            curr_g = grays[i]

            # A scratch is bright AND inconsistent with both neighbours
            diff_prev = np.abs(curr_g - prev_g)
            diff_next = np.abs(curr_g - next_g)
            incoherent = (diff_prev > _SCRATCH_THRESHOLD) & (diff_next > _SCRATCH_THRESHOLD)
            bright = curr_g > 180

            raw = (incoherent & bright).astype(np.uint8)
            # Dilate slightly to cover scratch edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
            mask = cv2.dilate(raw, kernel, iterations=1).astype(bool)
            masks.append(mask)

        return masks

    # ── Inpainting ────────────────────────────────────────────────────────────

    @staticmethod
    def _inpaint_single(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """OpenCV Telea inpainting on a single frame (fallback)."""
        mask_u8 = mask.astype(np.uint8) * 255
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        inpainted = cv2.inpaint(bgr, mask_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

    def _propainter_inpaint(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Delegate to ProPainter recurrent network."""
        results = self._model.inpaint(frames, masks)  # type: ignore[union-attr]
        return results

    # ── Build model ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_model(device: torch.device) -> object:
        try:
            # ProPainter ships as a standalone project — try vendored import
            from restorax.restorers.artifact_removal.propainter_arch import ProPainterPipeline  # type: ignore[import]
        except ImportError as exc:
            raise RestorerLoadError(
                "ProPainter arch not vendored — cannot load ScratchRemoval restorer. "
                "Add restorax/restorers/artifact_removal/propainter_arch.py."
            ) from exc

        try:
            from restorax.config import settings

            weight_dir = Path(settings.model_dir) / "propainter"
            if not weight_dir.exists():
                weight_dir.mkdir(parents=True)
                _download_propainter_weights(weight_dir)
            model = ProPainterPipeline(weight_dir=str(weight_dir), device=device)
            logger.info("ProPainter arch loaded from vendored module")
            return model
        except Exception as exc:
            raise RestorerLoadError(f"Failed to load ProPainter weights: {exc}") from exc


def _download_propainter_weights(weight_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RestorerLoadError("huggingface_hub required to download ProPainter weights.") from exc

    snapshot_download(repo_id=_HF_REPO, local_dir=str(weight_dir))
    logger.info("ProPainter weights downloaded to %s", weight_dir)
