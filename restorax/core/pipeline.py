from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from restorax.core.restorer import BaseRestorer, RestorerParams

if TYPE_CHECKING:
    from restorax.video.reader import VideoReader
    from restorax.video.writer import VideoWriter

logger = logging.getLogger(__name__)


@dataclass
class Stage:
    restorer: BaseRestorer
    params: RestorerParams
    enabled: bool = True


@dataclass
class Pipeline:
    name: str
    stages: list[Stage]
    # Frame chunking config — temporal models need a window of frames
    chunk_size: int = 16
    # Overlap between consecutive chunks to avoid boundary artifacts
    chunk_overlap: int = 2


class PipelineRunner:
    """
    Orchestrates sequential-chunk execution of a Pipeline.

    For each overlapping chunk of frames:
      1. Apply every enabled Stage in order (sequential, not parallel).
      2. Trim the overlap region from the output chunk edges.
      3. Write trimmed frames to the VideoWriter.
      4. Discard processed frames — memory stays constant regardless of length.

    The chunk_overlap prevents visible seams at boundaries for temporal models
    (BasicVSR++, RIFE) that look at neighbouring frames. Overlap frames are
    processed but not written to avoid duplicates.
    """

    def run(
        self,
        pipeline: Pipeline,
        reader: VideoReader,
        writer: VideoWriter,
        progress_cb: Callable[[float], None] | None = None,
    ) -> None:
        total = max(reader.meta.frame_count, 1)
        frames_written = 0

        for chunk_idx, (chunk, is_first, is_last) in enumerate(
            self._iter_chunks(reader, pipeline.chunk_size, pipeline.chunk_overlap)
        ):
            processed = chunk

            current_cs = "rgb"  # VideoReader always outputs RGB
            for stage in pipeline.stages:
                if not stage.enabled:
                    continue

                caps = stage.restorer.capabilities
                # Convert color space if the stage requires a different input format
                if caps.input_color_space != current_cs:
                    processed = [_convert_cs(f, current_cs, caps.input_color_space) for f in processed]

                if caps.requires_temporal:
                    processed = stage.restorer.process_sequence(processed, stage.params)
                else:
                    processed = [
                        stage.restorer.process_frame(f, stage.params) for f in processed
                    ]

                current_cs = caps.output_color_space

            # Convert back to RGB before writing (VideoWriter expects RGB)
            if current_cs != "rgb":
                processed = [_convert_cs(f, current_cs, "rgb") for f in processed]

            # Trim overlap to avoid writing duplicate frames at boundaries
            trimmed = self._trim_overlap(
                processed, pipeline.chunk_overlap, is_first, is_last
            )

            for frame in trimmed:
                writer.write_frame(frame)
                frames_written += 1

            if progress_cb is not None:
                progress_cb(min(frames_written / total, 1.0))

        if progress_cb is not None:
            progress_cb(1.0)

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _iter_chunks(
        reader: VideoReader,
        chunk_size: int,
        overlap: int,
    ) -> Iterator[tuple[list[np.ndarray], bool, bool]]:
        """
        Yield (chunk, is_first, is_last) tuples.

        Each chunk has up to chunk_size + overlap frames. The overlap frames
        at the start of a non-first chunk are the tail of the previous chunk.
        """
        buffer: list[np.ndarray] = []
        first_chunk = True

        for frame in reader:
            buffer.append(frame)
            if len(buffer) >= chunk_size + overlap:
                yield buffer[:chunk_size + overlap], first_chunk, False
                # Keep the last `overlap` frames as the start of the next chunk
                buffer = buffer[chunk_size:]
                first_chunk = False

        # Yield remaining frames as the last chunk
        if buffer:
            yield buffer, first_chunk, True

    @staticmethod
    def _trim_overlap(
        frames: list[np.ndarray],
        overlap: int,
        is_first: bool,
        is_last: bool,
    ) -> list[np.ndarray]:
        """
        Remove the overlap region from the end of each non-last chunk.
        First chunk: keep all frames from the start.
        Non-last chunk: drop the last `overlap` frames (they appear in next chunk).
        Last chunk: keep everything.
        """
        if is_last:
            return frames
        trim_end = len(frames) - overlap
        return frames[:trim_end]


def load_pipeline_from_yaml(path: str | Path, registry: object) -> Pipeline:
    """Load a Pipeline from a YAML preset file."""
    import yaml

    from restorax.core.exceptions import PipelineConfigError
    from restorax.core.registry import ModelRegistry

    assert isinstance(registry, ModelRegistry)

    with open(path) as f:
        config = yaml.safe_load(f)

    import torch

    from restorax.config import settings

    device = torch.device(settings.device)
    stages: list[Stage] = []
    for stage_cfg in config.get("stages", []):
        if not stage_cfg.get("enabled", True):
            continue  # disabled stages are inert — don't resolve a possibly-unregistered restorer
        restorer = registry.get(stage_cfg["restorer"], device)
        params = RestorerParams(
            scale=stage_cfg.get("scale", restorer.capabilities.scale_factor),
            tile_size=stage_cfg.get("tile_size", 0),
            tile_overlap=stage_cfg.get("tile_overlap", 32),
            half_precision=stage_cfg.get("half_precision", True),
            extra=stage_cfg.get("extra", {}),
        )
        stages.append(Stage(restorer=restorer, params=params, enabled=stage_cfg.get("enabled", True)))

    return Pipeline(
        name=config.get("name", Path(path).stem),
        stages=stages,
        chunk_size=config.get("chunk_size", 16),
        chunk_overlap=config.get("chunk_overlap", 2),
    )


def compute_output_fps(pipeline: Pipeline, original_fps: float) -> float:
    """
    Calculate the output video fps after all enabled pipeline stages.

    Most restorers have temporal_scale=1 (fps unchanged). RIFE has
    temporal_scale=2, meaning it inserts one mid-frame between every pair,
    doubling the output frame rate.

    Example: original_fps=24, RIFE stage active → output_fps=48.
    """
    multiplier = 1
    for stage in pipeline.stages:
        if stage.enabled:
            multiplier *= stage.restorer.capabilities.temporal_scale
    return original_fps * multiplier


def _convert_cs(frame: np.ndarray, src: str, dst: str) -> np.ndarray:
    """Convert a frame between color spaces used by restorers."""
    from restorax.video.utils import from_rgb, to_rgb

    if src == dst:
        return frame
    # Normalise to RGB then convert to dst
    rgb = to_rgb(frame, src)
    return from_rgb(rgb, dst)
