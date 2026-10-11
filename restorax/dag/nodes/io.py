from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from restorax.dag.context import ExecutionContext
from restorax.dag.node import Node, NodeResult, Port
from restorax.dag.serializer import dag_node_type


@dag_node_type("video_input")
class VideoInputNode(Node):
    """Read a video file and emit all overlapping frame chunks."""

    def __init__(
        self,
        id: str,
        name: str,
        video_path: str = "",
        chunk_size: int = 16,
        chunk_overlap: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(id, name)
        self.video_path = video_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def input_ports(self) -> list[Port]:
        return []

    @property
    def output_ports(self) -> list[Port]:
        return [Port("chunks", list), Port("meta", object)]

    async def execute(self, ctx: ExecutionContext, inputs: dict[str, Any]) -> NodeResult:
        from restorax.core.pipeline import PipelineRunner
        from restorax.video.reader import VideoReader

        path = self.video_path or ctx.config.get("input_path", "")
        chunks: list[list[np.ndarray]] = []

        with VideoReader(path) as reader:
            meta = reader.meta
            for chunk, _is_first, _is_last in PipelineRunner._iter_chunks(
                reader, self.chunk_size, self.chunk_overlap
            ):
                chunks.append(list(chunk))

        return NodeResult(outputs={"chunks": chunks, "meta": meta})

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": self.video_path,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoInputNode:
        return cls(
            id=data["id"],
            name=data["name"],
            video_path=data.get("video_path", ""),
            chunk_size=data.get("chunk_size", 16),
            chunk_overlap=data.get("chunk_overlap", 2),
        )


@dag_node_type("video_output")
class VideoOutputNode(Node):
    """Write processed frame chunks to a video file."""

    def __init__(
        self,
        id: str,
        name: str,
        output_path: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(id, name)
        self.output_path = output_path

    @property
    def input_ports(self) -> list[Port]:
        return [Port("chunks", list), Port("meta", object), Port("fps", float)]

    @property
    def output_ports(self) -> list[Port]:
        return [Port("output_path", str)]

    async def execute(self, ctx: ExecutionContext, inputs: dict[str, Any]) -> NodeResult:
        from restorax.video.writer import VideoWriter

        chunks: list[list[np.ndarray]] = inputs["chunks"]
        meta = inputs["meta"]
        fps: float = inputs.get("fps") or meta.fps
        path = self.output_path or ctx.config.get("output_path", "")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        first_frame = chunks[0][0] if chunks and chunks[0] else None
        out_h, out_w = (
            first_frame.shape[:2] if first_frame is not None else (meta.height, meta.width)
        )

        with VideoWriter(path, meta=meta, out_width=out_w, out_height=out_h, fps=fps) as writer:
            for chunk in chunks:
                for frame in chunk:
                    writer.write_frame(frame)

        return NodeResult(outputs={"output_path": path})

    def to_dict(self) -> dict[str, Any]:
        return {"output_path": self.output_path}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoOutputNode:
        return cls(id=data["id"], name=data["name"], output_path=data.get("output_path", ""))
