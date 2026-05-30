from __future__ import annotations

from typing import Any

import numpy as np

from restorax.dag.context import ExecutionContext
from restorax.dag.node import Node, NodeResult, Port
from restorax.dag.serializer import dag_node_type


@dag_node_type("restore")
class RestoreNode(Node):
    """Apply a single video restorer to frame chunks."""

    def __init__(
        self,
        id: str,
        name: str,
        restorer_name: str = "",
        params_dict: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(id, name)
        self.restorer_name = restorer_name
        self.params_dict: dict[str, Any] = params_dict or {}

    @property
    def input_ports(self) -> list[Port]:
        return [Port("chunks", list)]

    @property
    def output_ports(self) -> list[Port]:
        return [Port("chunks", list)]

    async def execute(self, ctx: ExecutionContext, inputs: dict[str, Any]) -> NodeResult:
        from restorax.core.restorer import RestorerParams

        chunks: list[list[np.ndarray]] = inputs["chunks"]
        restorer = ctx.registry.get(self.restorer_name, ctx.device)
        params = RestorerParams(**self.params_dict)
        caps = restorer.capabilities

        out_chunks: list[list[np.ndarray]] = []
        total = max(len(chunks), 1)

        for i, chunk in enumerate(chunks):
            if caps.requires_temporal:
                processed = restorer.process_sequence(chunk, params)
            else:
                processed = [restorer.process_frame(f, params) for f in chunk]
            out_chunks.append(processed)
            ctx.progress_emitter.emit(self.id, (i + 1) / total)

        return NodeResult(outputs={"chunks": out_chunks})

    def to_dict(self) -> dict[str, Any]:
        return {"restorer_name": self.restorer_name, "params_dict": self.params_dict}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RestoreNode:
        return cls(
            id=data["id"],
            name=data["name"],
            restorer_name=data.get("restorer_name", ""),
            params_dict=data.get("params_dict", {}),
        )


@dag_node_type("audio_restore")
class AudioRestoreNode(Node):
    """Run the audio restoration pipeline and remux into the output video."""

    def __init__(self, id: str, name: str, **kwargs: Any) -> None:
        super().__init__(id, name)

    @property
    def input_ports(self) -> list[Port]:
        return [Port("video_path", str)]

    @property
    def output_ports(self) -> list[Port]:
        return [Port("video_path", str)]

    async def execute(self, ctx: ExecutionContext, inputs: dict[str, Any]) -> NodeResult:
        from restorax.tasks.job_tasks import _run_audio_pipeline

        video_path: str = inputs["video_path"]
        preset_path: str = ctx.config.get("pipeline_preset_path", "")
        if preset_path:
            _run_audio_pipeline(preset_path, video_path, video_path, ctx.device)
        return NodeResult(outputs={"video_path": video_path})

    def to_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioRestoreNode:
        return cls(id=data["id"], name=data["name"])
