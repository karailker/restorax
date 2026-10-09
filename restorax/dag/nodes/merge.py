from __future__ import annotations

from typing import Any, Literal

import numpy as np

from restorax.dag.context import ExecutionContext
from restorax.dag.node import Node, NodeResult, Port
from restorax.dag.serializer import dag_node_type


@dag_node_type("merge")
class MergeNode(Node):
    """
    Fan-in: combine branch outputs.
    strategy='blend'  -> equal-weight pixel average across all branches per frame.
    strategy='select' -> pass through branch at select_index unchanged.
    """

    def __init__(
        self,
        id: str,
        name: str,
        strategy: Literal["blend", "select"] = "blend",
        select_index: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(id, name)
        self.strategy = strategy
        self.select_index = select_index

    @property
    def input_ports(self) -> list[Port]:
        return [Port("branch_outputs", list), Port("meta", object)]

    @property
    def output_ports(self) -> list[Port]:
        return [Port("chunks", list), Port("meta", object)]

    async def execute(self, ctx: ExecutionContext, inputs: dict[str, Any]) -> NodeResult:
        branch_outputs: list[list[list[np.ndarray]]] = inputs["branch_outputs"]
        meta = inputs.get("meta")

        if not branch_outputs:
            return NodeResult(outputs={"chunks": [], "meta": meta})

        merged: list[list[np.ndarray]]
        if self.strategy == "select":
            idx = min(self.select_index, len(branch_outputs) - 1)
            merged = branch_outputs[idx]
        else:
            # blend: per-frame pixel average
            n_branches = len(branch_outputs)
            n_chunks = len(branch_outputs[0])
            merged = []
            for chunk_idx in range(n_chunks):
                blended_chunk: list[np.ndarray] = []
                n_frames = len(branch_outputs[0][chunk_idx])
                for frame_idx in range(n_frames):
                    frames = np.stack(
                        [branch_outputs[b][chunk_idx][frame_idx] for b in range(n_branches)],
                        axis=0,
                    ).astype(np.float32)
                    blended = np.mean(frames, axis=0).clip(0, 255).astype(np.uint8)
                    blended_chunk.append(blended)
                merged.append(blended_chunk)

        return NodeResult(outputs={"chunks": merged, "meta": meta})

    def to_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "select_index": self.select_index}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MergeNode:
        return cls(
            id=data["id"],
            name=data["name"],
            strategy=data.get("strategy", "blend"),
            select_index=data.get("select_index", 0),
        )
