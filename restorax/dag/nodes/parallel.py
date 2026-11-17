from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from restorax.dag.context import ExecutionContext
from restorax.dag.node import Node, NodeResult, Port
from restorax.dag.serializer import dag_node_type

logger = logging.getLogger(__name__)


@dataclass
class BranchConfig:
    """A named sequence of (restorer_name, params_dict) steps forming one branch."""

    name: str
    restorer_steps: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "restorer_steps": self.restorer_steps}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BranchConfig:
        return cls(
            name=data["name"],
            restorer_steps=[tuple(s) for s in data.get("restorer_steps", [])],  # type: ignore[misc]
        )


@dag_node_type("parallel")
class ParallelNode(Node):
    """
    Fan-out: run N branches on the same input frame chunks sequentially.
    Each branch is a BranchConfig — an ordered list of restorers to apply.
    Emits per-branch progress events via ProgressEmitter.
    """

    def __init__(
        self,
        id: str,
        name: str,
        branches: list[BranchConfig] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(id, name)
        self.branches: list[BranchConfig] = branches or []

    @property
    def input_ports(self) -> list[Port]:
        return [Port("chunks", list), Port("meta", object)]

    @property
    def output_ports(self) -> list[Port]:
        return [Port("branch_outputs", list), Port("meta", object)]

    async def execute(self, ctx: ExecutionContext, inputs: dict[str, Any]) -> NodeResult:
        from restorax.core.restorer import RestorerParams

        chunks: list[list[np.ndarray]] = inputs["chunks"]
        meta = inputs.get("meta")
        branch_outputs: list[list[list[np.ndarray]]] = []
        branch_info: list[dict[str, Any]] = []

        for branch_idx, branch in enumerate(self.branches):
            branch_chunks: list[list[np.ndarray]] = [list(c) for c in chunks]
            total_steps = max(len(branch.restorer_steps), 1)

            for step_idx, (restorer_name, params_dict) in enumerate(branch.restorer_steps):
                restorer = ctx.registry.get(restorer_name, ctx.device)
                params = RestorerParams(**params_dict)
                caps = restorer.capabilities
                out_chunks: list[list[np.ndarray]] = []

                n_chunks = max(len(branch_chunks), 1)
                for chunk_i, chunk in enumerate(branch_chunks):
                    if caps.requires_temporal:
                        processed = restorer.process_sequence(chunk, params)
                    else:
                        processed = [restorer.process_frame(f, params) for f in chunk]
                    out_chunks.append(processed)

                    overall = (step_idx + (chunk_i + 1) / n_chunks) / total_steps
                    ctx.progress_emitter.emit(self.id, overall, branch_index=branch_idx)

                branch_chunks = out_chunks

            branch_outputs.append(branch_chunks)
            branch_info.append(
                {
                    "index": branch_idx,
                    "name": branch.name,
                    "preview_path": self._write_preview(ctx, branch_idx, branch_chunks),
                }
            )

        return NodeResult(
            outputs={"branch_outputs": branch_outputs, "meta": meta},
            metadata={"branches": branch_info},
        )

    @staticmethod
    def _write_preview(
        ctx: ExecutionContext, branch_idx: int, branch_chunks: list[list[np.ndarray]]
    ) -> str | None:
        """
        Save one PNG of the branch's first frame so the UI can compare branches.

        Written to <work_dir>/branches/branch_<i>.png — cheap (one frame per branch) and
        the only artifact the comparison view needs; branch frames themselves stay in RAM.
        """
        first = next((chunk[0] for chunk in branch_chunks if chunk), None)
        if first is None:
            return None
        try:
            from PIL import Image

            out_dir = ctx.work_dir / "branches"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"branch_{branch_idx}.png"
            Image.fromarray(first).save(path)
            return str(path)
        except Exception as exc:  # noqa: BLE001 - a missing preview must not fail the job
            logger.warning("branch preview write failed: %s", exc)
            return None

    def to_dict(self) -> dict[str, Any]:
        return {"branches": [b.to_dict() for b in self.branches]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParallelNode:
        return cls(
            id=data["id"],
            name=data["name"],
            branches=[BranchConfig.from_dict(b) for b in data.get("branches", [])],
        )
