"""Integration test: full ParallelNode + MergeNode round-trip with synthetic frames."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from restorax.dag.context import ExecutionContext, ProgressEmitter
from restorax.dag.edge import Edge
from restorax.dag.executor import DAGExecutor
from restorax.dag.graph import DAG
from restorax.dag.nodes.merge import MergeNode
from restorax.dag.nodes.parallel import BranchConfig, ParallelNode
from restorax.dag.serializer import DAGSerializer


def _make_ctx(tmp_path: Path) -> ExecutionContext:
    caps = MagicMock()
    caps.requires_temporal = False
    restorer = MagicMock()
    restorer.capabilities = caps
    restorer.process_frame.side_effect = lambda frame, params: frame  # identity

    registry = MagicMock()
    registry.get.return_value = restorer

    emitter = MagicMock(spec=ProgressEmitter)
    return ExecutionContext(
        run_id="integration-run",
        job_id="integration-job",
        work_dir=tmp_path,
        device=MagicMock(),
        registry=registry,
        progress_emitter=emitter,
        logger=MagicMock(),
    )


def _make_frames(n: int = 4) -> list[list[np.ndarray]]:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    return [[frame, frame]]  # 1 chunk, 2 frames


def test_parallel_then_merge_blend(tmp_path: Path):
    branches = [
        BranchConfig(name="branch_a", restorer_steps=[("real_esrgan", {})]),
        BranchConfig(name="branch_b", restorer_steps=[("waifu2x", {})]),
    ]
    nodes = {
        "parallel": ParallelNode(id="parallel", name="Parallel", branches=branches),
        "merge": MergeNode(id="merge", name="Merge", strategy="blend"),
    }
    edges = [Edge("parallel", "branch_outputs", "merge", "branch_outputs")]
    dag = DAG(id="test-dag", name="Test", nodes=nodes, edges=edges)

    ctx = _make_ctx(tmp_path)
    chunks = _make_frames()
    run = asyncio.run(
        DAGExecutor().execute(
            dag, ctx, initial_inputs={"parallel": {"chunks": chunks, "meta": None}}
        )
    )

    assert run.succeeded
    merge_result = run.node_results["merge"]
    assert "chunks" in merge_result.outputs
    assert len(merge_result.outputs["chunks"]) == 1
    assert len(merge_result.outputs["chunks"][0]) == 2  # 2 frames per chunk


def test_dag_serialization_roundtrip_with_parallel(tmp_path: Path):
    branches = [BranchConfig(name="b1", restorer_steps=[("real_esrgan", {"scale": 4})])]
    nodes = {
        "p": ParallelNode(id="p", name="P", branches=branches),
        "m": MergeNode(id="m", name="M", strategy="select", select_index=0),
    }
    edges = [Edge("p", "branch_outputs", "m", "branch_outputs")]
    dag = DAG(id="roundtrip", name="RoundTrip", nodes=nodes, edges=edges)

    data = DAGSerializer.to_dict(dag)
    restored = DAGSerializer.from_dict(data)

    assert restored.id == "roundtrip"
    assert "p" in restored.nodes
    assert isinstance(restored.nodes["m"], MergeNode)
    assert restored.nodes["m"].strategy == "select"
