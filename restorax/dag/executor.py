from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from restorax.core.exceptions import NodeExecutionError
from restorax.dag.context import ExecutionContext
from restorax.dag.edge import Edge
from restorax.dag.graph import DAG
from restorax.dag.node import Node, NodeResult, NodeState


@dataclass
class DAGRun:
    run_id: str
    dag_id: str
    job_id: str
    node_states: dict[str, NodeState] = field(default_factory=dict)
    node_results: dict[str, NodeResult] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    failed_node_id: str | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.failed_node_id is None and self.completed_at is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dag_id": self.dag_id,
            "job_id": self.job_id,
            "node_states": {k: v.value for k, v in self.node_states.items()},
            "failed_node_id": self.failed_node_id,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class DAGExecutor:
    """
    Executes a DAG by topological level. Nodes in the same level are run
    sequentially (single-GPU constraint). Applies RetryPolicy per node.
    On node failure, downstream nodes are marked SKIPPED.
    """

    async def execute(
        self,
        dag: DAG,
        ctx: ExecutionContext,
        initial_inputs: dict[str, dict[str, Any]] | None = None,
    ) -> DAGRun:
        """
        Execute the DAG. `initial_inputs` optionally seeds input ports of
        root nodes (nodes with no incoming edges): {node_id: {port_name: value}}.
        """
        run = DAGRun(
            run_id=ctx.run_id,
            dag_id=dag.id,
            job_id=ctx.job_id,
            node_states={nid: NodeState.PENDING for nid in dag.nodes},
        )

        for level in dag.topological_levels():
            for node_id in level:
                if run.node_states[node_id] == NodeState.SKIPPED:
                    continue

                node = dag.nodes[node_id]
                inputs = self._collect_inputs(node_id, dag, run, initial_inputs or {})

                run.node_states[node_id] = NodeState.RUNNING
                ctx.progress_emitter.emit(node_id, 0.0)

                t0 = time.perf_counter()
                try:
                    result = await self._execute_with_retry(node, ctx, inputs)
                    result.duration_seconds = time.perf_counter() - t0
                    run.node_states[node_id] = NodeState.SUCCEEDED
                    run.node_results[node_id] = result
                    ctx.progress_emitter.emit(node_id, 1.0, status="succeeded")
                except NodeExecutionError as exc:
                    run.node_states[node_id] = NodeState.FAILED
                    run.failed_node_id = node_id
                    run.error = str(exc)
                    ctx.progress_emitter.emit(node_id, 0.0, status="failed")
                    self._skip_downstream(node_id, dag, run)
                    run.completed_at = datetime.now(timezone.utc)
                    return run

        run.completed_at = datetime.now(timezone.utc)
        return run

    async def dry_run(self, dag: DAG, ctx: ExecutionContext) -> DAGRun:
        """Validate data flow without executing nodes. Marks all nodes SUCCEEDED."""
        run = DAGRun(
            run_id=ctx.run_id,
            dag_id=dag.id,
            job_id=ctx.job_id,
            node_states={nid: NodeState.SUCCEEDED for nid in dag.nodes},
        )
        run.completed_at = datetime.now(timezone.utc)
        return run

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _execute_with_retry(
        self, node: Node, ctx: ExecutionContext, inputs: dict[str, Any]
    ) -> NodeResult:
        policy = node.retry_policy
        last_exc: Exception = RuntimeError("unreachable")

        for attempt in range(policy.max_retries + 1):
            if attempt > 0:
                delay = (
                    policy.delay_seconds * (2 ** (attempt - 1))
                    if policy.backoff == "exponential"
                    else policy.delay_seconds
                )
                await asyncio.sleep(delay)
                ctx.progress_emitter.emit(node.id, 0.0, status="retrying")

            try:
                return await node.execute(ctx, inputs)
            except policy.retry_on as exc:
                last_exc = exc
                if attempt < policy.max_retries:
                    ctx.logger.warning(
                        "node retrying",
                        node_id=node.id,
                        attempt=attempt + 1,
                        error=str(exc),
                    )

        raise NodeExecutionError(node.id, policy.max_retries + 1, last_exc)

    def _collect_inputs(
        self,
        node_id: str,
        dag: DAG,
        run: DAGRun,
        initial_inputs: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        # Seed from initial_inputs for root nodes
        if node_id in initial_inputs:
            inputs.update(initial_inputs[node_id])
        # Override/supplement from upstream node results
        for edge in dag.edges:
            if edge.target_node_id == node_id:
                upstream_result = run.node_results.get(edge.source_node_id)
                if upstream_result is not None:
                    inputs[edge.target_port] = upstream_result.outputs.get(edge.source_port)
        return inputs

    def _skip_downstream(self, failed_node_id: str, dag: DAG, run: DAGRun) -> None:
        queue = [failed_node_id]
        visited: set[str] = set()
        while queue:
            current = queue.pop(0)
            for edge in dag.edges:
                if edge.source_node_id == current and edge.target_node_id not in visited:
                    run.node_states[edge.target_node_id] = NodeState.SKIPPED
                    visited.add(edge.target_node_id)
                    queue.append(edge.target_node_id)
