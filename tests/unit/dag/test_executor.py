from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from restorax.core.exceptions import NodeExecutionError
from restorax.dag.context import ExecutionContext, ProgressEmitter
from restorax.dag.edge import Edge
from restorax.dag.executor import DAGExecutor, DAGRun
from restorax.dag.graph import DAG
from restorax.dag.node import Node, NodeResult, NodeState, Port, RetryPolicy


class _AddOneNode(Node):
    @property
    def input_ports(self):
        return [Port("value", int)]

    @property
    def output_ports(self):
        return [Port("value", int)]

    async def execute(self, ctx, inputs):
        return NodeResult(outputs={"value": inputs["value"] + 1})


class _FailNode(Node):
    def __init__(self, *args, fail_on_attempt: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._attempts = 0
        self._fail_on_attempt = fail_on_attempt

    @property
    def input_ports(self):
        return [Port("value", int)]

    @property
    def output_ports(self):
        return [Port("value", int)]

    async def execute(self, ctx, inputs):
        if self._attempts <= self._fail_on_attempt:
            self._attempts += 1
            raise ValueError("intentional failure")
        return NodeResult(outputs={"value": inputs["value"]})


def _make_ctx() -> ExecutionContext:
    emitter = MagicMock(spec=ProgressEmitter)
    logger = MagicMock()
    logger.warning = MagicMock()
    return ExecutionContext(
        run_id="run-1",
        job_id="job-1",
        work_dir=Path("/tmp/test_dag"),
        device=MagicMock(),
        registry=MagicMock(),
        progress_emitter=emitter,
        logger=logger,
    )


def test_linear_dag_executes_in_order():
    nodes = {
        "n0": _AddOneNode(id="n0", name="N0"),
        "n1": _AddOneNode(id="n1", name="N1"),
        "n2": _AddOneNode(id="n2", name="N2"),
    }
    edges = [
        Edge("n0", "value", "n1", "value"),
        Edge("n1", "value", "n2", "value"),
    ]
    dag = DAG(id="test", name="Test", nodes=nodes, edges=edges)
    ctx = _make_ctx()
    executor = DAGExecutor()
    run = asyncio.run(executor.execute(dag, ctx, initial_inputs={"n0": {"value": 0}}))

    assert run.succeeded
    assert run.node_results["n2"].outputs["value"] == 3


def test_failed_node_marks_downstream_skipped():
    nodes = {
        "n0": _AddOneNode(id="n0", name="N0"),
        "n1": _FailNode(id="n1", name="Fail"),
        "n2": _AddOneNode(id="n2", name="N2"),
    }
    edges = [
        Edge("n0", "value", "n1", "value"),
        Edge("n1", "value", "n2", "value"),
    ]
    dag = DAG(id="test", name="Test", nodes=nodes, edges=edges)
    ctx = _make_ctx()
    executor = DAGExecutor()
    run = asyncio.run(executor.execute(dag, ctx, initial_inputs={"n0": {"value": 0}}))

    assert not run.succeeded
    assert run.failed_node_id == "n1"
    assert run.node_states["n2"] == NodeState.SKIPPED


def test_retry_policy_retries_and_succeeds():
    # Fails on attempt 0, succeeds on attempt 1
    fail_node = _FailNode(
        id="n0", name="Flaky",
        fail_on_attempt=0,
        retry_policy=RetryPolicy(max_retries=1, delay_seconds=0),
    )
    dag = DAG(
        id="test", name="Test",
        nodes={"n0": fail_node},
        edges=[],
    )
    ctx = _make_ctx()
    executor = DAGExecutor()
    run = asyncio.run(executor.execute(dag, ctx, initial_inputs={"n0": {"value": 5}}))
    assert run.succeeded
    assert run.node_results["n0"].outputs["value"] == 5


def test_dry_run_marks_all_succeeded_without_executing():
    executed = []

    class _TrackNode(Node):
        @property
        def input_ports(self):
            return []

        @property
        def output_ports(self):
            return []

        async def execute(self, ctx, inputs):
            executed.append(self.id)
            return NodeResult(outputs={})

    dag = DAG(
        id="dry", name="Dry",
        nodes={"n0": _TrackNode(id="n0", name="N0")},
        edges=[],
    )
    ctx = _make_ctx()
    run = asyncio.run(DAGExecutor().dry_run(dag, ctx))
    assert run.succeeded
    assert executed == []  # nothing actually ran
