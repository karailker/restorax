# Pipeline DAG Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a custom DAG orchestration engine (`restorax/dag/`) that supports parallel restoration branches, typed ports, per-node retry policies, and per-branch progress reporting — all running inside a single Celery task without Celery canvas primitives.

**Architecture:** Five phases — (1) core engine abstractions, (2) built-in video node types, (3) Celery + DB integration, (4) REST API endpoints, (5) WebSocket extension + dry_run. The existing `Pipeline`/`PipelineRunner` system is untouched throughout.

**Tech Stack:** Python 3.11, asyncio, SQLAlchemy (async), Redis pub/sub, FastAPI, Celery, pytest.

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `restorax/dag/__init__.py` | Create | Public exports |
| `restorax/dag/errors.py` | Create | DAGValidationError, NodeExecutionError, PortTypeMismatchError |
| `restorax/dag/node.py` | Create | Port, NodeState, NodeResult, RetryPolicy, Node ABC |
| `restorax/dag/edge.py` | Create | Edge dataclass |
| `restorax/dag/graph.py` | Create | DAG with validation + topological sort |
| `restorax/dag/context.py` | Create | ExecutionContext, ProgressEmitter |
| `restorax/dag/executor.py` | Create | DAGRun, DAGExecutor |
| `restorax/dag/serializer.py` | Create | dag_node_type decorator, DAGSerializer |
| `restorax/dag/nodes/__init__.py` | Create | Node type exports |
| `restorax/dag/nodes/io.py` | Create | VideoInputNode, VideoOutputNode |
| `restorax/dag/nodes/restore.py` | Create | RestoreNode, AudioRestoreNode |
| `restorax/dag/nodes/parallel.py` | Create | ParallelNode, BranchConfig |
| `restorax/dag/nodes/merge.py` | Create | MergeNode |
| `restorax/dag/nodes/map_node.py` | Create | MapNode |
| `restorax/dag/nodes/control.py` | Create | ChoiceNode, PassNode |
| `restorax/core/exceptions.py` | Modify | Add DAGValidationError, NodeExecutionError |
| `restorax/db/models.py` | Modify | Add dag_run JSON column to JobModel |
| `restorax/tasks/job_tasks.py` | Modify | Add run_dag_job Celery task |
| `restorax/api/routers/pipelines.py` | Modify | Add POST/GET /pipelines/dag |
| `restorax/api/routers/jobs.py` | Modify | Add dag_id field, /branches, /merge |
| `restorax/api/schemas/pipeline.py` | Modify | Add DAGCreateRequest, DAGResponse |
| `restorax/api/schemas/job.py` | Modify | Add BranchResponse, MergeRequest |
| `restorax/api/routers/ws.py` | Modify | Forward branch_index in progress events |
| `tests/unit/dag/__init__.py` | Create | |
| `tests/unit/dag/test_graph.py` | Create | DAG validation, topological sort |
| `tests/unit/dag/test_executor.py` | Create | Execution, retry, failure propagation |
| `tests/unit/dag/test_serializer.py` | Create | Round-trip serialization |
| `tests/unit/dag/test_nodes.py` | Create | Node type unit tests |
| `tests/unit/test_dag_api.py` | Create | API endpoint tests |

---

## Task 1 — Error types

**Files:**
- Modify: `restorax/core/exceptions.py`
- Create: `tests/unit/dag/__init__.py`

- [ ] **Step 1: Add exceptions to core/exceptions.py**

Append to the bottom of `restorax/core/exceptions.py`:

```python
class DAGValidationError(RestoraXError):
    """Raised when a DAG fails structural validation (cycles, unknown ports, type mismatches)."""


class NodeExecutionError(RestoraXError):
    """Raised when a node's execute() fails after all retries are exhausted."""

    def __init__(self, node_id: str, attempt: int, cause: Exception) -> None:
        super().__init__(f"Node '{node_id}' failed on attempt {attempt}: {cause}")
        self.node_id = node_id
        self.attempt = attempt
        self.__cause__ = cause


class PortTypeMismatchError(DAGValidationError):
    """Raised when an edge connects ports with incompatible type hints."""
```

- [ ] **Step 2: Create test package**

Create `tests/unit/dag/__init__.py` (empty file).

- [ ] **Step 3: Write test**

Create `tests/unit/dag/test_errors.py`:

```python
from restorax.core.exceptions import DAGValidationError, NodeExecutionError, PortTypeMismatchError


def test_node_execution_error_carries_cause():
    cause = ValueError("weights missing")
    err = NodeExecutionError("my_node", attempt=2, cause=cause)
    assert "my_node" in str(err)
    assert err.node_id == "my_node"
    assert err.attempt == 2
    assert err.__cause__ is cause


def test_port_type_mismatch_is_dag_validation_error():
    err = PortTypeMismatchError("port type mismatch")
    assert isinstance(err, DAGValidationError)
```

- [ ] **Step 4: Run test**

```bash
conda run -n restorax python -m pytest tests/unit/dag/test_errors.py -q
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add restorax/core/exceptions.py tests/unit/dag/__init__.py tests/unit/dag/test_errors.py
git commit -m "feat(dag): add DAGValidationError, NodeExecutionError, PortTypeMismatchError"
```

---

## Task 2 — Core node abstractions

**Files:**
- Create: `restorax/dag/node.py`
- Create: `restorax/dag/__init__.py`

- [ ] **Step 1: Create `restorax/dag/node.py`**

```python
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from restorax.dag.context import ExecutionContext


@dataclass
class Port:
    name: str
    type_hint: type | None = None


@dataclass
class RetryPolicy:
    max_retries: int = 0
    delay_seconds: float = 1.0
    backoff: Literal["fixed", "exponential"] = "fixed"
    retry_on: tuple[type[Exception], ...] = field(default_factory=lambda: (Exception,))


class NodeState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class NodeResult:
    outputs: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0


class Node(ABC):
    """Base class for all DAG nodes. Stateless — all run state lives in DAGRun."""

    def __init__(
        self,
        id: str,
        name: str,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.id = id
        self.name = name
        self.retry_policy = retry_policy or RetryPolicy()

    @property
    @abstractmethod
    def input_ports(self) -> list[Port]: ...

    @property
    @abstractmethod
    def output_ports(self) -> list[Port]: ...

    @abstractmethod
    async def execute(self, ctx: ExecutionContext, inputs: dict[str, Any]) -> NodeResult: ...

    def to_dict(self) -> dict[str, Any]:
        """Serialise node-specific config. Override in subclasses."""
        return {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Node:
        """Deserialise from dict. Override in subclasses."""
        return cls(id=data["id"], name=data["name"])
```

- [ ] **Step 2: Create `restorax/dag/__init__.py`**

```python
from restorax.dag.edge import Edge
from restorax.dag.graph import DAG
from restorax.dag.node import Node, NodeResult, NodeState, Port, RetryPolicy
from restorax.dag.executor import DAGExecutor, DAGRun
from restorax.dag.context import ExecutionContext, ProgressEmitter
from restorax.dag.serializer import DAGSerializer, dag_node_type

__all__ = [
    "DAG", "Node", "Edge", "Port", "NodeState", "NodeResult", "RetryPolicy",
    "DAGExecutor", "DAGRun", "ExecutionContext", "ProgressEmitter",
    "DAGSerializer", "dag_node_type",
]
```

- [ ] **Step 3: Write test for Node**

Add to `tests/unit/dag/test_graph.py` (create the file):

```python
from __future__ import annotations

import pytest
from restorax.dag.node import Node, NodeResult, NodeState, Port, RetryPolicy
from restorax.core.exceptions import DAGValidationError


class _EchoNode(Node):
    """Test node that echoes its 'data' input to 'data' output."""

    @property
    def input_ports(self):
        return [Port("data")]

    @property
    def output_ports(self):
        return [Port("data")]

    async def execute(self, ctx, inputs):
        return NodeResult(outputs={"data": inputs.get("data")})


def test_node_has_id_and_name():
    node = _EchoNode(id="n1", name="Echo")
    assert node.id == "n1"
    assert node.name == "Echo"


def test_default_retry_policy():
    node = _EchoNode(id="n1", name="Echo")
    assert node.retry_policy.max_retries == 0


def test_custom_retry_policy():
    policy = RetryPolicy(max_retries=3, delay_seconds=0.5, backoff="exponential")
    node = _EchoNode(id="n1", name="Echo", retry_policy=policy)
    assert node.retry_policy.max_retries == 3
    assert node.retry_policy.backoff == "exponential"
```

- [ ] **Step 4: Run test**

```bash
conda run -n restorax python -m pytest tests/unit/dag/test_graph.py -q
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add restorax/dag/__init__.py restorax/dag/node.py tests/unit/dag/test_graph.py
git commit -m "feat(dag): add Node ABC, Port, RetryPolicy, NodeState, NodeResult"
```

---

## Task 3 — Edge and DAG graph with validation

**Files:**
- Create: `restorax/dag/edge.py`
- Create: `restorax/dag/graph.py`

- [ ] **Step 1: Create `restorax/dag/edge.py`**

```python
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Edge:
    source_node_id: str
    source_port: str
    target_node_id: str
    target_port: str
```

- [ ] **Step 2: Create `restorax/dag/graph.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict, deque

from restorax.core.exceptions import DAGValidationError, PortTypeMismatchError
from restorax.dag.edge import Edge
from restorax.dag.node import Node


@dataclass
class DAG:
    """
    Immutable directed acyclic graph of Nodes connected by Edges.
    Validates structure on construction — raises DAGValidationError on any violation.
    """
    id: str
    name: str
    nodes: dict[str, Node]
    edges: list[Edge]

    def __post_init__(self) -> None:
        self._validate()

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        self._check_node_references()
        self._check_port_names()
        self._check_port_types()
        self._check_no_cycles()

    def _check_node_references(self) -> None:
        for edge in self.edges:
            if edge.source_node_id not in self.nodes:
                raise DAGValidationError(
                    f"Edge references unknown source node '{edge.source_node_id}'"
                )
            if edge.target_node_id not in self.nodes:
                raise DAGValidationError(
                    f"Edge references unknown target node '{edge.target_node_id}'"
                )

    def _check_port_names(self) -> None:
        for edge in self.edges:
            src_node = self.nodes[edge.source_node_id]
            src_ports = {p.name for p in src_node.output_ports}
            if edge.source_port not in src_ports:
                raise DAGValidationError(
                    f"Node '{edge.source_node_id}' has no output port '{edge.source_port}'. "
                    f"Available: {src_ports}"
                )
            tgt_node = self.nodes[edge.target_node_id]
            tgt_ports = {p.name for p in tgt_node.input_ports}
            if edge.target_port not in tgt_ports:
                raise DAGValidationError(
                    f"Node '{edge.target_node_id}' has no input port '{edge.target_port}'. "
                    f"Available: {tgt_ports}"
                )

    def _check_port_types(self) -> None:
        src_port_map: dict[tuple[str, str], type | None] = {}
        for node_id, node in self.nodes.items():
            for port in node.output_ports:
                src_port_map[(node_id, port.name)] = port.type_hint

        for edge in self.edges:
            src_type = src_port_map.get((edge.source_node_id, edge.source_port))
            tgt_node = self.nodes[edge.target_node_id]
            tgt_type = next(
                (p.type_hint for p in tgt_node.input_ports if p.name == edge.target_port), None
            )
            if src_type is not None and tgt_type is not None and src_type != tgt_type:
                raise PortTypeMismatchError(
                    f"Edge {edge.source_node_id}.{edge.source_port} ({src_type.__name__}) "
                    f"→ {edge.target_node_id}.{edge.target_port} ({tgt_type.__name__}): "
                    f"incompatible types"
                )

    def _check_no_cycles(self) -> None:
        """Kahn's algorithm: if topological sort consumes all nodes, graph is acyclic."""
        levels = self._kahn_sort()
        visited = {nid for level in levels for nid in level}
        if len(visited) != len(self.nodes):
            raise DAGValidationError(
                "DAG contains a cycle. Nodes not reachable via topological sort: "
                f"{set(self.nodes) - visited}"
            )

    # ── Topology ──────────────────────────────────────────────────────────────

    def topological_levels(self) -> list[list[str]]:
        """
        Return nodes grouped by level. All nodes in the same level can run
        concurrently (no intra-level dependencies). Level 0 = no upstream deps.
        """
        return self._kahn_sort()

    def _kahn_sort(self) -> list[list[str]]:
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        successors: dict[str, list[str]] = defaultdict(list)

        for edge in self.edges:
            in_degree[edge.target_node_id] += 1
            successors[edge.source_node_id].append(edge.target_node_id)

        queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
        levels: list[list[str]] = []

        while queue:
            level = list(queue)
            queue.clear()
            levels.append(level)
            for nid in level:
                for successor in successors[nid]:
                    in_degree[successor] -= 1
                    if in_degree[successor] == 0:
                        queue.append(successor)

        return levels
```

- [ ] **Step 3: Write DAG validation tests**

Append to `tests/unit/dag/test_graph.py`:

```python
from restorax.dag.edge import Edge
from restorax.dag.graph import DAG
from restorax.core.exceptions import DAGValidationError, PortTypeMismatchError


class _FrameNode(Node):
    @property
    def input_ports(self):
        return [Port("frames", list)]

    @property
    def output_ports(self):
        return [Port("frames", list)]

    async def execute(self, ctx, inputs):
        return NodeResult(outputs={"frames": inputs.get("frames", [])})


class _IntNode(Node):
    @property
    def input_ports(self):
        return [Port("value", int)]

    @property
    def output_ports(self):
        return [Port("value", int)]

    async def execute(self, ctx, inputs):
        return NodeResult(outputs={"value": inputs.get("value", 0)})


def _make_linear_dag(n_nodes: int = 2) -> DAG:
    nodes = {f"n{i}": _FrameNode(id=f"n{i}", name=f"Node{i}") for i in range(n_nodes)}
    edges = [
        Edge(source_node_id=f"n{i}", source_port="frames",
             target_node_id=f"n{i+1}", target_port="frames")
        for i in range(n_nodes - 1)
    ]
    return DAG(id="test", name="Test", nodes=nodes, edges=edges)


def test_valid_dag_builds_without_error():
    dag = _make_linear_dag(3)
    assert len(dag.nodes) == 3


def test_unknown_source_node_raises():
    nodes = {"n0": _EchoNode(id="n0", name="N0")}
    edges = [Edge("ghost", "data", "n0", "data")]
    with pytest.raises(DAGValidationError, match="unknown source node"):
        DAG(id="t", name="t", nodes=nodes, edges=edges)


def test_unknown_port_raises():
    nodes = {"n0": _EchoNode(id="n0", name="N0"), "n1": _EchoNode(id="n1", name="N1")}
    edges = [Edge("n0", "nonexistent_port", "n1", "data")]
    with pytest.raises(DAGValidationError, match="no output port"):
        DAG(id="t", name="t", nodes=nodes, edges=edges)


def test_cycle_raises():
    nodes = {
        "a": _EchoNode(id="a", name="A"),
        "b": _EchoNode(id="b", name="B"),
    }
    edges = [
        Edge("a", "data", "b", "data"),
        Edge("b", "data", "a", "data"),
    ]
    with pytest.raises(DAGValidationError, match="cycle"):
        DAG(id="t", name="t", nodes=nodes, edges=edges)


def test_port_type_mismatch_raises():
    nodes = {
        "a": _FrameNode(id="a", name="A"),
        "b": _IntNode(id="b", name="B"),
    }
    edges = [Edge("a", "frames", "b", "value")]
    with pytest.raises(PortTypeMismatchError):
        DAG(id="t", name="t", nodes=nodes, edges=edges)


def test_topological_levels_linear():
    dag = _make_linear_dag(3)
    levels = dag.topological_levels()
    assert len(levels) == 3
    assert levels[0] == ["n0"]
    assert levels[1] == ["n1"]
    assert levels[2] == ["n2"]


def test_topological_levels_parallel_roots():
    nodes = {
        "a": _EchoNode(id="a", name="A"),
        "b": _EchoNode(id="b", name="B"),
        "c": _EchoNode(id="c", name="C"),
    }
    # a and b both feed c — a and b are independent (level 0)
    edges = [
        Edge("a", "data", "c", "data"),
    ]
    # Note: c only has one "data" input port, so only one edge to it.
    # Just test that a and b are in the same level.
    nodes2 = {
        "a": _EchoNode(id="a", name="A"),
        "b": _EchoNode(id="b", name="B"),
    }
    dag2 = DAG(id="t", name="t", nodes=nodes2, edges=[])
    levels2 = dag2.topological_levels()
    assert len(levels2) == 1
    assert set(levels2[0]) == {"a", "b"}
```

- [ ] **Step 4: Run tests**

```bash
conda run -n restorax python -m pytest tests/unit/dag/test_graph.py -q
```

Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add restorax/dag/edge.py restorax/dag/graph.py tests/unit/dag/test_graph.py
git commit -m "feat(dag): add Edge, DAG with cycle detection and topological sort"
```

---

## Task 4 — ExecutionContext and ProgressEmitter

**Files:**
- Create: `restorax/dag/context.py`

- [ ] **Step 1: Create `restorax/dag/context.py`**

```python
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from restorax.core.registry import ModelRegistry

logger = logging.getLogger(__name__)


class ProgressEmitter:
    """
    Publishes per-node, per-branch progress events to Redis pub/sub.
    Uses the same channel prefix as ProgressReporter so the existing
    WebSocket layer forwards events to the browser unchanged.
    """

    _CHANNEL_PREFIX = "restorax:job_progress:"

    def __init__(self, job_id: str, redis_url: str) -> None:
        self._job_id = job_id
        self._redis_url = redis_url
        self._redis: Any = None

    def _get_redis(self) -> Any:
        if self._redis is None:
            import redis as _redis
            self._redis = _redis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    def emit(
        self,
        node_id: str,
        progress: float,
        branch_index: int = 0,
        status: str = "running",
    ) -> None:
        payload = json.dumps({
            "job_id": self._job_id,
            "node_id": node_id,
            "branch_index": branch_index,
            "progress": round(progress, 4),
            "status": status,
        })
        try:
            self._get_redis().publish(f"{self._CHANNEL_PREFIX}{self._job_id}", payload)
        except Exception as exc:
            logger.warning("ProgressEmitter publish failed: %s", exc)


@dataclass
class ExecutionContext:
    """Per-run context passed to every node.execute() call."""

    run_id: str
    job_id: str
    work_dir: Path
    device: "torch.device"
    registry: "ModelRegistry"
    progress_emitter: ProgressEmitter
    logger: Any  # structlog.BoundLogger
    config: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 2: Write test**

Create `tests/unit/dag/test_context.py`:

```python
from unittest.mock import MagicMock, patch
from restorax.dag.context import ProgressEmitter


def test_progress_emitter_publishes_to_redis():
    with patch("redis.from_url") as mock_from_url:
        mock_redis = MagicMock()
        mock_from_url.return_value = mock_redis

        emitter = ProgressEmitter(job_id="job-123", redis_url="redis://localhost:6379/0")
        emitter.emit(node_id="restore_1", progress=0.5, branch_index=1, status="running")

        mock_redis.publish.assert_called_once()
        channel, payload_str = mock_redis.publish.call_args[0]
        assert "job-123" in channel

        import json
        payload = json.loads(payload_str)
        assert payload["node_id"] == "restore_1"
        assert payload["branch_index"] == 1
        assert payload["progress"] == 0.5
        assert payload["status"] == "running"


def test_progress_emitter_swallows_redis_errors():
    with patch("redis.from_url") as mock_from_url:
        mock_redis = MagicMock()
        mock_redis.publish.side_effect = ConnectionError("redis down")
        mock_from_url.return_value = mock_redis

        emitter = ProgressEmitter(job_id="job-xyz", redis_url="redis://localhost:6379/0")
        emitter.emit("n1", 0.3)  # must not raise
```

- [ ] **Step 3: Run test**

```bash
conda run -n restorax python -m pytest tests/unit/dag/test_context.py -q
```

Expected: 2 passed

- [ ] **Step 4: Commit**

```bash
git add restorax/dag/context.py tests/unit/dag/test_context.py
git commit -m "feat(dag): add ExecutionContext and ProgressEmitter"
```

---

## Task 5 — DAGRun and DAGExecutor

**Files:**
- Create: `restorax/dag/executor.py`

- [ ] **Step 1: Create `restorax/dag/executor.py`**

```python
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
                run_state_key = f"{node.id}"
                ctx.progress_emitter.emit(node.id, 0.0, status="retrying")

            try:
                return await node.execute(ctx, inputs)
            except policy.retry_on as exc:
                last_exc = exc
                if attempt < policy.max_retries:
                    ctx.logger.warning(  # type: ignore[attr-defined]
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
```

- [ ] **Step 2: Write executor tests**

Create `tests/unit/dag/test_executor.py`:

```python
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
```

- [ ] **Step 3: Run tests**

```bash
conda run -n restorax python -m pytest tests/unit/dag/test_executor.py -q
```

Expected: 4 passed

- [ ] **Step 4: Commit**

```bash
git add restorax/dag/executor.py tests/unit/dag/test_executor.py
git commit -m "feat(dag): add DAGRun and DAGExecutor with retry and downstream skip"
```

---

## Task 6 — DAG serializer

**Files:**
- Create: `restorax/dag/serializer.py`

- [ ] **Step 1: Create `restorax/dag/serializer.py`**

```python
from __future__ import annotations

from typing import Any

from restorax.core.exceptions import DAGValidationError
from restorax.dag.edge import Edge
from restorax.dag.graph import DAG
from restorax.dag.node import Node

_NODE_REGISTRY: dict[str, type[Node]] = {}


def dag_node_type(type_id: str):
    """Class decorator that registers a Node subclass under a string type ID."""
    def decorator(cls: type[Node]) -> type[Node]:
        _NODE_REGISTRY[type_id] = cls
        cls._dag_type_id = type_id  # type: ignore[attr-defined]
        return cls
    return decorator


class DAGSerializer:
    """Converts DAG ↔ plain dict (JSON-safe). Registered node types only."""

    @staticmethod
    def to_dict(dag: DAG) -> dict[str, Any]:
        return {
            "schema_type": "dag",
            "id": dag.id,
            "name": dag.name,
            "nodes": [
                {
                    "type": getattr(node, "_dag_type_id", type(node).__name__),
                    "id": node.id,
                    "name": node.name,
                    **node.to_dict(),
                }
                for node in dag.nodes.values()
            ],
            "edges": [
                {
                    "source_node_id": e.source_node_id,
                    "source_port": e.source_port,
                    "target_node_id": e.target_node_id,
                    "target_port": e.target_port,
                }
                for e in dag.edges
            ],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> DAG:
        nodes: dict[str, Node] = {}
        for node_data in data.get("nodes", []):
            type_id = node_data.get("type")
            if type_id not in _NODE_REGISTRY:
                raise DAGValidationError(
                    f"Unknown node type '{type_id}'. "
                    f"Registered types: {list(_NODE_REGISTRY)}"
                )
            cls = _NODE_REGISTRY[type_id]
            nodes[node_data["id"]] = cls.from_dict(node_data)

        edges = [Edge(**e) for e in data.get("edges", [])]
        return DAG(
            id=data["id"],
            name=data["name"],
            nodes=nodes,
            edges=edges,
        )
```

- [ ] **Step 2: Write serializer tests**

Create `tests/unit/dag/test_serializer.py`:

```python
from __future__ import annotations

from restorax.dag.edge import Edge
from restorax.dag.graph import DAG
from restorax.dag.node import Node, NodeResult, Port
from restorax.dag.serializer import DAGSerializer, dag_node_type
from restorax.core.exceptions import DAGValidationError
import pytest


@dag_node_type("_test_echo")
class _SerEchoNode(Node):
    def __init__(self, id: str, name: str, label: str = "", **kwargs):
        super().__init__(id, name)
        self.label = label

    @property
    def input_ports(self):
        return [Port("data")]

    @property
    def output_ports(self):
        return [Port("data")]

    async def execute(self, ctx, inputs):
        return NodeResult(outputs={"data": inputs.get("data")})

    def to_dict(self):
        return {"label": self.label}

    @classmethod
    def from_dict(cls, data):
        return cls(id=data["id"], name=data["name"], label=data.get("label", ""))


def _make_two_node_dag() -> DAG:
    nodes = {
        "src": _SerEchoNode(id="src", name="Source", label="hello"),
        "dst": _SerEchoNode(id="dst", name="Dest", label="world"),
    }
    edges = [Edge("src", "data", "dst", "data")]
    return DAG(id="ser-test", name="Serialization Test", nodes=nodes, edges=edges)


def test_roundtrip_preserves_structure():
    dag = _make_two_node_dag()
    data = DAGSerializer.to_dict(dag)
    restored = DAGSerializer.from_dict(data)

    assert restored.id == dag.id
    assert restored.name == dag.name
    assert set(restored.nodes) == {"src", "dst"}
    assert restored.nodes["src"].label == "hello"  # type: ignore[attr-defined]
    assert len(restored.edges) == 1
    assert restored.edges[0].source_node_id == "src"


def test_schema_type_is_dag():
    data = DAGSerializer.to_dict(_make_two_node_dag())
    assert data["schema_type"] == "dag"


def test_unknown_node_type_raises():
    data = {
        "id": "t", "name": "t",
        "nodes": [{"type": "totally_unknown", "id": "x", "name": "X"}],
        "edges": [],
    }
    with pytest.raises(DAGValidationError, match="Unknown node type"):
        DAGSerializer.from_dict(data)
```

- [ ] **Step 3: Run tests**

```bash
conda run -n restorax python -m pytest tests/unit/dag/test_serializer.py -q
```

Expected: 3 passed

- [ ] **Step 4: Commit**

```bash
git add restorax/dag/serializer.py tests/unit/dag/test_serializer.py
git commit -m "feat(dag): add DAGSerializer with type registry and round-trip serialization"
```

---

## Task 7 — Video I/O node types

**Files:**
- Create: `restorax/dag/nodes/__init__.py`
- Create: `restorax/dag/nodes/io.py`

- [ ] **Step 1: Create `restorax/dag/nodes/__init__.py`** (empty)

- [ ] **Step 2: Create `restorax/dag/nodes/io.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
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

        # Infer output dimensions from first frame of first chunk
        first_frame = chunks[0][0] if chunks and chunks[0] else None
        out_h, out_w = (first_frame.shape[:2] if first_frame is not None else (meta.height, meta.width))

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
```

- [ ] **Step 3: Write tests**

Create `tests/unit/dag/test_nodes.py`:

```python
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from restorax.dag.nodes.io import VideoInputNode, VideoOutputNode


def test_video_input_node_ports():
    node = VideoInputNode(id="vi", name="Input", video_path="/tmp/video.mp4")
    assert node.input_ports == []
    assert any(p.name == "chunks" for p in node.output_ports)
    assert any(p.name == "meta" for p in node.output_ports)


def test_video_input_node_roundtrip():
    node = VideoInputNode(id="vi", name="Input", video_path="/tmp/v.mp4", chunk_size=8)
    data = {"type": "video_input", **{"id": node.id, "name": node.name}, **node.to_dict()}
    restored = VideoInputNode.from_dict(data)
    assert restored.chunk_size == 8
    assert restored.video_path == "/tmp/v.mp4"


def test_video_output_node_ports():
    node = VideoOutputNode(id="vo", name="Output")
    assert any(p.name == "chunks" for p in node.input_ports)
    assert any(p.name == "output_path" for p in node.output_ports)
```

- [ ] **Step 4: Run tests**

```bash
conda run -n restorax python -m pytest tests/unit/dag/test_nodes.py -q
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add restorax/dag/nodes/__init__.py restorax/dag/nodes/io.py tests/unit/dag/test_nodes.py
git commit -m "feat(dag): add VideoInputNode and VideoOutputNode"
```

---

## Task 8 — RestoreNode and AudioRestoreNode

**Files:**
- Create: `restorax/dag/nodes/restore.py`

- [ ] **Step 1: Create `restorax/dag/nodes/restore.py`**

```python
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
```

- [ ] **Step 2: Add tests**

Append to `tests/unit/dag/test_nodes.py`:

```python
from unittest.mock import MagicMock
from pathlib import Path
from restorax.dag.nodes.restore import RestoreNode
from restorax.dag.context import ExecutionContext, ProgressEmitter
import asyncio


def _make_ctx_for_restore(restorer_mock):
    registry = MagicMock()
    registry.get.return_value = restorer_mock
    emitter = MagicMock(spec=ProgressEmitter)
    return ExecutionContext(
        run_id="r1", job_id="j1", work_dir=Path("/tmp"),
        device=MagicMock(), registry=registry,
        progress_emitter=emitter, logger=MagicMock(),
    )


def test_restore_node_calls_process_frame():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    caps = MagicMock()
    caps.requires_temporal = False
    restorer = MagicMock()
    restorer.capabilities = caps
    restorer.process_frame.return_value = frame

    node = RestoreNode(id="r1", name="R1", restorer_name="real_esrgan")
    ctx = _make_ctx_for_restore(restorer)
    chunks = [[frame, frame]]
    result = asyncio.run(node.execute(ctx, {"chunks": chunks}))

    assert "chunks" in result.outputs
    assert restorer.process_frame.call_count == 2


def test_restore_node_roundtrip():
    node = RestoreNode(id="r1", name="R1", restorer_name="waifu2x", params_dict={"scale": 2})
    data = {"type": "restore", "id": node.id, "name": node.name, **node.to_dict()}
    restored = RestoreNode.from_dict(data)
    assert restored.restorer_name == "waifu2x"
    assert restored.params_dict["scale"] == 2
```

- [ ] **Step 3: Run tests**

```bash
conda run -n restorax python -m pytest tests/unit/dag/test_nodes.py -q
```

Expected: 5 passed

- [ ] **Step 4: Commit**

```bash
git add restorax/dag/nodes/restore.py tests/unit/dag/test_nodes.py
git commit -m "feat(dag): add RestoreNode and AudioRestoreNode"
```

---

## Task 9 — ParallelNode and MergeNode

**Files:**
- Create: `restorax/dag/nodes/parallel.py`
- Create: `restorax/dag/nodes/merge.py`

- [ ] **Step 1: Create `restorax/dag/nodes/parallel.py`**

```python
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from restorax.dag.context import ExecutionContext, ProgressEmitter
from restorax.dag.node import Node, NodeResult, Port
from restorax.dag.serializer import dag_node_type


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

        for branch_idx, branch in enumerate(self.branches):
            # Deep-copy chunks so each branch starts from the same input
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
                    ctx.progress_emitter.emit(
                        self.id, overall, branch_index=branch_idx
                    )

                branch_chunks = out_chunks

            branch_outputs.append(branch_chunks)

        return NodeResult(outputs={"branch_outputs": branch_outputs, "meta": meta})

    def to_dict(self) -> dict[str, Any]:
        return {"branches": [b.to_dict() for b in self.branches]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParallelNode:
        return cls(
            id=data["id"],
            name=data["name"],
            branches=[BranchConfig.from_dict(b) for b in data.get("branches", [])],
        )
```

- [ ] **Step 2: Create `restorax/dag/nodes/merge.py`**

```python
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
    strategy='blend'  → equal-weight pixel average across all branches per frame.
    strategy='select' → pass through branch at select_index unchanged.
    select_index is set dynamically via POST /jobs/{id}/merge before execution,
    or defaults to 0.
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

        if self.strategy == "select":
            idx = min(self.select_index, len(branch_outputs) - 1)
            merged = branch_outputs[idx]
        else:
            # blend: per-frame pixel average
            n_branches = len(branch_outputs)
            n_chunks = len(branch_outputs[0])
            merged: list[list[np.ndarray]] = []
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
```

- [ ] **Step 3: Add tests**

Append to `tests/unit/dag/test_nodes.py`:

```python
import asyncio
from restorax.dag.nodes.parallel import ParallelNode, BranchConfig
from restorax.dag.nodes.merge import MergeNode


def test_merge_node_blend_averages_frames():
    frame_a = np.full((4, 4, 3), 100, dtype=np.uint8)
    frame_b = np.full((4, 4, 3), 200, dtype=np.uint8)
    # branch_outputs: 2 branches, 1 chunk each, 1 frame each
    branch_outputs = [[[frame_a]], [[frame_b]]]
    node = MergeNode(id="m1", name="Merge", strategy="blend")
    result = asyncio.run(node.execute(MagicMock(), {"branch_outputs": branch_outputs}))
    merged_frame = result.outputs["chunks"][0][0]
    assert merged_frame.dtype == np.uint8
    np.testing.assert_allclose(merged_frame, 150, atol=1)


def test_merge_node_select_picks_correct_branch():
    frame_a = np.full((4, 4, 3), 10, dtype=np.uint8)
    frame_b = np.full((4, 4, 3), 99, dtype=np.uint8)
    branch_outputs = [[[frame_a]], [[frame_b]]]
    node = MergeNode(id="m1", name="Merge", strategy="select", select_index=1)
    result = asyncio.run(node.execute(MagicMock(), {"branch_outputs": branch_outputs}))
    np.testing.assert_array_equal(result.outputs["chunks"][0][0], frame_b)


def test_branch_config_roundtrip():
    bc = BranchConfig(name="branch_a", restorer_steps=[("real_esrgan", {"scale": 4})])
    restored = BranchConfig.from_dict(bc.to_dict())
    assert restored.name == "branch_a"
    assert restored.restorer_steps[0][0] == "real_esrgan"
```

- [ ] **Step 4: Run tests**

```bash
conda run -n restorax python -m pytest tests/unit/dag/test_nodes.py -q
```

Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add restorax/dag/nodes/parallel.py restorax/dag/nodes/merge.py tests/unit/dag/test_nodes.py
git commit -m "feat(dag): add ParallelNode (fan-out) and MergeNode (blend/select)"
```

---

## Task 10 — MapNode, ChoiceNode, PassNode

**Files:**
- Create: `restorax/dag/nodes/map_node.py`
- Create: `restorax/dag/nodes/control.py`

- [ ] **Step 1: Create `restorax/dag/nodes/map_node.py`**

```python
from __future__ import annotations

from typing import Any

from restorax.dag.context import ExecutionContext
from restorax.dag.node import Node, NodeResult, Port
from restorax.dag.serializer import dag_node_type


@dag_node_type("map")
class MapNode(Node):
    """
    Apply a sub-DAG to each item in a list sequentially.
    Analogous to AWS Step Functions Map state.
    Useful for batch-processing multiple video clips with the same pipeline.
    """

    def __init__(self, id: str, name: str, sub_dag_dict: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(id, name)
        self.sub_dag_dict: dict[str, Any] = sub_dag_dict or {}

    @property
    def input_ports(self) -> list[Port]:
        return [Port("items", list)]

    @property
    def output_ports(self) -> list[Port]:
        return [Port("results", list)]

    async def execute(self, ctx: ExecutionContext, inputs: dict[str, Any]) -> NodeResult:
        from restorax.dag.executor import DAGExecutor
        from restorax.dag.serializer import DAGSerializer
        import uuid

        items: list[Any] = inputs["items"]
        results: list[Any] = []

        if not self.sub_dag_dict:
            return NodeResult(outputs={"results": items})

        sub_dag = DAGSerializer.from_dict(self.sub_dag_dict)

        for i, item in enumerate(items):
            item_ctx = ExecutionContext(
                run_id=f"{ctx.run_id}-map-{i}",
                job_id=ctx.job_id,
                work_dir=ctx.work_dir / f"map_{i}",
                device=ctx.device,
                registry=ctx.registry,
                progress_emitter=ctx.progress_emitter,
                logger=ctx.logger,
                config=ctx.config,
            )
            item_ctx.work_dir.mkdir(parents=True, exist_ok=True)
            root_node_id = sub_dag.topological_levels()[0][0]
            run = await DAGExecutor().execute(
                sub_dag, item_ctx, initial_inputs={root_node_id: {"data": item}}
            )
            last_node_id = sub_dag.topological_levels()[-1][0]
            last_result = run.node_results.get(last_node_id)
            results.append(last_result.outputs.get("data") if last_result else None)
            ctx.progress_emitter.emit(self.id, (i + 1) / max(len(items), 1))

        return NodeResult(outputs={"results": results})

    def to_dict(self) -> dict[str, Any]:
        return {"sub_dag_dict": self.sub_dag_dict}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MapNode:
        return cls(id=data["id"], name=data["name"], sub_dag_dict=data.get("sub_dag_dict"))
```

- [ ] **Step 2: Create `restorax/dag/nodes/control.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from restorax.dag.context import ExecutionContext
from restorax.dag.node import Node, NodeResult, Port
from restorax.dag.serializer import dag_node_type


@dataclass
class ChoiceRule:
    field: str                    # key to read from input "meta" dict
    operator: Literal["eq", "gt", "lt", "gte", "lte", "ne"]
    value: Any
    branch_index: int


@dag_node_type("choice")
class ChoiceNode(Node):
    """
    Conditional routing: evaluates rules against input metadata,
    outputs branch_index indicating which downstream path to activate.
    Analogous to AWS Step Functions Choice state.
    """

    def __init__(self, id: str, name: str, rules: list[ChoiceRule] | None = None, default_branch: int = 0, **kwargs: Any) -> None:
        super().__init__(id, name)
        self.rules: list[ChoiceRule] = rules or []
        self.default_branch = default_branch

    @property
    def input_ports(self) -> list[Port]:
        return [Port("meta", object)]

    @property
    def output_ports(self) -> list[Port]:
        return [Port("branch_index", int)]

    async def execute(self, ctx: ExecutionContext, inputs: dict[str, Any]) -> NodeResult:
        meta = inputs.get("meta") or {}
        meta_dict = meta if isinstance(meta, dict) else vars(meta) if hasattr(meta, "__dict__") else {}

        ops = {
            "eq": lambda a, b: a == b,
            "ne": lambda a, b: a != b,
            "gt": lambda a, b: a > b,
            "lt": lambda a, b: a < b,
            "gte": lambda a, b: a >= b,
            "lte": lambda a, b: a <= b,
        }

        for rule in self.rules:
            field_val = meta_dict.get(rule.field)
            if field_val is not None and ops[rule.operator](field_val, rule.value):
                return NodeResult(outputs={"branch_index": rule.branch_index})

        return NodeResult(outputs={"branch_index": self.default_branch})

    def to_dict(self) -> dict[str, Any]:
        return {
            "rules": [{"field": r.field, "operator": r.operator, "value": r.value, "branch_index": r.branch_index} for r in self.rules],
            "default_branch": self.default_branch,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChoiceNode:
        rules = [ChoiceRule(**r) for r in data.get("rules", [])]
        return cls(id=data["id"], name=data["name"], rules=rules, default_branch=data.get("default_branch", 0))


@dag_node_type("pass")
class PassNode(Node):
    """Identity node. Passes all inputs through as outputs unchanged."""

    @property
    def input_ports(self) -> list[Port]:
        return [Port("data")]

    @property
    def output_ports(self) -> list[Port]:
        return [Port("data")]

    async def execute(self, ctx: ExecutionContext, inputs: dict[str, Any]) -> NodeResult:
        return NodeResult(outputs={"data": inputs.get("data")})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PassNode:
        return cls(id=data["id"], name=data["name"])
```

- [ ] **Step 3: Add tests**

Append to `tests/unit/dag/test_nodes.py`:

```python
from restorax.dag.nodes.control import ChoiceNode, ChoiceRule, PassNode


def test_pass_node_echoes_input():
    node = PassNode(id="p1", name="Pass")
    result = asyncio.run(node.execute(MagicMock(), {"data": "hello"}))
    assert result.outputs["data"] == "hello"


def test_choice_node_matches_rule():
    rule = ChoiceRule(field="width", operator="gt", value=1920, branch_index=1)
    node = ChoiceNode(id="c1", name="Choice", rules=[rule], default_branch=0)
    result = asyncio.run(node.execute(MagicMock(), {"meta": {"width": 3840}}))
    assert result.outputs["branch_index"] == 1


def test_choice_node_default_when_no_match():
    node = ChoiceNode(id="c1", name="Choice", rules=[], default_branch=2)
    result = asyncio.run(node.execute(MagicMock(), {"meta": {}}))
    assert result.outputs["branch_index"] == 2
```

- [ ] **Step 4: Run all dag node tests**

```bash
conda run -n restorax python -m pytest tests/unit/dag/ -q
```

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add restorax/dag/nodes/map_node.py restorax/dag/nodes/control.py tests/unit/dag/test_nodes.py
git commit -m "feat(dag): add MapNode, ChoiceNode, PassNode"
```

---

## Task 11 — DB migration and run_dag_job Celery task

**Files:**
- Modify: `restorax/db/models.py`
- Modify: `restorax/tasks/job_tasks.py`

- [ ] **Step 1: Add dag_run column to JobModel**

In `restorax/db/models.py`, add one line to `JobModel` after the `metrics` column:

```python
    dag_run: Mapped[dict | None] = mapped_column(JSON, nullable=True)
```

- [ ] **Step 2: Add run_dag_job to job_tasks.py**

Add these imports near the top of `restorax/tasks/job_tasks.py` (after existing imports):

```python
import uuid as _uuid
```

Then add this task function after the existing `run_job` task:

```python
@celery_app.task(bind=True, base=JobTask, name="restorax.tasks.job_tasks.run_dag_job")
def run_dag_job(
    self: Task,
    job_id: str,
    dag_id: str,
    input_path: str,
    output_path: str,
) -> dict:
    """
    Execute a DAG pipeline on a video file.
    Loads the DAG from DB, builds ExecutionContext, runs DAGExecutor.
    No Celery canvas — all orchestration is internal.
    """
    import asyncio as _asyncio

    from restorax.dag import DAGExecutor
    from restorax.dag.context import ExecutionContext, ProgressEmitter
    from restorax.dag.serializer import DAGSerializer
    from restorax.dag.nodes import io, restore, parallel, merge, map_node, control  # noqa: F401 — registers node types

    reporter = ProgressReporter(job_id)
    _update_job_db(job_id, status="running", started_at=datetime.now(timezone.utc))
    reporter.update(0.0, status="running")

    device_str = settings.device
    if device_str.startswith("cuda") and "CUDA_VISIBLE_DEVICES" in os.environ:
        device_str = "cuda:0"
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

    # Load DAG definition from DB
    async def _load_dag():
        from restorax.db.repositories.pipeline_repo import PipelineRepository
        from restorax.db.session import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            repo = PipelineRepository(session)
            from restorax.core.exceptions import PipelineConfigError
            try:
                template = await repo.get(dag_id)
            except PipelineConfigError:
                raise ValueError(f"DAG '{dag_id}' not found in database")
        return DAGSerializer.from_dict(template.config)

    dag = _asyncio.run(_load_dag())

    work_dir = Path(output_path).parent / "dag_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    emitter = ProgressEmitter(job_id=job_id, redis_url=settings.redis_url)
    ctx = ExecutionContext(
        run_id=str(_uuid.uuid4()),
        job_id=job_id,
        work_dir=work_dir,
        device=device,
        registry=_get_registry(),
        progress_emitter=emitter,
        logger=logger,
        config={"input_path": input_path, "output_path": output_path},
    )

    dag_run = _asyncio.run(DAGExecutor().execute(dag, ctx))

    # Persist DAGRun state into the job record
    _update_job_db(job_id, dag_run=dag_run.to_dict())

    if not dag_run.succeeded:
        raise RuntimeError(dag_run.error or "DAG execution failed")

    _update_job_db(
        job_id, status="completed",
        progress=1.0, output_path=output_path,
        completed_at=datetime.now(timezone.utc),
    )
    reporter.complete(output_path)
    return {"output_path": output_path}
```

- [ ] **Step 3: Add dag_run to update_status in job_repo**

Read `restorax/db/repositories/job_repo.py` and check if `update_status` accepts `**kwargs` or explicit fields. If it uses `**kwargs`, no change needed. If it uses explicit fields, add `dag_run: dict | None = None` and handle it.

Run a quick check:
```bash
grep -n "dag_run\|def update_status" /mnt/f/wsl_repo/restorax/restorax/db/repositories/job_repo.py
```

If `update_status` does not accept `dag_run`, add it as an accepted kwarg by checking if the method uses `**kwargs` to set attributes. The pattern is:

```python
# In update_status, add to the setattr loop or explicit assignment:
if "dag_run" in kwargs and kwargs["dag_run"] is not None:
    job.dag_run = kwargs["dag_run"]
```

- [ ] **Step 4: Write test for run_dag_job**

Add to `tests/unit/test_job_tasks.py` (create if not exists):

```python
"""Tests for DAG job task integration."""
from unittest.mock import MagicMock, patch
import pytest


def test_run_dag_job_task_is_registered():
    from restorax.tasks.job_tasks import run_dag_job
    assert run_dag_job.name == "restorax.tasks.job_tasks.run_dag_job"
```

- [ ] **Step 5: Run test**

```bash
conda run -n restorax python -m pytest tests/unit/test_job_tasks.py -q
```

Expected: 1 passed

- [ ] **Step 6: Commit**

```bash
git add restorax/db/models.py restorax/tasks/job_tasks.py tests/unit/test_job_tasks.py
git commit -m "feat(dag): add dag_run DB column and run_dag_job Celery task"
```

---

## Task 12 — API schemas and /pipelines/dag endpoints

**Files:**
- Modify: `restorax/api/schemas/pipeline.py`
- Modify: `restorax/api/routers/pipelines.py`

- [ ] **Step 1: Add DAG schemas to pipeline.py**

Append to `restorax/api/schemas/pipeline.py`:

```python
class DAGCreateRequest(BaseModel):
    id: str = Field(..., description="Unique DAG ID slug (e.g. 'film_restoration_dag')")
    name: str
    description: str = ""
    config: dict = Field(..., description="Serialised DAG dict from DAGSerializer.to_dict()")


class DAGResponse(BaseModel):
    id: str
    name: str
    description: str
    config: dict
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
```

- [ ] **Step 2: Add DAG routes to pipelines.py**

Append to `restorax/api/routers/pipelines.py` (after existing DELETE route):

```python
# ── DAG endpoints ─────────────────────────────────────────────────────────────

from restorax.api.schemas.pipeline import DAGCreateRequest, DAGResponse
from restorax.dag.serializer import DAGSerializer
from restorax.core.exceptions import DAGValidationError


@router.post("/dag", response_model=DAGResponse, status_code=status.HTTP_201_CREATED, tags=["dag"])
async def create_dag(
    req: DAGCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> DAGResponse:
    """Create a DAG pipeline. Config must be a valid DAGSerializer.to_dict() output."""
    # Validate the DAG structure before persisting
    try:
        DAGSerializer.from_dict(req.config)
    except (DAGValidationError, Exception) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid DAG config: {exc}")

    repo = PipelineRepository(db)
    p = PipelineTemplateModel(
        id=req.id,
        name=req.name,
        description=req.description,
        config=req.config,  # already has schema_type: "dag"
    )
    try:
        created = await repo.create(p)
    except Exception:
        raise HTTPException(status_code=409, detail=f"DAG '{req.id}' already exists")
    return DAGResponse.model_validate(created)


@router.get("/dag/{dag_id}", response_model=DAGResponse, tags=["dag"])
async def get_dag(dag_id: str, db: AsyncSession = Depends(get_db)) -> DAGResponse:
    repo = PipelineRepository(db)
    try:
        p = await repo.get(dag_id)
    except PipelineConfigError:
        raise HTTPException(status_code=404, detail=f"DAG '{dag_id}' not found")
    return DAGResponse.model_validate(p)
```

- [ ] **Step 3: Write API tests**

Create `tests/unit/test_dag_api.py`:

```python
"""Tests for DAG pipeline API endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from restorax.dag.nodes.control import PassNode
from restorax.dag.serializer import DAGSerializer
from restorax.dag.graph import DAG
from restorax.dag.edge import Edge


def _minimal_dag_config() -> dict:
    """Build a valid 1-node DAG config for API testing."""
    from restorax.dag.nodes.control import PassNode
    dag = DAG(
        id="test-dag",
        name="Test DAG",
        nodes={"p1": PassNode(id="p1", name="Pass")},
        edges=[],
    )
    return DAGSerializer.to_dict(dag)


@pytest.fixture(scope="module")
def client():
    from restorax.api.app import app
    return TestClient(app)


def test_create_dag_returns_201(client):
    config = _minimal_dag_config()
    config["id"] = "api-test-dag-001"

    with patch("restorax.api.routers.pipelines.PipelineRepository") as MockRepo:
        mock_repo = AsyncMock()
        mock_model = MagicMock()
        mock_model.id = "api-test-dag-001"
        mock_model.name = "Test DAG"
        mock_model.description = ""
        mock_model.config = config
        from datetime import datetime, timezone
        mock_model.created_at = datetime.now(timezone.utc)
        mock_model.updated_at = datetime.now(timezone.utc)
        mock_repo.create.return_value = mock_model
        MockRepo.return_value = mock_repo

        resp = client.post("/pipelines/dag", json={
            "id": "api-test-dag-001",
            "name": "Test DAG",
            "config": config,
        })

    assert resp.status_code == 201
    assert resp.json()["id"] == "api-test-dag-001"


def test_create_dag_invalid_config_returns_422(client):
    resp = client.post("/pipelines/dag", json={
        "id": "bad-dag",
        "name": "Bad",
        "config": {"schema_type": "dag", "id": "bad", "name": "bad", "nodes": [{"type": "nonexistent_type", "id": "x", "name": "X"}], "edges": []},
    })
    assert resp.status_code == 422
```

- [ ] **Step 4: Run tests**

```bash
conda run -n restorax python -m pytest tests/unit/test_dag_api.py -q
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add restorax/api/schemas/pipeline.py restorax/api/routers/pipelines.py tests/unit/test_dag_api.py
git commit -m "feat(dag): add POST/GET /pipelines/dag API endpoints"
```

---

## Task 13 — /jobs extensions: dag_id, /branches, /merge

**Files:**
- Modify: `restorax/api/schemas/job.py`
- Modify: `restorax/api/routers/jobs.py`

- [ ] **Step 1: Add BranchResponse and MergeRequest schemas**

Read `restorax/api/schemas/job.py`, then append:

```python
class BranchInfo(BaseModel):
    branch_index: int
    name: str
    status: str
    progress: float
    output_path: str | None = None


class BranchListResponse(BaseModel):
    job_id: str
    branches: list[BranchInfo]


class MergeRequest(BaseModel):
    strategy: str = Field(..., description="'blend' or 'select'")
    branch_index: int = Field(0, description="Used when strategy='select'")
```

- [ ] **Step 2: Extend POST /jobs with dag_id**

In `restorax/api/routers/jobs.py`, make `pipeline_id` optional and add `dag_id`. Change the existing signature line:

```python
    pipeline_id: str = Form(...),
```
to:
```python
    pipeline_id: str | None = Form(None),
    dag_id: str | None = Form(None, description="DAG pipeline ID (alternative to pipeline_id)"),
```

Then add a guard after the form parameters (before the file save block):

```python
    if pipeline_id is None and dag_id is None:
        raise HTTPException(status_code=422, detail="Either pipeline_id or dag_id is required")
```

Only resolve the preset path when using a sequential pipeline:

```python
    preset_path = _resolve_preset(pipeline_id) if pipeline_id else None
```

Replace the existing `preset_path = _resolve_preset(pipeline_id)` line with the above.

After that, add:

```python
    # Dispatch appropriate Celery task
    if dag_id is not None:
        from restorax.tasks.job_tasks import run_dag_job
        task = run_dag_job.apply_async(
            kwargs={
                "job_id": job_id,
                "dag_id": dag_id,
                "input_path": str(input_path),
                "output_path": output_path,
            }
        )
    else:
        from restorax.tasks.job_tasks import run_job
        task = run_job.apply_async(
            kwargs={
                "job_id": job_id,
                "pipeline_preset_path": preset_path,
                "input_path": str(input_path),
                "output_path": output_path,
                "restore_audio": restore_audio,
            }
        )
```

Replace the existing `task = run_job.apply_async(...)` block with this conditional.

Also update `JobModel` creation to handle the case where `pipeline_id` is None (when `dag_id` is provided):

```python
    job_model = JobModel(
        id=job_id,
        status="queued",
        input_path=str(input_path),
        pipeline_id=dag_id or pipeline_id,   # store whichever was provided
        ...
    )
```

- [ ] **Step 3: Add /branches and /merge endpoints**

Append to `restorax/api/routers/jobs.py`:

```python
from restorax.api.schemas.job import BranchInfo, BranchListResponse, MergeRequest


@router.get("/{job_id}/branches", response_model=BranchListResponse)
async def get_job_branches(job_id: str, db: AsyncSession = Depends(get_db)) -> BranchListResponse:
    """Return per-branch status, progress, and output paths for a DAG job."""
    repo = JobRepository(db)
    try:
        job = await repo.get(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    dag_run: dict = job.dag_run or {}
    node_states: dict = dag_run.get("node_states", {})

    # Find parallel nodes from dag_run and extract branch info
    branches: list[BranchInfo] = []
    for node_id, state in node_states.items():
        if "parallel" in node_id.lower() or "branch" in node_id.lower():
            branches.append(BranchInfo(
                branch_index=len(branches),
                name=node_id,
                status=state,
                progress=1.0 if state == "succeeded" else 0.0,
            ))

    if not branches:
        # No DAG run data yet — return empty
        return BranchListResponse(job_id=job_id, branches=[])

    return BranchListResponse(job_id=job_id, branches=branches)


@router.post("/{job_id}/merge", response_model=JobResponse)
async def merge_job_branches(
    job_id: str,
    req: MergeRequest,
    db: AsyncSession = Depends(get_db),
) -> JobResponse:
    """
    Trigger merge for a DAG job that has completed its parallel branches.
    Updates the MergeNode strategy/select_index in the stored DAGRun and
    marks the job for re-execution of the merge step.
    """
    repo = JobRepository(db)
    try:
        job = await repo.get(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    dag_run: dict = job.dag_run or {}
    if not dag_run:
        raise HTTPException(status_code=409, detail="Job has no DAG run data (not a DAG job or not yet executed)")

    # Store merge decision in job metrics for the worker to pick up
    metrics = dict(job.metrics or {})
    metrics["merge_strategy"] = req.strategy
    metrics["merge_branch_index"] = req.branch_index
    await repo.update_status(job_id, metrics=metrics)

    job = await repo.get(job_id)
    return _to_response(job)
```

- [ ] **Step 4: Add tests**

Append to `tests/unit/test_dag_api.py`:

```python
def test_get_branches_for_nonexistent_job_returns_404(client):
    with patch("restorax.api.routers.jobs.JobRepository") as MockRepo:
        from restorax.core.exceptions import JobNotFoundError
        mock_repo = AsyncMock()
        mock_repo.get.side_effect = JobNotFoundError("not found")
        MockRepo.return_value = mock_repo
        resp = client.get("/jobs/no-such-job/branches")
    assert resp.status_code == 404


def test_merge_request_schema_validates():
    from restorax.api.schemas.job import MergeRequest
    req = MergeRequest(strategy="select", branch_index=2)
    assert req.strategy == "select"
    assert req.branch_index == 2

    req2 = MergeRequest(strategy="blend")
    assert req2.branch_index == 0  # default
```

- [ ] **Step 5: Run tests**

```bash
conda run -n restorax python -m pytest tests/unit/test_dag_api.py -q
```

Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add restorax/api/schemas/job.py restorax/api/routers/jobs.py tests/unit/test_dag_api.py
git commit -m "feat(dag): extend /jobs with dag_id, add /branches and /merge endpoints"
```

---

## Task 14 — WebSocket per-branch events + nodes package import

**Files:**
- Modify: `restorax/dag/nodes/__init__.py`
- Modify: `restorax/api/routers/ws.py`

- [ ] **Step 1: Register all node types in nodes/__init__.py**

```python
# Import all node modules to trigger @dag_node_type registration
from restorax.dag.nodes import control, io, map_node, merge, parallel, restore

__all__ = ["control", "io", "map_node", "merge", "parallel", "restore"]
```

- [ ] **Step 2: Update WebSocket to forward branch_index**

The existing WebSocket router already forwards all JSON from Redis pub/sub — it will forward `branch_index` automatically since it does `await websocket.send_json(data)` without filtering fields. No code change needed.

Verify by checking that the existing ws.py sends the full payload:

```bash
grep -n "send_json\|data" /mnt/f/wsl_repo/restorax/restorax/api/routers/ws.py
```

Expected output shows `await websocket.send_json(data)` where `data` is the full parsed JSON — confirming per-branch events flow through without modification.

- [ ] **Step 3: Run full unit suite**

```bash
conda run -n restorax python -m pytest tests/unit/ -q --tb=short
```

Expected: all pass (same count as before + new dag tests)

- [ ] **Step 4: Push**

```bash
git add restorax/dag/nodes/__init__.py
git commit -m "feat(dag): register all node types on package import"
git push origin main
```

---

## Task 15 — Full integration test and cleanup

**Files:**
- Create: `tests/unit/dag/test_integration.py`

- [ ] **Step 1: Write end-to-end DAG test with synthetic frames**

Create `tests/unit/dag/test_integration.py`:

```python
"""Integration test: full ParallelNode + MergeNode round-trip with synthetic frames."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from restorax.dag.context import ExecutionContext, ProgressEmitter
from restorax.dag.edge import Edge
from restorax.dag.executor import DAGExecutor
from restorax.dag.graph import DAG
from restorax.dag.node import NodeResult, Port
from restorax.dag.nodes.control import PassNode
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
        DAGExecutor().execute(dag, ctx, initial_inputs={"parallel": {"chunks": chunks, "meta": None}})
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
```

- [ ] **Step 2: Run integration tests**

```bash
conda run -n restorax python -m pytest tests/unit/dag/test_integration.py -v
```

Expected: 2 passed

- [ ] **Step 3: Run full unit suite final check**

```bash
conda run -n restorax python -m pytest tests/unit/ -q --tb=short
```

Expected: all pass

- [ ] **Step 4: Push**

```bash
git add tests/unit/dag/test_integration.py
git commit -m "test(dag): add end-to-end integration test for ParallelNode + MergeNode"
git push origin main
```

---

## Completion Criteria

- [ ] `POST /pipelines/dag` accepts a `DAGSerializer.to_dict()` payload, validates it, and returns 201
- [ ] `DAGValidationError` raised at `DAG()` construction for cycles, unknown nodes, unknown ports
- [ ] `RetryPolicy` causes failed nodes to retry up to `max_retries` times before failing
- [ ] Failed node marks all downstream nodes `SKIPPED`
- [ ] `ParallelNode` with 2 branches runs both branches sequentially, emits per-branch progress
- [ ] `MergeNode` blend produces pixel-average output; select passes through correct branch
- [ ] `POST /jobs` with `dag_id` dispatches `run_dag_job` Celery task
- [ ] `GET /jobs/{id}/branches` returns branch list
- [ ] `POST /jobs/{id}/merge` stores merge decision
- [ ] `dry_run` marks all nodes SUCCEEDED without calling `execute()`
- [ ] `DAGSerializer.to_dict()` / `from_dict()` round-trip preserves all node configs
- [ ] All new tests pass, existing suite unaffected
