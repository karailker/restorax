# Pipeline DAG Engine — Design Spec

**Date:** 2026-05-29
**Sub-project:** 2 — Pipeline DAG Engine
**Status:** Approved for implementation

---

## Overview

RestoraX ships a sequential pipeline runner (`PipelineRunner`) that processes video frames through an ordered list of restorers. Sub-project 2 extends this with a full DAG orchestration engine — `restorax/dag/` — that supports parallel branches, merge strategies, typed data ports, per-node retry policies, and per-branch progress reporting.

The engine is designed as an **internal module** now and architected for extraction as a standalone open-source Python package (`restorax-dag`) in the future. It draws from:

- **Dagster** — typed input/output ports, `ExecutionContext`, per-node retry policies
- **AWS Step Functions** — explicit state types (Task, Parallel, Map, Choice, Pass), JSON serialization, per-state Catch/Retry
- **Apache Airflow** — topological execution, DAG validation at build time, operator hierarchy

The existing `Pipeline` / `PipelineRunner` / YAML preset system is **unchanged**. The DAG engine is purely additive.

---

## Constraints

- No Celery canvas primitives (no `group`, `chain`, `chord`) — the DAG executor runs within a single `run_dag_job` Celery task
- Parallel branches execute **sequentially on one GPU** — true multi-GPU parallelism is a future phase
- Merge strategy: **equal-weight pixel average** for blend; user-selected index for select
- Backward compatible: existing `POST /jobs` with `pipeline_id` continues to work unchanged

---

## Core Abstractions

### `Port`

```python
@dataclass
class Port:
    name: str
    type_hint: type | None = None  # validated at DAG construction
```

Named, typed connection point declared on a node. Type compatibility is checked when edges are added to the DAG — mismatched port types raise `DAGValidationError` at build time, never at runtime.

### `Node` (ABC)

```python
class Node(ABC):
    id: str
    name: str
    input_ports: list[Port]
    output_ports: list[Port]
    retry_policy: RetryPolicy

    @abstractmethod
    async def execute(self, ctx: ExecutionContext, inputs: dict[str, Any]) -> NodeResult:
        ...
```

Stateless — all execution state lives in `DAGRun`, not in the node. Nodes are reusable across runs.

### `RetryPolicy`

```python
@dataclass
class RetryPolicy:
    max_retries: int = 0
    delay_seconds: float = 1.0
    backoff: Literal["fixed", "exponential"] = "fixed"
    retry_on: tuple[type[Exception], ...] = (Exception,)
```

Per-node retry configuration. The executor applies this transparently — nodes do not implement retry logic themselves.

### `Edge`

```python
@dataclass
class Edge:
    source_node_id: str
    source_port: str
    target_node_id: str
    target_port: str
```

Directed connection from one node's output port to another's input port.

### `DAG`

```python
@dataclass(frozen=True)
class DAG:
    id: str
    name: str
    nodes: dict[str, Node]
    edges: list[Edge]
```

Immutable. Validates on construction:
1. Cycle detection via topological sort (raises `DAGValidationError` on cycle)
2. All referenced node IDs exist
3. All referenced port names exist on their nodes
4. Source/target port type hints are compatible (if both declared)

`dag.topological_levels() -> list[list[str]]` returns nodes grouped by execution level — nodes in the same level have no dependency between them (used by executor).

### `NodeState`

```python
class NodeState(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCEEDED = "succeeded"
    FAILED    = "failed"
    SKIPPED   = "skipped"
    RETRYING  = "retrying"
```

### `NodeResult`

```python
@dataclass
class NodeResult:
    outputs: dict[str, Any]          # keyed by output port name
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
```

### `DAGRun`

```python
@dataclass
class DAGRun:
    run_id: str
    dag_id: str
    job_id: str
    node_states: dict[str, NodeState]
    node_results: dict[str, NodeResult]
    started_at: datetime
    completed_at: datetime | None = None
    failed_node_id: str | None = None
    error: str | None = None
```

One `DAGRun` per job execution. Persisted in the `JobModel.metrics` JSON column so it survives worker restarts.

### `ExecutionContext`

```python
@dataclass
class ExecutionContext:
    run_id: str
    job_id: str
    work_dir: Path          # base dir for intermediate outputs
    device: torch.device
    registry: ModelRegistry
    progress_emitter: ProgressEmitter
    logger: structlog.BoundLogger
    config: dict[str, Any]  # DAG-level runtime config (overrides)
```

Passed to every `node.execute()` call. Nodes access GPU, registry, and progress through the context — never through globals. This is the pattern that makes the engine extractable: swap out the RestoraX-specific fields and the rest of the engine is generic.

### `ProgressEmitter`

```python
class ProgressEmitter:
    def emit(
        self,
        node_id: str,
        progress: float,          # 0.0–1.0
        branch_index: int = 0,
        status: str = "running",
    ) -> None: ...
```

Publishes `{run_id, node_id, branch_index, progress, status}` to Redis pub/sub channel `job:{job_id}:progress`. The existing WebSocket layer (`ws.py`) is extended to forward per-branch events to the client.

### `DAGExecutor`

```python
class DAGExecutor:
    async def execute(self, dag: DAG, ctx: ExecutionContext) -> DAGRun:
        ...

    async def dry_run(self, dag: DAG, ctx: ExecutionContext) -> DAGRun:
        """Validate data flow without executing nodes."""
        ...
```

Algorithm:
1. Build `DAGRun` with all nodes in `PENDING`
2. Compute topological levels
3. For each level:
   - For each node in level (in level-order, sequential):
     - Collect inputs from upstream `NodeResult.outputs` via edges
     - Set node state → `RUNNING`, emit progress event
     - Call `node.execute(ctx, inputs)` with retry wrapper
     - On success: set state → `SUCCEEDED`, store `NodeResult`
     - On failure after retries: set state → `FAILED`, mark run failed, stop
4. Return completed `DAGRun`

Nodes within the same topological level that have no data dependency between them are candidates for true parallelism in a future multi-GPU phase — the executor API does not need to change, only the execution strategy within a level.

---

## Built-in Node Types

### `VideoInputNode`
- **Input ports:** none
- **Output ports:** `chunks: list[list[np.ndarray]]`, `meta: VideoMeta`
- Reads video using `VideoReader`, yields overlapping frame chunks (chunk_size, chunk_overlap from DAG-level config)

### `VideoOutputNode`
- **Input ports:** `chunks: list[list[np.ndarray]]`, `meta: VideoMeta`, `fps: float`
- **Output ports:** `output_path: str`
- Writes processed frames using `VideoWriter`

### `RestoreNode`
- **Input ports:** `chunks: list[list[np.ndarray]]`
- **Output ports:** `chunks: list[list[np.ndarray]]`
- **Config:** `restorer_name: str`, `params: RestorerParams`
- Loads restorer from `ctx.registry`, applies per-chunk, emits progress via `ctx.progress_emitter`

### `ParallelNode`
- **Input ports:** `chunks: list[list[np.ndarray]]`
- **Output ports:** `branch_outputs: list[list[list[np.ndarray]]]` (one per branch)
- **Config:** `branch_dags: list[DAG]` — each sub-DAG is a linear chain of `RestoreNode`s
- Runs each branch sequentially using a nested `DAGExecutor` call; outputs saved to `{work_dir}/branch_{i}/`
- Emits per-branch progress with `branch_index=i`

### `MergeNode`
- **Input ports:** `branch_outputs: list[list[list[np.ndarray]]]`
- **Output ports:** `chunks: list[list[np.ndarray]]`
- **Config:** `strategy: Literal["blend", "select"]`, `select_index: int = 0`
- `blend`: equal-weight pixel average across all branches per frame
- `select`: pass through the branch at `select_index`

### `MapNode`
- **Input ports:** `items: list[T]`
- **Output ports:** `results: list[U]`
- **Config:** `sub_dag: DAG`
- Applies `sub_dag` to each item in `items` sequentially. Step Functions `Map` state analogy — useful for batch processing multiple clips.

### `AudioRestoreNode`
- **Input ports:** `video_path: str`
- **Output ports:** `video_path: str` (same path, audio remuxed in-place)
- Runs the audio pipeline pass (Demucs/VoiceFixer/RNNoise) and remuxes

### `ChoiceNode`
- **Input ports:** `meta: VideoMeta`
- **Output ports:** `branch_index: int`
- **Config:** `rules: list[ChoiceRule]` — each rule is `(field, op, value) → branch_index`
- Routes execution conditionally (e.g. skip upscaling for 4K inputs)

### `PassNode`
- **Input ports:** `data: Any`
- **Output ports:** `data: Any`
- Identity node — useful for wiring and testing

---

## DAG Serialization

`DAGSerializer` uses a **type registry** pattern (same as Dagster's resource config):

```python
# Each node type registers a string type_id
@dag_node_type("restore")
class RestoreNode(Node): ...

# Serialize
dag_dict = DAGSerializer.to_dict(dag)
# {"id": "...", "nodes": [{"type": "restore", "restorer_name": "real_esrgan", ...}], "edges": [...]}

# Deserialize
dag = DAGSerializer.from_dict(dag_dict)
```

Stored in `PipelineTemplateModel.config` with `"schema_type": "dag"` discriminator field. Existing sequential pipelines have `"schema_type": "sequential"` (or no field — backward compat).

---

## Database Changes

**No new tables.** `PipelineTemplateModel.config` already stores arbitrary JSON. Add `"schema_type": "dag"` discriminator.

`JobModel` gains one column:

```python
dag_run: Mapped[dict | None] = mapped_column(JSON, nullable=True)
# Stores serialized DAGRun state — node states, results, timestamps
```

This enables job resume after worker restart and per-branch status queries.

---

## API Changes

### New endpoints

```
POST   /pipelines/dag           — create DAG pipeline (stores in PipelineTemplateModel)
GET    /pipelines/dag/{id}      — get DAG definition
GET    /jobs/{id}/branches      — per-branch: {branch_index, status, progress, output_path}
POST   /jobs/{id}/merge         — {"strategy": "blend"} | {"strategy": "select", "branch_index": 1}
```

### Modified endpoints

```
POST /jobs — gains optional dag_id form field
             if dag_id present: dispatches run_dag_job task
             if pipeline_id present: existing behavior unchanged
```

---

## New Celery Task

```python
@celery_app.task(bind=True, base=JobTask, name="restorax.tasks.job_tasks.run_dag_job")
def run_dag_job(self, job_id: str, dag_id: str, input_path: str, output_path: str) -> dict:
    """Execute a DAG pipeline on a video file."""
    ...
```

No Celery canvas. The `DAGExecutor` handles all orchestration internally.

---

## Module Layout

```
restorax/dag/
  __init__.py          ← public exports: DAG, Node, Edge, Port, DAGExecutor, DAGRun, ExecutionContext
  node.py              ← Node ABC, Port, NodeState, NodeResult, RetryPolicy
  edge.py              ← Edge
  graph.py             ← DAG (immutable, validates on __init__, topological_levels())
  executor.py          ← DAGExecutor, DAGRun
  context.py           ← ExecutionContext, ProgressEmitter
  errors.py            ← DAGValidationError, NodeExecutionError, PortTypeMismatchError
  serializer.py        ← DAGSerializer, dag_node_type decorator, type registry
  nodes/
    __init__.py
    io.py              ← VideoInputNode, VideoOutputNode
    restore.py         ← RestoreNode, AudioRestoreNode
    parallel.py        ← ParallelNode
    merge.py           ← MergeNode
    map_node.py        ← MapNode
    control.py         ← ChoiceNode, PassNode
```

---

## Error Handling

- `DAGValidationError` — raised at DAG construction, never at runtime
- `NodeExecutionError` — wraps the original exception with `node_id`, `attempt_number`, context
- Failed node → `DAGRun.failed_node_id` set, all downstream nodes set to `SKIPPED`
- `DAGRun` persisted to DB on every state transition — no state lost on worker crash
- Unrecoverable failure emits final progress event `{status: "failed"}` via Redis

---

## Testing Strategy

- **Unit tests** — each `Node` type tested in isolation with mock `ExecutionContext`
- **DAG validation tests** — cycle detection, port type mismatch, dangling references
- **Executor tests** — topological ordering, retry logic, failure propagation, `SKIPPED` downstream
- **Integration tests** — full `ParallelNode` + `MergeNode` round-trip with synthetic frames
- **API tests** — `/pipelines/dag` CRUD, `/jobs/{id}/branches`, `/jobs/{id}/merge`

---

## Future: Standalone Package

When extracted as `restorax-dag`:
- Remove `restorax.*` imports from `context.py` — replace with generic `resource` dict
- `RestoreNode`, `VideoInputNode`, etc. move to `restorax` as a plugin package
- `ProgressEmitter` becomes an abstract base; Redis implementation stays in `restorax`
- Package ships with `PassNode`, `MapNode`, `ChoiceNode`, `ParallelNode`, `MergeNode` as built-ins
- No video/audio/GPU dependencies in the core package
