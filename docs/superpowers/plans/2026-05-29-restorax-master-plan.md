# RestoraX Master Plan

**Last updated:** 2026-05-29
**Owner:** İlker Kara

This document is the single source of truth for all active and planned development tracks. Each sub-project has its own spec + implementation plan. This document describes scope, order, dependencies, and status at a high level.

---

## Execution Order

```
Sub-project 1 — Backend Foundations          ✅ COMPLETE
Sub-project 2 — Pipeline DAG Engine          🔄 IN PROGRESS (spec approved)
Sub-project 3 — Modern UI                    ⏳ PENDING (depends on Sub-project 2 API)
Sub-project 4 — ComfyUI Node Pack            ⏳ PENDING (independent)
Track E       — Documentation Refactor       ⏳ LAST (after all sub-projects ship)
```

---

## Sub-project 1 — Backend Foundations ✅ COMPLETE

**Spec:** `docs/superpowers/specs/2026-05-04-backend-hardening-design.md`
**Plan:** `docs/superpowers/plans/2026-05-20-backend-foundations.md`

### What shipped
| Feature | File | Commit |
|---|---|---|
| Audio restorers in `GET /models` (24 total) | `api/routers/models.py` | `75018d3` |
| Global exception handlers (`RestoraXError` → structured JSON) | `api/app.py` | `44127b5` |
| `GET /health/celery` (queue depth + worker count) | `api/routers/health.py` | `7157a6f` |
| CLI `models` command handles audio capabilities | `cli.py` | `3221e9b` |
| 391 unit tests passing, 1 skipped | all test files | `3221e9b` |

### Completion criteria met
- ✅ `GET /models` returns all 24 restorers including Demucs, VoiceFixer, RNNoise
- ✅ `RestorerLoadError` → 503, `RestorerNotFoundError`/`JobNotFoundError` → 404, `PipelineConfigError` → 422
- ✅ `GET /health/celery` returns `{status, workers, active_tasks, queued_tasks}`
- ✅ 10 new tests, no regressions

---

## Sub-project 2 — Pipeline DAG Engine 🔄 IN PROGRESS

**Spec:** `docs/superpowers/specs/2026-05-29-pipeline-dag-design.md`
**Plan:** TBD (writing-plans skill → `docs/superpowers/plans/2026-05-29-pipeline-dag-engine.md`)

### Goal
Replace the limitation of purely sequential pipelines with a full DAG orchestration engine. Users can define parallel restoration branches, compare outputs side-by-side, and select or blend the best result. The engine is designed to be extracted as a standalone open-source library (`restorax-dag`) in the future.

### Key design decisions
- **No Celery canvas** — custom DAG executor runs inside a single `run_dag_job` Celery task
- **Additive** — existing `Pipeline`/`PipelineRunner`/YAML presets unchanged
- **Sequential GPU execution** — parallel branches run one-after-another on single GPU
- **Per-branch progress** — WebSocket emits `{node_id, branch_index, progress}` events
- **Typed ports** — inspired by Dagster; type errors caught at DAG construction, not runtime
- **State machine** — every node has `PENDING | RUNNING | SUCCEEDED | FAILED | SKIPPED | RETRYING`

### Architecture
```
restorax/dag/
  __init__.py        node.py        edge.py        graph.py
  executor.py        context.py     errors.py      serializer.py
  nodes/
    io.py            restore.py     parallel.py    merge.py
    map_node.py      control.py
```

### New API surface
```
POST   /pipelines/dag           create DAG pipeline
GET    /pipelines/dag/{id}      get DAG definition
POST   /jobs                    gains dag_id field (alongside existing pipeline_id)
GET    /jobs/{id}/branches      per-branch status + progress + output paths
POST   /jobs/{id}/merge         {"strategy":"blend"} | {"strategy":"select","branch_index":N}
```

### Phases

**Phase 1 — Core engine (no video integration)**
- `restorax/dag/` module: `Node`, `Port`, `Edge`, `DAG`, `NodeState`, `NodeResult`, `RetryPolicy`
- `DAGExecutor` with topological sort and retry wrapper
- `DAGSerializer` with type registry
- `DAGValidationError` cycle detection and port type checks
- Full unit tests for engine (no GPU, mock nodes)

**Phase 2 — Built-in node types**
- `VideoInputNode`, `VideoOutputNode`
- `RestoreNode`, `AudioRestoreNode`
- `ParallelNode`, `MergeNode`
- `MapNode`, `ChoiceNode`, `PassNode`
- Integration tests with synthetic frames

**Phase 3 — Celery + DB integration**
- `run_dag_job` Celery task
- `JobModel.dag_run` JSON column (Alembic migration)
- `ExecutionContext` wired to real `ModelRegistry`
- `ProgressEmitter` → Redis pub/sub
- WebSocket layer extended for per-branch events

**Phase 4 — API endpoints**
- `POST/GET /pipelines/dag`
- `POST /jobs` extended with `dag_id`
- `GET /jobs/{id}/branches`
- `POST /jobs/{id}/merge`
- API-level tests

**Phase 5 — Polish and documentation**
- `dry_run` mode in `DAGExecutor`
- `DAGRun` persistence for worker-crash recovery
- Developer guide: how to write a custom node type
- Update `GET /models` to include DAG capability flag

### Completion criteria
- [ ] `POST /pipelines/dag` stores a DAG with 2 parallel branches
- [ ] `POST /jobs` with `dag_id` executes both branches sequentially
- [ ] `GET /jobs/{id}/branches` returns per-branch progress and output paths
- [ ] `POST /jobs/{id}/merge` with `strategy=blend` produces blended output
- [ ] `DAGValidationError` raised for cyclic DAGs at construction time
- [ ] Retry policy retries failed nodes up to `max_retries` times
- [ ] All new tests pass, no regressions in existing suite

---

## Sub-project 3 — Modern UI ⏳ PENDING

**Spec:** TBD
**Plan:** TBD
**Depends on:** Sub-project 2 API (`/jobs/{id}/branches`, `/jobs/{id}/merge`)

### Goal
New React 18 frontend replacing any placeholder UI. Full pipeline builder with drag-and-drop DAG canvas, real-time job monitoring, and side-by-side branch comparison.

### Tech stack
- **React 18** + **Vite** — build tooling
- **shadcn/ui** — component library (Tailwind-based)
- **@xyflow/react (ReactFlow)** — DAG canvas
- **Dark mode** by default
- **Mobile-responsive**
- **NOT Next.js**

### Three core views

**Dashboard**
- Recent jobs list with status badges and thumbnail previews
- Quick-launch preset cards (Film Restoration, Anime Upscale, Noise Reduction, etc.)
- GPU status widget (from `GET /health/celery`)
- Queue depth indicator

**Pipeline Builder (ReactFlow canvas)**
- Sidebar palette of all 24 restorers (from `GET /models`), grouped by category
- Drag restorer → canvas to add `RestoreNode`
- Connect nodes with edges (port-to-port)
- `ParallelNode` renders as a swim-lane container
- `MergeNode` renders below parallel branches
- Inline param config panel (click node → right sidebar)
- Save/load DAG to API (`POST/GET /pipelines/dag`)

**Job Detail**
- Real-time progress via WebSocket — per-branch progress bars
- Side-by-side branch video comparison (scrubber-synced)
- Blend slider or winner-select UI
- Confirm merge → `POST /jobs/{id}/merge`
- Download completed output

### Phases
- Phase 1: Vite + React 18 + shadcn/ui scaffold, routing, auth placeholder
- Phase 2: Dashboard view with job list + GPU status
- Phase 3: Pipeline Builder (ReactFlow canvas + node types)
- Phase 4: Job Detail with WebSocket progress + branch comparison
- Phase 5: Merge UI (blend slider + winner select)

---

## Sub-project 4 — ComfyUI Node Pack ⏳ PENDING

**Spec:** TBD
**Plan:** TBD
**Depends on:** nothing (independent of Sub-projects 2 and 3)

### Goal
All 24 RestoraX restorers available as ComfyUI custom nodes, distributed via ComfyUI-Manager. One-click install from the ComfyUI-Manager community list.

### Tech constraints
- ComfyUI `IMAGE` = `torch.Tensor (B, H, W, C)` float32 [0,1] ↔ RestoraX `np.ndarray (H, W, 3)` uint8 — shared conversion helpers in `_base.py`
- Temporal restorers (VRT, RIFE, EvTexture, FlashVSR) accept batched `IMAGE` (B>1 = frame sequence)
- Audio nodes use ComfyUI `AUDIO` type (available ≥ ComfyUI 0.3)
- Lazy weight download: `restorax download-models` on first node execution

### File layout
```
comfyui_nodes/               ← root-level, ComfyUI-Manager clones here
  __init__.py                ← NODE_CLASS_MAPPINGS + NODE_DISPLAY_NAME_MAPPINGS
  _base.py                   ← BaseRestoraXNode (tensor↔numpy, registry singleton)
  nodes/
    super_resolution.py      ← RealESRGANNode, VRTNode, Waifu2xNode, MambaIRNode, ...
    face_restoration.py      ← CodeFormerNode, GFPGANNode, DicFaceNode, ...
    colorization.py          ← DDColorNode
    frame_interpolation.py   ← RIFENode
    deinterlacing.py         ← AIDeinterlaceNode
    artifact_removal.py      ← ScratchRemovalNode
    hdr.py                   ← HDRTVDMNode
    stabilization.py         ← GAVSNode, VideoStabilizationNode
    audio.py                 ← DemucsNode, VoiceFixerNode, RNNoiseNode
  requirements.txt           ← restorax as pip dep
  comfyui_manifest.json      ← for ComfyUI-Manager registration
```

### Phases
- Phase 1: `_base.py` with tensor/numpy conversion + registry singleton
- Phase 2: Super-resolution nodes (10 restorers)
- Phase 3: Face restoration + colorization + frame interpolation nodes
- Phase 4: Stabilization + deinterlacing + HDR + artifact removal nodes
- Phase 5: Audio nodes (Demucs, VoiceFixer, RNNoise)
- Phase 6: `comfyui_manifest.json` + PR to ComfyUI-Manager community list

---

## Track E — Documentation Refactor ⏳ LAST

**Plan:** `docs/superpowers/plans/2026-05-03-doc-refactor.md`
**Depends on:** ALL sub-projects shipped

Execute the existing doc-refactor plan after all sub-projects are complete so the documentation reflects the final state of the codebase.

---

## GitHub Achievements Progress

| Achievement | Status | How |
|---|---|---|
| Pair Extraordinaire | ✅ | Co-authored commits with Claude |
| YOLO | ✅ | Merged PRs without review |
| Pull Shark | ✅ | Merged multiple PRs |
| Arctic Code Vault Contributor | ✅ | Prior work |
| Quickdraw | 🔄 triggered | Closed issue #3 within seconds |
| Galaxy Brain | 🔄 triggered | Marked answer in Discussion #2 |
| Heart On Your Sleeve | 🔄 triggered | ❤️ reaction on Discussion #2 |

---

## Active Discussions

- [#2 — How does Pipeline DAG handle temporal models across parallel branches?](https://github.com/karailker/restorax/discussions/2)

## Open Issues

- [#4 — Pipeline DAG Engine tracking issue](https://github.com/karailker/restorax/issues/4)
