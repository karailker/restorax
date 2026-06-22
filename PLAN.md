# RestoraX — Project Plan

**Last updated:** 2026-06-22
**Owner:** İlker Kara

This is the single source of truth for RestoraX's history and roadmap — what shipped, what's in flight, and what's next. It replaces the original phase-based `PLAN.md`/`PROGRESS.md` and the `docs/superpowers/` master plan, which are now archived (see [Archive](#6-archive) below) rather than duplicated here.

---

## 1. What RestoraX Is

An open-source AI video restoration platform for old films, home videos, and archival footage: 25 restoration models (super-resolution, face restoration, colorization, frame interpolation, deinterlacing, scratch/dust removal, HDR conversion, stabilization, audio) unified behind a REST API + WebSocket progress, a Celery/Redis job queue, and a visual ComfyUI-style node-graph pipeline builder.

**Positioning:** a *video-restoration studio*, not a general creative sandbox. We match ComfyUI's graph-editing power (typed sockets, parallel branches, undo/redo, node search) but differentiate on domain depth — temporal/audio/codec-aware nodes, built-in quality metrics, restoration presets, and production-grade job orchestration that a general-purpose tool doesn't need.

**Interop:** the React/ReactFlow builder is the primary surface; a ComfyUI custom-node pack (Sub-project 4, below) extends reach without making ComfyUI a dependency of the core product.

---

## 2. Current State Snapshot

### Tech stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.11 |
| ML Backend | PyTorch 2.5+, CUDA 12.1 |
| API | FastAPI, async, Pydantic v2, WebSocket |
| Task Queue | Celery 5.3+ + Redis 7 (broker, result backend, pub/sub) |
| ORM / DB | SQLAlchemy 2.0 async + asyncpg/aiosqlite, PostgreSQL 16 (prod) / SQLite (dev) |
| Migrations | Alembic (env wired, **no revisions generated yet** — see [Open Gaps](#5-open-gaps--known-issues)) |
| Video I/O | PyAV |
| Observability | structlog, OpenTelemetry, Prometheus, optional Sentry |
| Frontend | **React 18 + Vite + TypeScript** (NOT Next.js) |
| UI components | Official shadcn/ui registry (Radix-based, `nova` preset) |
| Pipeline canvas | `@xyflow/react` (ReactFlow) |
| Before/After | `react-compare-slider` |
| Containerization | Docker + Docker Compose, NVIDIA Container Toolkit |
| Lint/Type | ruff, mypy |
| Testing | pytest + pytest-asyncio (backend), Vitest (frontend) |

### Model catalog (25 restorers, `restorax/api/routers/models.py::_RESTORER_CLASSES`)

| Category | Restorers |
| --- | --- |
| Super-Resolution | Real-ESRGAN x4, BasicVSR++, Upscale-A-Video, VRT, MambaIR, TDM, SeedVR, Waifu2x, FlashVSR, EvTexture |
| Face Restoration | CodeFormer, CodeFormer++, GFPGAN, DicFace |
| Colorization | DDColor |
| Frame Interpolation | RIFE v4.x |
| Deinterlacing | AI deinterlace, **YADIF (classical, weight-free)** |
| Artifact Removal | ProPainter-based scratch removal |
| HDR Conversion | HDRTVDM |
| Stabilization | Optical-flow stabilization, GaVS |
| Audio | Demucs, VoiceFixer, RNNoise (stub-approved exceptions; not runnable via DAG `restore` node — see gaps) |

Every restorer raises `RestorerLoadError` instead of silently falling back to a stub (enforced by `tests/unit/test_no_silent_stubs.py`). 15 restorers carry per-model `PARAM_SCHEMA` driving typed UI widgets.

### API surface

```text
GET  /health, /ready, /health/celery
POST   /jobs                      GET /jobs            GET /jobs/{id}
GET    /jobs/{id}/download        DELETE /jobs/{id}     POST /jobs/batch
GET    /jobs/{id}/branches        POST /jobs/{id}/merge
GET    /models
POST/GET/PUT/DELETE /pipelines    POST/GET /pipelines/dag, /pipelines/dag/{id}
WS     /ws/jobs/{job_id}/progress
```

### Repository structure (top level)

```text
restorax/
  core/          BaseRestorer ABC, Pipeline/PipelineRunner, ModelRegistry (LRU)
  dag/           typed-port DAG engine (node/edge/graph/executor/serializer + nodes/)
  restorers/     one sub-package per category (see catalog above)
  audio/         audio-specific pipeline/reader/writer/restorer base
  api/           FastAPI app, routers/, schemas/
  tasks/         Celery app, job_tasks.py (run_job, run_dag_job), progress.py
  db/            SQLAlchemy models, session, repositories
  storage/       StorageBackend protocol (local / S3-MinIO)
  video/         PyAV reader/writer, color/tiling utils
  metrics/       PSNR/SSIM/LPIPS/VMAF, NIQE/MUSIQ
  benchmarks/    benchmark CLI + VRAM monitor
frontend/        React 18 + Vite + shadcn/ui + ReactFlow
  src/components/{builder,dashboard,jobdetail,layout,ui}/
configs/presets/ YAML pipeline presets (sr_x4, classic_film, vhs_restoration, newsreel, ...)
docs/superpowers/archive/  historical plan/spec docs (see §6)
```

---

## 3. History — What Shipped

### 3.1 Original build-out (Phases 1–6, through 2026-04-30)

Foundation → MVP restorers → Web UI placeholder → extended restorer library → advanced models/performance → hardening. Delivered the core architecture still in use today: `BaseRestorer` ABC, chunked `PipelineRunner`, LRU `ModelRegistry`, PyAV video I/O, Celery/Redis job execution, REST API, Docker Compose. 309 backend + 25 frontend tests passing at the time. Frontend at this point was a placeholder later fully replaced (§3.6).

### 3.2 Backend Hardening (2026-05-04)

Production observability: `structlog` structured logging, OpenTelemetry traces, Prometheus metrics (`/metrics`), `/health` + `/ready` probes, request-ID propagation, optional Sentry APM. Wired into FastAPI app and Celery worker startup, all env-driven.

### 3.3 Real Model Activation (2026-05-06, "Track 2")

Vendored 15 architecture files from official repos, wired weight auto-download via `huggingface_hub`, and replaced every silent stub fallback with explicit `RestorerLoadError`. Audio stubs (Demucs/VoiceFixer/RNNoise) are the only approved stub exceptions. Canary test (`test_no_silent_stubs.py`) enforces this permanently.

### 3.4 Sub-project 1 — Backend Foundations (2026-05-20)

`GET /models` exposes all restorers including audio; global exception handlers map `RestorerLoadError`→503, `RestorerNotFoundError`/`JobNotFoundError`→404, `PipelineConfigError`→422; `GET /health/celery` (queue depth + worker count); CLI `models` command handles audio capabilities.

### 3.5 Sub-project 2 — Pipeline DAG Engine (2026-05-29/30)

Replaced purely-sequential pipelines with a full DAG orchestrator (`restorax/dag/`): typed multi-socket ports with construction-time validation, cycle detection, topological execution, per-node `RetryPolicy`, downstream-skip on failure, `ParallelNode`/`MergeNode` (blend or select), `MapNode`/`ChoiceNode`/`PassNode`. Runs inside a single `run_dag_job` Celery task (no Celery canvas) with per-branch progress over Redis pub/sub. New API: `POST/GET /pipelines/dag`, `dag_id` on `/jobs`, `/jobs/{id}/branches`, `/jobs/{id}/merge`. 430 unit tests passing at completion. Designed to be extractable as a standalone `restorax-dag` library later.

### 3.6 Sub-project 3 — Modern UI (in progress since 2026-05-30)

Full replacement of the placeholder frontend:

- **Foundation + views (PRs #10–11):** Vite + React 18 + TS + Tailwind v4, ReactFlow canvas. Three views: Dashboard (stats + preset quick-launch + jobs table), Pipeline Builder (drag-to-canvas, typed nodes, save/load DAG), Job Detail (live WS progress, branch comparison via `react-compare-slider`, merge panel).
- **Official shadcn/ui adoption (PR #12)** and **shadcn form controls (PR #13):** replaced hand-rolled primitives and native form elements with the official Radix-based registry.
- **README sync (PR #14).**
- **ComfyUI-parity milestones** (decided direction: match ComfyUI's graph UX, differentiate on restoration-domain depth — see §1):
  - **M1 — Workflow authoring ergonomics (PRs #15–16):** JSON export/import with node positions, 50-step undo/redo, `NodeSearch` command palette, right-click context menus, node duplication.
  - **M2 — Typed multi-socket ports (PR #17):** `ports.ts` single source of truth mirroring backend port names/types; `isValidConnection` blocks incompatible wires at the canvas level. Fixed two pre-existing backend-contract bugs in the process (bogus `"video"` port name; dropped `params_dict`).
  - **M3 — Schema-driven node widgets (PR #18):** added `PARAM_SCHEMA` to 15 backend restorers (the backend had none before); `ConfigPanel` renders typed widgets (number/bool/enum/multiselect) instead of a raw JSON textarea, with a scoped JSON escape hatch for `extra`.
  - **Review pass (PR #19):** fixed an `extra`-scoping bug that let a stray top-level JSON key crash `RestorerParams`, and a socket-handle visual misalignment.
  - **M4 — Canvas execution + live per-node progress (PR #20):** Run button drives the DAG directly from canvas state; per-node status border + progress bar fed by existing backend per-node lifecycle events. Not yet verified end-to-end against a live GPU worker.
- **CORS + audio palette fix (PR #21):** configurable `cors_origins` (was wildcard-with-credentials, broken/insecure in prod); `/models` now tags `kind: video|audio` and the builder palette filters out non-DAG-runnable audio restorers.

### 3.7 YADIF classical deinterlacer (branch `feat/yadif-deinterlace`, 2026-06-21, **on top of main, not yet merged**)

Weight-free motion-adaptive deinterlacer shelling out to system `ffmpeg`, with a numpy "bob" fallback and progressive-frame auto-detection (pass-through, never softens non-interlaced footage). Enabled by default in `classic_film.yaml` (previously a disabled placeholder stage). `cli.py` now registers all restorers via `registry.register_all()` instead of just Real-ESRGAN. Fixed a `pipeline.py` bug where disabled YAML stages still tried to resolve a (possibly-unregistered) restorer. 18 new tests; full suite green except one pre-existing, unrelated env-leakage failure (`test_config.py::test_default_device_is_cuda`).

### 3.8 SP3 live verification + fixes (2026-06-22)

Stood up the full dev stack (Redis, FastAPI/uvicorn, Celery worker, Vite) and drove it with the `browse` headless-browser skill (not just build/typecheck evidence) to confirm M1–M4 actually work end-to-end, then did a visual design pass. Found and fixed two real bugs:

- **Event-loop-blocking bug (`restorax/api/routers/health.py::celery_health`):** `inspect.active()`/`inspect.reserved()` are synchronous Celery broker calls invoked directly inside an `async def` route, blocking the single uvicorn worker's event loop for the full broker round-trip (~4s observed). This froze *every other concurrent request* (`/jobs`, `/models`) for that window — looked like a hung jobs-list bug in the UI but was actually loop starvation. Fixed by wrapping both calls in `asyncio.to_thread(...)`. Verified via concurrent-request test: `/jobs` now resolves in ~0.03s even while `/health/celery` is mid-flight.
- **No responsive layout (`frontend/src/components/layout/AppShell.tsx`):** sidebar nav was a fixed `w-60` block with no breakpoint handling — on narrow viewports (375px) it ate the full screen and clipped all page content. Added a `md:hidden` top bar with a hamburger trigger, an off-canvas drawer (`fixed` + `-translate-x-full`/`translate-x-0` + backdrop) below the `md` breakpoint, and kept the original static sidebar unchanged at `md:` and above. Verified via screenshots at 375px (collapsed + drawer-open) and 1280px (unchanged).

Desktop visual quality otherwise holds up: consistent dark shadcn theme, clear hierarchy, no other layout defects found across Dashboard and Pipeline Builder (including the restorer palette, which loads correctly once `/models` resolves — initial "empty palette" observation was the event-loop bug above, not a frontend defect).

M4 canvas-run flow (§4.1) was *not* re-verified this session (no job was actually submitted/run against live Celery+weights) — that item remains open.

---

## 4. Active / Next Tracks

```text
SP3 — Modern UI polish          ⏳ remaining items below
SP4 — ComfyUI Node Pack         ⏳ not started (independent of SP3)
Backend hardening follow-ups    ⏳ not started
Track E — Documentation Refactor ⏳ deliberately LAST (after everything above ships)
```

### 4.1 SP3 polish (deferred, needs verification against real backend + GPU)

- Branch-output image URLs: `BranchCompare` assumes `${API_BASE}/<output_path>` — confirm the backend actually serves branch artifacts at that path (vs. `/files/`, `/artifacts/`, or a dedicated per-branch frame endpoint).
- Parallel-branch authoring: builder saves `parallel` nodes with `branches: []` — there's no UI yet to author branch steps. Confirm backend's tolerance for empty branches or build the authoring UI.
- Preset IDs (`sr_x4`, `classic_film`, `vhs_restoration`, `newsreel`) used by Dashboard quick-launch — confirmed to exist in `configs/presets/` as of the 2026-05-31 review; re-verify if presets change.
- M4 canvas-run flow still has not been verified end-to-end with a real job actually submitted/run against live Redis + Celery + real weights (stack was confirmed live and wired correctly in §3.8, but no job was actually executed).
- Deferred from M4: inline per-node intermediate-frame previews (would need new backend per-node artifact endpoints — its own milestone).
- ~~Event-loop-blocking `/health/celery` call freezing concurrent requests~~ — **fixed 2026-06-22, §3.8.**
- ~~No mobile-responsive sidebar layout~~ — **fixed 2026-06-22, §3.8.**

### 4.2 Sub-project 4 — ComfyUI Node Pack (independent track)

All 25 restorers as ComfyUI custom nodes (`comfyui_nodes/`), distributed via ComfyUI-Manager.

- Tensor↔numpy conversion shared in `_base.py` (ComfyUI `IMAGE` = `(B,H,W,C)` float32 [0,1] ↔ RestoraX `(H,W,3)` uint8).
- Temporal restorers accept batched `IMAGE` (B>1 = sequence).
- Audio nodes use ComfyUI `AUDIO` type.
- Lazy weight download on first node execution.
- Phases: base conversion layer → SR nodes → face/color/interpolation nodes → stabilization/deinterlace/HDR/artifact nodes → audio nodes → manifest + ComfyUI-Manager PR.

### 4.3 Backend hardening follow-ups

- **Alembic migrations:** `alembic/versions/` is empty; schema is created via `Base.metadata.create_all` at startup. Fine for SQLite/dev; unversioned for Postgres prod (`create_all` won't `ALTER` existing tables). Generate an initial revision and wire it into deploy.
- **Test isolation:** full `pytest` run shows cross-file failures (each file passes alone). Root cause: async SQLAlchemy `AsyncAdaptedQueuePool` "Exception during reset" leaking across integration/system tests, plus env-var leakage into `test_config` override tests. Fix: per-test engine disposal + event-loop scoping in `conftest.py`, explicit env cleanup.
- **Frontend lint:** no ESLint config (eslint v9 needs flat `eslint.config.js`); only `tsc` typecheck today, no `lint` script.

### 4.4 Track E — Documentation Refactor (last)

Transform docs from working-notebook style into model cards, ADRs, an AI-tooling guide, `ROADMAP.md`, `CHANGELOG.md` — executed only after SP3/SP4 ship, so docs reflect final state rather than a moving target. Detailed plan/spec in archive (§6).

---

## 5. Open Gaps / Known Issues

(From the 2026-05-31 whole-project review; CORS and audio-palette items already fixed in §3.6/PR #21.)

- No Alembic migrations generated (§4.3).
- Test suite not isolated when run as a whole (§4.3).
- Audio restorers (Demucs/VoiceFixer/RNNoise) are listed by `/models` but are **not actually runnable via the DAG `restore` node** (they use `AudioRestorerParams`/`process()`, not `process_frame()`). Builder palette already filters them out (PR #21); the underlying DAG-incompatibility is still unresolved if audio-in-DAG is ever wanted.
- `models.py` reads `capabilities` via `object.__new__(cls)` (FRAGILE marker) — works today because `capabilities` is a pure property with no instance state, but is brittle if that ever changes.
- `config.py` `s3_secret_key` defaults to `"minioadmin"` — fine as a dev default, must be overridden in prod.
- No frontend ESLint config (§4.3).

---

## 6. Archive

Full step-by-step implementation history (TDD task lists, design specs) for every completed track in §3 lives in `docs/superpowers/archive/{plans,specs}/`, preserved for reference but superseded by this file as the source of truth going forward.
