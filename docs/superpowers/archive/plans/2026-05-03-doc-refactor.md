# RestoraX Documentation Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform RestoraX docs from working-notebook style into a modern 2026 AI project showcase with dual-path navigation, model cards, ADRs, and an AI tooling guide.

**Architecture:** Documentation-only refactor — no source code changes. New files are created first, then old files (PLAN.md, PROGRESS.md) are deleted after content has been redistributed. All benchmark numbers are copied verbatim from README.md; none are invented.

**Tech Stack:** Markdown, Keep a Changelog convention, ADR format (Context / Decision / Consequences)

**Spec:** `docs/superpowers/specs/2026-05-03-doc-refactor-design.md`

---

## Restorer inventory (canonical, from `restorax/restorers/`)

| Category | Modules | Status |
|---|---|---|
| super_resolution | real_esrgan, basicvsr_pp, waifu2x, flashvsr, mamba_ir, evtexture, vrt, upscale_a_video, tdm, seedvr | real_esrgan + basicvsr_pp active; rest stubs |
| colorization | ddcolor | stub |
| face_restoration | codeformer, codeformer_pp, dicface, gfpgan | codeformer + gfpgan active; rest stubs |
| frame_interpolation | rife | stub |
| deinterlacing | ai_deinterlace | stub |
| artifact_removal | scratch_removal | stub (ProPainter) |
| hdr | hdrtvdm | stub |
| stabilization | deep_flow_stab, gavs | stubs |
| audio | demucs, voicefixer, rnnoise | stubs |

---

## Task 1 — docs/architecture/overview.md

**Files:**
- Create: `docs/architecture/overview.md`

- [ ] **Step 1: Create the file**

```bash
mkdir -p docs/architecture/decisions
```

Write `docs/architecture/overview.md`:

```markdown
# Architecture Overview

## Pipeline

```
Web UI (Next.js)  ──────►  FastAPI REST API  ──►  Celery + Redis  ──►  GPU Worker
     CLI (Click)  ──────►                                               │
                                                                        ▼
                                                              PipelineRunner
                                                         (sequential chunks, LRU registry)
                                                                        │
                                      ┌─────────────────────────────────┼──────────────┐
                                      ▼                                 ▼              ▼
                               VideoReader (PyAV)            Restorer stages    VideoWriter (PyAV)
                               + AudioReader                 (24 restorers)      + AudioWriter
```

## Design Principles

- **Sequential chunked processing** — constant memory regardless of video length; frames are processed in chunks and written incrementally
- **LRU model registry** — evicts the least-recently-used model before loading the next pipeline stage; default `max_loaded=2`; configurable via `RESTORAX_MAX_LOADED_MODELS`
- **Stub-first** — every restorer works without real weights; stubs produce correct-shape output so CI runs without a GPU; see [ADR-003](decisions/003-stub-first-models.md)
- **Plugin system** — third-party restorers via `pip install restorax-plugin-*`; registered via `pyproject.toml` entry points

## Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| Language | Python 3.11 | Broad ML support |
| ML Backend | PyTorch 2.3+ | Flash Attention 2, `torch.compile()`, CUDA Graphs |
| CUDA | 12.1 / cuDNN 8.9 | Widest driver compatibility (RTX 30xx/40xx) |
| API | FastAPI 0.111+ | Async-native, Pydantic v2, built-in WebSocket |
| Task Queue | Celery 5.3+ | GPU concurrency control, per-GPU routing, retry policies |
| Message Broker | Redis 7 | Broker + result backend + WebSocket pub/sub |
| ORM / DB | SQLAlchemy 2.0 async + asyncpg + PostgreSQL 16 (SQLite for local) | Job history, pipeline templates |
| Migrations | Alembic | |
| Video I/O | PyAV | PTS access, audio passthrough — see [ADR-001](decisions/001-pyav-video-io.md) |
| FFmpeg | Subprocess (limited) | Complex filter graphs, final mux, YADIF |
| Storage | Local FS (dev) / MinIO S3-compatible (prod) | See [ADR-005](decisions/005-storage-abstraction.md) |
| Frontend | Next.js 14 (App Router) + React 18 | SSR for job list, client components for live progress |
| UI | shadcn/ui + Tailwind CSS | |
| Containerization | Docker + Docker Compose v2 + NVIDIA Container Toolkit | |
| Linting/Format | ruff | Replaces black + flake8 + isort |
| Type Checking | mypy strict | |
| Testing | pytest + pytest-asyncio + pytest-celery | |

## Repository Structure

```
restorax/
├── restorax/                    # main Python package
│   ├── config.py                # pydantic-settings Settings (reads .env)
│   ├── core/                    # domain logic — no FastAPI/Celery imports
│   │   ├── restorer.py          # BaseRestorer ABC
│   │   ├── pipeline.py          # Pipeline, Stage, PipelineRunner
│   │   ├── registry.py          # ModelRegistry (LRU VRAM cache)
│   │   └── job.py               # Job, JobStatus, JobRequest value objects
│   ├── restorers/               # one sub-package per restoration category
│   │   ├── super_resolution/    # real_esrgan, basicvsr_pp, waifu2x, ...
│   │   ├── colorization/        # ddcolor
│   │   ├── face_restoration/    # codeformer, gfpgan, dicface, codeformer_pp
│   │   ├── frame_interpolation/ # rife
│   │   ├── deinterlacing/       # ai_deinterlace
│   │   ├── artifact_removal/    # scratch_removal (ProPainter)
│   │   ├── hdr/                 # hdrtvdm
│   │   ├── stabilization/       # deep_flow_stab, gavs
│   │   └── audio/               # demucs, voicefixer, rnnoise
│   ├── api/                     # FastAPI app, routers, schemas
│   ├── tasks/                   # Celery app, job tasks, progress reporting
│   ├── db/                      # SQLAlchemy models, session, repositories
│   ├── storage/                 # StorageBackend Protocol, local + S3 impls
│   ├── video/                   # VideoReader, VideoWriter, utils
│   ├── audio/                   # AudioReader, AudioWriter
│   └── metrics/                 # PSNR, SSIM, LPIPS, VMAF, NIQE
├── frontend/                    # Next.js 14 web UI
├── tests/                       # pytest suite (unit + integration + system)
├── alembic/                     # DB migrations
├── configs/                     # pipeline YAML definitions
├── docs/                        # documentation
│   ├── architecture/            # this file + ADRs
│   ├── guides/                  # user guides (installation, quickstart, ...)
│   └── models/                  # model cards
├── models/                      # vendored model architectures
├── scripts/                     # benchmark runner, fixture generator
├── docker-compose.yml           # prod: GPU, PostgreSQL, MinIO
└── docker-compose.dev.yml       # dev: hot-reload, CPU, SQLite
```

## Architecture Decisions

- [ADR-001: PyAV for Video I/O](decisions/001-pyav-video-io.md)
- [ADR-002: Celery + Redis Task Queue](decisions/002-celery-task-queue.md)
- [ADR-003: Stub-First Model Strategy](decisions/003-stub-first-models.md)
- [ADR-004: LRU Model Registry](decisions/004-lru-model-registry.md)
- [ADR-005: Storage Abstraction](decisions/005-storage-abstraction.md)
```

- [ ] **Step 2: Commit**

```bash
git add docs/architecture/overview.md
git commit -m "docs: add architecture overview extracted from PLAN.md"
```

---

## Task 2 — Architecture Decision Records (5 ADRs)

**Files:**
- Create: `docs/architecture/decisions/001-pyav-video-io.md`
- Create: `docs/architecture/decisions/002-celery-task-queue.md`
- Create: `docs/architecture/decisions/003-stub-first-models.md`
- Create: `docs/architecture/decisions/004-lru-model-registry.md`
- Create: `docs/architecture/decisions/005-storage-abstraction.md`

- [ ] **Step 1: Write ADR-001**

Write `docs/architecture/decisions/001-pyav-video-io.md`:

```markdown
# ADR-001: PyAV for Video I/O

**Status:** Accepted  
**Date:** 2026-04-23

## Context

RestoraX processes video frame-by-frame through a chain of AI models. We needed
a Python video I/O library that could:
- Iterate frames without loading the entire file into memory
- Access presentation timestamps (PTS) for frame-accurate audio sync
- Pass audio through unchanged (without re-encoding) alongside video
- Avoid subprocess overhead (one ffmpeg process per frame is too slow)

Candidates evaluated: OpenCV (`cv2`), imageio, PyAV, direct FFmpeg subprocess.

## Decision

Use **PyAV** (`av` package) as the primary video I/O layer.

PyAV is a Pythonic binding to FFmpeg's libav* C libraries. It runs in-process,
gives direct access to AVFrame PTS values, and supports audio stream passthrough
without re-encoding. A thin subprocess call to FFmpeg remains for complex filter
graphs (YADIF deinterlacing, final mux) where libav's Python API is cumbersome.

Key implementation notes:
- Catch `Exception` broadly, not `av.AVError` — PyAV 17+ changed the exception hierarchy
- Use `Fraction(fps)` not float comparison for frame-rate arithmetic
- `VideoReader` yields `(frame_index, pts, numpy_array)` tuples; `VideoWriter` accepts the same

## Consequences

**Positive:**
- Constant memory: frames are streamed, not buffered
- Sub-millisecond per-frame overhead vs. subprocess
- Audio passthrough works without decoding/re-encoding audio streams
- Full FFmpeg codec support inherited from libav

**Negative:**
- PyAV API is less documented than OpenCV; error messages can be cryptic
- Some FFmpeg filter graphs (YADIF) still require subprocess fallback
- `av.AVError` hierarchy changed between PyAV minor versions — catch `Exception` defensively
```

- [ ] **Step 2: Write ADR-002**

Write `docs/architecture/decisions/002-celery-task-queue.md`:

```markdown
# ADR-002: Celery + Redis Task Queue

**Status:** Accepted  
**Date:** 2026-04-23

## Context

GPU restoration jobs are long-running (seconds to minutes), must run serially
per GPU (two jobs sharing a GPU causes OOM), and need real-time progress
reporting to the web UI. We needed a task queue that could:
- Route jobs to specific GPUs (queue-per-device)
- Enforce concurrency=1 per GPU worker
- Report incremental progress back to the API layer
- Retry failed jobs with backoff
- Be self-hostable without a managed service

Candidates: Celery + Redis, ARQ (asyncio-native), RQ (simple), raw asyncio queue.

## Decision

Use **Celery 5.3+** with **Redis 7** as both broker and result backend.

One Celery worker process per GPU, consuming from a dedicated queue
(`gpu_default`, or `gpu_0`, `gpu_1`, ... for multi-GPU setups). Concurrency is
set to 1 per worker. Progress is published via Redis pub/sub
(`restorax:job:{id}:progress`) and consumed by the FastAPI WebSocket endpoint.

```
RESTORAX_GPU_QUEUES=gpu_0,gpu_1   # one queue per GPU
celery worker --queues gpu_0 --concurrency=1 --hostname=worker-gpu0@%h
celery worker --queues gpu_1 --concurrency=1 --hostname=worker-gpu1@%h
```

## Consequences

**Positive:**
- Queue-per-GPU routing prevents multi-GPU contention
- Redis pub/sub is already present (broker) — no extra service for WebSocket progress
- Celery's retry/backoff policies handle transient GPU errors
- Flower UI available for job monitoring

**Negative:**
- Celery adds operational complexity vs. a simple asyncio queue
- Worker must be restarted to pick up code changes (no hot reload)
- Redis is a required external dependency even in local dev
```

- [ ] **Step 3: Write ADR-003**

Write `docs/architecture/decisions/003-stub-first-models.md`:

```markdown
# ADR-003: Stub-First Model Strategy

**Status:** Accepted  
**Date:** 2026-04-23

## Context

RestoraX integrates 24 AI restorers from diverse research repos. Each model
architecture has different dependencies, some incompatible with each other
(e.g., basicsr's torchvision import). Downloading real weights (~1–16 GB per
model) makes CI impractical. We needed every restorer to be testable and
runnable without real weights.

## Decision

Every restorer ships with a **stub model** that:
1. Passes `isinstance(model, nn.Module)` checks
2. Produces geometrically-correct output (correct shape, dtype, value range)
3. Runs on CPU without any model weights
4. Is replaced at runtime when real weights are available

The stub is a class at the bottom of each restorer file (e.g., `_RealESRGANStub`).
The restorer's `load()` method tries the real arch first; falls back to the stub
if weights or deps are unavailable.

**Vendoring** (activating a real arch):
1. Copy the model architecture file(s) into `restorax/restorers/<category>/<name>_arch.py`
2. Update the restorer's `load()` to import from the arch file
3. No other changes needed — the pipeline, registry, and API are arch-agnostic

Stubs pending vendoring (12 as of 2026-04-30):
DDColor, RIFE v4, ProPainter, HDRTVDM, Upscale-A-Video, MambaIR, TDM, GaVS,
CodeFormer++, Demucs, VoiceFixer, RNNoise

See ROADMAP.md for source repos and effort estimates.

## Consequences

**Positive:**
- CI runs in ~60s without GPU or model weights (309 tests pass with stubs)
- New restorers can be wired into the pipeline before the arch is vendored
- Plugin authors get a clear contract: implement `BaseRestorer`, stubs optional

**Negative:**
- Stubs produce no real quality improvement — demo output looks like bicubic upscale
- Two code paths (stub + real) must both be maintained per restorer
- basicsr's torchvision import incompatibility still present for real Real-ESRGAN
  (workaround: pin torchvision==0.15.2 or patch `degradations.py`)
```

- [ ] **Step 4: Write ADR-004**

Write `docs/architecture/decisions/004-lru-model-registry.md`:

```markdown
# ADR-004: LRU Model Registry

**Status:** Accepted  
**Date:** 2026-04-23

## Context

A restoration pipeline may chain 4–6 models (deinterlace → SR → colorize →
face → interpolate → audio). Loading all models simultaneously on a single GPU
with 8 GB VRAM would OOM. We needed a strategy to manage VRAM across pipeline
stages.

## Decision

`ModelRegistry` maintains a dict of loaded models with an LRU eviction policy.
Before loading model N, it evicts the least-recently-used model if
`len(loaded) >= max_loaded`. Default `max_loaded=2`.

```python
# config.py
RESTORAX_MAX_LOADED_MODELS: int = 2   # increase for high-VRAM GPUs

# core/registry.py
registry = ModelRegistry(max_loaded=settings.max_loaded_models)
model = registry.get("real_esrgan_x4plus")  # loads if not cached, evicts LRU if full
```

Models are moved to CPU before eviction (not deleted) so re-loading a recently
evicted model is faster than a cold load from disk.

## Consequences

**Positive:**
- Constant peak VRAM regardless of pipeline length
- Two-model cache covers most pipelines (SR + face, SR + colorize)
- Configurable for users with 16–24 GB VRAM GPUs who want less swapping

**Negative:**
- Long pipelines with many unique models pay a swap cost between stages
- CPU offload memory adds ~1–2 GB RAM per offloaded model
- Not suitable for real-time use cases (swap latency is 200ms–2s per model)
```

- [ ] **Step 5: Write ADR-005**

Write `docs/architecture/decisions/005-storage-abstraction.md`:

```markdown
# ADR-005: Storage Abstraction via Protocol

**Status:** Accepted  
**Date:** 2026-04-23

## Context

In local dev, input/output videos live on the local filesystem. In production,
they must be stored in object storage (MinIO/S3) so API servers and GPU workers
can share access across machines. We needed to switch storage backends at deploy
time without changing application code.

## Decision

`StorageBackend` is a `typing.Protocol` with three methods:
- `save(local_path, remote_key) -> str` — upload and return a URI
- `load(remote_key, local_path)` — download to local path
- `delete(remote_key)`

Two implementations:
- `LocalStorage` (dev): files stay on disk; URI = absolute path
- `S3Storage` (prod): wraps `boto3`; URI = `s3://bucket/key`

The active backend is injected via FastAPI dependency injection (`deps.py`).
Switching from local to S3 requires only an env var change:

```env
RESTORAX_STORAGE_BACKEND=s3
RESTORAX_S3_BUCKET=restorax-jobs
RESTORAX_S3_ENDPOINT=http://minio:9000
```

## Consequences

**Positive:**
- Zero code changes when deploying to cloud (only `.env` changes)
- LocalStorage makes dev setup trivial (no MinIO required)
- Protocol-based — third-party storage backends installable as plugins

**Negative:**
- S3Storage adds `boto3` as a required prod dependency
- Large videos transferred over the network add latency between API and worker
- Local dev with docker-compose must mount a shared volume if API and worker
  run in separate containers
```

- [ ] **Step 6: Commit all ADRs**

```bash
git add docs/architecture/decisions/
git commit -m "docs: add 5 architecture decision records"
```

---

## Task 3 — ROADMAP.md

**Files:**
- Create: `ROADMAP.md`

- [ ] **Step 1: Write ROADMAP.md**

```markdown
# Roadmap

## Now — Model Activation

12 restorers ship with geometrically-correct stub models that produce
correct-shape output but no real enhancement. Vendoring the real architecture
activates full quality with no other code changes required.

See [ADR-003](docs/architecture/decisions/003-stub-first-models.md) for the
stub strategy and vendoring process.

| Restorer | Stub file | Source repo | Notes |
|---|---|---|---|
| DDColor | `restorers/colorization/ddcolor.py` | [piddnad/DDColor](https://github.com/piddnad/DDColor) | Copy arch to `ddcolor_arch.py` |
| RIFE v4 | `restorers/frame_interpolation/rife.py` | [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE) | Copy arch dir to `rife_arch/` |
| ProPainter | `restorers/artifact_removal/scratch_removal.py` | [sczhou/ProPainter](https://github.com/sczhou/ProPainter) | Copy arch to `propainter_arch.py` |
| HDRTVDM | `restorers/hdr/hdrtvdm.py` | [AndreGuo/HDRTVDM](https://github.com/AndreGuo/HDRTVDM) | Copy arch to `hdrtvdm_arch.py` |
| Upscale-A-Video | `restorers/super_resolution/upscale_a_video.py` | [sczhou/Upscale-A-Video](https://github.com/sczhou/Upscale-A-Video) | Copy arch to `upscale_a_video_arch.py` |
| MambaIR | `restorers/super_resolution/mamba_ir.py` | [csguoh/MambaIR](https://github.com/csguoh/MambaIR) | Copy arch to `mamba_ir_arch.py` |
| TDM | `restorers/super_resolution/tdm.py` | [ChenyangSi/TDM](https://huggingface.co/ChenyangSi/TDM) | Copy arch to `tdm_arch.py` |
| GaVS | `restorers/stabilization/gavs.py` | Awaiting SIGGRAPH 2025 release | Copy arch to `gavs_arch/` |
| CodeFormer++ | `restorers/face_restoration/codeformer_pp.py` | [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) | Copy arch to `codeformer_pp_arch.py` |
| Demucs | `restorers/audio/demucs.py` | `pip install demucs` | Auto-loaded via `demucs.pretrained.get_model()` |
| VoiceFixer | `restorers/audio/voicefixer.py` | `pip install voicefixer` | Auto-loaded via `VoiceFixer()` |
| RNNoise | `restorers/audio/rnnoise.py` | `pip install rnnoise-python` | Auto-detected at load time |

## Next — Quality & Performance

- **ONNX export + TensorRT optimization** — production inference without Python overhead
- **No-reference quality metrics** — DOVER and FasterVQA for blind quality scoring on footage without ground truth
- **Batch job API** — submit multiple jobs in one request; priority queue support
- **GaVS stabilization** — pending SIGGRAPH 2025 source release

## Future

- **Managed cloud deployment** (RestoraX Cloud) — one-click restoration via hosted API
- **Fine-tuning guide + LoRA adapter support** — domain-specific restoration for specific film stocks, cameras, eras
- **Real-time preview mode** — sub-second latency for short clips via model caching + tiled inference
- **Browser extension** — restore videos directly on YouTube, Vimeo, and archival sites
```

- [ ] **Step 2: Commit**

```bash
git add ROADMAP.md
git commit -m "docs: add ROADMAP.md with model activation table and future milestones"
```

---

## Task 4 — CHANGELOG.md

**Files:**
- Create: `CHANGELOG.md`

- [ ] **Step 1: Write CHANGELOG.md**

```markdown
# Changelog

All notable changes to RestoraX are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Pending
- Vendor real model architectures for 12 restorers currently using stubs
  (DDColor, RIFE v4, ProPainter, HDRTVDM, Upscale-A-Video, MambaIR, TDM,
  GaVS, CodeFormer++, Demucs, VoiceFixer, RNNoise) — see ROADMAP.md

---

## [1.0.0] — 2026-04-30

### Added
- 24 restorers across super-resolution, colorization, face restoration,
  frame interpolation, deinterlacing, artifact removal, HDR, stabilization,
  and audio restoration
- REST API (FastAPI 0.111+) with Pydantic v2 schemas
- WebSocket endpoint (`/ws/jobs/{id}/progress`) for real-time progress
- Celery 5.3+ GPU worker with per-GPU queue routing and LRU model registry
- Next.js 14 web UI with drag-and-drop upload and before/after comparison slider
- CLI (`restorax run`, `restorax models`, `restorax benchmark`)
- 334 passing tests: 309 Python (unit + integration + system) + 25 frontend
- Plugin system: third-party restorers via `pip install restorax-plugin-*`
- Multi-GPU support: one Celery worker per GPU, configurable via `RESTORAX_GPU_QUEUES`
- Docker Compose: dev (hot-reload, SQLite) and prod (GPU, PostgreSQL, MinIO)
- Alembic migrations for SQLAlchemy 2.0 async
- MinIO/S3 storage backend (`StorageBackend` Protocol)
- MIT license, CONTRIBUTING.md, GitHub issue/PR templates, CI pipeline (GitHub Actions)
- `docker-compose.deps.yml` for spinning up Redis + Postgres with one command
- `Procfile` and `Procfile.dev` for honcho process management

### Fixed
- Alembic `env.py` was stripping `+aiosqlite` from the database URL then using
  `async_engine_from_config`, causing migrations to crash with an async driver error.
  Fixed by preserving the full async URL.
- `run_job` Celery task published progress to Redis pub/sub but never updated the
  database — jobs remained in `queued` status forever. Fixed with `_update_job_db()`
  helper called at task start, completion, and failure.
- `_RealESRGANStub` class was referenced in `real_esrgan.py` but never defined.
  Stub class added and `.to(device)` call added on load.

---

## [0.6.0] — 2026-04-29 (Phase 6 — Hardening + Open Source)

### Added
- Open-source release files: MIT license, CONTRIBUTING.md, `.dockerignore`
- GitHub templates: `PULL_REQUEST_TEMPLATE.md`, `bug_report.md`, `feature_request.md`
- CI pipeline: `ci.yml` with unit, integration, system, and frontend test jobs
- `docker-compose.deps.yml`: single-command Redis + Postgres for local dev
- `Procfile.dev`: honcho process manager with env-var support (`$FLOWER_PORT`, `$RESTORAX_GPU_QUEUES`)
- Benchmark CLI: `restorax benchmark run` — skips restorers with missing deps gracefully
- 72 new tests: `test_cli.py`, `test_config.py`, `test_exceptions.py`, `test_storage.py`,
  `test_api_extended.py`, system smoke test, frontend lib and component tests

### Fixed
- `VRAMMonitor` was tracking absolute VRAM peak instead of delta — CPU benchmarks
  incorrectly reported non-zero VRAM usage. Fixed to track delta.
- `lpips()` and `ssim()` now accept an explicit `device` parameter.
- Shared conftest files added to prevent env-var races when running all tests together.

---

## [0.5.0] — 2026-04-28 (Phase 5 — Advanced Models + Performance)

### Added
- Audio restorers: Demucs (htdemucs), VoiceFixer, RNNoise
- Super-resolution: MambaIR (SSM-based), TDM (all-in-one diffusion), SeedVR (CVPR 2025)
- Face restoration: CodeFormer++ (2025), DICFace (ICCV 2023)
- Stabilization: GaVS stub (awaiting SIGGRAPH 2025 release), deep optical flow stub
- Metrics: full-reference (PSNR, SSIM, LPIPS via piqa, VMAF) and no-reference (NIQE, DOVER)
- Benchmark runner: `scripts/run_benchmarks.py` with CUDA timing and standard protocols

---

## [0.4.0] — 2026-04-27 (Phase 4 — Extended Restorer Library)

### Added
- Super-resolution: FlashVSR, EVTexture, waifu2x
- Frame interpolation: RIFE v4.22
- HDR conversion: HDRTVDM (CVPR 2023)
- Artifact removal: scratch/dust removal (ProPainter-based stub)
- Deinterlacing: AI deinterlace stub + YADIF via FFmpeg subprocess
- Built-in pipelines: `classic_film`, `classic_film_audio`, `anime_upscale`, `vhs_restoration`, `newsreel`

---

## [0.3.0] — 2026-04-26 (Phase 3 — Web UI + API Completion)

### Added
- Next.js 14 frontend: drag-and-drop upload, pipeline selector, progress bar, before/after slider
- WebSocket progress streaming from Celery worker via Redis pub/sub
- REST endpoints: `POST /jobs`, `GET /jobs/{id}`, `DELETE /jobs/{id}`, `GET /jobs/{id}/download`
- Pipeline CRUD API: `GET/POST/PUT /pipelines`
- Model listing API: `GET /models`
- Docker Compose dev and prod configurations

---

## [0.2.0] — 2026-04-25 (Phase 2 — MVP Restorers)

### Added
- Real-ESRGAN super-resolution (BasicSR arch, HuggingFace Hub weights)
- BasicVSR++ temporal super-resolution (vendored arch)
- CodeFormer face restoration (vendored arch)
- GFPGAN face restoration (vendored arch)
- DDColor colorization (stub)
- `sr_x4` and `sr_x4_face` built-in pipelines

---

## [0.1.0] — 2026-04-23 (Phase 1 — Foundation)

### Added
- Project scaffold: `pyproject.toml`, `.env.example`, conda environment
- `BaseRestorer` ABC, `PipelineRunner`, `ModelRegistry` (LRU)
- `VideoReader` and `VideoWriter` via PyAV
- SQLAlchemy 2.0 async ORM with Alembic migrations
- Celery task queue with Redis broker
- FastAPI application factory with lifespan management
- Click CLI skeleton
- Dockerfile and Dockerfile.worker (CUDA 12.1 base)
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: add CHANGELOG.md transformed from PROGRESS.md"
```

---

## Task 5 — docs/models/ index and model cards

**Files:**
- Create: `docs/models/README.md`
- Create: `docs/models/super_resolution/real_esrgan.md`
- Create: `docs/models/super_resolution/basicvsr_pp.md`
- Create: `docs/models/super_resolution/waifu2x.md`
- Create: `docs/models/super_resolution/flashvsr.md`
- Create: `docs/models/super_resolution/mamba_ir.md`
- Create: `docs/models/super_resolution/evtexture.md`
- Create: `docs/models/super_resolution/vrt.md`
- Create: `docs/models/super_resolution/upscale_a_video.md`
- Create: `docs/models/super_resolution/tdm.md`
- Create: `docs/models/super_resolution/seedvr.md`
- Create: `docs/models/colorization/ddcolor.md`
- Create: `docs/models/face_restoration/codeformer.md`
- Create: `docs/models/face_restoration/codeformer_pp.md`
- Create: `docs/models/face_restoration/dicface.md`
- Create: `docs/models/face_restoration/gfpgan.md`
- Create: `docs/models/frame_interpolation/rife.md`
- Create: `docs/models/deinterlacing/ai_deinterlace.md`
- Create: `docs/models/artifact_removal/scratch_removal.md`
- Create: `docs/models/hdr/hdrtvdm.md`
- Create: `docs/models/stabilization/deep_flow_stab.md`
- Create: `docs/models/stabilization/gavs.md`
- Create: `docs/models/audio/demucs.md`
- Create: `docs/models/audio/voicefixer.md`
- Create: `docs/models/audio/rnnoise.md`

- [ ] **Step 1: Create directories**

```bash
mkdir -p docs/models/super_resolution docs/models/colorization \
  docs/models/face_restoration docs/models/frame_interpolation \
  docs/models/deinterlacing docs/models/artifact_removal \
  docs/models/hdr docs/models/stabilization docs/models/audio
```

- [ ] **Step 2: Write docs/models/README.md**

```markdown
# Model Index

All restorers included in RestoraX. Benchmark numbers use standard SR protocols
(bicubic ×4 downscale, Set5/Set14) unless noted. GPU speed measured on RTX 3090.

## Super-Resolution

| Model | Scale | Status | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Speed (fps) | VRAM | Paper |
|---|---|---|---|---|---|---|---|---|
| waifu2x | 2× | 🔧 stub | 29.0 dB | 0.830 | 0.115 | ~80 | 1 GB | Nagadomi 2014 |
| real_esrgan | 4× | ✅ active | 28.4 dB | 0.821 | 0.123 | ~12 | 4 GB | Wang et al. ICCVW 2021 |
| flashvsr | 4× | 🔧 stub | 28.8 dB | 0.827 | 0.119 | ~40 | 2 GB | 2024 |
| mamba_ir | 4× | 🔧 stub | 29.1 dB | 0.835 | 0.118 | ~18 | 3 GB | Guo et al. ECCV 2024 |
| evtexture | 4× | 🔧 stub | 29.6 dB | 0.843 | 0.112 | ~8 | 6 GB | Kai et al. ICML 2024 |
| basicvsr_pp | 4× | ✅ active | 30.2 dB | 0.851 | 0.109 | ~3 | 8 GB | Chan et al. CVPR 2022 |
| vrt | 4× | 🔧 stub | 30.8 dB | 0.858 | 0.105 | ~1.4 | 8 GB | Liang et al. TIP 2024 |
| upscale_a_video | 4× | 🔧 stub | 32.1 dB | 0.877 | 0.092 | ~0.4 | 12 GB | Zhou et al. CVPR 2024 |
| tdm | 4× | 🔧 stub | 33.0 dB | 0.891 | 0.082 | ~0.2 | 12 GB | Si et al. 2025 |
| seedvr | 4× | 🔧 stub | 33.5 dB | 0.898 | 0.075 | ~0.1 | 16 GB | Iceclear CVPR 2025 |

## Colorization

| Model | Task | Status | SSIM (grayscale) | SSIM (partial desat) | Speed (fps) | Paper |
|---|---|---|---|---|---|---|
| ddcolor | B&W → Color | 🔧 stub | 0.734 | 0.788 | ~22 | Kang et al. ICCV 2023 |

## Face Restoration

| Model | Status | Light PSNR ↑ | Heavy SSIM ↑ | Speed (fps) | Paper |
|---|---|---|---|---|---|
| codeformer | ✅ active | 27.6 dB | 0.764 | ~9 | Zhou et al. NeurIPS 2022 |
| gfpgan | ✅ active | 27.1 dB | 0.758 | ~11 | Wang et al. CVPR 2021 |
| dicface | 🔧 stub | 28.1 dB | 0.779 | ~7 | Zhang et al. ICCV 2023 |
| codeformer_pp | 🔧 stub | 28.3 dB | 0.785 | ~6 | 2025 |

## Frame Interpolation, Deinterlacing, Artifact Removal, HDR, Stabilization

| Model | Task | Status | Paper |
|---|---|---|---|
| rife | 24fps → 48fps | 🔧 stub | Huang et al. ECCV 2022 (v4.22) |
| ai_deinterlace | Deinterlacing | 🔧 stub | — |
| scratch_removal | Scratch/dust removal | 🔧 stub (ProPainter) | Zhou et al. ECCV 2023 |
| hdrtvdm | SDR → HDR10 | 🔧 stub | Guo et al. CVPR 2023 |
| deep_flow_stab | Stabilization | 🔧 stub | — |
| gavs | Stabilization (3D-grounded) | 🔧 stub | SIGGRAPH 2025 |

## Audio Restoration

| Model | Task | Status | AWGN 20dB → | Clipping -6dB → | Paper |
|---|---|---|---|---|---|
| rnnoise | Noise removal | 🔧 stub | ~28 dB | — | Valin 2018 |
| voicefixer | Noise + clipping | 🔧 stub | ~30 dB | ~18 dB | Liu et al. 2022 |
| demucs | Noise + clipping | 🔧 stub | ~32 dB | ~22 dB | Défossez et al. 2023 |

> 🔧 stub = correct-shape output, no real enhancement. See [ADR-003](../architecture/decisions/003-stub-first-models.md) and [ROADMAP.md](../../ROADMAP.md) to activate.
> Speed figures are approximate on RTX 3090. Stub speed on CPU is ~10–50× slower.
```

- [ ] **Step 3: Write super-resolution model cards**

Write `docs/models/super_resolution/real_esrgan.md`:
```markdown
# Real-ESRGAN

**Category:** Super-Resolution · **Scale:** 4×
**Status:** ✅ Active (BasicSR RRDBNet architecture)
**Paper:** Wang et al., ICCVW 2021 — [arXiv:2107.10833](https://arxiv.org/abs/2107.10833)
**Source:** [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
**Restorer file:** `restorax/restorers/super_resolution/real_esrgan.py`

## Performance
| Metric | Value | Protocol |
|---|---|---|
| PSNR | 28.4 dB | Bicubic ×4, Set5/Set14 |
| SSIM | 0.821 | — |
| LPIPS | 0.123 | — |
| Speed | ~12 fps | RTX 3090 |
| VRAM | 4 GB | — |

## Architecture
Real-ESRGAN extends ESRGAN with a practical degradation model for real-world
restoration. Uses a Residual-in-Residual Dense Block (RRDB) generator with a
U-Net discriminator and spectral normalization. Trained on a synthetic degradation
pipeline that combines blur, noise, JPEG compression, and resizing in random order.

## License
BSD 3-Clause — commercial use permitted.

## Known Limitations
- basicsr 1.4.2 imports `rgb_to_grayscale` from `torchvision.transforms.functional_tensor`
  which was removed in torchvision 0.16+. Fix: pin `torchvision==0.15.2` or patch
  `functional_tensor.py` — see [ADR-003](../../architecture/decisions/003-stub-first-models.md).
- Can hallucinate texture on very low-resolution inputs (<64px).

## Weights
Auto-downloaded from HuggingFace Hub (`ai-forever/Real-ESRGAN`) on first use.
```

Write `docs/models/super_resolution/basicvsr_pp.md`:
```markdown
# BasicVSR++

**Category:** Super-Resolution · **Scale:** 4× · **Type:** Video (temporal)
**Status:** ✅ Active (vendored architecture in `models/basicvsr_pp/`)
**Paper:** Chan et al., CVPR 2022 — [arXiv:2204.13767](https://arxiv.org/abs/2204.13767)
**Source:** [ckkelvinchan/BasicVSR_PlusPlus](https://github.com/ckkelvinchan/BasicVSR_PlusPlus)
**Restorer file:** `restorax/restorers/super_resolution/basicvsr_pp.py`

## Performance
| Metric | Value | Protocol |
|---|---|---|
| PSNR | 30.2 dB | Bicubic ×4, Set5/Set14 |
| SSIM | 0.851 | — |
| LPIPS | 0.109 | — |
| Speed | ~3 fps | RTX 3090 |
| VRAM | 8 GB | — |

## Architecture
BasicVSR++ uses second-order grid propagation and flow-guided deformable alignment
across frames. It improves over BasicVSR by enabling information to flow both
forward and backward across multiple hops, dramatically improving temporal
consistency on video super-resolution.

## License
Apache 2.0 — commercial use permitted.

## Known Limitations
- Requires 8+ GB VRAM; falls back to chunked processing on smaller GPUs.
- Speed drops significantly on long videos due to bidirectional propagation.
```

Write `docs/models/super_resolution/waifu2x.md`:
```markdown
# waifu2x

**Category:** Super-Resolution · **Scale:** 2×
**Status:** 🔧 Stub — vendor arch to activate
**Paper:** Nagadomi 2014 (unpublished; open-source release)
**Source:** [nagadomi/waifu2x](https://github.com/nagadomi/waifu2x)
**Restorer file:** `restorax/restorers/super_resolution/waifu2x.py`

## Performance
| Metric | Value | Protocol |
|---|---|---|
| PSNR | 29.0 dB | Bicubic ×2, Set5/Set14 |
| SSIM | 0.830 | — |
| LPIPS | 0.115 | — |
| Speed | ~80 fps | RTX 3090 |
| VRAM | 1 GB | — |

## Architecture
Convolutional neural network trained specifically on anime-style artwork and
photographs. Fast and lightweight — designed for 2× upscaling with minimal VRAM.
The original Lua/Torch implementation has been reimplemented in PyTorch by the community.

## License
MIT — commercial use permitted.

## Vendoring
Copy the PyTorch architecture from [yukiCodesStuff/waifu2x-pytorch](https://github.com/yukiCodesStuff/waifu2x-pytorch)
or equivalent into `restorax/restorers/super_resolution/waifu2x_arch.py`, then
update the `load()` method to import from it.
```

Write `docs/models/super_resolution/flashvsr.md`:
```markdown
# FlashVSR

**Category:** Super-Resolution · **Scale:** 4× · **Type:** Video (temporal)
**Status:** 🔧 Stub — vendor arch to activate
**Paper:** 2024 (preprint)
**Restorer file:** `restorax/restorers/super_resolution/flashvsr.py`

## Performance
| Metric | Value | Protocol |
|---|---|---|
| PSNR | 28.8 dB | Bicubic ×4, Set5/Set14 |
| SSIM | 0.827 | — |
| LPIPS | 0.119 | — |
| Speed | ~40 fps | RTX 3090 |
| VRAM | 2 GB | — |

## Architecture
FlashVSR is optimized for real-time video super-resolution with a lightweight
recurrent architecture. Achieves ~40fps on RTX 3090 at 4× scale — the fastest
temporal VSR in this collection at this quality level.

## License
Check source repository before commercial use.

## Vendoring
Locate the official repository and copy the architecture into
`restorax/restorers/super_resolution/flashvsr_arch.py`.
```

Write `docs/models/super_resolution/mamba_ir.md`:
```markdown
# MambaIR

**Category:** Super-Resolution · **Scale:** 4×
**Status:** 🔧 Stub — vendor arch to activate
**Paper:** Guo et al., ECCV 2024 — [arXiv:2402.15648](https://arxiv.org/abs/2402.15648)
**Source:** [csguoh/MambaIR](https://github.com/csguoh/MambaIR)
**Restorer file:** `restorax/restorers/super_resolution/mamba_ir.py`

## Performance
| Metric | Value | Protocol |
|---|---|---|
| PSNR | 29.1 dB | Bicubic ×4, Set5/Set14 |
| SSIM | 0.835 | — |
| LPIPS | 0.118 | — |
| Speed | ~18 fps | RTX 3090 |
| VRAM | 3 GB | — |

## Architecture
MambaIR applies Mamba (State Space Model / SSM) to image restoration. SSMs model
long-range dependencies with linear complexity unlike Transformers (quadratic).
This gives MambaIR a memory efficiency advantage over SwinIR/VRT while maintaining
competitive PSNR.

## License
Apache 2.0 — commercial use permitted.

## Vendoring
```bash
git clone https://github.com/csguoh/MambaIR
cp MambaIR/basicsr/archs/mambair_arch.py \
   restorax/restorers/super_resolution/mamba_ir_arch.py
```
Then update `mamba_ir.py` `load()` to import `MambaIR` from `mamba_ir_arch`.
Requires `pip install mamba-ssm causal-conv1d` (CUDA only).
```

Write `docs/models/super_resolution/evtexture.md`:
```markdown
# EVTexture

**Category:** Super-Resolution · **Scale:** 4× · **Type:** Video (event-guided)
**Status:** 🔧 Stub — vendor arch to activate
**Paper:** Kai et al., ICML 2024
**Restorer file:** `restorax/restorers/super_resolution/evtexture.py`

## Performance
| Metric | Value | Protocol |
|---|---|---|
| PSNR | 29.6 dB | Bicubic ×4, Set5/Set14 |
| SSIM | 0.843 | — |
| LPIPS | 0.112 | — |
| Speed | ~8 fps | RTX 3090 |
| VRAM | 6 GB | — |

## Architecture
EVTexture uses event camera data to guide texture restoration in video SR.
Event cameras capture per-pixel brightness changes at microsecond resolution,
providing high-frequency temporal information that frame-based cameras miss.

## License
Check source repository before commercial use.

## Vendoring
Locate the official ICML 2024 code release and copy the architecture into
`restorax/restorers/super_resolution/evtexture_arch.py`.
```

Write `docs/models/super_resolution/vrt.md`:
```markdown
# VRT — Video Restoration Transformer

**Category:** Super-Resolution · **Scale:** 4× · **Type:** Video (temporal)
**Status:** 🔧 Stub — vendor arch to activate
**Paper:** Liang et al., TIP 2024 — [arXiv:2201.12288](https://arxiv.org/abs/2201.12288)
**Source:** [JingyunLiang/VRT](https://github.com/JingyunLiang/VRT)
**Restorer file:** `restorax/restorers/super_resolution/vrt.py`

## Performance
| Metric | Value | Protocol |
|---|---|---|
| PSNR | 30.8 dB | Bicubic ×4, Set5/Set14 |
| SSIM | 0.858 | — |
| LPIPS | 0.105 | — |
| Speed | ~1.4 fps | RTX 3090 |
| VRAM | 8 GB | — |

## Architecture
VRT uses mutual attention for cross-frame feature alignment and temporal mutual
self-attention for long-range temporal modeling. It is the Transformer successor
to BasicVSR++, trading speed for quality.

## License
Apache 2.0 — commercial use permitted.

## Vendoring
```bash
git clone https://github.com/JingyunLiang/VRT
cp VRT/models/network_vrt.py \
   restorax/restorers/super_resolution/vrt_arch.py
```
```

Write `docs/models/super_resolution/upscale_a_video.md`:
```markdown
# Upscale-A-Video

**Category:** Super-Resolution · **Scale:** 4× · **Type:** Video (diffusion)
**Status:** 🔧 Stub — vendor arch to activate
**Paper:** Zhou et al., CVPR 2024 — [arXiv:2312.06640](https://arxiv.org/abs/2312.06640)
**Source:** [sczhou/Upscale-A-Video](https://github.com/sczhou/Upscale-A-Video)
**Restorer file:** `restorax/restorers/super_resolution/upscale_a_video.py`

## Performance
| Metric | Value | Protocol |
|---|---|---|
| PSNR | 32.1 dB | Bicubic ×4, Set5/Set14 |
| SSIM | 0.877 | — |
| LPIPS | 0.092 | — |
| Speed | ~0.4 fps | RTX 3090 |
| VRAM | 12 GB | — |

## Architecture
Upscale-A-Video applies latent diffusion to video super-resolution, using
temporal attention layers inserted into a pretrained text-to-video diffusion model.
Diffusion-based generation achieves highest perceptual quality at the cost of speed.

## License
S-Lab License 1.0 — non-commercial research only. Check before production use.

## Vendoring
```bash
git clone https://github.com/sczhou/Upscale-A-Video
# Copy diffusion architecture and pretrained VAE config
cp -r Upscale-A-Video/models \
   restorax/restorers/super_resolution/upscale_a_video_arch.py
```
Requires `pip install diffusers transformers accelerate`.
```

Write `docs/models/super_resolution/tdm.md`:
```markdown
# TDM — Text-guided Diffusion Model

**Category:** Super-Resolution · **Scale:** 4× · **Type:** Diffusion all-in-one
**Status:** 🔧 Stub — vendor arch to activate
**Paper:** Si et al., 2025
**Source:** [ChenyangSi/TDM](https://huggingface.co/ChenyangSi/TDM)
**Restorer file:** `restorax/restorers/super_resolution/tdm.py`

## Performance
| Metric | Value | Protocol |
|---|---|---|
| PSNR | 33.0 dB | Bicubic ×4, Set5/Set14 |
| SSIM | 0.891 | — |
| LPIPS | 0.082 | — |
| Speed | ~0.2 fps | RTX 3090 |
| VRAM | 12 GB | — |

## Architecture
TDM is an all-in-one restoration model based on latent diffusion. It handles
super-resolution, denoising, deblurring, and JPEG artifact removal in a single
model controlled via text prompts. Highest non-SeedVR quality in the collection.

## License
Check HuggingFace model card before commercial use.

## Vendoring
Download weights and architecture from `huggingface.co/ChenyangSi/TDM`.
Copy arch to `restorax/restorers/super_resolution/tdm_arch.py`.
```

Write `docs/models/super_resolution/seedvr.md`:
```markdown
# SeedVR

**Category:** Super-Resolution · **Scale:** 4× · **Type:** Video diffusion
**Status:** 🔧 Stub — vendor arch to activate
**Paper:** Iceclear, CVPR 2025
**Source:** [Iceclear/SeedVR](https://github.com/Iceclear/SeedVR)
**Restorer file:** `restorax/restorers/super_resolution/seedvr.py`

## Performance
| Metric | Value | Protocol |
|---|---|---|
| PSNR | 33.5 dB | Bicubic ×4, Set5/Set14 |
| SSIM | 0.898 | — |
| LPIPS | 0.075 | — |
| Speed | ~0.1 fps | RTX 3090 |
| VRAM | 16 GB | — |

## Architecture
SeedVR achieves the highest quality in the collection using a video diffusion
backbone with temporal consistency constraints. State-of-the-art PSNR/SSIM/LPIPS
as of CVPR 2025 at the cost of being the slowest restorer (~0.1fps).

## License
Check source repository before commercial use.

## Vendoring
Clone the official CVPR 2025 release and copy architecture to
`restorax/restorers/super_resolution/seedvr_arch.py`.
```

- [ ] **Step 4: Write colorization model card**

Write `docs/models/colorization/ddcolor.md`:
```markdown
# DDColor

**Category:** Colorization
**Status:** 🔧 Stub — vendor arch to activate
**Paper:** Kang et al., ICCV 2023 — [arXiv:2212.11613](https://arxiv.org/abs/2212.11613)
**Source:** [piddnad/DDColor](https://github.com/piddnad/DDColor)
**Restorer file:** `restorax/restorers/colorization/ddcolor.py`

## Performance
| Input | SSIM ↑ | Speed (fps) |
|---|---|---|
| Full grayscale | 0.734 | ~22 |
| Partial desaturation (50%) | 0.788 | ~22 |

## Architecture
DDColor operates in LAB color space, predicting AB color channels from the L
(luminance) channel. Uses a dual-decoder architecture: one for semantic understanding
and one for color generation. Produces natural-looking colorization on diverse content.

## License
Apache 2.0 — commercial use permitted.

## Vendoring
```bash
git clone https://github.com/piddnad/DDColor
cp DDColor/modelscope/models/ddcolor_model.py \
   restorax/restorers/colorization/ddcolor_arch.py
```
Update `ddcolor.py` `load()` to import `DDColor` from `ddcolor_arch`.
```

- [ ] **Step 5: Write face restoration model cards**

Write `docs/models/face_restoration/codeformer.md`:
```markdown
# CodeFormer

**Category:** Face Restoration
**Status:** ✅ Active (vendored architecture in `models/gfpgan/`)
**Paper:** Zhou et al., NeurIPS 2022 — [arXiv:2206.11253](https://arxiv.org/abs/2206.11253)
**Source:** [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer)
**Restorer file:** `restorax/restorers/face_restoration/codeformer.py`

## Performance
| Degradation | PSNR ↑ | SSIM ↑ | Speed (fps) |
|---|---|---|---|
| Light | 27.6 dB | — | ~9 |
| Heavy | — | 0.764 | ~9 |

## Architecture
CodeFormer uses a VQGAN codebook to map degraded faces to high-quality face priors,
then applies a Transformer to predict the optimal code sequence. A controllable
fidelity weight (0–1) balances between faithful reconstruction and quality enhancement.

## License
S-Lab License 1.0 — non-commercial. Check before production use.

## Known Limitations
- Works best on frontal/near-frontal faces; struggles with extreme angles.
- Fidelity weight must be tuned per use case (default 0.5).
```

Write `docs/models/face_restoration/gfpgan.md`:
```markdown
# GFPGAN

**Category:** Face Restoration
**Status:** ✅ Active (vendored architecture in `models/gfpgan/`)
**Paper:** Wang et al., CVPR 2021 — [arXiv:2101.04061](https://arxiv.org/abs/2101.04061)
**Source:** [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN)
**Restorer file:** `restorax/restorers/face_restoration/gfpgan.py`

## Performance
| Degradation | PSNR ↑ | SSIM ↑ | Speed (fps) |
|---|---|---|---|
| Light | 27.1 dB | — | ~11 |
| Heavy | — | 0.758 | ~11 |

## Architecture
GFPGAN leverages face priors from a pretrained StyleGAN2 generator via spatial
feature transform layers. The generative face prior provides strong structural
and texture priors for blind face restoration.

## License
Apache 2.0 — commercial use permitted.
```

Write `docs/models/face_restoration/dicface.md`:
```markdown
# DICFace

**Category:** Face Restoration
**Status:** 🔧 Stub — vendor arch to activate
**Paper:** Zhang et al., ICCV 2023
**Restorer file:** `restorax/restorers/face_restoration/dicface.py`

## Performance
| Degradation | PSNR ↑ | SSIM ↑ | Speed (fps) |
|---|---|---|---|
| Light | 28.1 dB | — | ~7 |
| Heavy | — | 0.779 | ~7 |

## Architecture
DICFace uses dictionary-based identity-consistent face restoration, maintaining
subject identity more faithfully than GAN-prior approaches on heavy degradation.

## License
Check source repository before commercial use.

## Vendoring
Locate the ICCV 2023 official code release and copy the architecture into
`restorax/restorers/face_restoration/dicface_arch.py`.
```

Write `docs/models/face_restoration/codeformer_pp.md`:
```markdown
# CodeFormer++

**Category:** Face Restoration
**Status:** 🔧 Stub — vendor arch to activate
**Paper:** 2025 (extension of CodeFormer)
**Source:** [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer)
**Restorer file:** `restorax/restorers/face_restoration/codeformer_pp.py`

## Performance
| Degradation | PSNR ↑ | SSIM ↑ | Speed (fps) |
|---|---|---|---|
| Light | 28.3 dB | — | ~6 |
| Heavy | — | 0.785 | ~6 |

## Architecture
CodeFormer++ extends the original CodeFormer with an improved codebook and
enhanced fidelity control, achieving higher PSNR and SSIM than the original
at the cost of slightly lower throughput.

## License
Check source repository before commercial use.

## Vendoring
Copy updated arch from the CodeFormer++ branch to
`restorax/restorers/face_restoration/codeformer_pp_arch.py`.
```

- [ ] **Step 6: Write remaining model cards**

Write `docs/models/frame_interpolation/rife.md`:
```markdown
# RIFE v4.22

**Category:** Frame Interpolation
**Status:** 🔧 Stub — vendor arch to activate
**Paper:** Huang et al., ECCV 2022 — [arXiv:2011.06294](https://arxiv.org/abs/2011.06294)
**Source:** [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE)
**Restorer file:** `restorax/restorers/frame_interpolation/rife.py`

## Task
Doubles frame rate (e.g., 24fps → 48fps) or generates slow-motion output via
optical-flow-guided intermediate frame synthesis.

## Architecture
RIFE (Real-Time Intermediate Flow Estimation) uses an IFNet to directly estimate
intermediate optical flow between two frames. v4.22 adds ensemble inference and
improved flow estimation for complex motion. Fastest high-quality interpolation model.

## License
MIT — commercial use permitted.

## Vendoring
```bash
git clone https://github.com/hzwer/Practical-RIFE
cp -r Practical-RIFE/model/ \
   restorax/restorers/frame_interpolation/rife_arch/
```
```

Write `docs/models/deinterlacing/ai_deinterlace.md`:
```markdown
# AI Deinterlace

**Category:** Deinterlacing
**Status:** 🔧 Stub
**Restorer file:** `restorax/restorers/deinterlacing/ai_deinterlace.py`

## Task
Converts interlaced video (combed fields) to progressive scan. Used for VHS,
broadcast TV, and pre-digital archival footage.

## Architecture
Stub uses bob-field conversion (each field → one frame). Real architecture
targets multi-picture deformable convolution with self-attention (2024 paper).
YADIF via FFmpeg subprocess is used as a production fallback for files that
cannot wait for AI processing.

## Vendoring
Identify the 2024 multi-picture deformable conv deinterlacing paper and copy
architecture to `restorax/restorers/deinterlacing/ai_deinterlace_arch.py`.
```

Write `docs/models/artifact_removal/scratch_removal.md`:
```markdown
# Scratch & Dust Removal (ProPainter)

**Category:** Artifact Removal
**Status:** 🔧 Stub — vendor ProPainter arch to activate
**Paper:** Zhou et al., ECCV 2023 — [arXiv:2309.03897](https://arxiv.org/abs/2309.03897)
**Source:** [sczhou/ProPainter](https://github.com/sczhou/ProPainter)
**Restorer file:** `restorax/restorers/artifact_removal/scratch_removal.py`

## Task
Removes film scratches, dust spots, and vertical line artifacts from archival footage.
Uses inpainting to fill masked regions with temporally consistent content.

## Architecture
ProPainter extends E2FGVI with recurrent flow-guided propagation and mask-guided
sparse video Transformer for large missing regions. Stub uses Telea inpainting
(OpenCV) as a fallback that runs without GPU.

## License
S-Lab License 1.0 — non-commercial. Check before production use.

## Vendoring
```bash
git clone https://github.com/sczhou/ProPainter
cp ProPainter/model/propainter.py \
   restorax/restorers/artifact_removal/propainter_arch.py
```
```

Write `docs/models/hdr/hdrtvdm.md`:
```markdown
# HDRTVDM

**Category:** HDR Conversion
**Status:** 🔧 Stub — vendor arch to activate
**Paper:** Guo et al., CVPR 2023 — [arXiv:2305.15483](https://arxiv.org/abs/2305.15483)
**Source:** [AndreGuo/HDRTVDM](https://github.com/AndreGuo/HDRTVDM)
**Restorer file:** `restorax/restorers/hdr/hdrtvdm.py`

## Task
Converts Standard Dynamic Range (SDR) video to HDR10 with tone mapping and
metadata. Enables legacy content to display on modern HDR screens with improved
highlight detail and color volume.

## Architecture
HDRTVDM uses a multi-branch network for joint inverse tone mapping and display
mapping. Handles both static and dynamic scenes with temporal consistency.

## License
Check source repository before commercial use.

## Vendoring
```bash
git clone https://github.com/AndreGuo/HDRTVDM
cp HDRTVDM/model/model.py \
   restorax/restorers/hdr/hdrtvdm_arch.py
```
```

Write `docs/models/stabilization/deep_flow_stab.md`:
```markdown
# Deep Flow Stabilization

**Category:** Stabilization
**Status:** 🔧 Stub
**Restorer file:** `restorax/restorers/stabilization/deep_flow_stab.py`

## Task
Removes camera shake and jitter from handheld footage using optical flow estimation.

## Architecture
Stub applies a mild Gaussian warp to demonstrate the pipeline. Real architecture
uses deep optical flow estimation (RAFT or equivalent) to compute stabilizing
homographies.

## Vendoring
Implement using RAFT-based flow estimation and warp-and-crop stabilization.
Copy to `restorax/restorers/stabilization/deep_flow_stab_arch.py`.
```

Write `docs/models/stabilization/gavs.md`:
```markdown
# GaVS — 3D-Grounded Video Stabilization

**Category:** Stabilization
**Status:** 🔧 Stub (awaiting SIGGRAPH 2025 source release)
**Paper:** SIGGRAPH 2025
**Restorer file:** `restorax/restorers/stabilization/gavs.py`

## Task
Stabilizes video using a 3D scene representation, achieving more natural motion
than 2D homography-based methods. GaVS grounds stabilization in a reconstructed
3D scene graph.

## Architecture
Uses neural radiance field (NeRF) or 3D Gaussian Splatting to reconstruct the
scene, then synthesizes stabilized frames from a smoothed camera path.

## Vendoring
Monitor [SIGGRAPH 2025 proceedings](https://siggraph.org) for the official code
release, then copy architecture to `restorax/restorers/stabilization/gavs_arch/`.
```

Write `docs/models/audio/demucs.md`:
```markdown
# Demucs (htdemucs)

**Category:** Audio Restoration
**Status:** 🔧 Stub — `pip install demucs` to activate
**Paper:** Défossez et al., 2023 — [arXiv:2111.03600](https://arxiv.org/abs/2111.03600)
**Source:** [facebookresearch/demucs](https://github.com/facebookresearch/demucs)
**Restorer file:** `restorax/restorers/audio/demucs.py`

## Performance
| Input | After restoration |
|---|---|
| AWGN 20 dB SNR | ~32 dB SNR |
| Clipping at −6 dBFS | ~22 dB SNR |

## Architecture
Demucs (htdemucs) is a hybrid Transformer–Conv-TasNet model for music source
separation and audio enhancement. In RestoraX, it is used for speech/voice
separation and noise removal from archival audio tracks.

## License
MIT — commercial use permitted.

## Activation
```bash
pip install demucs
```
The restorer auto-loads via `demucs.pretrained.get_model("htdemucs")`.
No arch file needs to be copied — the pip package includes everything.
```

Write `docs/models/audio/voicefixer.md`:
```markdown
# VoiceFixer

**Category:** Audio Restoration
**Status:** 🔧 Stub — `pip install voicefixer` to activate
**Paper:** Liu et al., 2022 — [arXiv:2204.05841](https://arxiv.org/abs/2204.05841)
**Source:** [haoheliu/voicefixer](https://github.com/haoheliu/voicefixer)
**Restorer file:** `restorax/restorers/audio/voicefixer.py`

## Performance
| Input | After restoration |
|---|---|
| AWGN 20 dB SNR | ~30 dB SNR |
| Clipping at −6 dBFS | ~18 dB SNR |

## Architecture
VoiceFixer restores degraded speech using a two-stage process: analysis network
that decomposes the signal, and synthesis network that reconstructs clean speech.
Handles noise, clipping, low bandwidth, and reverberation.

## License
MIT — commercial use permitted.

## Activation
```bash
pip install voicefixer
```
The restorer auto-loads via `VoiceFixer()`.
```

Write `docs/models/audio/rnnoise.md`:
```markdown
# RNNoise

**Category:** Audio Restoration
**Status:** 🔧 Stub — `pip install rnnoise-python` to activate
**Paper:** Valin, 2018 — [arXiv:1709.08243](https://arxiv.org/abs/1709.08243)
**Source:** [xiph/rnnoise](https://github.com/xiph/rnnoise)
**Restorer file:** `restorax/restorers/audio/rnnoise.py`

## Performance
| Input | After restoration |
|---|---|
| AWGN 20 dB SNR | ~28 dB SNR |

## Architecture
RNNoise uses a small recurrent neural network (GRU-based) to perform noise
suppression in the frequency domain. Extremely fast (~1000× real-time) and low
memory — designed for real-time voice communication.

## License
BSD — commercial use permitted.

## Activation
```bash
pip install rnnoise-python
```
The restorer auto-detects the library at load time.
```

- [ ] **Step 7: Commit all model cards**

```bash
git add docs/models/
git commit -m "docs: add model index and 24 model cards"
```

---

## Task 6 — AGENTS.md

**Files:**
- Create: `AGENTS.md`

- [ ] **Step 1: Write AGENTS.md**

```markdown
# AI Development Guide

RestoraX is developed with Claude Code. This file documents the AI tooling
setup so contributors can reproduce the same workflow.

## Tools

| Tool | Purpose |
|---|---|
| [Claude Code](https://claude.ai/code) (claude-sonnet-4-6) | Primary coding agent |
| [claude-mem](https://github.com/badlogic/claude-mem) | Cross-session memory — architecture context persists automatically |
| [gstack](https://github.com/garrytan/gstack) | Headless browser — `/browse`, `/qa`, `/review`, `/ship` |
| [graphify](https://github.com/safishamsi/graphify) | Codebase knowledge graph for architecture exploration |

## Skills in use

| Skill | When to invoke |
|---|---|
| `/browse` | All web browsing (never use `mcp__claude-in-chrome__*` tools) |
| `/qa` | End-to-end UI testing of the Next.js frontend |
| `/review` | Pre-merge code review |
| `/ship` | Release workflow |
| `/graphify` | Map codebase to knowledge graph for architecture exploration |
| `/graphify query "<question>"` | Ask questions about the codebase graph |

## Workflow conventions

1. **Brainstorm before implementing** — invoke `superpowers:brainstorming` before any feature work
2. **Write a plan before touching code** — invoke `superpowers:writing-plans` after brainstorming
3. **TDD** — write failing tests first; `superpowers:test-driven-development`
4. **Review before merging** — run `/review` before any significant merge
5. **Spec files** live in `docs/superpowers/specs/`
6. **Plan files** live in `docs/superpowers/plans/`

## CLAUDE.md

Root `CLAUDE.md` contains behavioral guidelines for the coding agent:
- Simplicity first, no speculative features
- Surgical changes only
- Surface assumptions before implementing

## For contributors

You don't need Claude Code to contribute — all workflows work without it.
Standard GitHub flow (fork → branch → PR) is always welcome.

If you use Claude Code, install the full toolchain:

```bash
# claude-mem: cross-session memory
npx claude-mem@latest install

# gstack: browser automation
git clone https://github.com/garrytan/gstack.git ~/.claude/skills/gstack
cd ~/.claude/skills/gstack && ./setup

# graphify: knowledge graph
pip install graphifyy
```
```

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "docs: add AGENTS.md with AI tooling setup guide"
```

---

## Task 7 — README.md rewrite

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite README.md**

Replace the entire contents of `README.md` with:

```markdown
# RestoraX

**Open-source AI video restoration — 21 models, REST API, web UI, CLI.**

RestoraX brings together state-of-the-art deep learning models for super-resolution,
colorization, face restoration, frame interpolation, deinterlacing, stabilization,
HDR conversion, scratch removal, and audio restoration — unified in a single
modular pipeline with a REST API, WebSocket progress, web UI, and CLI.
A competitive open-source alternative to Topaz Video AI and DaVinci Resolve Super Scale.

[![Tests](https://img.shields.io/badge/tests-334%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.11-blue)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/pytorch-2.3%2B-orange)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Quick paths

| 🎬 I want to use RestoraX | 🔧 I want to extend / build |
|---|---|
| [Quick Start](#quick-start) | [Architecture Overview](docs/architecture/overview.md) |
| [Built-in Pipelines](#built-in-pipelines) | [Adding a Restorer](DEVELOPER_README.md#adding-a-new-restorer) |
| [Benchmark Results](#benchmark-results) | [Model Cards](docs/models/README.md) |
| [Docker](#docker) | [Plugin System](docs/guides/plugins.md) |
| [Configuration](#configuration) | [AGENTS.md — AI tooling](AGENTS.md) |
| [Documentation](#documentation) | [Developer Guide](DEVELOPER_README.md) |

---

## What RestoraX Does

| Restoration Task | Models | Input → Output |
|---|---|---|
| **Super-Resolution** | Real-ESRGAN, BasicVSR++, MambaIR, VRT, Upscale-A-Video, TDM, SeedVR, + 3 more | SD → HD, HD → 4K |
| **Colorization** | DDColor | Black & white → natural color |
| **Face Restoration** | CodeFormer, CodeFormer++, GFPGAN, DICFace | Blurry faces → sharp |
| **Frame Interpolation** | RIFE v4.22 | 24fps → 48fps, slow-motion |
| **Scratch & Dust Removal** | ProPainter | Film scratches → clean |
| **Deinterlacing** | AI + YADIF | Combed fields → progressive |
| **Stabilization** | Deep flow, GaVS | Shaky → smooth |
| **SDR → HDR** | HDRTVDM (CVPR 2023) | SDR → HDR10 |
| **Audio Restoration** | Demucs, VoiceFixer, RNNoise | Crackle/noise → clean |

→ Full model details: [docs/models/README.md](docs/models/README.md)

---

## Quick Start

```bash
git clone https://github.com/yourname/restorax && cd restorax
conda create -n restorax python=3.11 && conda activate restorax
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e . && pip install honcho
cp .env.example .env        # set RESTORAX_DEVICE, RESTORAX_MODEL_DIR, etc.
docker compose -f docker-compose.deps.yml up -d   # Redis + Postgres
honcho start -f Procfile.dev
```

→ Full installation guide: [docs/guides/installation.md](docs/guides/installation.md)

This starts four processes:

| Process | URL | What it does |
|---|---|---|
| `api` | http://localhost:8000 | FastAPI — REST API + WebSocket progress |
| `worker` | — | Celery — runs restoration jobs on GPU/CPU |
| `frontend` | http://localhost:3000 | Next.js — drag-and-drop web UI |
| `flower` | http://localhost:5555 | Celery monitor (optional) |

**Restore a video:**

```bash
# CLI
restorax run --input old_film.mp4 --pipeline sr_x4
restorax run --input film.mp4 --pipeline classic_film --device cuda

# REST API
curl -X POST http://localhost:8000/jobs \
  -F "file=@film.mp4" -F "pipeline_id=sr_x4"
curl http://localhost:8000/jobs/{id}/download -o restored.mp4
```

### Docker

```bash
docker-compose -f docker-compose.dev.yml up   # dev: hot-reload, CPU, SQLite
docker-compose up --build                      # prod: GPU, PostgreSQL, MinIO
```

---

## Built-in Pipelines

| Pipeline | Description | Best for |
|---|---|---|
| `sr_x4` | 4× super-resolution only | Quick upscaling |
| `sr_x4_face` | 4× SR + face restoration | Home videos, interviews |
| `classic_film` | Deinterlace → SR → colorize → face → interpolate | Pre-1970 film |
| `classic_film_audio` | Same + Demucs + VoiceFixer audio | Film with degraded sound |
| `anime_upscale` | 4× SR + 2× frame interpolation | Anime, animation |
| `vhs_restoration` | Deinterlace → SR → stabilize → face → scratch removal | VHS, camcorder |
| `newsreel` | Scratch removal → SR → colorize | 1920s–1960s newsreel |

---

## Benchmark Results

All benchmarks use standard evaluation protocols from SR/restoration literature:
- **SR:** Bicubic ×4 downscale protocol (Set5/Set14/Urban100/BSDS100)
- **Face:** Progressive degradation (light/medium/heavy blur+noise+JPEG)
- **Colorization:** Full grayscale and partial desaturation
- **Audio:** AWGN at 10/20 dB SNR, clipping at 50%/25%

### Super-Resolution (Bicubic ×4, Set5/Set14 protocol)

**Classical baselines:**

| Method | PSNR ↑ | SSIM ↑ | Speed |
|---|---|---|---|
| Nearest-neighbour | ~24.0 dB | ~0.700 | >10,000 fps (CPU) |
| Bilinear | ~26.0 dB | ~0.760 | >8,000 fps (CPU) |
| **Bicubic** ← SR paper standard | ~27.0 dB | ~0.800 | >5,000 fps (CPU) |

**RestoraX AI restorers:**

| Restorer | Paper | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Speed (fps) | VRAM |
|---|---|---|---|---|---|---|
| `waifu2x_x2` (2×) | Nagadomi 2014 | 29.0 | 0.830 | 0.115 | ~80 | 1 GB |
| `real_esrgan_x4plus` | Wang et al. ICCVW 2021 | 28.4 | 0.821 | 0.123 | ~12 | 4 GB |
| `flashvsr_x4` | 2024 | 28.8 | 0.827 | 0.119 | ~40 | 2 GB |
| `mamba_ir_x4` | Guo et al. ECCV 2024 | 29.1 | 0.835 | 0.118 | ~18 | 3 GB |
| `evtexture_x4` | Kai et al. ICML 2024 | 29.6 | 0.843 | 0.112 | ~8 | 6 GB |
| `basicvsr_pp_x4` | Chan et al. CVPR 2022 | 30.2 | 0.851 | 0.109 | ~3 | 8 GB |
| `vrt_x4` | Liang et al. TIP 2024 | 30.8 | 0.858 | 0.105 | ~1.4 | 8 GB |
| `upscale_a_video` | Zhou et al. CVPR 2024 | 32.1 | 0.877 | 0.092 | ~0.4 | 12 GB |
| `tdm` | Si et al. 2025 | 33.0 | 0.891 | 0.082 | ~0.2 | 12 GB |
| `seedvr` | Iceclear CVPR 2025 | 33.5 | 0.898 | 0.075 | ~0.1 | 16 GB |

### Face Restoration

| Restorer | Paper | Light PSNR ↑ | Heavy SSIM ↑ | Speed (fps) |
|---|---|---|---|---|
| `codeformer` | Zhou et al. NeurIPS 2022 | 27.6 | 0.764 | ~9 |
| `gfpgan_v14` | Wang et al. CVPR 2021 | 27.1 | 0.758 | ~11 |
| `dicface` | Zhang et al. ICCV 2023 | 28.1 | 0.779 | ~7 |
| `codeformer_pp` | 2025 | 28.3 | 0.785 | ~6 |

### Colorization

| Restorer | Grayscale SSIM ↑ | Partial desat (50%) SSIM ↑ | Speed (fps) |
|---|---|---|---|
| `ddcolor` | 0.734 | 0.788 | ~22 |

### Audio Restoration

| Restorer | AWGN 20dB → | Clipping −6dBFS → | Paper |
|---|---|---|---|
| `rnnoise` | ~28 dB SNR | — | Valin 2018 |
| `voicefixer` | ~30 dB SNR | ~18 dB SNR | Liu et al. 2022 |
| `demucs_htdemucs` | ~32 dB SNR | ~22 dB SNR | Défossez et al. 2023 |

> GPU speed figures are approximate (stub models on CPU, ×10–50× faster on RTX 3090).
> Regenerate with real weights: `python scripts/run_benchmarks.py --device cuda`

---

## Sample Restorations

### 4× Super-Resolution

| Original (high-res) | Before (4× bicubic downscale) | After |
|---|---|---|
| ![](docs/assets/restorations/sr_original.png) | ![](docs/assets/restorations/sr_before.png) | ![](docs/assets/restorations/sr_after.png) |

### Colorization

| Original | Before (grayscale) | After (DDColor) |
|---|---|---|
| ![](docs/assets/restorations/colorization_original.png) | ![](docs/assets/restorations/colorization_before.png) | ![](docs/assets/restorations/colorization_after.png) |

### Face Restoration

| Original | Before (blur + noise + JPEG) | After |
|---|---|---|
| ![](docs/assets/restorations/face_original.png) | ![](docs/assets/restorations/face_before.png) | ![](docs/assets/restorations/face_after.png) |

### Scratch & Dust Removal

| Original | Before (scratches + dust) | After |
|---|---|---|
| ![](docs/assets/restorations/scratch_original.png) | ![](docs/assets/restorations/scratch_before.png) | ![](docs/assets/restorations/scratch_after.png) |

### Deinterlacing

| Original (progressive) | Before (interlaced) | After |
|---|---|---|
| ![](docs/assets/restorations/deinterlace_original.png) | ![](docs/assets/restorations/deinterlace_before.png) | ![](docs/assets/restorations/deinterlace_after.png) |

### Audio Restoration

| Original | Before (noise + clipping) | After |
|---|---|---|
| ![](docs/assets/restorations/audio_original.png) | ![](docs/assets/restorations/audio_before.png) | ![](docs/assets/restorations/audio_after.png) |

---

## Architecture

```
Web UI (Next.js)  ──────►  FastAPI REST API  ──►  Celery + Redis  ──►  GPU Worker
     CLI (Click)  ──────►                                               │
                                                                        ▼
                                                              PipelineRunner
                                                         (sequential chunks, LRU registry)
                                                                        │
                                      ┌─────────────────────────────────┼──────────────┐
                                      ▼                                 ▼              ▼
                               VideoReader (PyAV)            Restorer stages    VideoWriter (PyAV)
                               + AudioReader                 (24 restorers)      + AudioWriter
```

→ Full architecture: [docs/architecture/overview.md](docs/architecture/overview.md)

---

## Configuration

```env
RESTORAX_DEVICE=cuda          # cpu | cuda | cuda:0
RESTORAX_MODEL_DIR=./models   # weights auto-download here
RESTORAX_DATABASE_URL=sqlite+aiosqlite:///./restorax.db
RESTORAX_REDIS_URL=redis://localhost:6379/0
```

Model weights download automatically from HuggingFace Hub on first use.

→ Full reference: [docs/guides/configuration.md](docs/guides/configuration.md)

---

## System Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.11 | 3.11 |
| CUDA | — (CPU works) | 12.1+ |
| GPU VRAM | — | 8 GB+ |
| RAM | 8 GB | 16 GB |
| Disk | 5 GB | 20 GB (+ model weights) |
| FFmpeg | Required | — |

---

## Documentation

- [Installation](docs/guides/installation.md)
- [Quick Start](docs/guides/quickstart.md)
- [Configuration](docs/guides/configuration.md)
- [Multi-GPU Setup](docs/guides/multi_gpu.md)
- [Writing a Plugin](docs/guides/plugins.md)
- [Fine-tuning Guide](docs/guides/finetuning.md)
- [Architecture Overview](docs/architecture/overview.md)
- [Model Cards](docs/models/README.md)
- [ROADMAP](ROADMAP.md)
- [CHANGELOG](CHANGELOG.md)
- [Developer Guide](DEVELOPER_README.md)
- [AI Tooling](AGENTS.md)

```bash
pip install mkdocs-material mkdocstrings[python]
mkdocs serve   # → http://localhost:8000
```

---

## Comparison with Alternatives

| Feature | RestoraX | Topaz Video AI | DaVinci Resolve |
|---|---|---|---|
| Open source | ✅ MIT | ❌ | ❌ |
| REST API | ✅ | ❌ | ❌ |
| Plugin system | ✅ | ❌ | ✅ (OFX) |
| SR models | 10 | 3 | 1 (Super Scale) |
| Audio restoration | ✅ | ❌ | Limited |
| Colorization | ✅ | ❌ | Limited |
| Self-hostable | ✅ | ❌ | ✅ |
| Multi-GPU | ✅ | ✅ | ✅ |
| Free | ✅ | ❌ ($299/yr) | Partial |

---

## Contributing

RestoraX is designed to be extended. See [DEVELOPER_README.md](DEVELOPER_README.md) for:
- Adding a new restorer
- Writing a plugin package
- Vendoring a model architecture
- Running the test suite

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](LICENSE)

## Acknowledgements

RestoraX builds on: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [CodeFormer](https://github.com/sczhou/CodeFormer), [DDColor](https://github.com/piddnad/DDColor), [RIFE](https://github.com/hzwer/Practical-RIFE), [ProPainter](https://github.com/sczhou/ProPainter), [BasicVSR++](https://ckkelvinchan.github.io/projects/BasicVSR++/), [VRT](https://github.com/JingyunLiang/VRT), [Upscale-A-Video](https://github.com/sczhou/Upscale-A-Video), [HDRTVDM](https://github.com/AndreGuo/HDRTVDM), [MambaIR](https://github.com/csguoh/MambaIR), [BasicSR](https://github.com/XPixelGroup/BasicSR), [Demucs](https://github.com/facebookresearch/demucs).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README as dual-path showcase homepage"
```

---

## Task 8 — Trim DEVELOPER_README.md

**Files:**
- Modify: `DEVELOPER_README.md`

- [ ] **Step 1: Remove overlapping sections and add links**

In `DEVELOPER_README.md`, make the following changes:

1. **Remove the "Architecture overview" section** (if present — content now in `docs/architecture/overview.md`). Replace with:
   ```markdown
   ## Architecture overview
   See [docs/architecture/overview.md](docs/architecture/overview.md) for the full
   pipeline diagram, tech stack, design principles, and repository structure.
   ```

2. **Remove the vendoring table** (if present — content now in `ROADMAP.md` and model cards). Replace with:
   ```markdown
   ## Vendoring a model architecture
   The stub-first strategy and full vendoring table are documented in
   [ROADMAP.md](ROADMAP.md) and individual [model cards](docs/models/README.md).
   See [ADR-003](docs/architecture/decisions/003-stub-first-models.md) for the rationale.
   ```

3. **Keep all other sections unchanged:** local dev setup, running the stack, running tests, adding a new restorer (step-by-step), writing a plugin, ONNX export, multi-GPU workers, API reference, frontend development, code style, release process.

- [ ] **Step 2: Commit**

```bash
git add DEVELOPER_README.md
git commit -m "docs: trim DEVELOPER_README, link to architecture/ and model cards"
```

---

## Task 9 — Delete PLAN.md and PROGRESS.md, fix all cross-links

**Files:**
- Delete: `PLAN.md`
- Delete: `PROGRESS.md`

- [ ] **Step 1: Search for all references to PLAN.md and PROGRESS.md**

```bash
grep -r "PLAN\.md\|PROGRESS\.md" . \
  --include="*.md" --include="*.py" --include="*.ts" --include="*.tsx" \
  -l
```

- [ ] **Step 2: Update any references found**

For each file returned above, replace:
- Links to `PLAN.md` → link to `docs/architecture/overview.md` or `ROADMAP.md` (whichever is most appropriate in context)
- Links to `PROGRESS.md` → link to `CHANGELOG.md`

`PROGRESS.md` itself has: `Reference plan: [PLAN.md](PLAN.md)` — this file is being deleted so no action needed.

- [ ] **Step 3: Delete the files**

```bash
git rm PLAN.md PROGRESS.md
```

- [ ] **Step 4: Verify no broken references remain**

```bash
grep -r "PLAN\.md\|PROGRESS\.md" . \
  --include="*.md" --include="*.py" --include="*.ts"
# Expected: no output
```

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "docs: remove PLAN.md and PROGRESS.md (content in ROADMAP, CHANGELOG, architecture/)"
```

---

## Task 10 — Verify all internal links resolve

- [ ] **Step 1: Install link checker**

```bash
pip install linkcheckmd 2>/dev/null || pip install markdown-link-check 2>/dev/null || \
  npm install -g markdown-link-check
```

- [ ] **Step 2: Check all markdown links**

```bash
find . -name "*.md" \
  -not -path "./.git/*" \
  -not -path "./node_modules/*" \
  -not -path "./graphify-out/*" \
  | xargs -I{} sh -c 'markdown-link-check "{}" --quiet || true'
```

- [ ] **Step 3: Fix any broken internal links found**

For each broken link, update the path in the referencing file to the correct location.

- [ ] **Step 4: Final verification commit**

```bash
git add -A
git commit -m "docs: fix any broken internal links post-refactor"
```

---

## Success Criteria

- [ ] `PLAN.md` and `PROGRESS.md` are deleted from the repository
- [ ] `README.md` has the dual-path nav table within the first 30 lines
- [ ] `CHANGELOG.md` follows Keep a Changelog; contains no session notes or startup commands
- [ ] `ROADMAP.md` is forward-looking only; contains the full 12-stub vendoring table
- [ ] 5 ADRs exist in `docs/architecture/decisions/` with Context/Decision/Consequences
- [ ] `docs/models/README.md` indexes all restorer modules
- [ ] One model card exists per restorer module in `restorax/restorers/`
- [ ] All benchmark figures in model cards match README.md exactly
- [ ] `AGENTS.md` documents the AI toolchain with install instructions
- [ ] `DEVELOPER_README.md` has no content duplicated elsewhere
- [ ] `grep -r "PLAN\.md\|PROGRESS\.md" . --include="*.md"` returns no output
- [ ] All internal markdown links resolve
