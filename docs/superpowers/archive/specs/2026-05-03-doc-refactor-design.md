# RestoraX Documentation Refactor — Design Spec

**Date:** 2026-05-03  
**Goal:** Reposition RestoraX as a modern 2026-model AI showcase project.  
**Audience:** ML researchers/engineers AND developers/contributors (dual-path).  
**Scope:** Documentation only — no code changes.

---

## 1. Goals

- **Showcase/portfolio framing**: every doc should read like a well-maintained open-source project, not a working notebook.
- **Dual-path navigation**: two audiences (users, extenders) find their path immediately from the README.
- **Single source of truth**: no content duplicated across files; every file has one clear purpose.
- **Modern AI project signals**: model cards, ADRs, AI tooling integration doc, forward-looking roadmap.

---

## 2. File Map

### Deleted
| File | Reason |
|---|---|
| `PLAN.md` | Content distributed to `ROADMAP.md`, `docs/architecture/overview.md`, and ADRs |
| `PROGRESS.md` | Transformed into `CHANGELOG.md` |

### Modified
| File | Change |
|---|---|
| `README.md` | Full rewrite — dual-path showcase homepage |
| `DEVELOPER_README.md` | Trim to ~400 lines — remove sections that move to `docs/architecture/` |

### New
| File | Purpose |
|---|---|
| `CHANGELOG.md` | Keep a Changelog format, transformed from PROGRESS.md |
| `ROADMAP.md` | Forward-looking — model activation status + future milestones |
| `AGENTS.md` | AI tooling integration doc for Claude Code / gstack / claude-mem |
| `docs/architecture/overview.md` | Pipeline diagram, design principles, tech stack, repo tree |
| `docs/architecture/decisions/001-pyav-video-io.md` | ADR: why PyAV over subprocess FFmpeg |
| `docs/architecture/decisions/002-celery-task-queue.md` | ADR: GPU concurrency, per-queue routing |
| `docs/architecture/decisions/003-stub-first-models.md` | ADR: stub strategy, vendoring process, tradeoffs |
| `docs/architecture/decisions/004-lru-model-registry.md` | ADR: VRAM management via LRU eviction |
| `docs/architecture/decisions/005-storage-abstraction.md` | ADR: local vs MinIO/S3 via Protocol |
| `docs/models/README.md` | Index table — all restorers at a glance |
| `docs/models/<category>/<restorer>.md` | One file per restorer — canonical list from `restorax models` CLI, mirroring `restorax/restorers/` directory structure |

---

## 3. README.md Design

### Structure
```
# RestoraX
tagline: "Open-source AI video restoration — 21 models, REST API, web UI, CLI"
badges: tests | python | pytorch | license

## What it does
~2 paragraphs: problem (old films, home videos, archival footage), solution
(21 models unified in one pipeline), differentiator (open-source, API-first,
plugin system, competitive with Topaz Video AI).

## Quick paths  ← dual-path nav block
┌──────────────────────────┬─────────────────────────────────┐
│ 🎬 I want to use it      │ 🔧 I want to extend / build     │
│ → Quick Start (3 lines)  │ → Architecture Overview         │
│ → Built-in Pipelines     │ → Adding a Restorer             │
│ → Benchmark Results      │ → Model Cards (21 models)       │
│ → Docker                 │ → Plugin System                 │
│ → Configuration          │ → AGENTS.md (AI tooling)        │
└──────────────────────────┴─────────────────────────────────┘

## Model matrix
Condensed table: category | models | quality tier | VRAM range

## Quick Start
3-line summary (git clone, pip install, honcho start)
Link → docs/guides/quickstart.md for full instructions

## Built-in Pipelines  (existing table, unchanged)

## Benchmark Results   (existing tables, unchanged)

## Sample Restorations (existing, unchanged)

## Architecture        (existing ASCII diagram, unchanged)

## System Requirements (existing table, unchanged)

## Comparison with Alternatives (existing table, unchanged)

## Contributing        (3-sentence summary + links to DEVELOPER_README.md)

## Acknowledgements / License
```

### Key principles
- Dual-path nav block appears immediately after the intro — both audiences find their path in <10 seconds.
- Full install instructions move to `docs/guides/quickstart.md`; README links there.
- All factual content (benchmarks, pipelines, samples) is preserved exactly.

---

## 4. CHANGELOG.md Design

Follows [Keep a Changelog](https://keepachangelog.com/) conventions.

```
# Changelog

## [Unreleased]
### Pending
- Vendor real model architectures (12 restorers use stubs):
  DDColor, RIFE v4, ProPainter, HDRTVDM, Upscale-A-Video, MambaIR, TDM,
  GaVS (awaiting SIGGRAPH 2025 release), CodeFormer++, Demucs, VoiceFixer, RNNoise

## [1.0.0] — 2026-04-30
### Added
- 21 restorers across SR, colorization, face, interpolation, deinterlacing,
  stabilization, HDR, scratch removal, and audio
- REST API (FastAPI), WebSocket progress, Celery GPU worker, Next.js 14 UI, CLI
- 334 passing tests (309 Python + 25 frontend)
- Plugin system, multi-GPU routing, LRU model registry
- Docker (dev + prod), MinIO/S3 storage, Alembic migrations
- MIT license, CONTRIBUTING.md, CI pipeline, GitHub issue/PR templates

### Fixed
- Alembic async driver URL stripping crash on migrations
- run_job task not updating DB status (queued stuck forever)
- Real-ESRGAN stub class undefined + not moved to device

## Earlier phases (0.x)
Condensed summaries of Phases 1–6 (Foundation → Hardening)
No session-level notes, no startup commands
```

### What is stripped
- All "Next Session Plan" blocks
- Terminal A/B/C startup instructions (→ DEVELOPER_README.md)
- Raw session notes and debug observations
- Inline "fix basicsr/torchvision" option A/B/C lists (→ ADR-003)

---

## 5. ROADMAP.md Design

Forward-looking only. No session notes, no done/done/done phase tracking.

```
# Roadmap

## Now — Model Activation
12 restorers ship with geometrically-correct stubs. Vendoring the real
architecture activates full quality with no other code changes required.

| Restorer | Stub file | Source | Effort |
|---|---|---|---|
| DDColor | restorers/colorization/ddcolor.py | piddnad/DDColor | ~1 day |
| RIFE v4 | restorers/frame_interpolation/rife.py | hzwer/Practical-RIFE | ~1 day |
| ... (all 12) | | | |

See ADR-003 for the stub strategy and vendoring process.

## Next — Quality & Performance
- ONNX export + TensorRT optimization
- No-reference quality metrics (DOVER, FasterVQA) for blind scoring
- Batch job API + priority queue support
- GaVS stabilization (pending SIGGRAPH 2025 release)

## Future
- Managed cloud deployment (RestoraX Cloud)
- Fine-tuning guide + LoRA adapter support for domain-specific restoration
- Real-time preview mode (sub-second latency on short clips)
- Browser extension for video platform integration
```

---

## 6. docs/architecture/overview.md Design

Consolidates content currently split across PLAN.md and DEVELOPER_README.md.

### Sections
1. **Pipeline architecture** — the ASCII diagram from README + prose explanation
2. **Key design principles** — sequential chunked processing, LRU registry, stub-first, plugin system
3. **Tech stack** — the full tech stack table from PLAN.md (language, ML backend, API, queue, DB, video I/O, frontend, storage, tooling)
4. **Repository structure** — the annotated directory tree from PLAN.md
5. **Architecture decisions** — brief summary of each ADR with links

---

## 7. Architecture Decision Records (ADRs)

Each ADR: ~300–500 words. Format: **Status / Context / Decision / Consequences**.

| ADR | Decision | Key point |
|---|---|---|
| 001 | PyAV for video I/O | PTS access, audio passthrough, no subprocess-per-frame overhead vs cv2/ffmpeg subprocess |
| 002 | Celery + Redis | GPU concurrency control via per-GPU queues; retry policies; result backend doubles as WebSocket pub/sub |
| 003 | Stub-first models | Every restorer works without weights; stubs produce correct-shape output; enables CI without GPU |
| 004 | LRU model registry | Constant VRAM regardless of pipeline length; max_loaded=2 by default; configurable |
| 005 | Storage abstraction | `StorageBackend` Protocol; local FS in dev, MinIO/S3 in prod; zero code changes at deploy time |

---

## 8. Model Cards Design

### docs/models/README.md — index table
All 21 models in one table: category | model | status | scale/task | VRAM | speed (fps) | paper.

### Per-model card template
```markdown
# [Model Name]

**Category:** [SR / Colorization / Face / ...]
**Status:** [✅ Active | 🔧 Stub — vendor to activate]
**Paper:** [Author et al., Venue Year] — [arXiv link]
**Source repo:** [org/repo]

## Performance
| Metric | Value | Protocol |
| PSNR | X dB | Bicubic ×4, Set5/Set14 |
| SSIM | X.XXX | — |
| Speed | ~N fps | RTX 3090 |
| VRAM | N GB | — |

## Architecture
2–4 sentences: what makes this model distinctive architecturally.

## License
License name — commercial use permitted / restricted.

## Known Limitations
Bullet list of known failure modes, compat issues, hardware requirements.

## Vendoring
How to activate the real arch: what to install, what file to edit, what to replace.
Link to ADR-003 for the full vendoring strategy.
```

All benchmark numbers come directly from README.md — no invented figures.  
Models with no public benchmark data (stubs only) note "pending — run benchmarks with real weights."

---

## 9. AGENTS.md Design

```markdown
# AI Development Guide

RestoraX is developed with Claude Code. This file documents the AI tooling
setup so contributors can reproduce the same workflow.

## Tools
| Tool | Purpose |
|---|---|
| Claude Code (claude-sonnet-4-6) | Primary coding agent |
| claude-mem | Cross-session memory — architecture context persists automatically |
| gstack | Headless browser — /browse, /qa, /review, /ship |
| graphify | Codebase knowledge graph for architecture exploration |

## Skills in use
| Skill | When to invoke |
|---|---|
| /browse | All web browsing (replaces chrome MCP tools) |
| /qa | End-to-end UI testing |
| /review | Pre-merge code review |
| /ship | Release workflow |
| /graphify | Map codebase to knowledge graph |

## Workflow conventions
- Brainstorm before implementing any feature
- Write plan before touching code
- Run /review before merging significant changes
- CLAUDE.md contains behavioral guidelines (see root CLAUDE.md)

## For contributors
You don't need Claude Code to contribute.
If you use it, install the full toolchain:
  npx claude-mem@latest install
  git clone https://github.com/garrytan/gstack.git ~/.claude/skills/gstack && cd ~/.claude/skills/gstack && ./setup
```

---

## 10. DEVELOPER_README.md Changes

**Remove** (content moves elsewhere):
- Architecture overview section → `docs/architecture/overview.md`
- Vendoring table → `ROADMAP.md` + individual model cards

**Keep** (unchanged):
- Local dev setup (prerequisites, conda env, frontend deps, env file, starting services)
- Running the stack (honcho, manual start instructions)
- Running tests (pytest, npm test, coverage)
- Adding a new restorer (step-by-step guide)
- Writing a plugin package
- ONNX export
- Multi-GPU workers
- API reference
- Frontend development
- Code style (ruff, mypy)
- Release process

**Add** links to new docs where content was removed.

Expected result: ~400 lines (down from ~600), zero content overlap with other files.

---

## 11. Success Criteria

- [ ] PLAN.md and PROGRESS.md are deleted
- [ ] README.md has dual-path nav block and links to all major sections
- [ ] CHANGELOG.md follows Keep a Changelog; no session notes
- [ ] ROADMAP.md is forward-looking only; includes full vendoring table
- [ ] 5 ADRs written with Context/Decision/Consequences
- [ ] One model card per restorer (run `restorax models` for canonical list); all benchmark figures match README.md exactly; stubs noted as "pending real weights"
- [ ] AGENTS.md documents full toolchain with install instructions
- [ ] DEVELOPER_README.md has no content duplicated in other files
- [ ] All cross-file links resolve correctly
- [ ] No invented benchmark numbers; stubs noted as "pending real weights"
