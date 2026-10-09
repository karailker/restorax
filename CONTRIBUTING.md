# Contributing to RestoraX

Thank you for your interest in contributing! This document explains how to set up a development environment, run tests, and submit changes.

---

## Development setup

```bash
git clone https://github.com/yourname/restorax
cd restorax
conda create -n restorax python=3.11 && conda activate restorax
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
cp .env.example .env
```

Start external services:
```bash
docker compose -f docker-compose.deps.yml up -d
```

Start the app:
```bash
honcho start -f Procfile.dev
```

Optional but recommended: run the same checks as CI before every commit.
```bash
pip install pre-commit && pre-commit install
```

---

## Quality checks (all enforced in CI)

```bash
ruff check restorax/ tests/ comfyui_nodes/ scripts/   # lint, includes security rules (S)
ruff format --check restorax/ tests/ comfyui_nodes/ scripts/
mypy restorax/ --ignore-missing-imports --no-strict-optional
gitleaks detect --source . --redact                   # secret scan
pip-audit --skip-editable                             # Python dependency CVEs
cd frontend && npm audit --omit=dev --audit-level=high
```

mypy runs with `disallow_untyped_defs`, `disallow_any_generics` and `disallow_untyped_calls`: new code must be fully annotated. Vendored upstream architectures (`*_arch.py`, `propainter/`) are excluded from lint and type checks so they stay identical to upstream.

---

## Running tests

```bash
# All Python tests (unit + integration + system)
pytest tests/ -q

# Frontend typecheck and build
cd frontend && npm run typecheck && npm run build

# With coverage
pytest tests/ --cov=restorax --cov-report=term-missing
```

Tests are organized as:

| Layer | Directory | Notes |
|---|---|---|
| Unit | `tests/unit/` | No GPU, no real weights, fast |
| Integration | `tests/integration/` | FastAPI + SQLite in-process, Celery eager |
| System | `tests/system/` | Full app lifespan, mocked task dispatch |
| Frontend | `frontend/tests/` | Vitest + jsdom |

---

## Code style

```bash
ruff check restorax/ --fix   # lint + autofix
ruff format restorax/         # format
mypy restorax/ --strict       # type check
```

The CI enforces zero ruff errors and zero mypy errors.

---

## Adding a new restorer

1. Pick the category: `super_resolution`, `face_restoration`, `colorization`, etc.
2. Create `restorax/restorers/<category>/<name>.py`
3. Subclass `BaseRestorer`, implement `name`, `capabilities`, `load`, `unload`, `process_frame`
4. Provide a `_<Name>Stub` fallback that runs without real weights (e.g. bicubic resize)
5. Add a test in `tests/unit/restorers/`
6. Register the class in `restorax/api/routers/models.py` and `restorax/tasks/job_tasks.py`
7. Add an entry to `pyproject.toml` under `[project.entry-points."restorax.restorers"]`

See `restorax/restorers/super_resolution/mamba_ir.py` for a complete reference implementation.

---

## Adding a new pipeline preset

1. Create `configs/presets/<name>.yaml`
2. Define `stages` (and optionally `audio_stages`)
3. Test with `restorax run --input sample.mp4 --pipeline <name>`

---

## Submitting a PR

- Fork the repo and create a feature branch (`git checkout -b feat/my-change`)
- Add tests for your change
- Ensure `pytest tests/ -q` and `npm test` both pass
- Open a PR against `main` — CI will run automatically

---

## Reporting bugs

Open an issue at <https://github.com/yourname/restorax/issues> and include:
- OS, Python version, CUDA version
- Full error traceback
- Minimal reproduction steps
