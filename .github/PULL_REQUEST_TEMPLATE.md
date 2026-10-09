## Summary

Brief description of what this PR changes and why.

## Type of change

- [ ] Bug fix
- [ ] New restorer / model
- [ ] New pipeline preset
- [ ] API change
- [ ] Documentation
- [ ] Refactor / tests / CI

## Checklist

- [ ] Tests pass locally (`pytest tests/ -q` and `cd frontend && npm run typecheck`)
- [ ] New code has tests
- [ ] No new ruff errors (`ruff check restorax/ tests/ scripts/`)
- [ ] No new mypy errors (`mypy restorax/ --ignore-missing-imports`)
- [ ] PR description explains the "why", not just the "what"

## For new restorers

- [ ] Stub fallback works without real weights
- [ ] Registered in `restorax/api/routers/models.py`
- [ ] Registered in `restorax/tasks/job_tasks.py`
- [ ] Entry added to `pyproject.toml` entry points
- [ ] Unit test added in `tests/unit/restorers/`
