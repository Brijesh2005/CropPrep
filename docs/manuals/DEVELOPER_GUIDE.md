# Developer Guide

Practical onboarding for contributors. See also [DEVELOPMENT.md](../DEVELOPMENT.md)
and [CONTRIBUTING.md](../CONTRIBUTING.md).

## Daily loop

```bash
# Backend
cd backend
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm run dev          # http://localhost:5173

# Tests
pytest backend/app/tests -q
cd frontend && npm test -- --run
```

## Where things live

| Need | Look in |
|---|---|
| API routers | `backend/app/modules/<name>/router.py` |
| Business logic | `backend/app/modules/<name>/service.py` |
| ORM models | `backend/database/models/` + migrations |
| AI model code | `ai/models/` |
| Preprocessing | `ai/preprocessing/` |
| Training/eval | `ai/training/` |
| Explainability | `ai/explainability/` |
| ML quality | `quality/` |
| MLOps | `mlops/` |
| Frontend UI | `frontend/src/features/` |
| Deployment | `deployment/`, `docker-compose*.yml`, `Dockerfile.*` |

## Dependency rules

- `backend` depends on `ai.models` and `services.dataset_manager` (install all
  first-party packages with `make install`).
- `quality` is standalone (no backend imports); the backend stays decoupled.
- New runtime deps go into `requirements.txt`; pin exact versions.

## Adding an API endpoint

1. Add the handler in the module's `router.py` and schema in `schemas/`.
2. Register the router in `app/main.py` under `build_api_router`.
3. Write a test in `backend/app/tests/` using `client_with_fake_engine`.
4. Run `pytest backend/app/tests/test_<module>.py -q` and `make lint`.

## Debugging

- Structured logs: set `BACKEND_LOG__JSON_LOGS=false` for readable output.
- Metrics: `curl localhost:8000/metrics | grep cropfusion`.
- Tests with SQLite: the `settings` fixture points at an in-memory DB.
- `torch.compile` falls back to eager without a compiler - safe to ignore.

## Release checklist

Run `make release` locally (tests + frontend build). CI validates the rest
(compose, security, coverage). Tag `v<semver>` to trigger the Release workflow.
