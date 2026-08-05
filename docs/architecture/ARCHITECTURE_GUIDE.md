# CropFusion Architecture Guide

## Two independent platforms

CropFusion is organised as **two independent platforms** plus a small shared
contract layer:

```
┌─────────────────────────┐        ┌─────────────────────────┐
│     TRAINING PLATFORM   │        │    PREDICTION PLATFORM  │
│         training/       │        │       application/      │
│                         │        │                         │
│  models                 │        │  backend (FastAPI)      │
│  preprocessing          │        │  frontend (React)       │
│  training (engine)      │        │  database (PostGIS)     │
│  explainability         │        │  gis                    │
│  dataset_manager        │        │  inference              │
│  stam (ST alignment)    │        │  authentication         │
│  quality (gates)        │        │  monitoring             │
│  mlops (registry/CLI)   │        │  docker                 │
│  feature_engineering    │        │  admin, history, ...    │
│  evaluation / export /  │        │                         │
│  hyperparameter_search  │        │                         │
└───────────┬─────────────┘        └───────────┬─────────────┘
            │                                  │
            └──────────┬───────────────────────┘
                       ▼
               ┌──────────────┐
               │    shared/   │  schemas, DTOs, enums,
               │   contracts  │  interfaces, validation,
               │              │  serialization, config
               └──────────────┘
```

**Dependency rules**

- `training/*` may depend on `shared/*` only.
- `application/*` may depend on `shared/*` only.
- `training/*` and `application/*` must never import each other.
- `shared/*` depends on nothing inside the repo (standard library + third
  party only).

## What moved where

| Old (monolithic) | New (two-platform) |
| ---------------- | ------------------- |
| `ai/models` | `training/models` |
| `ai/preprocessing` | `training/preprocessing` |
| `ai/training` | `training/training` |
| `ai/explainability` | `training/explainability` |
| `services/dataset_manager` | `training/dataset_manager` |
| `services/spatial_alignment` | `training/stam` |
| `quality` | `training/quality` |
| `mlops` | `training/mlops` |
| `backend/app` | `application/backend/app` |
| `backend/database` | `application/database` |
| `backend/datasets` | `application/backend/datasets` |
| `frontend` | `application/frontend` |
| `gis` | `application/gis` |
| `tests` | `application/tests` |
| `research` | `docs/research` |
| `deployment/monitoring` | `application/monitoring` |
| `deployment/caddy` | `application/docker/caddy` |
| `nginx` | `application/docker/nginx` |
| Dockerfiles / compose / .dockerignore | `application/docker/` |
| `.env.example`, `.env.production.example` | `application/config/` |

## Import naming

First-party Python imports use the new module roots:

- `training.models.exporter`, `training.preprocessing.*`, `training.quality.*`,
  `training.mlops.*`, `training.dataset_manager.*`, `training.stam.*`, ...
- `app.*` (backend, unchanged — the FastAPI package stays named `app` at
  `application/backend/app`).
- `database.*` (unchanged — `application/database`).
- `shared.*` (new contract layer).

## Runtime bootstrap

Paths are wired through `application/backend/app/core/paths.py` and the pytest
`conftest.py` files, which add `application/backend`, `application` and the
repository root to `sys.path` (the repo root is what makes `training.*` /
`shared.*` importable — never add the `training/` directory itself, because it
contains the `training/training` sub-package). In Docker, the equivalent
`PYTHONPATH` is set in the image environment.

## Related documents

- [FOLDER_GUIDE.md](FOLDER_GUIDE.md) — folder-by-folder tour
- [MIGRATION_GUIDE.md](../migration/MIGRATION_GUIDE.md) — how the move was done
- [MIGRATION_REPORT.md](../migration/MIGRATION_REPORT.md) — detailed report
- [SOFTWARE_DESIGN_DOCUMENT.md](SOFTWARE_DESIGN_DOCUMENT.md) — original design
- [../research/ARCHITECTURE.md](../research/ARCHITECTURE.md) — system architecture
