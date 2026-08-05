# Migration Guide — Monolith → Two Platforms

This guide explains how the repository moved from a single monolithic layout
to two independent platforms (`training/` and `application/`) with a shared
contract layer (`shared/`). It is written for developers who need to update
their local checkout, their tooling, or code that referenced the old layout.

## Before / after

**Before**

```
ai/  services/  quality/  mlops/  backend/  frontend/  gis/  tests/
deployment/  research/  nginx/  Dockerfile.*  docker-compose*.yml
```

**After**

```
training/    # models, preprocessing, training, explainability, dataset_manager,
             #   stam, quality, mlops + new placeholder directories
application/ # backend (FastAPI app/), frontend, database, gis, monitoring,
             #   docker, config, tests + new placeholder directories
shared/      # schemas, dto, enums, interfaces, validation, utils, exceptions,
             #   config, constants, serialization, tests
docs/        # now organised into architecture/installation/usage/developer-guide/
             #   deployment/api/research/images/diagrams/migration
releases/    # tagged release archives
```

## What changed for developers

### Python imports

Imports were rewritten from the old module roots to the new ones:

| Old import | New import |
| ---------- | ---------- |
| `from ai.models.exporter import ModelExporter` | `from training.models.exporter import ModelExporter` |
| `from ai.preprocessing.*` | `from training.preprocessing.*` |
| `from ai.training.*` | `from training.training.*` |
| `from ai.explainability.*` | `from training.explainability.*` |
| `from services.dataset_manager.*` | `from training.dataset_manager.*` |
| `from services.spatial_alignment.*` | `from training.stam.*` |
| `from quality.*` | `from training.quality.*` |
| `from mlops.*` | `from training.mlops.*` |

Unchanged: the backend package remains `app` (now at
`application/backend/app`) and the database package remains `database` (now at
`application/database`), so `from app.main import create_app` and
`from database.seeds.runner import seed_database` keep working.

### sys.path / PYTHONPATH

Runtime code needs these roots on `sys.path`:

- `application/backend` — for `app.*`
- `application` — for `database.*`, `gis.*`
- `<repo root>` — for `training.*` and `shared.*`

> **Important**: add the **repository root**, not `training/` itself, to
> `sys.path`. The training engine lives at `training/training/`, so adding the
> `training/` directory directly makes `import training` resolve to that inner
> sub-package and breaks `import training.models`.

This is handled automatically by:

- `application/backend/app/core/paths.py` (backend bootstrap),
- pytest `conftest.py` files (`application/tests/conftest.py`,
  `application/backend/app/tests/conftest.py`, and the per-package
  `training/*/tests/conftest.py`),
- the Docker images (`PYTHONPATH` in `application/docker/Dockerfile.backend`).

### Tooling

- **Install**: first-party packages are installed from their new paths:
  `pip install -e ./training/models ./training/preprocessing ./training/training
  ./training/explainability ./training/dataset_manager ./training/stam`.
- **Tests**: `pytest.ini`/`pyproject.toml` `testpaths` now cover
  `application/tests`, `application/backend/app/tests`, `training`, `shared`.
- **Docker**: Dockerfiles and compose files live in `application/docker/`.
  Compose uses `context: ..` with
  `dockerfile: application/docker/Dockerfile.<image>`. Run compose with
  `-f application/docker/docker-compose.yml`.
- **Env files**: examples moved to `application/config/`
  (`.env.example`, `.env.production.example`). Copy them to the repo root
  `.env` as before.
- **Makefile / CI**: all `make` targets and GitHub Actions workflows were
  updated to the new paths.

## Docker volumes and runtime directories

- The backend runtime datasets directory moved from `backend/datasets/` to
  `application/backend/datasets/` (update any volume bind-mounts / `.gitignore`
  entries).
- Named volumes (`models`, `datasetdata`, `reports`, ...) are unchanged.

## Gotchas

- Old `git mv` failures on directories are expected on Windows; use
  `Move-Item` + `git add -A`.
- `scripts/build_docs.py` and the docs Dockerfile read `docs/` from the repo
  root — unaffected by the reorganisation, but `docs/` is now organised into
  subfolders.
- The MLOps console script is now `cropfusion-mlops = training.mlops.cli:app`,
  and the admin scheduler runs `python -m training.mlops.scheduler`.

## Rollback

Nothing was deleted. All old top-level directories (`ai`, `services`,
`quality`, `mlops`, `backend`, `frontend`, `gis`, `deployment`, `research`,
`nginx`) were moved (not removed), and the empty parent directories were
deleted. You can recover any file from the Git history of this branch.
