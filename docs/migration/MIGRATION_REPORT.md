# Migration Report — Monolith → Two Platforms

Detailed record of the repository reorganisation from a single monolithic
layout into two independent platforms plus a shared contract layer. Companion
to the [MIGRATION_GUIDE](MIGRATION_GUIDE.md), which focuses on *how* to update
tooling and code; this report records *what was done* and *how it was
verified*.

- **Status**: complete
- **Date**: 2026-08-05
- **Branch**: `main` (single commit, not yet pushed)

## 1. Scope and principles

- **Move-only**: files were relocated, not redesigned. No functional code was
  modified beyond import statements and path bootstrap logic.
- **Dependency rules**: `training` and `application` may depend only on
  `shared` (plus stdlib / third-party); never on each other.
- **Preserve runtime identity**: the FastAPI package stays `app` and the
  database package stays `database`, so existing entrypoints
  (`uvicorn app.main:app`, `from database.seeds.runner import seed_database`)
  are unchanged.
- **Nothing deleted**: every file was moved via `Move-Item` + `git add -A`;
  the old top-level directories were emptied and removed. Full history remains
  in Git.

## 2. Final top-level layout

```
training/    330 files   # models, preprocessing, training, explainability,
                         #   dataset_manager, stam, quality, mlops, kaggle,
                         #   + placeholder dirs (feature_engineering,
                         #     evaluation, experiments, export,
                         #     hyperparameter_search, config, tests)
application/ 388 files   # backend (FastAPI app package), frontend, database,
                         #   gis, monitoring, docker, config, tests,
                         #   + placeholder dirs (inference, authentication,
                         #     history, admin, models, inference_package)
shared/       13 files   # schemas, dto, enums, interfaces, validation, utils,
                         #   exceptions, config, constants, serialization, tests
docs/         51 files   # reorganised into architecture/ installation/ usage/
                         #   developer-guide/ deployment/ api/ research/
                         #   images/ diagrams/ migration/
releases/      3 files   # latest/, v1.0/, v1.1/ (placeholder archives)
scripts/       4 files
.github/      11 files   # workflows, dependabot, CODEOWNERS
cropfusion/    1 file    # legacy entrypoint wrapper (unchanged)
```

Excludes VCS, `.venv`, `node_modules`, and caches. Repo total: 846 files.

## 3. Directory migration inventory

| Old | New | Notes |
| --- | --- | --- |
| `ai/` | `training/` | `ai/training` → `training/training` (the training engine), `ai/models` → `training/models`, etc. |
| `services/dataset_manager/` | `training/dataset_manager/` | |
| `services/spatial_alignment/` | `training/stam/` | Renamed package name (kept the old name only in runtime logger identifiers — see §8). |
| `quality/` | `training/quality/` | |
| `mlops/` | `training/mlops/` | Console script renamed to `cropfusion-mlops = training.mlops.cli:app`. |
| `backend/` | `application/backend/` | `backend/app` → `application/backend/app` — the `app` package is preserved; `backend/datasets` → `application/backend/datasets`. |
| `frontend/` | `application/frontend/` | |
| `database/` | `application/database/` | `database` package preserved. |
| `gis/` | `application/gis/` | |
| `tests/` | `application/tests/` | |
| `deployment/` | `application/docker/` + `deployment/`-style docs in `docs/deployment/` | Dockerfiles, compose files, nginx, caddy, scripts all moved under `application/docker/`. |
| `research/` | `docs/research/` | Historical research documents. |
| `nginx/` | `application/docker/nginx/` | |
| `Dockerfile.*`, `docker-compose*.yml` | `application/docker/` | |
| — | `shared/` | New shared contract layer. |
| — | `releases/` | New; `latest/`, `v1.0/`, `v1.1/` placeholders. |

## 4. Python imports

~113 Python files were rewritten to the new module roots. Mapping:

| Old | New |
| --- | --- |
| `ai.models.*` | `training.models.*` |
| `ai.preprocessing.*` | `training.preprocessing.*` |
| `ai.training.*` | `training.training.*` |
| `ai.explainability.*` | `training.explainability.*` |
| `services.dataset_manager.*` | `training.dataset_manager.*` |
| `services.spatial_alignment.*` | `training.stam.*` |
| `quality.*` | `training.quality.*` |
| `mlops.*` | `training.mlops.*` |

Unchanged: `app.*` and `database.*`.

## 5. Path bootstrap

Runtime code adds exactly three roots to `sys.path`:

1. `application/backend` — for `app.*`
2. `application` — for `database.*`, `gis.*`
3. repo root — for `training.*` and `shared.*`

**Critical detail (verified by failures during migration)**: the **repository
root** must be added, *not* `training/`. The training engine lives at
`training/training/`, so adding the `training/` directory makes `import
training` resolve to that inner sub-package, breaking `import training.models`
and friends. With the repo root on `sys.path`, `import training` resolves to
`D:\CropPrep\training\__init__.py`.

The platform root packages `training/__init__.py`, `application/__init__.py`,
and `shared/__init__.py` were created (all `__version__ = "0.1.0"`).

Bootstrap is applied by:

- `application/backend/app/core/paths.py` — BACKEND_ROOT = parents[2],
  APPLICATION_ROOT = parents[3], REPO_ROOT = parents[4]; adds all three.
- `application/tests/conftest.py` — ROOT = parents[2] (repo root); adds
  `application/backend`, `application/gis`, repo root.
- `application/backend/app/tests/conftest.py` — adds `_BACKEND_ROOT`
  (parents[2]) and `_REPO_ROOT` (parents[4]).
- `application/tests/smoke/test_startup.py` — ROOT = parents[2] / "backend".
- Docker images — `PYTHONPATH=/app:/app/application:/app/application/backend`.

## 6. Tooling, Docker, CI/CD

- `pyproject.toml`: package `["cropfusion"]`; script
  `cropfusion-mlops = "training.mlops.cli:app"`; pytest `testpaths` =
  `application/tests application/backend/app/tests training shared`; pytest
  `pythonpath` = `[".", "application/backend"]`; ruff / mypy / bandit
  `src`, `exclude` updated.
- `pytest.ini`: rewritten with `testpaths`, `pythonpath`, and the full marker
  list.
- `Makefile`: `test-backend`, `lint`, `format`, `typecheck`, `security`, and
  compose targets point at the new paths.
- `environment.yml` / `requirements.txt`: first-party install entries now use
  `./training/{models,preprocessing,training,explainability,dataset_manager,stam}`.
- `.gitignore`: frontend build (`application/frontend`), dataset blobs
  (`application/backend/datasets`, `application/gis`, `**/gis/*.tif`), env
  files (`.env`).
- Env examples moved to `application/config/` (`.env.example`,
  `.env.production.example`).
- Docker: `application/docker/Dockerfile.{backend,admin,frontend,inference,docs}`;
  compose files use `context: ..` and
  `dockerfile: application/docker/Dockerfile.<image>`; volume bind-mounts
  rebased (`../monitoring/...`, `../training/quality/monitoring/grafana`,
  `./caddy/Caddyfile`, `../scripts/backup`).
- GitHub Actions: `ci.yml`, `docker.yml`, `deploy.yml`, `security.yml`,
  `release.yml` updated to the new paths; `dependabot.yml` updated (npm →
  `/application/frontend`, docker → `/application/docker`); `CODEOWNERS`
  created.

## 7. Docs and placeholders

- `docs/` reorganised into `architecture/`, `installation/`, `usage/`,
  `developer-guide/`, `deployment/`, `api/`, `research/`, `images/`,
  `diagrams/`, `migration/`. Historical records (phase reports, research)
  were moved as-is and are **not** rewritten.
- New guides: root `README.md`, `docs/README.md`,
  `docs/architecture/ARCHITECTURE_GUIDE.md`, `docs/architecture/FOLDER_GUIDE.md`,
  `MIGRATION_GUIDE.md` (this report), per-platform READMEs
  (`training/README.md`, `application/README.md`, `shared/README.md`).
- Five Mermaid diagrams in `docs/diagrams/`: `repository-structure.md`,
  `dependency.md`, `folder-relationship.md`, `migration-flow.md`,
  `import-dependency.md`.
- Placeholder skeletons created with `.gitkeep` (empty directories are not
  tracked by git): `training/{feature_engineering, evaluation, experiments,
  export, hyperparameter_search, config, tests}`,
  `application/{inference, authentication, history, admin, models,
  inference_package}`, `releases/{latest, v1.0, v1.1}`.
- `training/kaggle/` populated (README.md, requirements.txt, setup.py,
  .gitignore).

## 8. Intentional leftovers

These are deliberately unchanged:

- `training/stam/logger.py` uses runtime logger identifiers
  `cropfusion.spatial_alignment.*` (`ROOT_NAME`), and
  `training/dataset_manager/logger.py` similarly — they are self-consistent
  logger names, not imports, so no code change was required.
- `application/frontend/` contains `src/services/*.ts` — this is the frontend
  API-client layer, unrelated to the old Python `services/` directory.
- Historical documents (`docs/research/`, phase reports, `FINAL PROJECT
  SUMMARY.md`) are preserved as dated records of the old layout.

## 9. Verification

| Check | Result |
| ----- | ------ |
| `compileall -q training application shared scripts cropfusion` | exit 0 |
| Grep for stale imports `(from|import) (ai\|services\|quality\|mlops)` | 0 matches |
| Grep for stale docstring `:mod:`/`~services.*` refs | fixed (6 files) |
| Sanity imports (app.main, app.core.config/paths, database.seeds.runner, training.mlops.cli/scheduler, training.quality, training.stam.*, training.dataset_manager, training.preprocessing, training.models.{exporter,cropfusion,factory,config}) | all OK |
| `pytest --collect-only` backend / training / application/tests | 80 + 639 + 8 = 727 collected, 0 errors |
| `pytest -q application/tests` (smoke) | 8 passed |
| `pytest training/stam/tests training/models/tests` | 232 passed |
| Stale path references in non-doc config (Makefile, pyproject, workflows, compose) | 0 (verified by grep; matches found were the new valid paths) |
| Module READMEs referencing old install/run paths | all updated |

Pydantic `model_*` protected-namespace warnings are pre-existing and unrelated
to the migration.

## 10. Rollback / safety

Nothing was deleted. Any file can be recovered from Git history. The old
layout can be restored by reversing the mapping in §3 (git-recognises the
moves as renames; see the `{old => new}` entries in `git status`).

Git state: branch `main`, all changes staged, **nothing committed yet**.
