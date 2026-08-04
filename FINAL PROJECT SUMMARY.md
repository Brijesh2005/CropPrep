# CropFusion — Final Project Summary

**Version:** 1.0.0
**Date:** 2026-08-04
**Status:** All phases complete; final validation pass green.

## What was delivered in this final phase

The deployment / DevOps / MLOps / documentation / research-assets / release-packaging / open-source groundwork for the CropFusion monorepo. No prior Phase 1–11 functionality was redesigned.

### Deployment & DevOps
- **Containers:** `Dockerfile.backend` (multi-stage wheels, non-root `cropfusion` user, `/health` healthcheck, 2× uvicorn workers), `Dockerfile.frontend` (Vite build → nginx SPA), `Dockerfile.inference` (GPU warm replica on :8001), `Dockerfile.admin` (MLOps scheduler), `Dockerfile.docs` (static docs via `scripts/build_docs.py`).
- **Compose stacks (both validated with `docker compose config --quiet`):**
  - `docker-compose.yml` (dev): postgres + postgis 16, redis 7, backend, frontend :3000, docs :8080, prometheus :9090, grafana :3001, loki :3100, promtail, node-exporter; `gpu` / `mlops` / `devtools` profiles.
  - `docker-compose.prod.yml`: Caddy TLS, secrets, memory limits, backup service, docs service.
- **Networking:** `nginx.conf` (SPA + `/api` → backend:8000), `deployment/caddy/Caddyfile`.
- **Monitoring:** `deployment/monitoring/` Prometheus config + alert rules (backend down, 5xx > 5%, p95 > 2s, drift > 0.1, disk < 10%), Loki/Promtail logging, Grafana auto-provisioned datasources and dashboards.
- **Backups:** `scripts/backup/backup-db.sh`, `backup-assets.sh`, `restore-db.sh` (optional S3 offsite).

### MLOps & Reproducibility
- `mlops/` package: registry (keep 3 production versions), metric / latency-regression gates (fail closed), experiments, reports, scheduler, and the `cropfusion-mlops` CLI (`mlops.cli:app`).
- Root `pyproject.toml`: umbrella `cropfusion` package, console scripts, ruff/mypy/coverage/bandit config, `testpaths`, `pythonpath = ["."]`.
- Pinned `requirements.txt`, `requirements-dev.txt`, `environment.yml`, `Makefile`, `.editorconfig`, `VERSION`, `.env.example`, `.env.production.example`, `.dockerignore`, `.gitignore`.

### CI/CD & Open Source
- `.github/workflows/`: `ci.yml` (matrix per-package tests + lint + frontend + coverage + compose validation), `docker.yml` (GHCR multi-image), `security.yml` (Safety / Bandit / npm audit / Trivy / CodeQL), `release.yml` (tag-gated tarball + GitHub Release), `deploy.yml` (staging → production rsync + compose over SSH), `dependabot.yml`.
- `LICENSE` (Apache-2.0), `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, issue templates (bug / feature / security), `PULL_REQUEST_TEMPLATE.md`, `docs/CHANGELOG.md`, `docs/ROADMAP.md`, `docs/CITATION.md`.

### Documentation & Research Assets
- Docs site (25 pages built by `scripts/build_docs.py`): README, INSTALLATION, QUICKSTART, DEPLOYMENT, API, BACKEND, FRONTEND, DATABASE, MLOPS, DEVELOPMENT, TESTING, TROUBLESHOOTING, CONTRIBUTING, CHANGELOG, ROADMAP, CITATION, `docs/manuals/` (FARMER, ADMIN, RESEARCHER, DEVELOPER), `docs/releases/` (v1.0.0, UPGRADE), `docs/website/`.
- `research/`: `ARCHITECTURE.md`, `MODEL_ARCHITECTURE.md`, `DATASETS.md`, `BENCHMARKS.md`, `TRAINING_AND_EVAL.md`, `dataset_stats.json` (35,831 rows / 124 cols across 5 CSVs), `scripts/dataset_stats.py`.

### Bug fixed during validation
`torch.load(..., weights_only=True)` (torch 2.13 default) rejected two training-checkpoint payloads:
- numpy RNG state (ndarray) → now serialised to primitives via `_serialize_numpy_state` / `_deserialize_numpy_state` (`ai/training/checkpoint.py`);
- `Path` fields in config dumps → now stored via `model_dump(mode="json")` (`ai/training/checkpoint.py`, `ai/models/checkpoint.py`).

All 3 previously-failing tests (`test_checkpoint` ×2, `test_trainer::test_resume_continues_from_checkpoint`) now pass.

## Final validation results

| Suite | Result |
|---|---|
| backend/app/tests | 80 passed |
| quality | 63 passed |
| mlops | 15 passed |
| root `tests` | 8 passed |
| ai/training | 67 passed |
| ai/models | 131 passed |
| ai/preprocessing | 11 passed |
| ai/explainability | 1 passed |
| services (dataset_manager + spatial_alignment) | 218 passed |
| **Total** | **594 passed** |
| Docs build | 25 pages |
| `docker-compose.yml` + `docker-compose.prod.yml` | `config --quiet` OK |

Note: `ruff` / `mypy` are not installed in this local conda env; they run in CI (`ci.yml`, `security.yml`). Expected local environment notes (no MSVC compiler → `torch.compile` inductor falls back to lazy eager) remain documented in `docs/TROUBLESHOOTING.md`.

## How to run

```bash
# Tests (whole repo, honors root pytest.ini)
pytest

# Docs site
python scripts/build_docs.py

# Dev stack
docker compose up --build          # frontend :3000, backend, docs :8080, postgis, redis
docker compose --profile mlops --profile devtools up -d   # scheduler, prometheus, grafana, loki

# Production (with DOMAIN set)
docker compose -f docker-compose.prod.yml up -d

# MLOps CLI
cropfusion-mlops --help
cropfusion-mlops register yieldnet path/to/model.pt
cropfusion-mlops promote yieldnet 1.0.0
```
