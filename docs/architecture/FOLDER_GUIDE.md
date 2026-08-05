# Repository & Folder Guide

This guide tours the repository after the two-platform reorganisation.

## Root

| Path | Purpose |
| ---- | ------- |
| `training/` | Training Platform (see [training/README.md](../../training/README.md)) |
| `application/` | Prediction Platform (see [application/README.md](../../application/README.md)) |
| `shared/` | Platform-agnostic contracts (see [shared/README.md](../../shared/README.md)) |
| `docs/` | Documentation, organised by topic (see [docs/README.md](../README.md)) |
| `releases/` | Tagged release archives (`latest/`, `v1.0/`, `v1.1/`) |
| `scripts/` | Docs build, backups |
| `.github/` | CI/CD workflows, dependabot, CODEOWNERS |
| `cropfusion/` | Umbrella meta-package |
| `datasets/`, `Tabular_Datasets/` | Dataset caches (mostly git-ignored) |

## training/ — Training Platform

| Path | Purpose |
| ---- | ------- |
| `models/` | TabTransformer, Dual-CNN, temporal Transformer, fusion, exporters |
| `preprocessing/` | Tabular + satellite preprocessing |
| `training/` | Training engine, losses, metrics |
| `explainability/` | SHAP / saliency explanations |
| `dataset_manager/` | Dataset profiling, validation, versioning |
| `stam/` | Spatio-temporal alignment |
| `quality/` | Drift, fairness, optimization, monitoring |
| `mlops/` | Model registry, promotion gates, scheduler |
| `feature_engineering/`, `evaluation/`, `experiments/`, `export/`, `hyperparameter_search/`, `config/`, `tests/` | Placeholder scaffold for future training work |
| `kaggle/` | Kaggle notebooks and runtime |

## application/ — Prediction Platform

| Path | Purpose |
| ---- | ------- |
| `backend/app/` | FastAPI modular monolith (the `app` Python package) |
| `backend/datasets/` | Runtime dataset workspace (git-ignored) |
| `frontend/` | React 19 SPA (PWA) |
| `database/` | Enterprise DB layer + Alembic migrations |
| `gis/` | GIS / spatial services |
| `monitoring/` | Prometheus / Grafana / Loki configuration |
| `docker/` | Dockerfiles, compose files, nginx/Caddy configs |
| `config/` | Environment example files |
| `tests/` | Platform-wide QA suite |
| `inference/`, `authentication/`, `history/`, `admin/`, `models/`, `inference_package/` | Placeholder scaffold for future prediction work |

## shared/ — Contracts

| Path | Purpose |
| ---- | ------- |
| `schemas/` | Shared schema definitions |
| `dto/` | Cross-platform data-transfer objects |
| `enums/` | Shared enumerations |
| `interfaces/` | Abstract interfaces |
| `validation/` | Shared validation rules |
| `utils/` | Shared utilities |
| `exceptions/` | Shared exception hierarchy |
| `config/` | Shared configuration models |
| `constants/` | Shared constants |
| `serialization/` | Shared (de)serialization |
| `tests/` | Shared-layer tests |

## docs/

| Path | Purpose |
| ---- | ------- |
| `architecture/` | Architecture + folder guides, design document |
| `installation/` | Install / quickstart / troubleshooting |
| `usage/` | End-user guides (frontend) |
| `developer-guide/` | Development, testing, contributing |
| `deployment/` | Deployment, backend, database, MLOps |
| `api/` | HTTP API overview |
| `research/` | Research assets (was `research/` at repo root) |
| `images/` | Shared images |
| `diagrams/` | Generated architecture diagrams |
| `migration/` | Migration guide + report |
| `manuals/`, `releases/`, `website/` | Manuals, release notes, website content |
