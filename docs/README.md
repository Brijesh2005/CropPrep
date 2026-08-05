# CropFusion Documentation

## Getting started

- [installation/INSTALLATION.md](installation/INSTALLATION.md) — local (non-Docker) environment setup
- [installation/QUICKSTART.md](installation/QUICKSTART.md) — 5-minute Docker quick start
- [installation/TROUBLESHOOTING.md](installation/TROUBLESHOOTING.md) — common problems and fixes
- [deployment/DEPLOYMENT.md](deployment/DEPLOYMENT.md) — Docker Compose (dev + prod), TLS, monitoring, backups

## Architecture

- [architecture/ARCHITECTURE_GUIDE.md](architecture/ARCHITECTURE_GUIDE.md) — two-platform architecture
- [architecture/FOLDER_GUIDE.md](architecture/FOLDER_GUIDE.md) — folder-by-folder tour
- [architecture/SOFTWARE_DESIGN_DOCUMENT.md](architecture/SOFTWARE_DESIGN_DOCUMENT.md) — original design document
- [diagrams/](diagrams/) — architecture diagrams (incl. [R1.2 provider architecture](diagrams/r1-2-provider-architecture.md) and [R1.3 diagrams](diagrams/r1-3-shared-packages.md))
- [migration/MIGRATION_GUIDE.md](migration/MIGRATION_GUIDE.md) — migration to the two-platform layout
- [migration/MIGRATION_REPORT.md](migration/MIGRATION_REPORT.md) — detailed migration report
- [migration/MIGRATION_REPORT_R1.2.md](migration/MIGRATION_REPORT_R1.2.md) — Dataset Manager provider pattern
- [migration/MIGRATION_REPORT_R1.3.md](migration/MIGRATION_REPORT_R1.3.md) — shared framework extraction

## Shared framework (R1.3)

- [shared/SHARED.md](shared/SHARED.md) — overview of the `shared/` package
- [shared/ARCHITECTURE.md](shared/ARCHITECTURE.md) — how `shared/` fits the two platforms
- [shared/DEVELOPER_GUIDE.md](shared/DEVELOPER_GUIDE.md) — working with `shared/` from either platform
- [shared/CONFIGURATION.md](shared/CONFIGURATION.md) — configuration loading, env vars, precedence
- [shared/EXTENSION_GUIDE.md](shared/EXTENSION_GUIDE.md) — adding serializers, validators, providers
- [shared/CODING_STANDARDS.md](shared/CODING_STANDARDS.md) — conventions for `shared/` code

## Prediction platform (R1.4)

- [prediction/PREDICTION_PLATFORM.md](prediction/PREDICTION_PLATFORM.md) — overview of the Prediction Platform
- [prediction/ARCHITECTURE.md](prediction/ARCHITECTURE.md) — inference-only architecture
- [prediction/INFERENCE.md](prediction/INFERENCE.md) — the inference pipeline contract
- [prediction/MODEL_LOADING.md](prediction/MODEL_LOADING.md) — loading exported model artifacts
- [prediction/INFERENCE_PACKAGE.md](prediction/INFERENCE_PACKAGE.md) — the consumed artifact set
- [prediction/DEPLOYMENT.md](prediction/DEPLOYMENT.md) — inference-only Docker deployment
- [migration/MIGRATION_REPORT_R1.4.md](migration/MIGRATION_REPORT_R1.4.md) — inference-only preparation
- [diagrams/](diagrams/) — incl. [R1.4 prediction architecture](diagrams/r1-4-prediction-architecture.md)

## Kaggle training infrastructure (R2.1)

- [training/kaggle/docs/SETUP.md](../training/kaggle/docs/SETUP.md) — training setup guide
- [training/kaggle/docs/KAGGLE.md](../training/kaggle/docs/KAGGLE.md) — Kaggle guide
- [training/kaggle/docs/BOOTSTRAP.md](../training/kaggle/docs/BOOTSTRAP.md) — bootstrap guide
- [training/kaggle/docs/WORKSPACE.md](../training/kaggle/docs/WORKSPACE.md) — workspace guide
- [training/kaggle/docs/CONFIGURATION.md](../training/kaggle/docs/CONFIGURATION.md) — configuration guide
- [migration/MIGRATION_REPORT_R2.1.md](migration/MIGRATION_REPORT_R2.1.md) — Kaggle training infrastructure

## Guides

- [api/API.md](api/API.md) — HTTP API overview
- [deployment/BACKEND.md](deployment/BACKEND.md) — backend architecture & modules
- [usage/FRONTEND.md](usage/FRONTEND.md) — frontend development guide
- [deployment/DATABASE.md](deployment/DATABASE.md) — schema, migrations, PostGIS
- [deployment/MLOPS.md](deployment/MLOPS.md) — model registry, promotion gates, scheduler
- [developer-guide/DEVELOPMENT.md](developer-guide/DEVELOPMENT.md) — how to work in this repository
- [developer-guide/TESTING.md](developer-guide/TESTING.md) — running and writing tests
- [developer-guide/CONTRIBUTING.md](developer-guide/CONTRIBUTING.md) — contribution guidelines

## Project

- [CHANGELOG.md](CHANGELOG.md) — version history
- [ROADMAP.md](ROADMAP.md) — planned work
- [CITATION.md](CITATION.md) — how to cite CropFusion

## Manuals

- [manuals/FARMER_GUIDE.md](manuals/FARMER_GUIDE.md)
- [manuals/ADMIN_GUIDE.md](manuals/ADMIN_GUIDE.md)
- [manuals/RESEARCHER_GUIDE.md](manuals/RESEARCHER_GUIDE.md)
- [manuals/DEVELOPER_GUIDE.md](manuals/DEVELOPER_GUIDE.md)

## Reference

- Phase completion reports: `PHASE2_COMPLETION_REPORT.md` … `PHASE10_COMPLETION_REPORT.md`
- Research assets: [research/](research/) (architecture diagrams, benchmarks, dataset stats)

## Package documentation

Each package documents itself:

- `training/models/docs/` — architecture, layers, configuration, development
- `training/preprocessing/docs/`, `training/training/docs/`, `training/explainability/docs/`
- `training/dataset_manager/docs/`, `training/stam/docs/`
- `application/database/docs/`
