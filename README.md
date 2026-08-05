<div align="center">

# CropFusion

**Precision agriculture decision support** — multimodal crop-yield prediction,
spatial analysis, and drift/fairness monitoring in one platform.

Python 3.12 · FastAPI · React 19 · PyTorch · PostGIS · Docker

</div>

---

CropFusion fuses **tabular agronomic data** with **satellite vegetation-index
imagery** to recommend crops and predict yields at district/village
resolution, then explains every recommendation to non-expert users. It ships
as a full MLOps-ready platform: model registry with promotion gates, drift and
fairness monitoring, a production monitoring stack, and one-command Docker
deployments.

The repository is organised into two independent platform roots — a
**Training Platform** and a **Prediction Platform** — that share code only
through a common `shared/` contract layer:

```
cropfusion/
├── training/       # Training Platform: models, preprocessing, training,
│                   #   explainability, dataset manager, STAM, quality, MLOps
├── application/    # Prediction Platform: backend, frontend, database, GIS,
│                   #   monitoring, docker, config, tests
├── shared/         # Platform-agnostic schemas, DTOs, interfaces, validation
├── docs/           # architecture, installation, usage, deployment, research
├── releases/       # tagged release archives
├── scripts/        # docs build, backups
├── datasets/       # dataset caches (git-ignored)
└── .github/        # CI/CD, dependabot, CODEOWNERS
```

## Feature highlights

- **Multimodal model** — TabTransformer + Dual CNN (NDVI/EVI) + temporal
  Transformer + cross-modal gated fusion (`training/models`).
- **Enterprise backend** — modular FastAPI monolith: auth + RBAC, predictions,
  history, notifications, explainability, GIS, model/dataset registries
  (`application/backend`).
- **Dataset & spatial management** — dataset profiling/validation/versioning
  (`training/dataset_manager`) and spatio-temporal alignment
  (`training/stam`).
- **ML quality gates** — drift, fairness, and optimization tooling
  (`training/quality`) with Grafana dashboards.
- **MLOps** — model registry CLI, promotion gates, rollback, experiment
  tracking, scheduled monitoring (`training/mlops`).
- **Deployment** — Docker Compose (dev + prod) with Caddy TLS, Prometheus,
  Grafana, Loki, and automated backups.

## Platform roots

| Root | Platform | Contents |
| ---- | -------- | -------- |
| [`training/`](training/README.md) | Training | Model development, dataset management, evaluation, quality gates, experiment tracking |
| [`application/`](application/README.md) | Prediction | Backend, frontend, database, GIS, monitoring, inference |
| [`shared/`](shared/README.md) | Contracts | Schemas, DTOs, enums, interfaces, validation, serialization |

## Quick start

```bash
# One-command development stack (frontend :3000, backend :8000, docs :8080)
cp application/config/.env.example .env
docker compose -f application/docker/docker-compose.yml up -d

# Docs site
open http://localhost:8080
```

See [docs/installation/QUICKSTART.md](docs/installation/QUICKSTART.md) for the
step-by-step guide and [docs/installation/INSTALLATION.md](docs/installation/INSTALLATION.md)
for local (non-Docker) setup.

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
pip install -e ./training/models ./training/preprocessing ./training/training ./training/explainability
pip install -e ./training/dataset_manager ./training/stam
cd application/frontend && npm ci && cd ../..
make test      # full Python suite
make compose   # validate compose files
```

See [docs/developer-guide/DEVELOPMENT.md](docs/developer-guide/DEVELOPMENT.md),
[docs/developer-guide/CONTRIBUTING.md](docs/developer-guide/CONTRIBUTING.md)
and the [Repository & Folder Guide](docs/architecture/FOLDER_GUIDE.md).

## Documentation

- [docs/README.md](docs/README.md) — documentation index
- [docs/architecture/ARCHITECTURE_GUIDE.md](docs/architecture/ARCHITECTURE_GUIDE.md) — platform architecture
- [docs/architecture/FOLDER_GUIDE.md](docs/architecture/FOLDER_GUIDE.md) — folder-by-folder tour
- [docs/deployment/DEPLOYMENT.md](docs/deployment/DEPLOYMENT.md) — Docker, TLS, monitoring, backups
- [docs/deployment/MLOPS.md](docs/deployment/MLOPS.md) — model promotion workflow
- [docs/api/API.md](docs/api/API.md) — API overview
- [docs/research/ARCHITECTURE.md](docs/research/ARCHITECTURE.md) — system architecture
- [docs/migration/MIGRATION_GUIDE.md](docs/migration/MIGRATION_GUIDE.md) — two-platform migration guide

## Reproducibility

- `requirements.txt` — pinned runtime dependencies
- `environment.yml` — conda environment (Python 3.12)
- Git commit + dataset version recorded on every model registration

## License

Apache-2.0 — see [LICENSE](LICENSE). See [SECURITY.md](SECURITY.md) for the
vulnerability reporting policy.
