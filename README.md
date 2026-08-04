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

## Feature highlights

- **Multimodal model** — TabTransformer + Dual CNN (NDVI/EVI) + temporal
  Transformer + cross-modal gated fusion (`ai/models`).
- **Enterprise backend** — modular FastAPI monolith: auth + RBAC, predictions,
  history, notifications, explainability, GIS, model/dataset registries
  (`backend/`).
- **Dataset & spatial management** — dataset profiling/validation/versioning
  (`services/dataset_manager`) and spatio-temporal alignment
  (`services/spatial_alignment`).
- **ML quality gates** — drift, fairness, and optimization tooling (`quality/`)
  with Grafana dashboards.
- **MLOps** — model registry CLI, promotion gates, rollback, experiment
  tracking, scheduled monitoring (`mlops/`).
- **Deployment** — Docker Compose (dev + prod) with Caddy TLS, Prometheus,
  Grafana, Loki, and automated backups.

## Repository layout

```
cropfusion/
├── ai/            # models, preprocessing, training, explainability
├── backend/       # FastAPI modular monolith + enterprise database (Alembic)
├── frontend/      # React 19 + TypeScript + Vite SPA (PWA)
├── services/      # dataset_manager, spatial_alignment
├── quality/       # drift, fairness, monitoring, optimization
├── mlops/         # registry, gates, scheduler, experiments, reports
├── research/      # architecture diagrams, benchmarks, dataset stats
├── docs/          # installation, deployment, API, manuals, releases
├── deployment/    # compose configs, monitoring, TLS (Caddy)
├── scripts/       # docs build, backups
└── datasets/      # dataset caches (git-ignored)
```

## Quick start

```bash
# One-command development stack (frontend :3000, backend :8000, docs :8080)
cp .env.example .env
docker compose up -d

# Docs site
open http://localhost:8080
```

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for the step-by-step guide and
[docs/INSTALLATION.md](docs/INSTALLATION.md) for local (non-Docker) setup.

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
pip install -e ./ai/models ./ai/preprocessing ./ai/training ./ai/explainability
pip install -e ./services/dataset_manager ./services/spatial_alignment
cd frontend && npm ci && cd ..
make test      # full Python suite
make compose   # validate compose files
```

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) and
[CONTRIBUTING.md](CONTRIBUTING.md).

## Documentation

- [docs/README.md](docs/README.md) — documentation index
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — Docker, TLS, monitoring, backups
- [docs/MLOPS.md](docs/MLOPS.md) — model promotion workflow
- [docs/API.md](docs/API.md) — API overview
- [research/ARCHITECTURE.md](research/ARCHITECTURE.md) — system architecture
- [research/MODEL_ARCHITECTURE.md](research/MODEL_ARCHITECTURE.md) — neural architecture

## Reproducibility

- `requirements.txt` — pinned runtime dependencies
- `environment.yml` — conda environment (Python 3.12)
- Git commit + dataset version recorded on every model registration

## License

Apache-2.0 — see [LICENSE](LICENSE). See [SECURITY.md](SECURITY.md) for the
vulnerability reporting policy.
