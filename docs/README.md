# CropFusion Documentation

## Getting started

- [INSTALLATION.md](INSTALLATION.md) — local (non-Docker) environment setup
- [QUICKSTART.md](QUICKSTART.md) — 5-minute Docker quick start
- [DEPLOYMENT.md](DEPLOYMENT.md) — Docker Compose (dev + prod), TLS, monitoring, backups

## Guides

- [API.md](API.md) — HTTP API overview
- [BACKEND.md](BACKEND.md) — backend architecture & modules
- [FRONTEND.md](FRONTEND.md) — frontend development guide
- [DATABASE.md](DATABASE.md) — schema, migrations, PostGIS
- [MLOPS.md](MLOPS.md) — model registry, promotion gates, scheduler
- [DEVELOPMENT.md](DEVELOPMENT.md) — how to work in this repository
- [TESTING.md](TESTING.md) — running and writing tests
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — common problems and fixes

## Project

- [CONTRIBUTING.md](CONTRIBUTING.md) — contribution guidelines
- [CHANGELOG.md](CHANGELOG.md) — version history
- [ROADMAP.md](ROADMAP.md) — planned work
- [CITATION.md](CITATION.md) — how to cite CropFusion

## Manuals

- [manuals/FARMER_GUIDE.md](manuals/FARMER_GUIDE.md)
- [manuals/ADMIN_GUIDE.md](manuals/ADMIN_GUIDE.md)
- [manuals/RESEARCHER_GUIDE.md](manuals/RESEARCHER_GUIDE.md)
- [manuals/DEVELOPER_GUIDE.md](manuals/DEVELOPER_GUIDE.md)

## Reference

- [SOFTWARE_DESIGN_DOCUMENT.md](SOFTWARE_DESIGN_DOCUMENT.md) — the original design document
- Phase completion reports: `PHASE2_COMPLETION_REPORT.md` … `PHASE11_COMPLETION_REPORT.md`
- Research assets: `../research/` (architecture diagrams, benchmarks, dataset stats)

## Package documentation

Each package documents itself:

- `ai/models/docs/` — architecture, layers, configuration, development
- `ai/preprocessing/docs/`, `ai/training/docs/`, `ai/explainability/docs/`
- `services/dataset_manager/docs/`, `services/spatial_alignment/docs/`
