# Prediction Platform (`application/`)

The **Prediction Platform** is the production deployment surface: the FastAPI
backend, React frontend, enterprise database, GIS services, observability and
inference runtime. It depends only on `shared/` (never on `training/`).

## Layout

| Directory | Responsibility |
| --------- | -------------- |
| [`backend/`](backend/app/) | FastAPI modular monolith (`app` package): auth + RBAC, predictions, history, notifications, explainability, GIS, registries |
| [`frontend/`](frontend/) | React 19 + TypeScript + Vite SPA (PWA) |
| [`database/`](database/) | Enterprise database layer + Alembic migrations (PostGIS) |
| [`gis/`](gis/) | GIS / spatial services |
| [`inference/`](inference/) | Inference runtime (placeholder) |
| [`authentication/`](authentication/) | Authentication services (placeholder) |
| [`history/`](history/) | Prediction history services (placeholder) |
| [`admin/`](admin/) | Admin panel (placeholder) |
| [`models/`](models/) | Deployed model artifacts (placeholder) |
| [`inference_package/`](inference_package/) | Packaged inference library (placeholder) |
| [`monitoring/`](monitoring/) | Prometheus / Grafana / Loki configs |
| [`docker/`](docker/) | Dockerfiles, docker-compose files, nginx / Caddy configs |
| [`config/`](config/) | Environment example files (`.env.example`, `.env.production.example`) |
| [`tests/`](tests/) | Platform-wide QA suite |

## Running locally

```bash
# Backend (from application/backend/)
cp ../../application/config/.env.example ../../.env
pip install -e ../../training/models ../../training/preprocessing \
  ../../training/training ../../training/explainability
pip install -e ../../training/dataset_manager ../../training/stam
uvicorn app.main:app --reload --port 8000

# Frontend (from application/frontend/)
npm ci && npm run dev
```

Or bring up the full stack with
`docker compose -f application/docker/docker-compose.yml up -d`.
