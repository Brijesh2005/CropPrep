# Backend

The backend (`backend/`) is a **domain-driven modular monolith** built with
FastAPI. Modules are isolated and extractable into microservices; today they
share one process and one database for simplicity and data consistency.

## Layout

```
backend/
├── alembic.ini                 # Alembic configuration
├── database/
│   ├── migrations/versions/    # schema migrations
│   └── models/                 # SQLAlchemy ORM models
└── app/
    ├── main.py                 # application factory (create_app)
    ├── core/                   # config, database, security, logging, exceptions
    ├── modules/                # domain modules (auth, inference, gis, ...)
    ├── services/               # cross-cutting services (registry, cache, metrics)
    ├── middleware/             # prometheus, correlation-id, rate limiting
    ├── models/                 # ORM models (domain)
    ├── repositories/           # data-access layer
    ├── schemas/                # Pydantic request/response schemas
    ├── dependencies/           # DI + auth dependencies
    ├── events/                 # startup/shutdown lifecycle
    └── workers/                # background task definitions
```

## Modules

| Module | Responsibility |
|---|---|
| `auth` | registration, login, refresh, reset, email verification, sessions |
| `users` | profile, preferences, saved locations |
| `inference` | prediction engine wiring (model registry -> predictions) |
| `predictions` | prediction history + search |
| `explainability` | feature attribution + summaries |
| `gis` | location index + spatial lookups |
| `dataset` | dataset management API |
| `monitoring` | metrics snapshots, quality views |
| `admin` | dashboards, audit log, enterprise admin |
| `configuration` | runtime config endpoint |
| `health` | `/health`, `/live`, `/ready` |
| `history` | user history aggregation |

## Application factory

`create_app()` (in `app/main.py`) builds and wires everything:

1. Load settings (env > YAML > defaults).
2. Configure logging (loguru, structured JSON).
3. Build the DI container (database, cache, model registry, inference engine,
   explainability service, spatial index).
4. Register middleware (prometheus, tracing, correlation id, rate limiting).
5. Include routers.
6. Run startup: load + warm the model, refresh datasets, build spatial index.

## Configuration

`app/core/config.py` exposes typed settings (pydantic v2) under the
`BACKEND_<SECTION>__<KEY>` env convention:

```bash
BACKEND_DATABASE__URL=postgresql+asyncpg://cropfusion:cropfusion@postgres:5432/cropfusion
BACKEND_DATABASE__POSTGIS=true
BACKEND_SECURITY__SECRET_KEY=<strong-secret>
BACKEND_MODEL__CHECKPOINT_PATH=/app/models/yieldnet.pt
```

## Inference

The inference engine loads the model through `app/services/model_registry.py`,
warms it at startup, and serves predictions with caching. When no model is
available the engine returns an explicit heuristic fallback (never a crash).

## Observability

- `cropfusion_*` Prometheus metrics via `app/services/prometheus.py`.
- OpenTelemetry tracing (FastAPI + ASGI instrumentation).
- Structured JSON logs with correlation IDs.

## Running locally

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```
