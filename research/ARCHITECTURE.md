# CropFusion System Architecture

CropFusion is a precision-agriculture decision support platform. It ingests
tabular agronomic data plus satellite-derived vegetation indices, produces
crop-recommendation and yield-prediction outputs, and explains them to
non-expert users. This document summarises the architecture at the system
level; see [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) for the neural model.

## High-level system diagram

```mermaid
flowchart LR
    subgraph Frontend["Frontend (React 19 + Vite, PWA)"]
        UI[Farmer / Admin UI]
    end

    subgraph Backend["Backend (FastAPI modular monolith)"]
        API[API Gateway /auth /predict /explain /registry]
        AUTH[Auth + RBAC]
        INF[Inference Engine]
        EXPL[Explainability]
        HIST[History & Notifications]
        REG[Model & Dataset Registry]
        MON[Prometheus + OTel]
    end

    subgraph Data["Data layer"]
        PG[(PostgreSQL + PostGIS)]
        RD[(Redis)]
        FS[(Model/artifact store)]
    end

    subgraph QA["ML Quality (quality/)"]
        DRIFT[Drift Monitor]
        FAIR[Fairness Evaluator]
        OPT[Optimized Runtime]
    end

    subgraph MLOps["MLOps (mlops/)"]
        REGCLI[Model Registry CLI]
        GATES[Promotion Gates]
        SCHED[Scheduler]
        EXPT[Experiment Tracker]
    end

    UI -->|/api/v1| API
    API --> AUTH
    API --> INF --> EXPL
    API --> REG --> FS
    API --> HIST
    INF --> PG
    API --> PG
    API --> RD
    API -.metrics.-> MON
    INF --> QA
    QA -.verdicts.-> MON
    SCHED --> QA
    SCHED --> REG
    SCHED -->|reports| FS
```

## Deployment topology

```mermaid
flowchart LR
    USR[Farmer / Admin] -->|HTTPS| CAD[Caddy edge proxy]
    CAD --> FE[frontend nginx]
    CAD --> API[backend uvicorn]
    API --> PG[(postgis)]
    API --> RD[(redis)]
    API --> INF[inference worker*]
    INF --> PG
    ADMIN[admin scheduler] --> API
    ADMIN --> QA[quality checks]
    PROM[prometheus] --> API
    PROM --> NX[node-exporter]
    GRA[grafana] --> PROM
    LOKI[loki] <--> PTAIL[promtail]
    PTAIL --> API
```

\* The inference worker shares the backend image and is enabled with the
`gpu` profile for horizontal scaling.

## Component responsibilities

| Layer | Package | Responsibility |
|---|---|---|
| Datasets | `services/dataset_manager` | Single access point for all datasets; profile, validate, cache, export |
| Spatial | `services/spatial_alignment` | Spatio-temporal alignment manager (STAM) |
| Preprocessing | `ai/preprocessing` | Builds model-ready batches (tabular, NDVI/EVI, masks) |
| Models | `ai/models` | Multimodal neural architecture (forward + export) |
| Training | `ai/training` | Training loop, evaluators, benchmarks, visualisation |
| Explainability | `ai/explainability` | Feature attribution / summaries |
| Quality | `quality/` | Drift, fairness, monitoring exporters, optimisation |
| API | `backend/app` | Modular FastAPI monolith (Phase 7-11) |
| Enterprise DB | `backend/database` | Alembic migrations + SQLAlchemy models |
| UI | `frontend` | React SPA + PWA |
| MLOps | `mlops/` | Registry, gates, scheduler, experiments, reports |

## Environment matrix

| Environment | Purpose | Stack |
|---|---|---|
| Development | Local dev | `docker-compose.yml` on `localhost:3000` |
| Staging | Pre-prod validation | `docker-compose.prod.yml` + Caddy |
| Production | Live service | `docker-compose.prod.yml` + Caddy + TLS |

## Configuration model

Every package is 12-factor: environment variables (prefixed) override YAML
config files which override defaults. The backend uses the
`BACKEND_<SECTION>__<KEY>` convention resolved by pydantic-settings-compatible
loaders shared across packages.

## Observability

- Prometheus metrics under `/metrics` (`cropfusion_*` namespace).
- OpenTelemetry tracing (FastAPI instrumentation) with Loki log shipping.
- Grafana dashboards: `quality/monitoring/grafana/*.json` (ML quality +
  performance), provisioned datasources in `deployment/monitoring/grafana/`.
