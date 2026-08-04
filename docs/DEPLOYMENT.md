# Deployment

This guide covers running CropFusion with Docker Compose for development,
staging and production, plus the observability stack, backups, security
hardening and the MLOps scheduler.

## Contents

- [Topology](#topology)
- [Prerequisites](#prerequisites)
- [Development stack](#development-stack)
- [Production stack](#production-stack)
- [Configuration](#configuration)
- [Observability](#observability)
- [Backups](#backups)
- [Security hardening](#security-hardening)
- [MLOps scheduler](#mlops-scheduler)
- [Upgrades & rollback](#upgrades--rollback)
- [Troubleshooting](#troubleshooting)

## Topology

```mermaid
flowchart LR
    USR[Users] --> CAD[Caddy - TLS]
    CAD --> FE[frontend (nginx SPA)]
    CAD --> API[backend (uvicorn, :8000)]
    CAD --> GRA[grafana]
    API --> PG[(postgres + postgis)]
    API --> RD[(redis)]
    API -.-> INF[inference replica :8001]
    ADM[admin scheduler] --> API
    PROM[prometheus] --> API & INF & NX[node-exporter]
    PTAIL[promtail] --> LOKI[loki]
```

## Prerequisites

- Docker Engine 24+ with the Compose plugin.
- `git`, and for local (non-Docker) runs: Python 3.12 + Node 20.
- Recommended: 4 vCPU / 8 GB RAM (16 GB for full stack with GPU inference).

## Development stack

```bash
cp .env.example .env
docker compose up -d
```

| Service | URL |
|---|---|
| Frontend (SPA) | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| OpenAPI docs | http://localhost:8000/docs |
| Documentation site | http://localhost:8080 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 (admin/admin) |
| Loki | http://localhost:3100 |

Optional profiles:

```bash
docker compose --profile devtools up -d   # pgAdmin on :5050
docker compose --profile mlops up -d      # admin scheduler (MLOps)
docker compose --profile gpu up -d        # warm inference replica on :8001
```

## Production stack

```bash
cp .env.production.example .env
# 1. Strong secrets
#    - BACKEND_SECURITY__SECRET_KEY: 64+ random bytes
#    - POSTGRES_PASSWORD, REDIS_PASSWORD, GRAFANA_PASSWORD
#    - secrets/db_password.txt (mounts POSTGRES_PASSWORD into postgres)
# 2. Your domain (Caddy issues a Let's Encrypt certificate automatically)
DOMAIN=cropfusion.app

docker compose -f docker-compose.prod.yml up -d
```

Production characteristics:

- **TLS at the edge** via Caddy (auto-renewed Let's Encrypt).
- **Non-root containers** (backend runs as `cropfusion` user).
- **Pinned image versions**, restart policies, and memory limits.
- **Secrets** from `secrets/db_password.txt` (git-ignored) and the environment.
- **Persistent volumes** for Postgres, Redis, models, datasets, Prometheus,
  Grafana, Loki and backups.

## Configuration

All configuration is environment-driven (12-factor). The backend follows the
`BACKEND_<SECTION>__<KEY>` convention, e.g.:

| Variable | Purpose |
|---|---|
| `BACKEND_DATABASE__URL` | Async SQLAlchemy URL (`postgresql+asyncpg://...`) |
| `BACKEND_DATABASE__POSTGIS` | Enable PostGIS spatial support |
| `BACKEND_SECURITY__SECRET_KEY` | JWT signing secret |
| `BACKEND_SECURITY__SECURE_COOKIES` | `true` in production |
| `BACKEND_MONITORING__PROMETHEUS_ENABLED` | Expose `/metrics` |
| `BACKEND_MONITORING__TRACING_ENABLED` | OpenTelemetry tracing |
| `BACKEND_MODEL__WARMUP` | Warm the model at startup |
| `BACKEND_MODEL__CHECKPOINT_PATH` | Trained checkpoint to serve |
| `BACKEND_MODEL__MODEL_CONFIG_PATH` | YAML model config |

YAML config files are also supported via `BACKEND_CONFIG_FILE`; env wins over
YAML which wins over defaults.

## Observability

- **Metrics:** Prometheus scrapes `/metrics` on `backend:8000`
  (`cropfusion_*` namespace). Rule file: `deployment/monitoring/alerts.yml`.
- **Dashboards:** Grafana is provisioned with Prometheus + Loki datasources
  and auto-loads the ML-quality and performance dashboards from
  `quality/monitoring/grafana/`.
- **Logs:** the backend emits structured JSON (loguru); Promtail ships
  container logs to Loki; correlation IDs link requests across traces.

Drift/fairness verdicts are exported to Prometheus by the admin scheduler and
are available as gauges for alerting (`cropfusion_drift_score`, etc.).

## Backups

- **Database:** `scripts/backup/backup-db.sh` (run by the `backup` service in
  prod) dumps Postgres daily into `backups/`, prunes after
  `BACKUP_RETENTION_DAYS`, and optionally uploads to S3.
- **Assets (models/registry/reports/config):** `scripts/backup/backup-assets.sh`.
- **Restore:** `scripts/backup/restore-db.sh` with `RESTORE_FILE` set.

Offsite backup requires an S3-compatible endpoint:

```bash
BACKUP_S3_BUCKET=cropfusion-backups
BACKUP_S3_ENDPOINT=https://s3.example.com
BACKUP_AWS_ACCESS_KEY_ID=...
BACKUP_AWS_SECRET_ACCESS_KEY=...
```

## Security hardening

1. **Secrets:** never commit `.env`; use the platform secret store or
   `secrets/` (git-ignored). Rotate `BACKEND_SECURITY__SECRET_KEY` on breach.
2. **TLS:** Caddy enforces HTTPS + HSTS. Grafana runs behind the sub-path
   (`/grafana/`).
3. **Headers:** strict transport, `X-Content-Type-Options`, `X-Frame-Options`,
   `Referrer-Policy` are set at the edge.
4. **Containers:** non-root users, pinned tags, `readonly` where practical.
5. **Scans:** weekly dependency scans (Safety, Trivy, `npm audit`) and CodeQL
   run in CI (`.github/workflows/security.yml`); Dependabot opens PRs.
6. **Auth:** Argon2id password hashing, JWT with refresh tokens, per-user
   session limits, account lockout after failed attempts (Phase 10).

See [SECURITY.md](../SECURITY.md) for the disclosure policy.

## MLOps scheduler

The `admin` container runs `python -m mlops.scheduler` on an interval and:

1. Runs the **drift battery** (reference vs current data) and writes reports.
2. Runs the **fairness evaluator** (reads `reports/fairness_inputs.json`).
3. Verifies registry invariants (no duplicate production versions).

Enable it in dev:

```bash
docker compose --profile mlops up -d
MLOPS_INTERVAL_SECONDS=3600   # in .env
```

Model promotion is a human-in-the-loop workflow — see [MLOPS.md](MLOPS.md).

## Upgrades & rollback

- Pull the new images and recreate: `docker compose -f docker-compose.prod.yml pull && docker compose -f docker-compose.prod.yml up -d`.
- Database migrations run via Alembic; upgrade migration files are released
  with the version (`backend/database/migrations/versions/`).
- Application rollback: `docker compose ... up -d` with the previous image tag.
- Model rollback: `cropfusion-mlops rollback <model> <version>`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Frontend returns 502 | backend not ready; check `docker compose logs backend` |
| Postgres not healthy | check volume permissions / `POSTGRES_PASSWORD` file |
| `torch.compile` warnings | expected when no compiler backend exists; eager fallback is automatic |
| Prometheus shows inference down | `inference` service is profile-gated; run with `--profile gpu` |
| Grafana blank dashboards | confirm `quality/monitoring/grafana/*.json` exist (git-tracked) |
| TLS cert errors | ensure `DOMAIN` resolves to the host; check Caddy logs |
