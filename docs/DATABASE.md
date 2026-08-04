# Database

CropFusion uses **PostgreSQL with PostGIS** in production and **SQLite
(in-memory / file)** for tests. Schema management uses Alembic.

## Migrations

```bash
cd backend
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```

Migration files: `backend/database/migrations/versions/`. The baseline
migration creates the full enterprise schema (`initial_enterprise_schema.py`).

## Core tables

| Table | Purpose |
|---|---|
| `users` | accounts, password hashes, roles |
| `user_sessions` | active sessions per user |
| `refresh_tokens` | JWT refresh-token store |
| `user_locations` | saved farmer locations |
| `predictions` | prediction records + inputs |
| `explanation_records` | per-prediction explanations |
| `model_versions` | ML model registry entries |
| `dataset_versions` | dataset registry entries |
| `notifications` | user notifications |
| `audit_logs` | administrative audit trail |
| `feedback` | user feedback + resolution |
| `spatial_boundaries` | district/village geometry (PostGIS) |

## PostGIS

Spatial support is enabled with `BACKEND_DATABASE__POSTGIS=true` and the
`postgis/postgis` image in compose. The GIS module stores locations with
geometry and performs radius and nearest-neighbour lookups.

## Connection pooling

For PostgreSQL the backend uses async SQLAlchemy with a tuned pool:

| Setting | Default |
|---|---|
| `pool_size` | 10 |
| `max_overflow` | 20 |
| `pool_recycle_seconds` | 1800 |

## Backups

Database dumps are produced by `scripts/backup/backup-db.sh`; restore with
`scripts/backup/restore-db.sh`. See [DEPLOYMENT.md](DEPLOYMENT.md#backups).
