# Migrations (Phase 10)

Schema changes are managed with **Alembic** under `backend/database/migrations/`.

## Files

- `backend/alembic.ini` — Alembic configuration (script location,
  `sqlalchemy.url` deliberately empty; the environment resolves it).
- `database/migrations/env.py` — async environment. Loads the URL from the
  app's `DatabaseSettings` (env var `BACKEND_DATABASE__URL`, YAML config file,
  or defaults), and imports both the Phase 2–9 models (`app.models`) and the
  Phase 10 models (`database.models`) so autogenerate sees the full metadata.
- `database/migrations/script.py.mako` — revision template.
- `database/migrations/versions/54a4a02d525f_initial_enterprise_schema.py` —
  the initial enterprise schema (25 tables + indexes/constraints).

## Usage

```bash
cd backend
python -m alembic upgrade head       # apply all migrations
python -m alembic downgrade base     # roll everything back
python -m alembic check              # assert the model matches the schema (no drift)
python -m alembic revision --autogenerate -m "describe change"
```

## PostGIS handling

The ORM keeps geometry columns (`administrative_boundaries.centroid/geometry`,
`spatial_locations.point`, `user_locations.point`) as portable binary so the
models run identically on SQLite (dev/tests). The initial migration detects
PostgreSQL and promotes them:

- **upgrade**: `CREATE EXTENSION IF NOT EXISTS postgis`; `ALTER COLUMN ...
  TYPE geometry(Point|MultiPolygon, 4326) USING ST_SetSRID(ST_GeomFromWKB(...),
  4326)`; GIST index per column.
- **downgrade**: drops the GIST indexes and converts columns back to `bytea`
  via `ST_AsEWKB(...)`.

On SQLite (or any non-PostgreSQL dialect) the geometry hooks are skipped and
the columns remain binary; the spatial repository uses a haversine + bounding
box fallback there, while the PostgreSQL path uses native PostGIS queries.

## Verification

- `python -m alembic upgrade head && python -m alembic downgrade base && python -m alembic upgrade head`
  runs cleanly against SQLite.
- `python -m alembic check` reports `No new upgrade operations detected.`
