# CropFusion Enterprise Data Layer (Phase 10)

The enterprise data layer under `backend/database/` implements the Phase 10
hardening of the CropFusion backend: a production-grade relational schema,
security, RBAC, audit, registries, analytics, notifications and feedback —
all wired into the Phase 8 FastAPI application without redesigning the
existing Phase 2–9 modules.

## Layout

| Path | Responsibility |
| --- | --- |
| `database/models/` | 25 ORM tables (SQLAlchemy 2, PostgreSQL 17+/PostGIS-ready) |
| `database/repositories/` | Data access layer (`DataRepository` extends Phase 8 `BaseRepository`) |
| `database/services/` | Business services (auth, profile, history, analytics, registry, spatial, notifications, feedback, audit, experiments, config) |
| `database/security/` | Argon2id + policy, account lockout, token/session services |
| `database/policies/` | Permission catalog (23), role matrix (5 roles), enforcer |
| `database/seeds/` | Idempotent bootstrap seeds (roles, users, catalog, boundaries) |
| `database/migrations/` | Alembic migrations (async, settings-driven, PostGIS promotion) |
| `database/api/` | Phase 10 FastAPI routers + schemas |
| `app/dependencies/enterprise.py` | Per-request service factories |

## Stack

- **Python 3.12**, SQLAlchemy 2.0 (async), FastAPI, Pydantic v2.
- **PostgreSQL 17+ with PostGIS** in production (asyncpg); SQLite (aiosqlite)
  for development and tests.
- **Redis** for session cache, spatial cache, analytics cache, notification
  unread counts and rate limiting — with an automatic in-memory fallback.
- **Alembic** for schema migrations; **Argon2id** for password hashing.

## Quick start

```bash
cd backend
python -m alembic upgrade head            # apply migrations
# seed the database (roles, demo users, catalog, ICRISAT districts):
BACKEND_SEED__ON_STARTUP=true python -m uvicorn app.main:app
```

The application seeds automatically at startup when
`BACKEND_SEED__ON_STARTUP=true`. Point `BACKEND_SEED__CSV_PATH` at
`Tabular_Datasets/ICRISAT-District Level Data.csv` for the full 311-district
boundary set (falls back to a small synthetic set otherwise).

See [API.md](API.md) for the enterprise API reference and
[MIGRATIONS.md](MIGRATIONS.md) for the migration/PostGIS story.
