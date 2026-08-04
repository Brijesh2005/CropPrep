# Troubleshooting

## Backend fails to start

- Check settings: the backend requires a model checkpoint or model config
  (`BACKEND_MODEL__CHECKPOINT_PATH` / `BACKEND_MODEL__MODEL_CONFIG_PATH`) unless
  `warmup=false` and the fallback path is acceptable.
- `uvicorn app.main:app` must run from the `backend/` directory.
- Verify `services.dataset_manager` and `ai.models` are installed (see
  [INSTALLATION.md](INSTALLATION.md)).

## Frontend can't reach the API

- In Docker, the frontend proxies `/api` to `backend:8000` via nginx; ensure
  the backend service is healthy.
- Locally, rebuild with the correct `VITE_API_BASE_URL` (e.g. `http://localhost:8000`).
- Browser console shows CORS issues: set the allowed origins in backend config.

## Postgres / PostGIS issues

- `docker compose logs postgres` - check credentials match `.env`.
- The PostGIS image needs time to initialize on first run; wait for the
  healthcheck (`pg_isready`).
- Migration errors: run `alembic upgrade head` inside the backend container.

## `torch.compile` warnings or slow startup

- CropFusion requires a C compiler for torch inductor. Without one, the
  `OptimizedRuntime` automatically falls back to eager execution - this is
  expected and safe (see `quality/optimization/runtime.py`).

## Prometheus shows the inference target down

- The `inference` service only runs with `docker compose --profile gpu up -d`.
  This is expected in default dev mode.

## Grafana dashboards are empty

- Dashboards are provisioned from `quality/monitoring/grafana/`; confirm the
  files exist and the datasource provisioning mounted correctly
  (`deployment/monitoring/grafana/provisioning/`).

## TLS certificate errors in production

- Ensure `DOMAIN` in `.env` resolves to the host and ports 80/443 are open.
- Check Caddy logs: `docker compose -f docker-compose.prod.yml logs caddy`.

## Test failures due to imports

- Run pytest from the package directory (each package sets `pythonpath`) or
  from the repo root (root `pytest.ini` adds the root to `sys.path`).

## Model promotion blocked

- The promotion gates fail closed. Inspect the gate report:
  `cat reports/releases/<model>-<version>-*.json` and address the failing gate.

## More help

- Open an issue using the templates in `.github/ISSUE_TEMPLATE/`.
- Security problems: see [SECURITY.md](../SECURITY.md).
