# API

The CropFusion backend is a modular FastAPI monolith. Interactive docs are
served at `/docs` (Swagger UI) and the OpenAPI schema at `/openapi.json`.

## Conventions

- All business endpoints are under the `/api/v1` prefix.
- Health endpoints live at the root: `/health`, `/live`, `/ready`.
- Prometheus metrics: `GET /metrics` (when `BACKEND_MONITORING__PROMETHEUS_ENABLED=true`).
- Requests are JSON; errors follow `{"detail": {...}}` shape.
- Authentication uses bearer JWTs (`Authorization: Bearer <token>`).

## Module map

| Area | Base path | Purpose |
|---|---|---|
| Auth | `/api/v1/auth` | register, login, refresh, password reset, sessions, email verification |
| Users | `/api/v1/users` | profile, preferences, saved locations |
| Predictions | `/api/v1/predict` | crop + yield prediction by coordinates, map overlay |
| History | `/api/v1/predictions` | prediction history, search, filters |
| Explain | `/api/v1/explain` | feature attribution + plain-language summaries |
| GIS | `/api/v1/gis` | location index, spatial lookups |
| Dataset | `/api/v1/datasets` | dataset management API |
| Monitoring | `/api/v1/monitoring` | metrics snapshots, drift/fairness views |
| Admin | `/api/v1/admin` | dashboard, audit log, enterprise admin |
| Registry | `/api/v1/registry` | model + dataset registry (register, activate, validate) |
| Catalog | `/api/v1/catalog` | crops and seasons reference data |
| Spatial | `/api/v1/spatial` | boundary resolution, locations, nearest queries |
| Notifications | `/api/v1/notifications` | user notifications |
| Feedback | `/api/v1/feedback` | user feedback submission + admin resolution |
| Config | `/api/v1/config` | runtime configuration |

## Auth flow

1. `POST /api/v1/auth/register` `{email, password, full_name}`
2. `POST /api/v1/auth/login` (form `username`/`password`) → `access_token`, `refresh_token`
3. Send `Authorization: Bearer <access_token>` on protected endpoints.

## Example: prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"lon": 74.801, "lat": 13.099, "year": 2020, "season": "Kharif"}'
```

Response (abridged):

```json
{
  "recommended_crop": "Paddy",
  "crop_probs": {"Paddy": 0.81, "Wheat": 0.14},
  "expected_yield": 6.12,
  "confidence": 0.81,
  "model_version": "yieldnet-1.0",
  "inference_time_ms": 12.5,
  "fallback": false
}
```

## Model registry

- `GET /api/v1/registry/models` — list registered models
- `POST /api/v1/registry/models` — register a model
- `POST /api/v1/registry/models/activate?name=...&version=...` — activate
- `GET/POST /api/v1/registry/datasets` — dataset registry

## Versioning

API responses include the backend version; the overall project version is in
`VERSION` at the repository root.
