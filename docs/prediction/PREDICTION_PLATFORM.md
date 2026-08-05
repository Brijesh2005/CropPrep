# CropFusion Prediction Platform Guide

The Prediction Platform (`application/`) is the serving half of CropFusion. It
exposes the trained model over an HTTP API and, in the near future, a dedicated
inference-only process.

R1.4 prepares the platform for inference-only operation: the architecture
skeleton, the exported-artifact contract and the config templates. **Nothing is
implemented yet** — no inference, no model loading, no backend API changes.

## What the platform contains

| Path | Role | R1.4 status |
| --- | --- | --- |
| `application/backend/` | FastAPI API + embedded inference engine | unchanged (existing) |
| `application/inference/` | Future inference runtime: engine, loader, cache, explainability, versioning, validation, services | skeleton / ports |
| `application/gis/` | Location resolution: reverse geocoding → spatial → historical context | boundary data (existing) + ports |
| `application/history/` | Prediction-history storage contract | ports (no schema change) |
| `application/inference_package/` | Consumed artifact set (manifest + README) | contract |
| `application/models/` | Exported model weights | layout / naming convention |
| `application/config/` | Multi-YAML templates for the inference process | templates |
| `application/database/` | Enterprise schema + services + API routers | existing (unchanged) |
| `application/docker/` | Container builds, incl. the new standalone inference image | updated |

## Dependency rules

The platform depends on `shared/` only — never on the Training Platform for
utilities:

```
application → shared
training     → shared
shared       → stdlib + third-party only
```

The one remaining `application → training` coupling is the deliberate
execution delegation in `application/backend` (it runs the real model, STAM and
preprocessors). The new inference skeleton (`application/inference`,
`application/gis`, `application/history`, `application/inference_package`,
`application/models`) imports **no** `training` module (grep-verified).

## The farmer-mode interface

The future `POST /predict` accepts **latitude + longitude only**:

```json
{ "lon": 74.90, "lat": 12.85, "include_explanation": false }
```

Season, target year and historical context are auto-resolved from the request
date and the location (see `application/gis`). This matches
`application_mode == "farmer"` already configured in
`application/backend/app/core/config.py`.

## Next steps (future phases)

1. Implement the `application/inference` ports (loader → engine).
2. Wire `application/gis` resolvers to the shapefiles + exported parquet.
3. Bind `application/history.PredictionHistoryStore` to the enterprise schema.
4. Add the inference-only process entrypoint and serve `application/config/`.
5. Swap the backend's execution delegation behind the new ports.

## References

- [ARCHITECTURE.md](ARCHITECTURE.md) — the inference-only architecture
- [INFERENCE.md](INFERENCE.md) — the inference pipeline contract
- [MODEL_LOADING.md](MODEL_LOADING.md) — how exported weights are loaded
- [INFERENCE_PACKAGE.md](INFERENCE_PACKAGE.md) — the consumed artifact set
- [DEPLOYMENT.md](DEPLOYMENT.md) — inference-only Docker deployment
- [MIGRATION_REPORT_R1.4](../migration/MIGRATION_REPORT_R1.4.md) — what R1.4 changed
