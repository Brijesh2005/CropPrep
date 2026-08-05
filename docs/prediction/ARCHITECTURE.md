# CropFusion Inference-Only Architecture

The inference-only architecture separates the serving runtime from the
training stack. R1.4 lays it out as ports and value objects; a later phase
supplies implementations.

## Layering

```
┌─────────────────────────────────────────────────────────────┐
│ application/config      multi-YAML templates (CF_ env)       │
├─────────────────────────────────────────────────────────────┤
│ application/gis         LocationResolver chain               │
│   reverse_geocoding ──► spatial_resolver ──► historical_context
├─────────────────────────────────────────────────────────────┤
│ application/inference   runtime ports + DTOs                 │
│   services ──► engine ──► loaders · cache · explainability   │
│   versioning · validation                                    │
├─────────────────────────────────────────────────────────────┤
│ application/history     PredictionHistoryStore port          │
├─────────────────────────────────────────────────────────────┤
│ application/inference_package   consumed artifact contract   │
│ application/models              exported weights layout      │
├─────────────────────────────────────────────────────────────┤
│ shared/                config · enums · schemas · versioning │
│                        validation · interfaces · exceptions  │
└─────────────────────────────────────────────────────────────┘
```

Everything in the box depends on `shared/` only. `training/` stays out of the
inference-only process.

## Flow

1. **Request** — `POST /predict {lon, lat}` (farmer mode, no year/season).
2. **GIS resolution** — `LocationResolver.resolve(GeoPoint, day)` composes the
   three resolvers into a `GeoContext` (place, admin hierarchy, season,
   climatology, target year).
3. **Orchestration** — `PredictionService.predict` maps `GeoContext` to
   `PredictionContext`, checks the cache, and calls `InferenceEngine.predict`.
4. **Serving** (future) — the engine runs the loaded `ModelPackage` forward and
   returns a `PredictionResult`.
5. **Persistence** — `PredictionHistoryStore.save` writes to the existing
   `predictions` / `prediction_metadata` / `explanations` tables.
6. **Response** — the result (optionally with an explanation summary) is
   returned and cached.

## Key files

| Concern | File |
| --- | --- |
| DTOs | `application/inference/models.py` |
| Engine port | `application/inference/engine/__init__.py` |
| Loader port | `application/inference/loaders/__init__.py` |
| Service port | `application/inference/services/__init__.py` |
| Cache port | `application/inference/cache/__init__.py` |
| Explainer port | `application/inference/explainability/__init__.py` |
| Version resolver port | `application/inference/versioning/__init__.py` |
| Package validator port | `application/inference/validation/__init__.py` |
| GIS value objects | `application/gis/models.py` |
| GIS chain facade | `application/gis/resolver.py` |
| History store port | `application/history/store.py` |
| Artifact manifest | `application/inference_package/manifest.py` |

## Architecture diagrams

- [r1-4-prediction-architecture](../diagrams/r1-4-prediction-architecture.md)
- [r1-4-inference-flow](../diagrams/r1-4-inference-flow.md)
- [r1-4-model-loading-flow](../diagrams/r1-4-model-loading-flow.md)
- [r1-4-inference-package-flow](../diagrams/r1-4-inference-package-flow.md)
- [r1-4-deployment-flow](../diagrams/r1-4-deployment-flow.md)
