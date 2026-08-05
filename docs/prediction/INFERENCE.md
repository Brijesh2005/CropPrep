# CropFusion Inference Guide

Defines the inference pipeline the Prediction Platform will serve, and the
contracts R1.4 fixed for it. R1.4 implements none of it; it pins the seams so a
later phase can build the real engine.

## Pipeline

A prediction starts from a raw coordinate and ends as a stored result:

```
POST /predict  (lon, lat)
        │
        ▼
LocationResolver            application/gis/resolver.py
  ReverseGeocoder           location_index.parquet        → place
  SpatialResolver           District/Taluk shapefiles      → village/taluk/district
  HistoricalContextResolver historical_context.parquet    → season + climatology
        │
        ▼
PredictionService           application/inference/services → orchestrates
  InferenceEngine           application/inference/engine   → model forward (future)
  PredictionCache           application/inference/cache    → memoise per location/day
  PredictionHistoryStore    application/history            → persist result
        │
        ▼
PredictionResult            application/inference/models
```

## Contracts

| Port | Location | Responsibility |
| --- | --- | --- |
| `InferenceEngine` | `inference/engine` | async `predict(request, context) -> PredictionResult`, `status()` |
| `ModelLoader` | `inference/loaders` | load exported artifacts into a `ModelPackage` |
| `ModelVersionResolver` | `inference/versioning` | pick pinned / latest exported model |
| `PredictionCache` | `inference/cache` | keyed (lon, lat, date) result cache |
| `PredictionExplainer` | `inference/explainability` | serving-time explanation summary |
| `InferencePackageValidator` | `inference/validation` | check the exported package manifest |
| `PredictionService` | `inference/services` | orchestration seam the API calls |
| `PredictionHistoryStore` | `history/store` | save/search prediction history |
| `ReverseGeocoder` | `gis/reverse_geocoding` | point → nearest known place |
| `SpatialResolver` | `gis/spatial_resolver` | place → administrative hierarchy |
| `HistoricalContextResolver` | `gis/historical_context` | point+date → season/climatology |
| `LocationResolver` | `gis/resolver` | full GIS chain facade |

## Key decisions

- **Location-only input**: the engine never receives `year`/`season`; the GIS
  layer derives them. DTO: `PredictionRequest` in `inference/models.py`.
- **No embedded algorithms**: the engine consumes exported artifacts (scalers,
  encoder, weights). STAM and preprocessing code stays in the Training Platform.
- **Async boundaries**: every runtime port is `async`; the GIS resolvers are
  sync (pure geometry/table lookups) and are called off the event loop.

## Conventions

- Ports import only stdlib + `shared.*`; never `training.*`.
- DTOs are `@dataclass(slots=True)` value objects with `.to_dict()` (matching
  `shared.schemas`).
- Version types reuse `shared.versioning.ModelVersion`.
- Validation results reuse `shared.validation.ValidationResult`.

## Status / probes

A future engine must expose `status()` returning readiness, the served model
version and device so `/health` / `/ready` can report accurately without
reaching into internals.
