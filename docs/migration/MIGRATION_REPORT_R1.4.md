# Migration Report — R1.3 → R1.4 (Inference-Only Prediction Platform)

Detailed record of the R1.4 phase: preparing the Prediction Platform
(`application/`) to run **inference only**, decoupled from the Training
Platform. Companion to [MIGRATION_REPORT_R1.3](MIGRATION_REPORT_R1.3.md).

- **Status**: complete
- **Date**: 2026-08-05
- **Branch**: `main` (not yet committed)

## 1. Scope and principles

- **Architecture preparation only**: R1.4 built skeletons, ports, contracts,
  config templates, Docker prep and documentation. It did **not** implement
  inference, did **not** load a model, and did **not** modify backend
  endpoints.
- **Training Platform untouched**: no training algorithms, models, STAM,
  loss functions, evaluation or explainability code was modified.
- **Existing backend untouched**: `application/backend` (FastAPI, embedded
  engine, model registry, execution delegation into `training.*`) is unchanged
  and remains the current serving path.
- **No schema changes**: `PredictionHistory` storage reuses the existing
  `predictions` / `prediction_metadata` / `explanations` tables; the
  `application/history` work is a port over them.
- **Dependency rules preserved**: new code depends on `shared/` only —
  grep-verified: zero `import training` in `application/inference`,
  `application/gis`, `application/history`, `application/inference_package`,
  `application/models`.

## 2. New components — `application/inference/`

| Package | Port / DTO | Responsibility |
| --- | --- | --- |
| `inference/models.py` | `PredictionRequest`, `PredictionContext`, `PredictionResult` | Canonical DTOs; `PredictionRequest` is **lon/lat only** |
| `inference/engine` | `InferenceEngine` | async `predict(request, context)` + `status()` |
| `inference/loaders` | `ModelLoader`, `ModelPackage` | load exported artifacts; never `training.*` |
| `inference/services` | `PredictionService` | orchestration seam the API calls |
| `inference/cache` | `PredictionCache` | (lon, lat, date)-keyed result cache |
| `inference/explainability` | `PredictionExplainer` | serving-time explanation summary |
| `inference/versioning` | `ModelVersionResolver` | pinned / latest exported model version |
| `inference/validation` | `InferencePackageValidator` | manifest check via `shared.validation` |

## 3. New components — `application/gis/`

Boundary data already shipped in this directory (`District/`, `Taluk/`,
`Dakshina_Kannada/`, `kml/`). R1.4 added the resolution architecture:

| Module | Port | Responsibility |
| --- | --- | --- |
| `gis/models.py` | `GeoPoint`, `ResolvedPlace`, `AdminContext`, `HistoricalContext`, `GeoContext` | GIS value objects |
| `gis/reverse_geocoding` | `ReverseGeocoder` | point → nearest known place |
| `gis/spatial_resolver` | `SpatialResolver` | place → village/taluk/district |
| `gis/historical_context` | `HistoricalContextResolver` | point+date → season/climatology |
| `gis/resolver.py` | `LocationResolver` | chain facade: Location → Reverse Geocoding → Spatial Resolver → Historical Context → GeoContext |

## 4. New components — `application/history/`

`PredictionHistoryStore` port (`history/store.py`) + DTOs (`HistoryFilters`,
`HistoryRecord`, `HistoryPage`). Storage targets the **existing** enterprise
schema; no schema change.

## 5. New components — `application/inference_package/` and `application/models/`

- `inference_package/manifest.py` — single source of truth for the expected
  artifact set (metadata.db, historical_context.parquet, location_index.parquet,
  feature_scalers.pkl, label_encoder.pkl, model_config.yaml,
  dataset_version.json, model_version.json, metrics.json, README.md).
- `inference_package/README.md` + `.gitignore` — the package is **consumed,
  never generated**.
- `models/README.md` — expected `cropfusion.pt`; future `cropfusion_v1/v2/
  latest.pt`. No weights committed; no loading in R1.4.

## 6. New components — `application/config/`

Multi-YAML templates for the future inference process:

- `application.yaml`, `model.yaml`, `inference.yaml`, `logging.yaml`,
  `security.yaml`, `database.yaml`
- Resolution convention: `CF_<SECTION>__<KEY>` env > YAML > defaults
  (reuses `shared.constants.ENV_PREFIX_SHARED = "CF_"` and
  `shared.config.load_yaml_config`).

## 7. New component — Docker

- `application/docker/Dockerfile.inference.standalone` — inference-only image
  that ships `shared/` + the inference skeleton but **no** `training/`.
  R1.4 entrypoint imports every inference-layer package (verifies the image is
  self-contained). `Dockerfile.backend` and `Dockerfile.inference` are
  unchanged.

## 8. Documentation

6 new guides in `docs/prediction/`:

- `PREDICTION_PLATFORM.md`, `INFERENCE.md`, `DEPLOYMENT.md`,
  `ARCHITECTURE.md`, `MODEL_LOADING.md`, `INFERENCE_PACKAGE.md`

5 new diagrams in `docs/diagrams/`:

- `r1-4-prediction-architecture.md`, `r1-4-inference-flow.md`,
  `r1-4-model-loading-flow.md`, `r1-4-inference-package-flow.md`,
  `r1-4-deployment-flow.md`

`docs/README.md` updated with a Prediction Platform section.

## 9. Folder / file changes

| Change | Location |
| --- | --- |
| Added package (7 subpackages + DTOs) | `application/inference/` |
| Added manifest + README + .gitignore | `application/inference_package/` |
| Added README + .gitignore + `__init__.py` | `application/models/` |
| Added 6 YAML templates | `application/config/` |
| Added 4 modules + 3 subpackages + README | `application/gis/` |
| Added store port + DTOs | `application/history/` |
| Added standalone inference Dockerfile | `application/docker/` |
| Added 6 docs | `docs/prediction/` |
| Added 5 diagrams | `docs/diagrams/` |
| Added migration report | `docs/migration/MIGRATION_REPORT_R1.4.md` |

## 10. Import changes

- **New packages** import only stdlib + `shared.*`. Grep-verified zero
  `import/from training` in `inference`, `gis`, `history`,
  `inference_package`, `models`.
- Internal `application` imports used deliberately:
  `history → inference` (result DTO), `gis → shared.enums`.
- `application/backend` was **not** changed; its `training.*` execution
  delegation (`model_registry.py`, `modules/inference/service.py`,
  `core/app_container.py`) remains and is documented as the future work to
  re-point behind the new ports.

## 11. Verification

| Check | Result |
| ----- | ------ |
| Import of all new packages (`application.inference.*`, `inference_package`, `gis.*`, `history`, `models`) | OK (`0.1.0`, 10 manifest artifacts) |
| Grep: `import/from training` in new packages | none |
| `pytest shared/tests` | **101 passed** (re-run) |
| `pytest application/backend/app/tests` | **80 passed** (re-run) |

## 12. Future work

- Implement `ModelLoader` / `InferenceEngine` / `PredictionService` and wire
  them to the exported artifacts.
- Implement the GIS resolvers against the shapefiles + exported parquet.
- Bind `PredictionHistoryStore` to the existing repositories.
- Add the inference-only process entrypoint and load `application/config/*`.
- Re-point `application/backend`'s execution delegation behind the new ports.

## 13. Known limitations

- R1.4 is contract-only: the inference path does not yet produce predictions.
- The inference package and model weights are not yet produced by the training
  export pipeline, so they cannot be loaded.
- The standalone Docker image's entrypoint is a placeholder (verification
  only) until the serving process is implemented.
- `application/backend/app/core/paths.py` still injects `training` into
  `sys.path` for the current embedded engine (unchanged, out of scope).

## 14. Rollback / safety

All R1.4 changes are additive: new packages, new config templates, new docs,
a new Dockerfile, and one new test. Removing the new directories restores the
R1.3 state exactly; nothing existing was modified, so all prior test suites
remain the safety net.
