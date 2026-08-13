# Backend Core — Release-Package Inference (drop-in files)

## What this is

R5's inference stack was **fully wired but to the wrong thing**: `/predict`
already worked end-to-end, but through `training.models.ModelFactory`, STAM
and the live Dataset Manager — exactly what the R6 spec forbids ("must NEVER
... load GeoTIFFs, train models ... only loads `cropfusion_release/`").

These files replace that wiring with a loader that reads **only**
`cropfusion_release/`, plus fill one real gap (`app/models/prediction.py`
was imported everywhere but didn't exist in the R5 zip).

## Copy these files into your project (same relative paths)

**New files —  just add them:**
```
application/inference_package/release/__init__.py
application/inference_package/release/manifest.py
application/inference_package/release/loader.py
application/backend/app/services/release_model_registry.py
application/backend/app/modules/inference/location_resolver.py
application/backend/app/modules/inference/feature_builder.py
application/backend/app/modules/model_info/__init__.py
application/backend/app/modules/model_info/router.py
application/backend/app/models/__init__.py
application/backend/app/models/prediction.py
```

**Replace these existing files entirely (same path, overwrite):**
```
application/backend/app/modules/inference/service.py
application/backend/app/modules/inference/schemas.py
application/backend/app/modules/predictions/service.py
application/backend/app/modules/predictions/schemas.py
application/backend/app/modules/predictions/dependencies.py
application/backend/app/core/app_container.py
application/backend/app/api/router.py
```

No manual edits needed — `api/router.py` already has the `/model` route wired in.

## Configuration

Point `model.checkpoint_path` at your `cropfusion_release/` directory (the
field is reused rather than adding a new setting, so `config/model.yaml` /
`BACKEND_MODEL__CHECKPOINT_PATH` don't need schema changes):

```yaml
model:
  checkpoint_path: "/srv/cropfusion_release"   # the release dir, not a .pt file
  device: "auto"
```

## Release package your Training Platform export step must produce

```
cropfusion_release/
  model/cropfusion.pt          # TorchScript preferred (torch.jit.save) —
                                # fully decouples inference from training code.
                                # A raw state_dict also works; the loader then
                                # imports training.models.ModelFactory just to
                                # rebuild the architecture shape.
  metadata/metadata.db
  metadata/historical_context.parquet   # columns: village, district, season,
                                         # year, + whatever configs/model.yaml's
                                         # feature_order needs
  metadata/location_index.parquet       # columns: village, district, taluk, lon, lat
  metadata/village_metadata.parquet     # columns: village, district, + static features
  preprocess/scaler.pkl          # sklearn-style .transform()
  preprocess/label_encoder.pkl   # sklearn-style .classes_
  configs/model.yaml              # must include `feature_order: [...]`,
                                   # and `input_dim` for warmup
  configs/inference.yaml
  version/manifest.json           # {"format": "cropfusion_release",
                                   #  "schema_version": 1,
                                   #  "model_version": "...", "dataset_version": "..."}
  version/checksum.json           # {"files": {"model/cropfusion.pt": "<sha256>", ...}}
  reports/metrics.json
```

`ReleasePackageLoader` validates file presence → manifest format/schema →
per-file sha256 → then loads everything, in that order, and raises
`ReleasePackageError` with a specific reason on any mismatch.

## What's intentionally NOT in this batch

- **GIS map / reverse geocoding UI** — `app_container.py` now sources GIS
  markers from `location_index.parquet` (cheap, works), but the interactive
  map frontend, village search, and shapefile-based boundaries are a
  separate chunk.
- **Auth, History endpoints, Docker, Monitoring dashboards, React frontend**
  — pick these up as separate chunks so each stays reviewable; the existing
  `/predict` routing (`predictions/router.py`, `predictions/dependencies.py`)
  already calls the new service, so nothing else in the auth/history modules
  needs to change for this chunk to work.
- **Full Explainability integration** — `PredictionService._explain()` is a
  placeholder (ranks features by scaled magnitude). Wiring the real
  Explainability module against release-package artifacts (without importing
  training-time explainer internals) is worth its own chunk.

Say the word and I'll do GIS/frontend, auth+history+DB, or docker/monitoring next.
