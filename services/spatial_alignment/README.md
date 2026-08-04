# CropFusion Spatial-Temporal Alignment Module (STAM)

The **research contribution** of the CropFusion platform. STAM transforms
**tabular agricultural data + multi-temporal Sentinel-2 vegetation indices**
into **one unified multimodal agricultural observation** per (location, year,
season).

No AI model touches the raw datasets. Every training sample (Phase 4 feature
engineering and the AI module) passes through STAM — and STAM itself reads
data **only** through the Dataset Manager (Phase 2).

---

## What STAM does

```
GPS Location / Map Click
        │
        ▼
Nearest Dataset Location   ──  Spatial Index (KDTree + haversine)
        │
        ▼
Administrative Area        ──  Boundary Index (Shapely STRtree)
        │
        ▼
Season + Year              ──  Season Calendar (Kharif/Rabi/Summer/custom)
        │
        ▼
Tabular Features           ──  Dataset Manager CSV records
        │
        ▼
NDVI + EVI Sequences       ──  ordered, validated NDVI/EVI pairs
        │
        ▼
Agricultural Observation   ──  strongly-typed multimodal sample
```

## Quick start

```python
from services.dataset_manager import DatasetManager
from services.spatial_alignment import STAM

manager = DatasetManager.from_config()
manager.download()            # primary Kaggle dataset
manager.generate_metadata()   # index every file

stam = STAM(manager)          # or STAM.from_config(manager, "stam.yaml")
stam.initialize()             # build spatial + boundary indexes

# GPS / map click → one observation
obs = stam.build_observation(lon=74.87, lat=13.09, year=2020, season="Kharif")
print(obs.crop, obs.yield_value, obs.quality.overall_score)

# Nearest dataset point (for map snapping)
nearest = stam.find_nearest(74.87, 13.09)

# Ordered image sequence (paths + metadata, lazy pixels)
seq = stam.build_sequence(74.87, 13.09, year=2020, season="Kharif")

# Fixed-size raster patch (128x128 etc.)
patch = stam.get_patch(obs.sequence.pairs[0].ndvi.path, 74.87, 13.09, size=128)

# Re-run quality control
report = stam.validate(obs)

# Diagnostics
print(stam.summary())
```

## Configuration

Settings resolve env (`ST_*`) > YAML (`ST_CONFIG_FILE` / `--config`) >
defaults. Key options:

| Setting | Default | Purpose |
|---------|---------|---------|
| `ST_PATCH__SIZE` | `128` | Patch edge (px) — 128/224/256 |
| `ST_SPATIAL__MAX_SEARCH_RADIUS_KM` | `5.0` | Nearest-location radius |
| `ST_SPATIAL__DISTANCE_THRESHOLD_KM` | `5.0` | Low-confidence threshold |
| `ST_TEMPORAL__TOLERANCE_DAYS` | `15` | Date-matching tolerance |
| `ST_IMAGE__RESOLUTION` | `R10m` | Preferred band (R10m/R20m) |
| `ST_IMAGE__REQUIRE_PAIRS` | `true` | Strict NDVI+EVI pairing |
| `ST_QUALITY__MAX_TEMPORAL_GAP_DAYS` | `60` | Max gap between dates |
| `ST_ADMIN__BOUNDARIES` | `[]` | Boundary shapefile/GeoJSON paths |
| `ST_TABULAR__TABLE` | auto | Agricultural record CSV |

Generate a template:

```bash
python -c "from services.spatial_alignment import save_stam_config_template; save_stam_config_template('stam.yaml')"
```

## The output: `AgriculturalObservation`

A strongly-typed pydantic model containing:

* `observation_id` (UUID) + `created_at`
* `location` — lon/lat, distance to dataset point, admin hierarchy
* `temporal` — year, season, planting/harvest window, observation dates
* `tabular` — matched record + feature fields
* `sequence` — ordered NDVI/EVI pairs, paths, CRS, resolution, gaps
* `quality` — `QualityReport` (score 0-100 + per-issue flags)
* `crop` / `yield_value` — training labels sourced from the tabular record
* `provenance` + `to_train_dict()` — ready for Phase 4 feature engineering

## Tests

```bash
cd services/spatial_alignment
pytest
```

## Documentation

* [Architecture](docs/ARCHITECTURE.md)
* [Data flow](docs/DATA_FLOW.md)
* [Sequence diagram](docs/SEQUENCE.md)
* [Modules](docs/MODULES.md)
* [Developer guide](docs/DEVELOPER.md)
