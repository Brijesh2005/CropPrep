# CropFusion — Phase 3 Completion Report

**Phase:** Spatial-Temporal Alignment Module (STAM)
**Status:** ✅ Complete
**Date:** 2026-08-02
**Tests:** 91 passed · (Phase 2 regression: 127 passed)

---

## ✔ Files Created

```
services/spatial_alignment/
├── __init__.py             # public API
├── stam.py                 # STAM facade (public API)
├── matcher.py              # matching + Dataset Manager adapters
├── spatial_index.py        # KDTree + Shapely STRtree boundary index
├── temporal_index.py       # SeasonCalendar + TemporalIndex
├── sequence_builder.py     # ImagePairBuilder + ObservationSequenceBuilder
├── patch_generator.py      # SpatialPatchGenerator + RasterPatch
├── coordinate_transform.py # CRS / pixel↔world math
├── validators.py           # quality control → QualityReport
├── cache.py                # DMS-backed observation/index cache
├── observation.py          # strongly-typed pydantic models
├── interfaces.py           # ports
├── config.py               # StamConfig (+ template generator)
├── exceptions.py           # ST-* errors
├── logger.py               # structured logging
├── pyproject.toml
├── README.md
├── docs/  (ARCHITECTURE, DATA_FLOW, SEQUENCE, MODULES, DEVELOPER)
└── tests/  (14 test modules)

Plus a Phase 3 extension to the Dataset Manager:
services/dataset_manager/manager.py   # + load_geometries() + admin_dir
services/dataset_manager/config.py    # + admin_dir field
```

## ✔ Classes

| Class | Responsibility |
|-------|----------------|
| `STAM` | Facade: initialize / build_observation / find_nearest / build_sequence / get_patch / validate / summary |
| `SpatialTemporalMatcher` | spatial + admin + temporal + tabular + image matching |
| `KDTreeSpatialIndex` | nearest dataset point (cKDTree + haversine, radius, dedupe) |
| `BoundaryIndex` | point-in-polygon admin resolution (STRtree) |
| `SeasonCalendar` / `TemporalIndex` | season windows (crossing-year aware), date/gap helpers |
| `ImagePairBuilder` / `ObservationSequenceBuilder` | ordered NDVI/EVI pairing + sequence assembly |
| `SpatialPatchGenerator` / `RasterPatch` | fixed-size patches, edge correction, padding, masks |
| `assess_quality` | QualityReport (score 0-100) |
| `DatasetManagerStamCache` | stam:-namespaced caching |
| `AgriculturalObservation` (+ nested models) | the typed output sample |

## ✔ Public API

```
STAM(manager, config).initialize()
STAM.build_observation(lon, lat, year=None, season=None, month=None, reference_date=None, resolution=None)
STAM.find_nearest(lon, lat, max_radius_km=None)          → dict
STAM.build_sequence(lon, lat, year, season, ...)          → SequenceBuildResult
STAM.get_patch(image_path, lon, lat, size=None, ...)      → RasterPatch
STAM.validate(observation)                                → QualityReport
STAM.summary()                                            → dict
STAM.from_config(manager, config_path)
```

## ✔ Data-access rule

STAM reads **only** through the Dataset Manager: image metadata
(`query_metadata`), rasters (`load_image(window=...)`), tabular
(`list_csvs`/`load_csv`), boundaries (`load_geometries` — a small Phase 3
extension to the DMS). No filesystem scans.

## ✔ Performance

* Spatial index: KDTree O(log n) nearest queries; haversine for exact km.
* Boundary lookup: Shapely STRtree (R-tree) containment.
* **Lazy pixels** — sequences hold paths+metadata; arrays only via
  `get_patch` (windowed reads, no full-raster loads).
* Parallel sequence building; metadata header reads only.
* Caching of observations, nearest results, temporal contexts and indexes.

## ✔ Integration points

* **Phase 4 (preprocessing)** consumes `AgriculturalObservation` directly;
  `STAM.get_patch` is the patch extractor.
* **GIS/backend** can reuse `find_nearest` + admin resolution for map snapping.
* `to_train_dict()` gives a compact representation for research scripts.

## ✔ Known limitations

* Lightweight TIFF parser resolves CRS only via rasterio; without GDAL, CRS is
  None (documented).
* Synthetic test rasters use a degree transform in EPSG:4326; real Sentinel-2
  products (UTM) exercise the pyproj point-projection path.
* Image pairing is strict on bounding-box equality — real scenes with
  slightly different footprints would need a tolerance (config extension).

## ✔ Future improvements

* Configurable bbox tolerance for pairing.
* Real Sentinel-2 integration test (needs the Kaggle download).
* Multi-resolution fusion (R10m + R20m) within one observation.

---

**Awaiting:** `"Proceed to Phase 5"` (Phase 4 preprocessing is delivered below).
