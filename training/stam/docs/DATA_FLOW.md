# STAM — Data Flow

## Observation construction (`build_observation`)

```
Query (lon, lat, year?, season?, month?, date?)
   │
   ▼
① Coordinates validated (range check)          [ST-Q-COORD]
   │
   ▼
② Observation cache lookup  ── HIT ──► return cached observation
   │ (MISS)
   ▼
③ Spatial match: KDTree nearest dataset point  [ST-SPATIAL]
   │  + haversine distance, radius threshold
   ▼
④ Admin resolution: STRtree point-in-polygon    [ST-ADMIN]
   │  → village / taluk / district / state
   ▼
⑤ Temporal resolve: year + season window        [ST-TEMPORAL]
   │  (defaults / inference from reference date)
   ▼
⑥ Tabular match (via Dataset Manager)          [ST-TABULAR]
   │  village → district fallback → year/season subset
   ▼
⑦ Image match: NDVI + EVI records              [ST-IMAGE]
   │  filtered to season window + resolution
   ▼
⑧ Sequence build: pair by date → validate → sort → gaps
   │  (CRS/resolution/bbox checks, duplicates, missing sides)
   ▼
⑨ Quality control → QualityReport (score 0-100)
   │
   ▼
⑩ Assemble AgriculturalObservation  → cache store → return
```

## Patch extraction (`get_patch`)

```
(lon, lat) + image path
   │
   ▼
Image metadata (CRS, pixel_size, bounds, width/height)   ← Dataset Manager
   │
   ▼
Point → raster CRS (pyproj) → pixel (row, col)  (affine inverse)
   │
   ▼
Centred window (size × size) → clamp to raster extent (edge correction)
   │
   ▼
Windowed read (never full raster) → pad to size + validity mask
   │
   ▼
RasterPatch (array, mask, bounds, CRS, resolution)
```

## Data provenance

Every observation records:

* Dataset Manager metadata count used,
* NDVI/EVI record counts, paired count, duplicate dates,
* STAM configuration snapshot,
* whether it was served from cache.

This makes every training sample fully reproducible (Phase 4 + the AI module
can trace exactly which files and metadata produced it).
