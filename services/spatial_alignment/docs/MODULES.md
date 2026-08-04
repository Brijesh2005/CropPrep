# STAM — Module Documentation

## `stam.py` — STAM facade
*Public API:* `initialize()`, `build_observation()`, `find_nearest()`,
`build_sequence()`, `get_patch()`, `validate()`, `summary()`.

Orchestrates the matcher, sequence builder, patch generator, quality pass and
cache. `build_observation()` is the single entry point from GPS/map clicks to
`AgriculturalObservation`.

## `matcher.py` — matching + DMS adapters
* `SpatialTemporalMatcher` — `find_nearest`, `resolve_admin`, `location_info`,
  `resolve_temporal`, `match_tabular`, `match_images`, `initialize`.
* DMS adapters implementing the STAM ports:
  * `DatasetManagerImageSource` (metadata records),
  * `DatasetManagerImageReader` (windowed reads),
  * `DatasetManagerTabularSource` (CSV record matching, district fallback),
  * `DatasetManagerBoundaryProvider` (boundary GeoDataFrames),
  * `DatasetManagerLocationCatalog` (boundary centroids + image centroids).

## `spatial_index.py` — spatial search
* `KDTreeSpatialIndex` — cKDTree candidate pruning + haversine exact distance,
  radius filtering, metre-tolerance deduplication.
* `BoundaryIndex` — Shapely STRtree point-in-polygon (returns all levels).
* `haversine_km`, `LocationPoint`, `NearestMatch`, `BoundaryHit`.

**Why KDTree over BallTree/RTree** (nearest points): 2-D nearest search is
KDTree's sweet spot — O(log n) queries, tiny memory, no extra C deps. BallTree
suits high-dimensional/non-Euclidean metrics (overkill here). RTree/STRtree is
used precisely where it shines: bounding-box/polygon containment.

## `temporal_index.py` — temporal search
* `SeasonCalendar` — configurable season definitions; crossing-year seasons
  (Rabi Nov→Mar) attributed to the *planting* year.
* `TemporalIndex` — `sort_unique`, `nearest`, `in_range`, `gaps`,
  `within_tolerance`, `dedupe_by_date`.

## `sequence_builder.py` — image sequences
* `ImagePairBuilder` — pairs NDVI+EVI by observation date; validates
  resolution / CRS / bounding box; flags duplicates and missing sides.
* `ObservationSequenceBuilder` — orders pairs by date, detects gaps, produces
  `SequenceBuildResult` (paths + metadata only; pixels stay lazy).

## `patch_generator.py` — raster patches
* `SpatialPatchGenerator` — point → raster-CRS → pixel → centred window →
  edge correction → pad → validity mask. Returns `RasterPatch`.
* No full-raster loads; windowed reads only. Optional reflect/constant padding.

## `coordinate_transform.py` — CRS & pixel math
* `normalise_crs`, `crs_to_epsg`, `validate_crs`, `assert_same_crs`,
  `transform_point`, `transform_points`, `world_to_pixel`,
  `pixel_to_world`, `geographic_to_raster_index`, `patch_window`,
  `window_affine`, `window_bounds`.

## `validators.py` — quality control
* `assess_quality()` — coordinates, distance threshold, tabular match, image
  presence, missing NDVI/EVI sides, temporal gaps, pairing mismatches →
  `QualityReport` (score 0-100, `passed`).

## `observation.py` — typed models
* `AgriculturalObservation` (the output sample), plus `LocationInfo`,
  `AdminLocation`, `TemporalInfo`, `TabularFeatures`, `SequenceInfo`,
  `ImagePairRef`, `ImageRecordRef`, `QualityReport`, `QualityIssue`.

## `cache.py` — caching
* `DatasetManagerStamCache` — `stam:`-namespaced keys on the DMS cache for
  observations, nearest-location results, temporal contexts and indexes.

## `config.py` — configuration
* `StamConfig` + sections (`Patch`, `Spatial`, `Temporal`, `Image`, `Quality`,
  `Cache`, `Admin`, `Tabular`, `Seasons`).
* `load_stam_config` (env > YAML > defaults), `save_stam_config_template`.

## `exceptions.py` / `logger.py` / `interfaces.py`
Typed `ST-*` errors, JSON-capable logging (reuses DMS formatters), and the
hexagonal ports.
