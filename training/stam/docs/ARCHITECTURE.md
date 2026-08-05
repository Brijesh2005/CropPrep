# STAM — Architecture

## Overview

STAM is a hexagonal module inside `services/spatial_alignment`. The
`STAM` facade orchestrates ports defined in `interfaces.py`; the production
adapters wrap the **Dataset Manager** — the only sanctioned data access path.

```
                  ┌────────────────────────────────────────────┐
                  │               STAM (facade)               │
                  │  initialize · build_observation ·          │
                  │  find_nearest · build_sequence ·          │
                  │  get_patch · validate · summary            │
                  └───────┬─────────────────────────┬─────────┘
                          │                         │
                matcher.py│ (orchestrator)          │ patch_generator.py
                  ┌───────▼──────────┐       ┌──────▼─────────┐
                  │ SpatialTemporal  │       │ SpatialPatch   │
                  │ Matcher          │       │ Generator      │
                  └──┬──────┬─────┬──┘       └──────┬─────────┘
                     │      │     │                 │
   ┌─────────────────┘      │     └────────────┐    │
   ▼                        ▼                  ▼    ▼
┌─────────────┐   ┌──────────────────┐  ┌────────────────┐  ┌────────────────┐
│spatial_index│   │ temporal_index   │  │sequence_builder│  │coordinate_      │
│KDTree +     │   │ SeasonCalendar + │  │ImagePairBuilder│  │transform        │
│BoundaryIndex│   │ TemporalIndex    │  │ObservationSeq  │  │(CRS/pixel math) │
└─────────────┘   └──────────────────┘  └────────────────┘  └────────────────┘
        │                    │                   │                  │
        └────────────────────┴─────────┬─────────┴──────────────────┘
                                       ▼
                        ┌──────────────────────────┐
                        │   Dataset Manager (DMS)  │  ← THE ONLY data path
                        │  metadata · CSV · rasters│
                        └──────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
              metadata.db         tabular CSVs        GeoTIFFs (lazy)
```

## Design rules

1. **Dataset Manager only** — STAM never scans the filesystem; every read
   goes through `DatasetManager` (metadata queries, `load_csv`,
   `load_image(window=...)`, `load_geometries`).
2. **Ports & adapters** — `interfaces.py` declares
   `ImageMetadataSource`, `ImageReader`, `TabularSource`, `BoundaryProvider`,
   `LocationCatalog`, `StamCache`; the concrete adapters in `matcher.py` /
   `cache.py` wrap the DMS.
3. **Lazy pixels** — sequences store *references* (paths + metadata). Arrays
   are produced on demand via `get_patch` (windowed reads; never full loads).
4. **No duplicated raster loading** — a raster's header is read once into the
   metadata store; patches read only their window.
5. **Deterministic output** — observations are fully typed pydantic models,
   serialisable for caching and Phase 4 consumption.

## Module responsibilities

| Module | Responsibility |
|--------|----------------|
| `stam.py` | Facade + public API + orchestration + observation cache |
| `matcher.py` | Spatial/temporal/tabular matching + DMS adapters |
| `spatial_index.py` | KDTree nearest-point + Shapely STRtree boundary containment |
| `temporal_index.py` | Season calendar (crossing-year aware) + date helpers |
| `sequence_builder.py` | Ordered NDVI/EVI time series + image pair validation |
| `patch_generator.py` | Fixed-size patches, edge correction, padding, masks |
| `coordinate_transform.py` | CRS normalisation, projection, pixel↔world math |
| `validators.py` | Quality control → `QualityReport` |
| `cache.py` | Observation/index/nearest caching on the DMS cache |
| `observation.py` | Strongly-typed models (`AgriculturalObservation`, ...) |
| `config.py` / `exceptions.py` / `logger.py` / `interfaces.py` | Cross-cutting |
