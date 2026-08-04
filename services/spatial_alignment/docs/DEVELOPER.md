# STAM — Developer Guide

## Layout

```
services/spatial_alignment/
├── __init__.py          # public API
├── stam.py              # facade
├── matcher.py           # matching + DMS adapters
├── spatial_index.py     # KDTree + boundary index
├── temporal_index.py    # season calendar + date helpers
├── sequence_builder.py  # NDVI/EVI sequence + pairing
├── patch_generator.py   # fixed-size patches
├── coordinate_transform.py
├── validators.py        # quality control
├── cache.py             # DMS-backed cache
├── observation.py       # typed models
├── interfaces.py        # ports
├── config.py            # StamConfig
├── exceptions.py        # ST-* errors
├── logger.py
├── README.md / docs/
└── tests/
```

## Conventions

| Thing | Convention |
|-------|------------|
| Modules | `snake_case.py` |
| Classes | `PascalCase` |
| Public API verbs | `initialize / build_* / find_* / get_* / validate / summary` |
| Error codes | `ST-<AREA>-<NNN>` (quality codes `ST-Q-*`) |
| Cache keys | `stam:<thing>:<id>` |
| Env vars | `ST_<SECTION>__<FIELD>` |

## Data-access rule

**STAM must never read the filesystem directly.** Every data read goes through
the Dataset Manager:

* image metadata → `DatasetManager.query_metadata / get_metadata`
* raster pixels → `DatasetManager.load_image(window=...)`
* tabular → `DatasetManager.list_csvs / load_csv / preview_csv`
* boundaries → `DatasetManager.load_geometries`

## Adding a capability

1. Add the port to `interfaces.py` if new.
2. Implement the adapter (DMS-backed) in `matcher.py` (or a new module).
3. Expose it on `SpatialTemporalMatcher` and/or the `STAM` facade.
4. Add tests under `tests/` using the synthetic dataset fixtures.
5. Document in `docs/` and `README.md`.

## Testing

```bash
cd services/spatial_alignment
pytest
```

The suite uses a **real Dataset Manager** pointed at a synthetic dataset
(tabular CSV + NDVI/EVI GeoTIFFs + boundary GeoJSON) so the full
DMS → STAM integration is covered — no mocks for the data path, no network.

## Extending seasons

Add a definition to `StamConfig.seasons` (or the YAML):

```yaml
seasons:
  - name: Kharif
    start_month: 6
    end_month: 10
  - name: Rabi
    start_month: 11
    end_month: 3
  - name: Summer
    start_month: 4
    end_month: 5
```
