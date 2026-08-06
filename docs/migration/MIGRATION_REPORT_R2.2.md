# Migration Report — R2.1 → R2.2 (Dataset Manager Expansion)

Detailed record of the R2.2 phase: expanding the **Dataset Manager** with a
provider registry, multi-provider configuration, spatial index, patch
extraction, historical context building, extended metadata persistence,
statistics, reports and extended validation. Companion to
[MIGRATION_REPORT_R2.1](MIGRATION_REPORT_R2.1.md).

- **Status**: complete
- **Date**: 2026-08-06
- **Branch**: `main` (not yet committed)

## 1. Scope and principles

- **Dataset Manager only**: R2.2 added the provider layer, spatial /
  temporal / patch capabilities, statistics, reports and validation extensions
  inside `training/dataset_manager`. STAM, training, models, preprocessing and
  explainability were **not modified** — the new `HistoricalContextBuilder`
  explicitly gathers raw context without executing STAM.
- **Registry-first architecture**: providers are registered in a
  `ProviderRegistry` and resolved by name / kind; the manager never constructs
  or holds providers directly (legacy attributes are wired *through* the
  registry to preserve the R1.2 surface).
- **Same `metadata.db`**: the extended repository adds four tables
  (`provider_metadata`, `spatial_metadata`, `temporal_metadata`,
  `patch_metadata`) next to the existing `metadata_records` — no new database.
- **Existing suites preserved**: all R1.x tests keep passing; the new
  behaviour is covered by 8 new R2.2 test modules.
- **No preprocessing**: patch extraction returns raw band windows; the
  historical context builder returns raw per-year observations.
- **Best-effort extensions**: spatial / temporal / provider validation checks
  and index auto-build degrade gracefully when their dependencies (spatial
  index, metadata store, provider registry) are not wired.

## 2. New components — `training/dataset_manager/`

| Module | Component | Responsibility |
| --- | --- | --- |
| `provider_registry.py` | Provider Registry | register/resolve by name & kind, priority ordering, availability/health/capabilities/discovery |
| `spatial_index.py` | Spatial Index | villages/districts by name, KD-tree nearest, radius/bbox/coordinate queries, `build_records_from_frame` |
| `patch_extractor.py` | Patch Extractor | geographic patch extraction: locate raster → CRS conversion → windowed read → edge padding → patch record |
| `historical_context_builder.py` | Historical Context | multi-year per-location observations (tabular + NDVI/EVI records, dates, quality) |
| `metadata_repository.py` | Extended Metadata Repository | provider/spatial/temporal/patch tables in `metadata.db` |
| `statistics.py` | Statistics | aggregate tabular + image statistics into `DatasetStatistics` |
| `reports.py` | Reports | seven JSON report families (inventory/csv/image/provider/spatial/temporal/validation) |

Extended (modified) modules: `validator.py` (temporal/spatial/CRS/duplicate/provider
checks), `manager.py` (registry wiring, `_auto_build_spatial_index`, R2.2 API),
`models.py` (spatial/temporal/patch/historical/statistics dataclasses),
`interfaces.py` (new ports), `config.py` (`providers.registry` settings),
`cli.py` (spatial/location/extract-patch/historical-context/reports/statistics/
providers/health/availability/discovery/search commands), `providers/*`
(`name` override support).

## 3. Provider registry

* Defaults: `git_repository_tabular` (tabular) and `kaggle_hub_image` (image).
* `settings.providers.registry.providers[]` entries can **override** (enable/
  disable/priority), or **add** providers of a supported kind (multi-provider /
  future plugins).
* Resolution honours `enabled` (disabled ⇒ `DatasetNotFoundError`) and sorts by
  `priority` (highest wins); equal priorities fall back to registration order.
* Manager legacy attributes (`tabular_provider`, `image_provider`) are wired
  through the registry bypassing the enabled check, so a disabled provider is
  still constructible.

## 4. New public APIs

| Method | Purpose |
| --- | --- |
| `manager.get_patch(lat, lon, size, ...)` | geographic patch extraction |
| `manager.get_location(name/coords, k/radius/tolerance)` | location resolution |
| `manager.historical_context_builder.build(...)` | multi-year observation context |
| `manager.statistics()` | aggregate dataset statistics |
| `manager.availability() / health() / discovery() / provider_manifests()` | provider introspection |
| `manager.spatial_metadata() / temporal_metadata(...)` | index / availability summaries |
| `generate_reports(manager, report_dir)` | write the seven report families |

## 5. Extended validation checks

| Code | Check |
| --- | --- |
| `V-TEMP-001/002` | duplicate observation records / year-range coverage |
| `V-SPAT-001..003` | latitude/longitude ranges, duplicate location names |
| `V-CRS-001` | mixed coordinate systems across rasters |
| `V-META-004` | metadata-store duplicate records |
| `V-PROV-000/001` | provider registry readability / provider unavailability |

## 6. Documentation

- 5 new guides under `training/dataset_manager/docs/`: PROVIDERS.md,
  SPATIAL.md, EXTRACTION.md, METADATA_REPOSITORY.md, REPORTS.md.
- 6 new diagrams under `docs/diagrams/`: `r2-2-provider-registry.md`,
  `r2-2-spatial-index.md`, `r2-2-patch-extraction.md`,
  `r2-2-historical-context.md`, `r2-2-metadata-repository.md`,
  `r2-2-reports-validation.md`.

## 7. Verification

- 8 new R2.2 test modules covering the registry, spatial index, patch
  extraction, historical context, metadata repository, statistics, reports and
  extended validation.
- `pytest training/dataset_manager/tests` → **242 passed**.
- Full repo `pytest` (application + backend + training + shared) → **1013
  passed, 0 failed**.
- Two pre-existing repo-wide issues fixed to make the full suite runnable:
  - `pytest.ini` now uses `--import-mode=importlib` (seven packages each ship a
    `tests/test_config.py`; default import mode raised "import file mismatch").
  - `shared/tests/test_logging.py` counted all handlers instead of file
    handlers; pytest's caplog attaches `LogCaptureHandler`s to non-propagating
    loggers, so the assertion now targets `logging.FileHandler`.
- R2.2 internal fixes during verification: reserved `name` LogRecord `extra` key
  (3 sites → `provider_name`/`report_name`), `lookup_district` returns only
  district-kind records, `_registered_provider` wiring, provider `name` override
  for config-registered providers, r22 fixture georeferencing (raster origin +
  EPSG:4326) and provider-config merging.

## 8. Migration impact

| Aspect | Impact |
| --- | --- |
| STAM / training / models / preprocessing / explainability | None — not modified |
| `application/*` (Prediction Platform) | None |
| `shared/*` | `test_logging.py` assertion hardened (see §7) |
| Dataset Manager R1.x surface | Preserved — legacy attributes + methods unchanged |
| `metadata.db` | Extended with 4 new tables (idempotent schema) |
| `pytest.ini` | `--import-mode=importlib` added to `addopts` |
