# Migration Report — R2.2 → R2.3 (Training Sample Generation & Dataset Export)

Detailed record of the R2.3 phase: building the **training-sample generation
layer** — an observation-corpus generator over STAM, per-modality feature
builders, corpus statistics + class balancing, quality-control reports, and
portable dataset export (JSON / Parquet / torch). Companion to
[MIGRATION_REPORT_R2.2](MIGRATION_REPORT_R2.2.md).

- **Status**: complete
- **Date**: 2026-08-06
- **Branch**: `main` (not yet committed)

## 1. Scope and principles

- **Generation + export only**: R2.3 builds the data layer R3 training will
  consume. It does **not** implement the model, training loop, losses,
  evaluation or prediction. STAM, Dataset Manager, preprocessing, models and
  explainability were reused, **not modified** for training behaviour.
- **Reuse over re-implementation**: the corpus generator, feature builders,
  statistics and exporters are built on the existing `AgriculturalObservation`,
  `SeasonResolver` (via STAM), `CropFusionDataset`/`split_observations`
  (via `build_cropfusion_datasets`) and Dataset Manager catalogue APIs.
- **Self-describing artifacts**: every exported row carries `sample_id`,
  `quality_score`, `year`, `season` and `location_id`, so downstream consumers
  never need the corpus object or the Dataset Manager.
- **Keep failures**: the resolver keeps rejected **and** error cells in the
  corpus (default) so QC can separate "no data" (rejected) from "generation
  crashed" (error).
- **Existing suites preserved**: no existing test was modified; R2.2 gates
  still pass (see §8).
- **Label safety**: raw tabular columns that are training labels
  (`label_columns`) are excluded from generic feature fields, so features never
  leak the resolved labels.

## 2. New components

| Package | Module | Responsibility |
| --- | --- | --- |
| `training/stam/` | `observation_resolver.py` | grid plan over locations × years × seasons → per-cell STAM resolve → `ObservationCorpus` (accepted / rejected / error) with JSON cache |
| `training/feature_engineering/` | new package | `config.py`, `tabular.py`, `image.py`, `temporal.py`, `builder.py` (registry + feature frame), `utils.py`, `statistics.py` (`CorpusStatistics`), `balancing.py` (`BalancingReport`), `dataset.py` (preprocessing bridge), `exceptions.py`, `logger.py` |
| `training/quality/samples/` | new sub-package | `report.py` (`SampleQualityReport`), `build_report`, `exceptions.py`, `logger.py` — corpus QC JSON report |
| `training/export/` | new package | `config.py` (`ExportConfig`), `records.py` (normalisation + meta attach), `exporters.py` (`JsonExporter` / `ParquetExporter` / `TorchExporter`), `builder.py` (`export_dataset` + manifest), `exceptions.py`, `logger.py` |

Extended (modified) STAM modules: `exceptions.py` (added `SampleResolutionError`
`ST-RESOLVE-001`, `SampleCellError` `ST-RESOLVE-002`), `__init__.py` (export the
resolver surface).

## 3. Observation resolver (corpus generation)

- `ObservationResolverConfig`: bbox / `max_locations` / `seasons` /
  `min_quality_score` / `include_rejected` / `include_errors` / `use_cache`.
- `plan()` builds `SamplingCell` × `ObservationPlan`; with `infer_years` it
  intersects the tabular table years with the image catalog years so the grid
  only contains datable cells.
- `resolve()` isolates per-cell failures: exceptions become
  `status="error"` with `error={"code", "message"}`; the rest of the corpus
  continues.
- `ObservationCorpus` exposes `accepted()`, `rejected()`, `errors()`,
  `status_counts()` and JSON `save()`/`load()` for resumable large runs.

## 4. Feature engineering

- `TabularFeatureBuilder` (tab.*): location features, resolved crop/yield
  labels, generic fields — **labels excluded from fields** via
  `label_columns = ["crop", "yield", "yield_value", "yield_kg"]`.
- `ImageFeatureBuilder` (img.*): pair/gap/coverage sequence stats from NDVI/EVI
  records; optional per-date patch statistics via an injected extractor.
- `TemporalFeatureBuilder` (tmp.*): year, season, days-in-season,
  observation-date range.
- `FeatureBuilderRegistry` merges enabled modalities into one row;
  `build_feature_frame` turns a corpus (accepted observations) or observation
  list into a rectangular `pandas.DataFrame` (missing keys → `NaN`).
- `build_cropfusion_datasets` bridges accepted observations into the existing
  leakage-free `split_observations` / `CropFusionDataset` train/val/test trio.
- Config loaded with **env (`FE_`) > YAML (`FE_CONFIG_FILE`) > defaults**.

## 5. Statistics, balancing, QC

- `CorpusStatistics.summarize(corpus)`: status counts, quality-score summary
  (read from **accepted samples**, not observations), per-crop/year/season/
  location distributions, yield stats, `missing_labels`.
- `BalancingReport.from_corpus(corpus, label_key="crop")`: class counts,
  minority/majority ratio, `imbalance_ratio`, `balance_score` (0..1).
- `SampleQualityReport.from_corpus(corpus)`: acceptance rate, issue-code and
  severity histograms, per-key acceptance rates, top error codes; written as
  `sample_quality_report.json` via `build_report(corpus, output_dir)`.

## 6. Export

- `export_dataset(frame, corpus, config)` → `manifest.json` + one artifact per
  configured format:
  - `json` — array of JSON-safe records (`NaN` → `null`),
  - `jsonl` — NDJSON stream,
  - `parquet` — pandas/pyarrow,
  - `torch` — `{sample_id, features: float32 tensor, feature_names, n_samples}`.
- `attach_meta` adds `sample_id` / `year` / `season` / `location_id` /
  `quality_score` from the corpus (accepted order aligned to frame rows).
- Config loaded with **env (`EX_`) > YAML (`EX_CONFIG_FILE`) > defaults**;
  unsupported formats rejected at validation and at the registry (`EX-FORMAT-001`).

## 7. Documentation

- 5 guides: `training/stam/docs/OBSERVATION_RESOLVER.md`,
  `training/feature_engineering/docs/FEATURES.md`,
  `training/feature_engineering/docs/STATISTICS.md`,
  `training/quality/samples/docs/QUALITY_REPORTS.md`,
  `training/export/docs/EXPORT.md`.
- 5 diagrams under `docs/diagrams/`: `r2-3-sample-generation.md`,
  `r2-3-feature-building.md`, `r2-3-statistics-balancing.md`,
  `r2-3-quality-reports.md`, `r2-3-dataset-export.md`.

## 8. Verification

- New test modules (79 tests total):
  - `training/stam/tests/test_observation_resolver.py` — 15 passed,
  - `training/feature_engineering/tests/` (conftest + 9 modules) — 38 passed,
  - `training/quality/samples/tests/test_report.py` — 9 passed,
  - `training/export/tests/test_export.py` — 17 passed.
- Full repo `pytest` (application + backend + training + shared) → **1013 +
  79 = 1092 passed, 0 failed**.
- Internal fixes during verification: `CorpusStatistics.summarize` quality
  scores read from accepted resolved samples; tabular builder skips
  `label_columns`; `status_counts` seeded with all three statuses;
  `pd.NA` identity check in record normalisation; synthetic-data assumptions in
  the QC tests relaxed (issue codes may legitimately be empty).

## 9. Migration impact

| Aspect | Impact |
| --- | --- |
| STAM | `observation_resolver.py` added; `exceptions.py` + `__init__.py` extended only |
| Dataset Manager | None — consumed read-only |
| Preprocessing | None — `split_observations`/`CropFusionDataset` reused via the FE bridge |
| Models / training engine / evaluation | None — not modified |
| `application/*` (Prediction Platform) | None |
| `shared/*` | None — `CropFusionError` reused |
| New packages | `feature_engineering`, `export`; `quality/samples` sub-package |
| Artifact surface | `data/out/datasets/{prefix}.{json,jsonl,parquet,pt}` + `manifest.json` (git-ignored outputs) |
