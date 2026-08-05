# CropFusion Datasets

Source data lives in `Tabular_Datasets/` and is managed through the
`services/dataset_manager` package (profiling, validation, caching, export).
Statistics are produced by `research/scripts/dataset_stats.py` and refreshed
into `research/DATASETS.md`.

## Summary (computed snapshot)

- **Total records:** 35,831 rows across 5 source files (124 columns total).
- **Primary sources:**
  - ICRISAT district-level data (16,146 rows × 80 cols) - crop-wise area,
    production and yield by district across India.
  - `cropdata_updated.csv` (16,411 rows × 7 cols) - soil/climate suitability
    samples.
  - `data_season.csv` (3,158 rows × 12 cols, 2004-2019) - seasonal rainfall,
    temperature, irrigation and yield observations.
  - All-India crop-wise area/production/yield (98 rows × 17 cols) - 5-season
    national aggregates.
  - Small evaluation set (18 rows × 8 cols).

## Data lifecycle

1. **Ingest:** `dataset_manager` registers a dataset version, computes a
   checksum and stores metadata in `datasets/.cropfusion/`.
2. **Profile / validate:** schema, ranges, missingness and cross-field checks.
3. **Align:** STAM (`services/spatial_alignment`) joins district/village
   geography with time series.
4. **Preprocess:** `ai/preprocessing` builds model batches (tabular tensors +
   NDVI/EVI sequences + temporal masks).
5. **Version:** dataset versions are recorded in the backend registry
   (`dataset_versions` table) and are prerequisites for model registrations.

## Governance

- Every dataset version carries a checksum and validation status
  (`pending | validating | valid | invalid`).
- Drift monitoring compares production samples against a reference dataset
  (`MLOPS_DRIFT_REFERENCE_DATA`).
- Dataset statistics JSON snapshot: `research/dataset_stats.json`.
