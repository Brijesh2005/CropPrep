# PHASE 2 — MASTER DATASET REPORT

> **Date:** 2026-09-09
> **Status:** COMPLETE — new master dataset built from approved sources; yield target explicitly UNRESOLVED; no model training performed.

---

## 1. What was delivered

| Deliverable | Path | Notes |
|---|---|---|
| Master dataset | `data/master.csv` | 226,854 rows, class labels only |
| Image manifest | `data/image_manifest.csv` | 226,854 rows, image_status VALID/MISSING + honest `sentinel2_status` |
| Split manifest | `data/split_manifest.json` | Spatial leave-one-taluk-out split |
| Dataset builder | `training/prepare_data.py` | Self-contained; no frozen corpus / STAM / dataset_manager / MLOps |
| Source inventory | `docs/PHASE_2_SOURCE_INVENTORY.md` | Every approved + excluded source documented |
| Schema | `docs/MASTER_DATASET_SCHEMA.md` | Column provenance, mapping, leakage notes |
| Yield decision | `docs/YIELD_TARGET_ANALYSIS.md` | **UNRESOLVED** (final recommendation) |

## 2. Data sources used (approved only)

1. **Karnataka OGD Crop Survey** — `govt_crop_survey_data/ogd_unified_all_hoblis.csv` (261,906 rows, 4 DK taluks) + `govt_crop_survey_data/ogd_putturu_kharif_2020_21.csv` (124,848 rows, Puttur).
2. **DK_Features** (`training/datasets/tabular/DK_Features_2018..2023.csv`) — inspected; ONLY 4 columns (`Season, Year, Yield_Proxy_NPP, District`); sole numeric column is the excluded NPP proxy → **contributes no usable features**.
3. **Sentinel-2 (Kaggle** `shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada`**)**: **86.9 GB, NOT downloaded**; no local imagery. Manifest is honest (`sentinel2_status=NOT_DOWNLOADED` on every row); no fabricated matches.

**Excluded & verified absent from the new contract:** `data_season.csv`, `ICRISAT-District Level Data.csv`, `cropdata_updated.csv`, `All-India*`, `dataset.csv`, and `Yield_Proxy_NPP` as supervised target.

## 3. Yield target: UNRESOLVED (explicit decision)

- No per-observation production/yield column exists in any approved source.
- OGD `Crop_Extent` is a **parcel-area** string (`acres-ares-sq.m`) — dimensionally area, not yield; using it as yield would be an invented unit change.
- DK_Features only exposes `Yield_Proxy_NPP` (district NPP composite), which is **excluded as a target**.
- The frozen corpus has **no yield column** either.
- `data/master.csv` keeps `yield_target` as an **empty column**; no fabricated values. Fusion evaluation must treat **crop classification** as the current supervised task; yield regression is explicitly out of scope pending a real target source.

## 4. Master dataset summary

- **Unit:** one OGD crop-survey observation (crop at a surveyed lat/lon, one season+year).
- **Dedup:** kept first row per `(lat, lon, year, season, crop_label, hobli, village)` → **0 duplicate sample_ids**.
- **Geographic filter:** DK bounding box (lat 12.4–13.4, lon 74.5–76.0) → dropped 1 stray row (lat 23.64N, lon 88.85E, West Bengal).
- **Classes:** coconut 199,105 · pepper 27,545 · coffee 173 · cardamom 31.

| Split | Taluks | Samples | Coconut | Pepper | Coffee | Cardamom |
|---|---|---|---|---|---|---|
| train | Belthangady, Mangalore, Bantwal | 104,753 | 99,504 | 5,173 | 71 | 5 |
| val | Puttur | 87,558 | 72,687 | 14,823 | 46 | 2 |
| test | Sullia | 34,543 | 26,914 | 7,549 | 56 | 24 |

**Image status:** VALID 211,494 (93.2%, OGD survey photo URL) · MISSING 15,360 (6.8%).

## 5. Validation results (all 25 checks PASSED)

- Row counts match `split_manifest.json`; 0 duplicate sample_ids; 0 missing critical fields; 0 coords outside DK.
- Taluk→split mapping correct; split & per-split class counts match manifest.
- **Leakage:** 0 exact-coordinate rows shared between different splits; `yield_target` empty (no invented target); no excluded-dataset columns in `master.csv`.
- `image_manifest.csv` sample_ids ↔ `master.csv` exactly; `image_status` ∈ {VALID, MISSING, INVALID}; VALID rows all have URLs; `sentinel2_status` all `NOT_DOWNLOADED`.
- **Reproducibility:** rebuild produces byte-identical `master.csv`.
- `prepare_data.py` functional code has zero references to excluded datasets / frozen corpus / STAM / dataset_manager.

## 6. Active-contract cleanup (excluded-dataset references removed)

- `training/preprocessing/data_contract.py` — docstring updated; `_KG_HA_SOURCES` no longer lists `data_season` / `icrisat` / `yeilds`; marked LEGACY. All 13 unit tests pass.
- `training/config/stam.yaml` — removed `data_season.csv` and `ICRISAT-District Level Data.csv` table blocks; marked LEGACY config; only approved DK_Features tables remain (with note that `Yield_Proxy_NPP` is feature-only). YAML parse OK.
- Excluded files remain on disk (undeleted); they are simply outside the active data contract.

## 7. Reproducibility

```
python training/prepare_data.py --repo-root D:\CropPrep
```
Deterministic (sha256-based sample_id; same input files → same output bytes).

## 8. Image plan & next-phase notes

- OGD survey photos (kodi URLs) cover 93.2% of samples — usable as an image modality TODAY without download.
- Sentinel-2 GeoTIFFs require the 86.9 GB Kaggle dataset (or a certified alternative). When acquired, link via sample `year`+`latitude/longitude`; the manifest schema is ready (`year`, `lat`, `lon`, `sentinel2_status`).
- Any future re-balancing MUST preserve the spatial split; do NOT re-split randomly.

## 9. Out of scope (NOT done, per directive)

- No model training (tabular/image/fusion).
- No evaluation baselines were run.
- No frozen-corpus pipeline imports; no compatibility layer.
- No Phase-3 work started.