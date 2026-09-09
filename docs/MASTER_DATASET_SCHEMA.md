# MASTER DATASET SCHEMA

Status: COMPLETE — final schema of the new active master dataset (`data/master.csv`).

## 1. Unit of one master row

**One row == one OGD crop-survey observation**: a sampled parcel (lat/lon) where a single crop was recorded in a given season+year by the Karnataka OGD crop survey.

- Derived from: `govt_crop_survey_data/ogd_unified_all_hoblis.csv` + `govt_crop_survey_data/ogd_putturu_kharif_2020_21.csv`.
- One `Survey_id` can produce multiple rows (a survey visit records several crops).
- Deduplication key: `(latitude, longitude, year, season, crop_label, hobli, village)`.
- Geographic filter: coordinates must fall inside the Dakshina Kannada bounding box (lat 12.4–13.4, lon 74.5–76.0). One raw row (lat 23.64N, lon 88.85E — inland West Bengal) was dropped by this filter.

## 2. Files produced

| File | Contents | Row count |
|---|---|---|
| `data/master.csv` | All observation-level fields | 226,854 |
| `data/image_manifest.csv` | Per-sample image linkage + status | 226,854 |
| `data/split_manifest.json` | Spatial split definition + counts | — |

## 3. `data/master.csv` schema

| Column | Type | Source | Provenance | Inference availability | Leakage risk |
|---|---|---|---|---|---|
| `sample_id` | str | derived | `obs_` + sha256(lat\|lon\|year\|season\|crop\|hobli\|village)[:12]. Stable, deterministic. | needed to join manifest/split | none (not a feature) |
| `survey_id` | str | OGD `Survey_id` | raw; kept for lineage only, NOT a feature | — | not used as feature |
| `latitude` | float | OGD `Latitude` | decimal degrees | yes | spatial split already separated |
| `longitude` | float | OGD `Longtitude` | decimal degrees | yes | spatial split already separated |
| `taluk` | str | OGD `Taluk_Name` | Belthangady/Mangalore/Bantwal/Puttur/Sullia | yes (geo admin) | used to define split; keep as non-feature by default |
| `hobli` | str | OGD `Hobli_Name` | admin sub-unit | yes | near-dup grouping aid |
| `village` | str | OGD `Village_Name` | admin unit | yes | near-dup grouping aid |
| `year` | str | OGD `Years` → leading year | `2020`/`2021` | yes | none |
| `season` | str | OGD `Season` | `Kharif`/`Rabi` | yes | none |
| `month` | str | OGD `CropSurveyDate` → MM | survey month | yes | adjacent to label timing; treat carefully |
| `crop_label` | str | mapped | OGD `Cropname` → class (see §4) | yes | this IS the classification label |
| `crop_extent_ogd` | str | OGD `Crop_Extent` | parcel-area string `A-AAAA-SS.SS` (acres-ares-sq.m), NOT yield. Kept raw for optional future use. | yes | area is measured at survey time; may be treated as covariate later |
| `image_url` | str | OGD `Image_url` | survey field photo URL (kodi.karnataka.gov.in) | prod would need hosted copy | n/a (link, not feature) |
| `split` | str | derived | train/val/test per spatial rule (§5) | — | defines protocol |
| `yield_target` | str | — | **UNRESOLVED** (see `docs/YIELD_TARGET_ANALYSIS.md`); empty string, never a fabricated column | — | absent by design |

## 4. Crop label mapping

| OGD `Cropname` | `crop_label` | Class id (mirrors frozen corpus convention) |
|---|---|---|
| `Coconut` | coconut | 4 |
| `Betel Nuts (Areca nuts)` | coconut | 4 |
| `Pepper (Black)` | pepper | 3 |
| `Coffee arabica` | coffee | 2 |
| `Coffee robusta` | coffee | 2 |
| `Cardamom` | cardamom | 1 |

Rationale: this is the **demonstrable equivalence** already used by the reference (frozen corpus) corpus: OGD `Coconut`+`Betel Nuts (Areca nuts)` → coconut; both coffee species → coffee. No biologically-distinct crop is merged (pepper vs cardamom vs coconut vs coffee remain separate classes). `Urad`→blackgram (2 samples only) is excluded — not a supervised class.

Final distribution (226,854): coconut 199,105 · pepper 27,545 · coffee 173 · cardamom 31.

## 5. Spatial split (`data/split_manifest.json`)

Rule: leave-one-taluk-out. Row gets exactly one split from its taluk.

| Split | Taluks | Samples | Coconut | Pepper | Coffee | Cardamom |
|---|---|---|---|---|---|---|
| train | Belthangady, Mangalore, Bantwal | 104,753 | 99,504 | 5,173 | 71 | 5 |
| val | Puttur | 87,558 | 72,687 | 14,823 | 46 | 2 |
| test | Sullia | 34,543 | 26,914 | 7,549 | 56 | 24 |

Total 226,854 (one raw observation with coordinates outside Dakshina Kannada —
lat 23.64N lon 88.85E — was dropped by the DK bounding-box filter; see
`docs/PHASE_2_SOURCE_INVENTORY.md` flag 6).

## 6. `data/image_manifest.csv` schema

| Column | Type | Meaning |
|---|---|---|
| `sample_id` | str | join key to `master.csv` |
| `image_url` | str | OGD survey photo URL, or empty |
| `year` | str | sample year (join key for future Sentinel-2) |
| `latitude` / `longitude` | float | sample coords (join key for future Sentinel-2) |
| `image_source` | str | `ogd_survey_photo` | `none` |
| `image_status` | str | `VALID` (URL present) | `MISSING` (none) | `INVALID` (reserved) |
| `sentinel2_status` | str | `NOT_DOWNLOADED` for all rows (86.9 GB Kaggle dataset not fetched; no fabricated matches) |

Current counts: VALID 211,494 (93.2%) · MISSING 15,360 (6.8%).

## 7. Known gaps / decisions

- **Yield target UNRESOLVED** — no defensible yield field exists in approved sources. `yield_target` column is present but empty; downstream must not train yield regression until resolved.
- **Sentinel-2 imagery not linked** — not downloaded (86.9 GB). Manifest is honest (`NOT_DOWNLOADED`). When downloaded, map via lat/lon+year.
- **DK_Features contributes nothing** — only 4 columns; sole numeric column `Yield_Proxy_NPP` excluded as target; no coordinates.
- **No environment grid features** in the new master (the 27 ndvi/evi/soil/etc. fields existed only inside the frozen corpus via the old matcher). Re-introducing them requires a future approved derivation.
- **If we later sub-select a balanced subset** (e.g. to mirror the frozen-corpus scale), the spatial split MUST be preserved — never re-split randomly.