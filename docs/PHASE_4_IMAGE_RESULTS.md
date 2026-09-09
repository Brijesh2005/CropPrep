# PHASE 4 — SENTINEL-2 IMAGE BRANCH RESULTS

> **Date:** 2026-09-09
> **Status:** CODE & LOCAL VERIFICATION COMPLETE — canonical pipeline built and
> smoke-tested end-to-end on synthetic rasters (CPU). **Real 93 GB image
> benchmark PENDING on Kaggle GPU** (all result cells below marked `[KAGGLE
> RUN]`). No fusion or yield work done.

---

## 1. Scope & rules honored

- Dataset of record: Phase 2 output `data/master.csv` (NOT modified) and the
  fixed spatial split `data/split_manifest.json` (leave-one-taluk-out).
  - TRAIN = Belthangady/Mangalore/Bantwal · VAL = Puttur · TEST = Sullia.
- Image branch only: crop classification from Sentinel-2 derived rasters.
  No feature-concat fusion model, no yield regression.
- The Sentinel-2 imagery ship in the public Kaggle dataset
  `shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada`
  (note the trailing `n`; earlier drafts dropped it).
- **The ground-truth presence/absence of every required raster is verified per
  sample**; a sample is usable only for the value actually verified to exist.
- Model selection on the **validation split only**; the test split (Sullia) is
  evaluated **once**, after selection.
- Kaggle dataset file-layout is **not assumed**: discovery auto-classifies each
  `.tif` (`1.4.1`) and the match is enforced by geodata (`1.5`).

## 2. Kaggle dataset structure (as provided)

Facts verified 2026-09-09 from the official Kaggle site (CLI upload-era views +
JSON-LD + notebook asset listing to `dataset-metadata.json`):

| Item | Value |
|---|---|
| Handle | `shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada` |
| datasetId / version | 10915019 / 10 (updated 2026-09-02) |
| Size (site listing) | ~93.4 GB (`totalBytes` view ~148 GB) |
| Owner shown | "Shathananda Bhat N" |
| Content per description | Sentinel-2 imagery `.tif` indices **NDVI, EVI, GCI, Moisture Index**, 2021–2024; weather/soil CSVs; field records |
| Claims on page | Page shows "90%+ accuracy" marketing — **not adopted, unverified** |

The dataset-view file listing is not enumerable programmatically before
Kaggle runtime (the files API returns an empty view for this upload; the
`ListDatasetFiles` RPC is forbidden from this CLI). The on-disk layout is
therefore discovered **at Kaggle runtime** by `training/image_data.py`, and the
inventory that the wrapper prints (`kaggle/train_image.py`) must be pasted into
`§20`. Legacy expectations (from `training/config/kaggle.yaml`, reference only)
were `YYYY-MM-DD_NDVI.tif`-style R10m composites; the discovery code supports
dated/year-only/static files and all seven index tokens.

## 3. Sample↔image correspondence (verified)

`data/image_manifest.csv` has **no `image_id`**; its `image_url` is the OGD
field-photo link. The match key is geodata:

- **(latitude, longitude, year, season)** from `master.csv`.
- Season windows: **Kharif** = Jun–Oct of Y (field mid **Sep Y**); **Rabi** =
  Nov Y–Mar Y+1 (field mid **Jan Y+1**).
- A dated file matches when its date is inside the window; the **nearest dated
  frame** wins (deterministic tie-break). Year-only files match that year;
  static composite files are always eligible.
- Sentinel-2 files are discovered and each is classified from its filename
  (`token` in {NDVI, EVI, GCI, MOISTURE, NDWI, SAVI, NDRE, RAW}, `date` in
  `YYYY-MM-DD`/`YYYYMMDD`/year-only/static).

## 4. Coverage — exact counts

Manifest-level facts (from `data/image_index.csv`, derived from the manifests;
`[KAGGLE RUN]` cells are the real VALID counts to be filled on Kaggle):

| Split | Samples | image_status VALID | image_status MISSING |
|---|---|---|---|
| train | 104,753 | [Open] | [Open] |
| val | 87,558 | [Open] | [Open] |
| test | 34,543 | [Open] | [Open] |
| **total** | **226,854** | 211,494 | 15,360 |

`sentinel2_status` is `NOT_DOWNLOADED` for all 226,854 rows (the imagery lives
only in the Kaggle dataset). Per-split real counts:
`[KAGGLE RUN]` → paste the `coverage_report.json` image into this section.

## 5. Validity classification (VALID / MISSING / INVALID)

A sample is **VALID** iff every active channel resolves to a real, in-grid,
single-band GeoTIFF that is inside the spatial split's sample patch; otherwise
**MISSING** (no raster at all / partial coverage) or **INVALID** (raster
exists but patch out of grid, or mixed valid/invalid bands). No zero-fill, no
duplicate-channel substitution. Detailed per-channel reasons differenced into a
`validation_detail` column.

## 6. Input channels (set at runtime, never assumed)

Discovered channels for both local verification runs (simulated rasters):
`['EVI', 'GCI', 'NDVI', 'NDWI', 'RAW']`. On Kaggle the real set `[KAGGLE RUN]`
is expected near `['NDVI','EVI','GCI','MOISTURE', ...]` per the description.
Model input = the realized channel order; a `C != 3` stem is adapted via the
documented **Kratzert mean-init** (channel-equal per filter), never by
duplicating channels.

## 7. Patch extraction & normalization

- Patch window `patch_size` (default 64) around the sample coords, resampled to
  `img_size` (default 128) only if needed. Float32; single-band rasters stay
  single-band.
- Normalization: per-channel **p2/p98 clip + z-score using train-split
  statistics only** (deterministic subsample), written to `preprocessing.json`.
- Train-only augmentation: H/V flip + rotation ±10° (fill = channel mean).
  Val/test deterministic, no augmentation.

## 8. Class imbalance

Phase 3 imbalance (coconut 99.5k vs pepper 5.2k vs coffee 71 vs cardamom 5 on
train) persists. Strategy reused from Phase 3: **class-weighted CE with the
`cap10` scheme computed from Train labels only** (balanced weights capped at
10× the majority weight). No oversampling, no synthetic data, no rare-class
deletion.

## 9. Models & training protocol

- Benchmarks: **EfficientNetV2-S** (embed 1280-d) and **ConvNeXt-Tiny** (embed
  768-d) via timm/torchvision (ImageNet pretrained on Kaggle).
- Two-stage per arch: (1) frozen backbone, train head @ lr 3e-3; (2) fine-tune
  all @ lr 1e-4, from best stage-1 weights. Early stop on **val macro F1**
  (patience 3). Seed 42.
- Selection criterion: **validation macro F1** only.
- Batch 64, stats-subsample 2500. Full-epoch cap configurable
  (`--max-train-samples`) for fast debugging.

## 10. Leakage controls (all enforced)

Verified against the **full 226,854-row index** (local, simulated rasters):

| Check | Result |
|---|---|
| duplicate `sample_id` | 0 |
| duplicate rows | 0 |
| coordinates in >1 split | 0 |
| identical (file, EXACT coord, channel) patches **across** splits | **0** (hard block) |
| identical patches **within** a split (re-surveyed parcel in diff. years) | 3,139 (flagged for transparency; not leakage) |
| approx <1px identical centers across splits (5-dp risk metric) | 0 (reported, not blocked) |
| shared scene files per split pair | 5/5/5 (documented, allowed) |

**Design note (documented, allowed):** the discovery may match multiple
spatially-disjoint districts to the same scene-level `.tif` composite. This is
not leakage because the real invariant (identical patch pixels across splits)
is zero and enforced; the shared-file counts are reported per split pair.
Augmentation is train-only; val/test evaluation is deterministic.

## 11. Local verification (already executed, CPU, synthetic rasters)

1. **Data layer smoke** — 7 synthetic GeoTIFFs (5 dated 2020-08-01-style
   single-band channels used + year-only + static variants). 60/60/60 crafted
   valid samples: coverage VALID 100 % per split (crop counts correct), leak
   `passed: true`, patch tensor `(5, 32, 32)`, per-channel stats sane
   (means ≈ 0.146/0.148/0.153/0.143/0.156). Full-index match + validation on
   all 226,854 rows: **64.4 s** on local CPU.
2. **Model smoke** — both archs forward (3- and 4-channel, `(B,128,128)`),
   `embed()` = 1280/768-d, Kratzert init verified (channel-equal per filter),
   checkpoint round-trip without internet.
3. **End-to-end mini-run** — `train_image.py` (32-train/20-val, `cap10`,
   2-stage × 1-2 epochs, `--no-pretrained`) → selection by val macro F1 and all
   artifacts written; then `evaluate_image.py --final-test` single test
   evaluation → `results_test.json`. Checkpoint loads with restored
   embed-dim 1280 and `forward` returns `(embed, logits)`.
   (Tiny figures: val macro F1 0.3939 — synthetic-data toy, **not a real
   result**.)

## 12. Bands discovered + preprocessing used (final)

`[KAGGLE RUN]` — paste `preprocessing.json` + the discovery log. Local:
`['EVI','GCI','NDVI','NDWI','RAW']`, p2/p98 clip + zscore, patch 64 → img 128.

## 13. Results — per-architecture validation (Puttur, untouched during fit)

| Architecture | stage-1 val macroF1 | stage-2 val macroF1 | selected stage |
|---|---|---|---|
| efficientnet_v2_s | [KAGGLE RUN] | [KAGGLE RUN] | [KAGGLE RUN] |
| convnext_tiny | [KAGGLE RUN] | [KAGGLE RUN] | [KAGGLE RUN] |

Selection: `[KAGGLE RUN]` (criterion: validation macro F1).

## 14. Selected model

`[KAGGLE RUN]` — arch, embed dim, checkpoint path, artifact bytes (all in
`best_standalone_image.json`).

## 15. Final test — Sullia, evaluated ONCE (post-selection)

`[KAGGLE RUN]` — macro F1, accuracy, weighted F1, and per-class P/R/F1 +
confusion matrix from `results_test.json`.

## 16. Rare-class behavior

`[KAGGLE RUN]` — coffee/cardamom precision/recall and any class collapses;
compare against the Phase-3 finding (all tabular models were 0.0 F1 on
coffee/cardamom due to coverage, not tuning). Report honestly.

## 17. Embedding interface (for Phase 5 fusion)

- `forward(x) -> (embed, logits)`; `embed()` = pooled feature vector of dim
  `embed_dim` (1280 EfficientNetV2-S / 768 ConvNeXt-Tiny).
- `training/image_models.py::ImageClassifier.embed_dim` + `save_model/load_model`
  round-trip verified (local smoke).
- Phase 5 will concatenate tabular + embedding → fusion MLP on the SAME
  val/test protocol.

## 18. Artifacts & paths

| Artifact | Path |
|---|---|
| best checkpoint | `models/image/best_image_model.pt` (from Kaggle `outputs/image/`) |
| per-arch checkpoints | `models/image/efficientnet_v2_s.pt`, `convnext_tiny.pt` |
| selection JSON | `models/image/best_standalone_image.json` |
| run config | `models/image/image_config.json` |
| normalization | `models/image/preprocessing.json` |
| coverage / leakage | `models/image/coverage_report.json`, `leakage_check.json` |
| resolved index | `models/image/image_index.csv` (also `data/image_index.csv`) |
| final test | `models/image/results_test.json` |
| source | `training/image_data.py`, `image_models.py`, `train_image.py`, `evaluate_image.py`; `kaggle/train_image.py` |

## 19. Comparison vs tabular (Phase 3)

| Branch | Best VAL macro F1 | Best TEST macro F1 | Notes |
|---|---|---|---|
| Tabular (standalone CatBoost+cap10) | 0.2602 | 0.2481 | sees only geo+time; val/test constant features → location classifier |
| Image (this phase) | [KAGGLE RUN] | [KAGGLE RUN] | |

**Recommended representation for Phase 5:** `[KAGGLE RUN]` — whichever branch
wins here, plus the tabular encoder (`tabular_encoder.pt`), concatenated into
the fusion MLP, provided FUSION > max(branches) on val.

## 20. Reproducibility record — `[KAGGLE RUN]` (fill after executing)

- Notebook URL + run id, execution date.
- Accelerator (T4/P100), `torch` / `torchvision` / `timm` / `rasterio` versions
  printed by `kaggle/train_image.py`.
- Kaggle dataset version mounted + the inventory tree the wrapper prints
  (`§2`).
- Exact command lines (from `kaggle/README.md`).
- SHA-256 of zipped `outputs/image`.

## 21. Local smoke→Kaggle parity notes

- Canonical scripts (`training/*`) are identical locally and on Kaggle; the
  Kaggle wrapper only sets paths + prints the dataset inventory.
- Local runs use `--no-pretrained` + `--max-train-samples` + 1-2 epochs;
  Kaggle runs use pretrained weights and full epochs — the code path is the
  same, only magnitude differs.
- `data/crop_mapping.json`-style Phase-2 mapping is imported through
  `CLASS_IDS/CLASS_NAMES`; nothing re-maps labels on Kaggle.

## 22. Limitations

- Kaggle dataset structure/true coverage only confirmed at runtime (files API
  closed). Counts in §2/§4 are manifest-level until the run.
- Rare classes (coffee 71 / cardamom 5 train rows) likely yield near-zero F1
  regardless of branch — inherited from coverage, not a tuning knob.
- Single-band index rasters per channel; no raw multispectral bands assumed.
- Scene-level composites force the documented "shared scene files, disjoint
  patches" model; exact-pixel leakage is 0 and enforced.
- GPU-image benchmark results are pending the Kaggle run; nothing in this
  document implies those results exist yet.