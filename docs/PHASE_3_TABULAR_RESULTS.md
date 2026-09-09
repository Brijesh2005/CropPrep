# PHASE 3 — TABULAR MODEL BENCHMARK RESULTS

> **Date:** 2026-09-09
> **Status:** COMPLETE — tabular branch benchmarked; best standalone model selected on validation; final test run ONCE. No image, fusion, or yield work done.

---

## 1. Scope & rules honored

- Active dataset: `data/master.csv` (Phase 2 output, NOT modified).
- Active split: `data/split_manifest.json` (spatial leave-one-taluk-out).
  - TRAIN = Belthangady/Mangalore/Bantwal · VAL = Puttur · TEST = Sullia.
- Frozen corpus: reference only; **not** used in training.
- Tabular features: **approved features only**, identical for every model.
- Test set used **only once**, for final reporting after model selection.
- No yield regression (target unresolved since Phase 2). No Sentinel-2. No image model. No fusion model.

## 2. Dataset used

| Item | Value |
|---|---|
| Total rows | 226,854 |
| Train / Val / Test | 104,753 / 87,558 / 34,543 |
| Feature count | **5** |
| Features | `latitude`, `longitude`, `year`, `month`, `season` (binary Kharif=0/Rabi=1) |

Class distribution (label counts):

| Class | Train | Val | Test |
|---|---|---|---|
| coconut | 99,504 | 72,687 | 26,914 |
| pepper | 5,173 | 14,823 | 7,549 |
| coffee | 71 | 46 | 56 |
| cardamom | 5 | 2 | 24 |

**Excluded from features (documented in `docs/MASTER_DATASET_SCHEMA.md`):** `sample_id`, `survey_id` (raw identifiers), `taluk` (split-defining), `hobli`/`village` (admin identity, near-dup grouping), `crop_label` (label), `crop_extent_ogd` (survey-time parcel area — unavailable at inference), `image_url` (image link), `split` (protocol), `yield_target` (unresolved). No invented agricultural features.

### Important structural fact
Val and Test are **100% Kharif-2020**; only Train contains 2021 (+535 Rabi rows). So `year`/`month`/`season` are constant on both evaluation splits and the models effectively classify on **location alone** at evaluation time. This is a genuine property of the approved data, not an artifact.

## 3. Preprocessing

- Type fixes only: `year`/`month` → int; `latitude`/`longitude` → float; `season` → 0/1.
- No scaling for tree models (they split natively on the identical numeric matrix).
- Neural encoder: `StandardScaler` **fit on Train only**, applied to val/test — no leakage.

## 4. Imbalance handling (investigated)

Extreme imbalance (coffee 0.076 %, cardamom 0.014 % of train). Five legitimate weight schemes were scanned on the **validation split only** (all derived from Train labels):

- `uniform` — no weighting (baseline)
- `balanced` — sklearn inverse-frequency weights (cardamom ≈ 5237× the coconut weight)
- `cap5` / `cap10` / `cap20` — balanced weights capped at 5× / 10× / 20× the majority-class weight

Findings from the investigation (see §5 scan table):

- **Uncapped balanced weights are destructively extreme** — CatBoost+balanced collapsed to val macro F1 0.065 (predicting rare classes everywhere). This motivated the cap family.
- **Uniform / cap5** sit at a "predict-majority" fixed point (macro F1 0.2268, accuracy 0.83, coffee/cardamom 0, pepper 0).
- **cap10 is the sweet spot**: it trades some coconut accuracy for a real pepper recall gain and the highest macro F1 (0.2505 mean over seeds).
- No synthetic data, no oversampling, no deletion of rare classes, no validation/test leakage. Every scheme is derived from Train labels only.

## 5. Scan results — VALIDATION (Puttur), mean over 3 seeds (2020/2021/2022)

Primary metric = **macro F1**. Full per-seed detail in `models/tabular/results_validation.json`.

| Model + scheme | Mean val macro F1 | Std | Per-seed macro F1 |
|---|---|---|---|
| **CatBoost + cap10** | **0.2505** | 0.0063 | 0.256 / 0.254 / 0.242 |
| XGBoost + balanced | 0.2283 | 0.0062 | 0.230 / 0.220 / 0.235 |
| CatBoost + cap5 | 0.2268 | 0.0000 | 0.227 / 0.227 / 0.227 |
| XGBoost + uniform | 0.2268 | 0.0000 | — |
| XGBoost + cap5 | 0.2268 | 0.0000 | — |
| XGBoost + cap10 | 0.2268 | 0.0000 | — |
| CatBoost + uniform | 0.2268 | 0.0000 | — |
| LightGBM + uniform | 0.2268 | 0.0000 | — |
| LightGBM + cap5 | 0.2267 | 0.0000 | — |
| LightGBM + cap10 | 0.2267 | 0.0000 | — |
| XGBoost + cap20 | 0.2111 | 0.0225 | 0.242 / 0.201 / 0.190 |
| LightGBM + balanced | 0.2021 | 0.0000 | — |
| LightGBM + cap20 | 0.1902 | 0.0000 | — |
| CatBoost + cap20 | 0.1842 | 0.0034 | 0.189 / 0.181 / 0.183 |
| CatBoost + balanced | 0.0648 | 0.0037 | 0.070 / 0.064 / 0.061 |

All three algorithms use the identical feature matrix, splits, labels, weights and metrics. Hyperparameters: `n_estimators/iterations=1500`, `max_depth=6`, `learning_rate=0.1`, `subsample=colsample=0.9`, `num_leaves=63` (LGBM), early stopping 100 rounds on the **weighted** validation loss, train-time class weights applied. Note: macro F1 identical to 0.2268 across many configs is the degenerate same-solution fixed point (≈ "always coconut"), not a stable coincidence.

## 6. Best standalone model (selection)

**Selection criterion:** mean validation macro F1 over 3 seeds (test untouched). Secondary: per-class robustness, stability (std), accuracy.

**Winner: CatBoost + cap10** (final seed 42, `models/tabular/best_standalone_tabular.model`, 63,544 bytes).

### VALIDATION RESULT (Puttur)

| Metric | Value |
|---|---|
| Macro F1 | **0.2602** |
| Accuracy | 0.7432 |
| Weighted F1 | 0.7362 |
| Macro precision / recall | 0.2609 / 0.2599 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| coconut | 0.837 | 0.858 | **0.847** | 72,687 |
| pepper | 0.207 | 0.181 | 0.193 | 14,823 |
| coffee | 0.000 | 0.000 | 0.000 | 46 |
| cardamom | 0.000 | 0.000 | 0.000 | 2 |

Confusion matrix (rows = true, cols = pred: cardamom, coffee, pepper, coconut):

```
cardamom [   0,   0,    1,    1]
coffee   [   0,   0,   12,   34]
pepper   [   0,   0, 2688,12135]
coconut  [   0,   0,10298,62389]
```

### FINAL TEST RESULT (Sullia) — evaluated exactly once, after selection

| Metric | Value |
|---|---|
| Macro F1 | **0.2481** |
| Accuracy | 0.7055 |
| Weighted F1 | 0.6772 |
| Macro precision / recall | 0.2512 / 0.2512 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| coconut | 0.780 | 0.867 | **0.821** | 26,914 |
| pepper | 0.225 | 0.138 | 0.171 | 7,549 |
| coffee | 0.000 | 0.000 | 0.000 | 56 |
| cardamom | 0.000 | 0.000 | 0.000 | 24 |

Confusion matrix:

```
cardamom [   0,   0,    3,   21]
coffee   [   0,   0,   10,   46]
pepper   [   0,   0, 1043, 6506]
coconut  [   0,   0, 3588,23326]
```

Test performance was used **only** for this final reporting, never for model/scheme selection.

## 7. Rare-class results — honest statement

**coffee and cardamom achieve F1 = 0.0 in every model × scheme on both val and test.** The model never predicts them correctly. This is the central, honest finding of Phase 3:

1. Train has only 71 coffee and 5 cardamom observations — far too few to learn a decision rule that generalizes.
2. The strict spatial split makes it worse: train coffee is in Belthangady, while val coffee (Puttur) and test coffee/cardamom (Sullia) sit in **different taluks**; location-only features cannot extrapolate 71 points to a new taluk.
3. **Overall accuracy is NOT the story**: accuracy ≈ 0.71–0.83 is driven by coconut (class balance 87.8%) and hides the near-total failure on rare classes. Phase 3 explicitly does **not** claim high performance from accuracy.

This is not a modeling failure that tuning fixes; it is a **data-coverage limit of the approved tabular-only inputs**.

## 8. Learned tabular representation (for future feature fusion)

A small learned encoder was trained on the identical features: `5 → Linear(64) → ReLU → Linear(16) → ReLU → embed(16) → Linear(4)`. Weighted CrossEntropy (cap10 weights), StandardScaler fit on Train, early stopping on val macro F1, Adam.

| Split | Macro F1 | Accuracy | Pepper F1 | Coconut F1 |
|---|---|---|---|---|
| VAL (Puttur) | **0.2620** | 0.7189 | 0.219 | 0.829 |
| TEST (Sullia) | **0.2180** | 0.4660 | 0.304 | 0.568 |

Assessment:
- On validation the 16-d learned embedding **matches** the best tree (0.262 vs 0.260) — a competitive, dense learned tabular representation exists.
- On test it is **less stable** (0.218 vs 0.248): it over-commits to pepper (higher pepper F1 than CatBoost, but sacrifices coconut, lowering accuracy). Coffee/cardamom remain at F1 0.
- **Conclusion:** a learned tabular representation is available and worth evaluating in the fusion model, but it is NOT clearly better than the tree model standalone. Do not assume the best standalone tree is automatically the best fusion representation — Phase 5 must test both.

## 9. Recommended representation for Phase 5 feature fusion

Two candidates should be compared inside the fusion MLP (both cheap):

1. **CatBoost probability vector (recommended first):** 4-d softmax output of `best_standalone_tabular.model` as the tabular branch input to the fusion MLP. Strongest standalone; simple; no new code paths.
2. **16-d learned embedding** from `tabular_encoder.pt` (features → scaler → encoder.embed) concatenated with image features. Competitive on validation; a genuine "learned tabular representation".

Phase 5 must report **FUSION > TABULAR > IMAGE** individually against these two baselines on the identical val protocol and untouched test set. Fusion should, at minimum, beat the CatBoost baseline (val macro F1 0.260 / test 0.248); the image branch (Phase 4) must beat/stand on its own before fusion comparisons are meaningful.

## 10. Reproducibility

- Scan seeds: 2020, 2021, 2022. Final/selected model seed: 42. Encoder seed: 42.
- All weight schemes computed from **Train** labels only; everything stored in `models/tabular/feature_spec.json`.
- Same matrix for all models (no differing feature engineering).
- `python training/train_tabular.py --repo-root .` reproduces the scan; `python training/evaluate_tabular.py --repo-root .` reproduces the final test evaluation.
- Timing (final run): CatBoost 3.4 s, encoder 1.6 s training; full 45-fit scan took a few minutes on CPU.
- Determinism: within library limits (CatBoost thread_count fixed to 4; LightGBM/XGBoost seeded).

## 11. Artifacts (`models/tabular/`)

| File | Contents | Size |
|---|---|---|
| `best_standalone_tabular.model` | Selected CatBoost (cap10) model | 63,544 B |
| `tabular_encoder.pt` | PyTorch state dict, 16-d learned encoder | ~19 KB |
| `scaler.joblib` | StandardScaler (fit on Train) | — |
| `feature_spec.json` | Features, class map, split rule, weight schemes, counts | — |
| `results_validation.json` | Full scan + selected validation metrics | — |
| `results_test.json` | Final test metrics (selected model + encoder) | — |

## 12. Limitations (do not overstate)

1. 5 features, effectively **location-only at evaluation** (year/month/season constant on val+test).
2. Rare classes (coffee/cardamom) at F1 0 — tabular-only inputs cannot resolve them under the strict spatial split.
3. Macro F1 ≈ 0.25–0.26 is the honest tabular ceiling; accuracy (≈0.71–0.83) must not be read as success.
4. The frozen corpus' 27 environmental features are NOT available to this pipeline by Phase-2 design — richer tabular features would require a new approved derivation.
5. The tabular branch is a **baseline**, not the final model. Phase 5 fusion must beat VAL and TEST numbers from §6 on the same protocol.
6. `crop_extent_ogd` (parcel area) was excluded pending confirmation it is available at production inference time; re-adding it (if approved) is the single most likely legitimate lever to raise the tabular ceiling.

## 13. Out of scope (next phases)

- **Phase 4:** Sentinel-2 image branch (86.9 GB download/conversion NOT started here).
- **Phase 5:** fusion MLP — must beat both this tabular baseline (macro F1 0.26 val / 0.25 test) and the image-only baseline.
- **Yield regression:** blocked on the unresolved Phase-2 yield target.