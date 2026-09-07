# Feature-Level Fusion campaign (`reports/feature_fusion`)

Simpler-is-better variant of the CropFusion architecture: tabular and imagery
encoders are fused at the **feature level** with masked temporal aggregation.

- **Architecture / models:** `training/models/feature_fusion.py`
- **Config:** `training/config/feature_fusion.yaml`
- **Runner:** `training/kaggle/scripts/train_feature_fusion.py`
- **Imagery cache:** `training/kaggle/scripts/extract_image_embeddings.py`
  → gitignored `artifacts/feature_fusion/image_embeddings/`
- **Leakage audit / shared helpers:** `training/kaggle/scripts/feature_fusion_utils.py`
- **Tests:** `training/kaggle/tests/test_feature_fusion.py`

## Reproduce
```bash
# optional: materialize + verify the imagery-embedding cache
python training/kaggle/scripts/extract_image_embeddings.py
python training/kaggle/scripts/extract_image_embeddings.py --verify

# the five ablations
python training/kaggle/scripts/train_feature_fusion.py --mode tabular                    # A/E: no location
python training/kaggle/scripts/train_feature_fusion.py --mode tabular --with-location    # F: +lat/lon
python training/kaggle/scripts/train_feature_fusion.py --mode imagery                    # B
python training/kaggle/scripts/train_feature_fusion.py --mode fusion --fusion-type concat # C
python training/kaggle/scripts/train_feature_fusion.py --mode fusion --fusion-type gated  # D

# aggregate metrics/csvs/figures + comparison row for the original campaign
python training/kaggle/scripts/train_feature_fusion.py --compile
```

## Artifacts
- `models/<tag>/` — per-run checkpoint, `metrics.json`, `run_metadata.json`,
  `training_history.csv`, `predictions.csv`, `confusion_matrix.csv`,
  `embeddings.npz` (fusion runs; val+test tabular/image/fused embeddings for
  PCA/t-SNE). Generated CSVs/NPZ/checkpoints are gitignored (reproducible).
- Top level (from `--compile`): `model_comparison.csv/.json`, `metrics.csv`,
  `training_history.csv`, `predictions.csv`, `run_metadata.json`, `figures/*.png`,
  plus this report.

## Constraints honored
Preprocessing/weights/threshold/early-stopping use train/val only; the frozen
test split is evaluated exactly once per model; leakage audit aborts on any
forbidden column (target/fraction/dominance/extent/benchmark/npp/yield/proxy) or
cross-split field overlap. See `feature_fusion_report.md` for results.