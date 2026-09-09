"""Phase 3 — FINAL TEST evaluation of the selected tabular model.

Loads the model selected in train_tabular.py (chosen purely on VALIDATION) and
evaluates it ONCE on the untouched TEST split (Sullia). No test-set tuning.

Also evaluates the learned tabular encoder on the same test split.

Usage:
    python training/evaluate_tabular.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from train_tabular import (
    CLASS_NAMES,
    FEATURE_COLUMNS,
    build_feature_frame,
    load_master,
    metrics_for,
)
from train_tabular import TabularEncoder


def load_best_model(spec_dir: Path):
    """Load the selected standalone tree model. Returns (name, predict_fn)."""
    val_report = json.loads((spec_dir / "results_validation.json").read_text(encoding="utf-8"))
    name = val_report["selected"]["model"]
    path = Path(val_report["selected"]["artifact_path"])
    feat_cols = json.loads((spec_dir / "feature_spec.json").read_text(encoding="utf-8"))["feature_columns"]

    def predict_classes(X):  # X = np array, shape (n, n_features)
        if name == "XGBoost":
            import xgboost as xgb
            from xgboost import XGBClassifier
            m = XGBClassifier()
            m.load_model(path)
            return np.asarray(m.predict(X), dtype=int)
        if name == "CatBoost":
            from catboost import CatBoostClassifier
            m = CatBoostClassifier()
            m.load_model(str(path))
            return np.asarray(m.predict(X), dtype=int).ravel()
        if name == "LightGBM":
            import lightgbm as lgb
            booster = lgb.Booster(model_file=str(path))
            probs = booster.predict(X)
            return np.asarray(probs, dtype=np.float64).argmax(axis=1)
        raise ValueError(f"unknown model {name}")

    return name, predict_classes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=str, default=".")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    spec_dir = repo_root / "models" / "tabular"
    if not spec_dir.exists():
        raise FileNotFoundError("run training/train_tabular.py first")

    # Rebuild features identically to training
    df = load_master(repo_root)
    frame = build_feature_frame(df)
    test = frame[frame["split"] == "test"]
    if len(test) != 34543:
        print(f"[evaluate] WARNING: test size {len(test)} != expected 34543")

    Xte = test[FEATURE_COLUMNS].to_numpy(np.float32)
    yte = test["label"].to_numpy()

    # ---- Selected standalone tree model: FINAL TEST -----------------------
    name, predict = load_best_model(spec_dir)
    pred_test = np.asarray(predict(Xte), dtype=int)
    test_metrics = metrics_for(yte, pred_test, CLASS_NAMES)

    # ---- Learned tabular encoder: FINAL TEST ------------------------------
    scaler = joblib.load(spec_dir / "scaler.joblib")
    enc = TabularEncoder(in_dim=Xte.shape[1])
    enc.load_state_dict(torch.load(spec_dir / "tabular_encoder.pt", map_location="cpu"))
    enc.eval()
    with torch.no_grad():
        pred_enc = enc(torch.tensor(scaler.transform(Xte), dtype=torch.float32)).argmax(1).numpy()
    enc_test_metrics = metrics_for(yte, pred_enc, CLASS_NAMES)

    # ---- Persist test results ----------------------------------------------
    sel_info = json.loads((spec_dir / "results_validation.json").read_text(encoding="utf-8"))["selected"]
    out = {
        "phase": 3,
        "split": "test (Sullia) — untouched during training/selection",
        "selected_model_final_test": {
            "name": name,
            "metrics": test_metrics,
            "artifact_path": str(Path(sel_info["artifact_path"])),
        },
        "tabular_encoder_final_test": {
            "metrics": enc_test_metrics,
            "artifact_path": str(spec_dir / "tabular_encoder.pt"),
        },
        "note": "Test performance used ONLY for final reporting, never for model selection.",
    }
    with open(spec_dir / "results_test.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    # ---- Console output ----------------------------------------------------
    val_report = json.loads((spec_dir / "results_validation.json").read_text(encoding="utf-8"))
    val_metrics = val_report["selected"]["validation_metrics"]
    enc_val = val_report["tabular_encoder"]["validation_metrics"]

    print("=" * 70)
    print("VALIDATION RESULT (Puttur) — reproduced from train phase")
    print("=" * 70)
    print(f"{name}: macro_f1={val_metrics['macro_f1']} accuracy={val_metrics['accuracy']} "
          f"weighted_f1={val_metrics['weighted_f1']}")
    print(f"  per-class f1: { {c: val_metrics['per_class'][c]['f1'] for c in CLASS_NAMES} }")
    print(f"TabularEncoder: macro_f1={enc_val['macro_f1']} "
          f"per-class f1={ {c: enc_val['per_class'][c]['f1'] for c in CLASS_NAMES} }")

    print()
    print("=" * 70)
    print("FINAL TEST RESULT (Sullia)")
    print("=" * 70)
    print(f"{name}: macro_f1={test_metrics['macro_f1']} accuracy={test_metrics['accuracy']} "
          f"weighted_f1={test_metrics['weighted_f1']}")
    print(f"  macro_precision={test_metrics['macro_precision']} "
          f"macro_recall={test_metrics['macro_recall']}")
    print("  per-class:")
    for c in CLASS_NAMES:
        pc = test_metrics["per_class"][c]
        print(f"    {c:<10} support={pc['support']:<6} precision={pc['precision']:.4f} "
              f"recall={pc['recall']:.4f} f1={pc['f1']:.4f}")
    print("  confusion matrix (rows=true, cols=pred):")
    print(f"    classes: {CLASS_NAMES}")
    for row, cname in zip(test_metrics["confusion_matrix"], CLASS_NAMES):
        print(f"    {cname:<10} {row}")

    print()
    print(f"TabularEncoder: macro_f1={enc_test_metrics['macro_f1']} "
          f"accuracy={enc_test_metrics['accuracy']}")
    print(f"  per-class f1: { {c: enc_test_metrics['per_class'][c]['f1'] for c in CLASS_NAMES} }")


if __name__ == "__main__":
    main()