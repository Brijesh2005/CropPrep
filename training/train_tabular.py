"""Phase 3 — high-performance tabular model benchmark (TRAIN / model selection).

Benchmarks XGBoost, CatBoost and LightGBM plus a lightweight learned tabular
encoder on the Phase-2 master dataset, using ONLY the frozen spatial split:

    TRAIN: Belthangady / Mangalore / Bantwal
    VAL:   Puttur
    TEST:  Sullia           (never used here — reserved for evaluate_tabular.py)

Imbalance handling is INVESTIGATED legitimately: several inverse-frequency
weight schemes (uniform / balanced / capped) are scanned jointly with the model
choice, ALL on the validation split only. Neither the test split nor the class
structure of it is used for any decision.

Primary metric: macro F1 (validation). Secondary: accuracy, weighted F1,
macro precision/recall, per-class metrics.

No yield model, no image model, no fusion model. No test-set tuning.

Usage:
    python training/train_tabular.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

CLASS_IDS = {"cardamom": 0, "coffee": 1, "pepper": 2, "coconut": 3}
CLASS_NAMES = [c for c, _ in sorted(CLASS_IDS.items(), key=lambda kv: kv[1])]

# Approved tabular features (see docs/MASTER_DATASET_SCHEMA.md).
# Excluded deliberately: sample_id, survey_id (raw ids); taluk (split-defining);
# hobli/village (admin identity, near-dup grouping aid); crop_label (label);
# crop_extent_ogd (survey-time area, unavailable at inference);
# image_url (image link); split (protocol); yield_target (unresolved, empty).
FEATURE_COLUMNS = ["latitude", "longitude", "year", "month", "season"]
CONTINUOUS_COLUMNS = ["latitude", "longitude", "year", "month"]
SEASON_ENCODE = {"Kharif": 0, "Rabi": 1}

SEEDS = [2020, 2021, 2022]
FINAL_SEED = 42
EARLY_STOPPING_ROUNDS = 100
MAX_ITERATIONS = 1500

WEIGHT_SCHEMES = ["uniform", "balanced", "cap5", "cap10", "cap20"]

TREE_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.1,
}

# ---------------------------------------------------------------------------
# Data + feature building (shared with evaluate_tabular.py)
# ---------------------------------------------------------------------------


def load_master(repo_root: Path) -> pd.DataFrame:
    """Load data/master.csv with stable dtypes."""
    df = pd.read_csv(repo_root / "data" / "master.csv")
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the approved feature matrix + label + split as a tidy frame.

    No feature is derived from anything except the approved master.csv columns
    and no one-hot encoding is needed: season is binary-encoded and all other
    features are numeric, so every model consumes the identical matrix.
    """
    out = df[["sample_id", "split", "crop_label"]].copy()
    for c in CONTINUOUS_COLUMNS:
        out[c] = df[c].astype(float)
    out["season"] = df["season"].map(SEASON_ENCODE).astype(float)
    out["label"] = df["crop_label"].map(CLASS_IDS).astype(int)
    return out


def train_val_test_frames(frame: pd.DataFrame):
    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] == "val"]
    test = frame[frame["split"] == "test"]
    return train, val, test


def make_weight_schemes(labels: np.ndarray) -> dict[str, dict[int, float]]:
    """Several legitimate weight schemes, ALL derived from TRAIN labels only.

    - uniform : all weights = 1 (no imbalance treatment)
    - balanced: sklearn 'balanced' inverse-frequency weights
    - capN    : balanced weights, relative-to-majority weight capped at N
    """
    counts = pd.Series(labels).value_counts().to_dict()
    n = float(sum(counts.values()))
    k = len(counts)
    balanced = {c: n / (k * cnt) for c, cnt in counts.items()}
    min_w = min(balanced.values())
    schemes: dict[str, dict[int, float]] = {"uniform": {c: 1.0 for c in counts}}
    schemes["balanced"] = balanced
    for cap in (5, 10, 20):
        schemes[f"cap{cap}"] = {c: min(w / min_w, float(cap)) for c, w in balanced.items()}
    return schemes


def weights_array(labels, scheme: dict[int, float]) -> np.ndarray:
    return np.asarray([scheme[int(c)] for c in labels], dtype=np.float64)


def metrics_for(y_true, y_pred, class_names: list[str]) -> dict[str, Any]:
    """Full metric block (macro primary, secondary, and per-class)."""
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    per_class = {
        name: {
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "support": int(report[name]["support"]),
        }
        for name in class_names
    }
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }


# ---------------------------------------------------------------------------
# Tree model wrappers
# ---------------------------------------------------------------------------


def fit_xgboost(Xtr, ytr, wtr, Xva, yva, wva, seed: int) -> tuple[Any, dict]:
    import xgboost as xgb

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASS_NAMES),
        n_estimators=MAX_ITERATIONS,
        max_depth=TREE_PARAMS["max_depth"],
        learning_rate=TREE_PARAMS["learning_rate"],
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        n_jobs=4,
        random_state=seed,
        eval_metric="mlogloss",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbosity=0,
    )
    t0 = time.time()
    model.fit(
        Xtr, ytr,
        sample_weight=wtr,
        eval_set=[(Xva, yva)],
        sample_weight_eval_set=[wva],
        verbose=False,
    )
    return model, {"train_seconds": round(time.time() - t0, 2)}


def fit_catboost(Xtr, ytr, wtr, Xva, yva, wva, seed: int) -> tuple[Any, dict]:
    from catboost import CatBoostClassifier, Pool

    model = CatBoostClassifier(
        iterations=MAX_ITERATIONS,
        learning_rate=TREE_PARAMS["learning_rate"],
        depth=TREE_PARAMS["max_depth"],
        loss_function="MultiClass",
        random_seed=seed,
        allow_writing_files=False,
        logging_level="Silent",
        thread_count=4,
        bootstrap_type="Bernoulli",
        subsample=0.9,
    )
    train_pool = Pool(Xtr, ytr, weight=wtr)
    val_pool = Pool(Xva, yva, weight=wva)
    t0 = time.time()
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )
    return model, {"train_seconds": round(time.time() - t0, 2)}


def fit_lightgbm(Xtr, ytr, wtr, Xva, yva, wva, seed: int) -> tuple[Any, dict]:
    import warnings

    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(CLASS_NAMES),
        n_estimators=MAX_ITERATIONS,
        max_depth=TREE_PARAMS["max_depth"],
        learning_rate=TREE_PARAMS["learning_rate"],
        num_leaves=2 ** TREE_PARAMS["max_depth"] - 1,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=4,
        random_state=seed,
        verbosity=-1,
    )
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            Xtr, ytr,
            sample_weight=wtr,
            eval_set=[(Xva, yva)],
            eval_sample_weight=[wva],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
        )
    return model, {"train_seconds": round(time.time() - t0, 2)}


FITTERS = {
    "XGBoost": fit_xgboost,
    "CatBoost": fit_catboost,
    "LightGBM": fit_lightgbm,
}


# ---------------------------------------------------------------------------
# Lightweight learned tabular encoder (for future feature fusion)
# ---------------------------------------------------------------------------


class TabularEncoder(nn.Module):
    """Small MLP: 5 features -> 64 -> 16-dim embedding -> 4 classes."""

    def __init__(self, in_dim: int, hidden: int = 64, embed_dim: int = 16,
                 n_classes: int = len(CLASS_NAMES)):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.relu2 = nn.ReLU()
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu2(self.fc2(self.relu1(self.fc1(x))))


def train_encoder(Xtr, ytr, wtr, Xva, yva, weight_scheme: dict[int, float],
                  seed: int, epochs: int = 40, patience: int = 6,
                  batch_size: int = 2048, embed_dim: int = 16) -> tuple[TabularEncoder, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)

    model = TabularEncoder(in_dim=Xtr.shape[1], embed_dim=embed_dim)
    # Class weights for the CE loss: same scheme as the winning tree config.
    w = torch.tensor([weight_scheme.get(int(i), 1.0) for i in range(len(CLASS_NAMES))],
                     dtype=torch.float32)
    w = w / w.sum() * len(CLASS_NAMES)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32)
    ytr_t = torch.tensor(np.asarray(ytr), dtype=torch.long)
    Xva_t = torch.tensor(Xva_s, dtype=torch.float32)
    yva_t = torch.tensor(np.asarray(yva), dtype=torch.long)

    n = Xtr_t.shape[0]
    best_f1, best_state, best_epoch, no_improve = -1.0, None, 0, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            out = model(Xtr_t[idx])
            loss = loss_fn(out, ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xva_t).argmax(1).numpy()
        macro_f1 = f1_score(yva, pred, average="macro", zero_division=0)
        if macro_f1 > best_f1:
            best_f1 = float(macro_f1)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_f1, 6),
        "train_seconds": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=str, default=str(REPO_ROOT))
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = repo_root / "models" / "tabular"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load + build features ---------------------------------------------
    df = load_master(repo_root)
    frame = build_feature_frame(df)
    train, val, test = train_val_test_frames(frame)
    if len(test) != 34543:
        raise RuntimeError(f"Unexpected test split size: {len(test)} — refusing to touch test in training")

    Xtr, ytr = train[FEATURE_COLUMNS].to_numpy(np.float32), train["label"].to_numpy()
    Xva, yva = val[FEATURE_COLUMNS].to_numpy(np.float32), val["label"].to_numpy()
    Xte, yte = test[FEATURE_COLUMNS].to_numpy(np.float32), test["label"].to_numpy()

    schemes = make_weight_schemes(ytr)
    scheme_weights = {name: weights_array(ytr, w) for name, w in schemes.items()}
    scheme_val_weights = {name: weights_array(yva, w) for name, w in schemes.items()}

    print("=" * 72)
    print("PHASE 3 — TABULAR BENCHMARK (TRAIN/VALIDATION)")
    print("=" * 72)
    print(f"Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print(f"Train {len(Xtr)} rows = {dict(Counter(ytr))} | Val {len(Xva)} | Test {len(Xte)} held")
    print(f"Weight schemes (train-only): { {n: {CLASS_NAMES[int(c)]: round(v,2) for c, v in s.items()} for n, s in schemes.items()} }")

    feature_spec = {
        "feature_columns": FEATURE_COLUMNS,
        "continuous_columns": CONTINUOUS_COLUMNS,
        "season_encode": SEASON_ENCODE,
        "class_ids": CLASS_IDS,
        "class_names": CLASS_NAMES,
        "split_rule": {
            "train_taluk": ["Belthangady", "Mangalore", "Bantwal"],
            "validation_taluk": "Puttur",
            "test_taluk": "Sullia",
        },
        "n_features": len(FEATURE_COLUMNS),
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "n_test": int(len(Xte)),
        "train_class_counts": {CLASS_NAMES[i]: int(c) for i, c in Counter(ytr).items()},
        "val_class_counts": {CLASS_NAMES[i]: int(c) for i, c in Counter(yva).items()},
        "test_class_counts": {CLASS_NAMES[i]: int(c) for i, c in Counter(yte).items()},
        "weight_schemes": {n: {CLASS_NAMES[int(c)]: v for c, v in s.items()} for n, s in schemes.items()},
    }
    with open(out_dir / "feature_spec.json", "w", encoding="utf-8") as f:
        json.dump(feature_spec, f, indent=2)

    # 2. Investigate (model x weight-scheme) on VALIDATION ONLY -------------
    scan = {}
    for mname, fitter in FITTERS.items():
        for sname in WEIGHT_SCHEMES:
            f1s, details = [], []
            for seed in SEEDS:
                model, tinfo = fitter(Xtr, ytr, scheme_weights[sname],
                                      Xva, yva, scheme_val_weights[sname], seed)
                m = metrics_for(yva, model.predict(Xva), CLASS_NAMES)
                f1s.append(m["macro_f1"])
                details.append({"seed": seed, "metrics": m, "train_seconds": tinfo["train_seconds"]})
            scan[f"{mname}+{sname}"] = {
                "model": mname,
                "scheme": sname,
                "per_seed_macro_f1": f1s,
                "mean_val_macro_f1": round(float(np.mean(f1s)), 6),
                "std_val_macro_f1": round(float(np.std(f1s)), 6),
                "seed_details": details,
            }
            print(f"[{mname}+{sname}] val macro_f1 = "
                  f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f} "
                  f"({[round(v,4) for v in f1s]})")

    # 3. Select best (model, scheme) on mean validation macro F1 ------------
    best_key = max(scan, key=lambda k: scan[k]["mean_val_macro_f1"])
    best_model, best_scheme = scan[best_key]["model"], scan[best_key]["scheme"]
    best_run = max(scan[best_key]["seed_details"], key=lambda d: d["metrics"]["macro_f1"])
    print("\n>>> SELECTED (by mean validation macro F1):", best_key)
    print(f"    per-seed macro F1: {scan[best_key]['per_seed_macro_f1']}")

    # 4. Retrain the selected model + scheme at FINAL seed, save artifact ----
    fitter = FITTERS[best_model]
    model_final, tinfo_final = fitter(Xtr, ytr, scheme_weights[best_scheme],
                                      Xva, yva, scheme_val_weights[best_scheme], FINAL_SEED)
    val_metrics_final = metrics_for(yva, model_final.predict(Xva), CLASS_NAMES)
    val_metrics_final.update({"seed": FINAL_SEED, "train_seconds": tinfo_final["train_seconds"]})

    if best_model == "XGBoost":
        artifact = out_dir / "best_standalone_tabular.json"
        model_final.save_model(str(artifact))
    elif best_model == "LightGBM":
        artifact = out_dir / "best_standalone_tabular.model"
        model_final.booster_.save_model(str(artifact))
    else:
        artifact = out_dir / "best_standalone_tabular.model"
        model_final.save_model(str(artifact))

    # 5. Learned tabular encoder (winner scheme weights) ---------------------
    enc, enc_info = train_encoder(Xtr, ytr, scheme_weights[best_scheme], Xva, yva,
                                  schemes[best_scheme], seed=FINAL_SEED, epochs=args.epochs)
    scaler = StandardScaler().fit(Xtr)
    with torch.no_grad():
        pred_val_enc = enc(torch.tensor(scaler.transform(Xva), dtype=torch.float32)).argmax(1).numpy()
    enc_val_metrics = metrics_for(yva, pred_val_enc, CLASS_NAMES)
    enc_val_metrics.update(enc_info)
    enc_path = out_dir / "tabular_encoder.pt"
    torch.save(enc.state_dict(), enc_path)
    scaler_path = out_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    # 6. Persist -------------------------------------------------------------
    report_data = {
        "phase": 3,
        "split": "spatial leave-one-taluk-out (frozen)",
        "feature_count": len(FEATURE_COLUMNS),
        "imbalance_method": "inverse-frequency class weights (scan: uniform/balanced/cap5/cap10/cap20), train-only",
        "selection_criterion": "mean validation macro F1 over 3 seeds (test untouched)",
        "validation_split": "Puttur",
        "test_used_in_training_or_selection": False,
        "scan": {k: {kk: vv for kk, vv in v.items() if kk != "seed_details"}
                 for k, v in scan.items()},
        "scan_detail": {k: v["seed_details"] for k, v in scan.items()},
        "selected": {
            "combo": best_key,
            "model": best_model,
            "scheme": best_scheme,
            "final_seed": FINAL_SEED,
            "validation_metrics": val_metrics_final,
            "artifact_path": str(artifact),
            "artifact_size_bytes": artifact.stat().st_size,
        },
        "tabular_encoder": {
            "architecture": "Linear(5)-ReLU-Linear(64)-ReLU-embed(16)-head(4)",
            "imbalance_method": "weighted CrossEntropy (balanced, train-only)",
            "validation_metrics": enc_val_metrics,
            "artifact_path": str(enc_path),
            "artifact_size_bytes": enc_path.stat().st_size,
            "scaler_path": str(scaler_path),
        },
    }
    with open(out_dir / "results_validation.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, default=str)

    # 7. Console summary -------------------------------------------------------
    print("\n" + "=" * 72)
    print("VALIDATION RESULTS (Puttur) — full scan")
    print("=" * 72)
    for k, v in sorted(scan.items(), key=lambda kv: -kv[1]["mean_val_macro_f1"]):
        print(f"{k:<20} macro_f1={v['mean_val_macro_f1']:.4f} ± {v['std_val_macro_f1']:.4f} "
              f"acc={v['seed_details'][0]['metrics']['accuracy']:.4f}")
    print("=" * 72)
    print(f"SELECTED: {best_key} (final seed {FINAL_SEED})")
    print(f"  val macro_f1={val_metrics_final['macro_f1']} "
          f"acc={val_metrics_final['accuracy']} "
          f"weighted_f1={val_metrics_final['weighted_f1']}")
    print(f"  per-class f1: { {c: val_metrics_final['per_class'][c]['f1'] for c in CLASS_NAMES} }")
    print(f"  confusion matrix (rows=true): {val_metrics_final['confusion_matrix']}")
    print(f"\nTABULAR ENCODER val macro_f1={enc_val_metrics['macro_f1']} "
          f"acc={enc_val_metrics['accuracy']} (best epoch {enc_info['best_epoch']})")
    print("Artifacts ->", out_dir)
    print("NEXT: python training/evaluate_tabular.py --repo-root .")


if __name__ == "__main__":
    main()