"""Train / evaluate the feature-level fusion crop models.

Architecture (feature fusion, simpler-is-better hypothesis)
-----------------------------------------------------------
    tabular  -> TabularFeatureEncoder           256-D
    imagery  -> ImageFeatureEncoder (per year)  512-D
             -> TemporalAttentionPool (masked)  512-D field vector
    fusion   -> FeatureFusion concat|gated      256-D
    head     -> Linear -> 1 logit (binary coconut/pepper)

Benchmarks (same encoder blocks, same population, same protocol):
    --mode tabular   tabular-only                (ablation A / E: no geo)
    --mode tabular --with-location               (ablation F: +lat/lon)
    --mode imagery   imagery-only                (ablation B)
    --mode fusion --fusion-type concat           (ablation C)
    --mode fusion --fusion-type gated            (ablation D)

Honest-evaluation constraints (load-bearing)
---------------------------------------------
* Frozen R5.10/R5.9 inputs + deterministic balanced population (3,985 fields;
  train 1,540 / val 1,465 / test 980); NEVER subset the test split.
* Preprocessing (median/std imputation, categorical codes, imagery scaling,
  pos-weight) fit on TRAIN ONLY.
* Early stopping, model selection and the classification threshold use the
  VALIDATION split ONLY. The test split is evaluated exactly once per model.
* Leakage audit aborts the run if any forbidden column or split overlap.
* AUC, balanced accuracy etc. computed with sklearn on raw test predictions.

Example
-------
    python train_feature_fusion.py --mode fusion --fusion-type concat
    python train_feature_fusion.py --mode tabular --with-location
    python train_feature_fusion.py --compile
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

from training.kaggle.scripts.feature_fusion_utils import (  # noqa: E402
    FIG_DIR, IMG_CACHE_FILE, MODELS_DIR, REPORT_DIR, audit_all,
    git_head_revision, load_image_cache, load_pipeline, metrics_row,
    save_image_cache, select_threshold, split_hash, verify_image_cache,
    write_py_json,
)
from training.kaggle.scripts.r5_10_temporal_field_target import (  # noqa: E402
    _metrics_full,
)
from training.models.feature_fusion import (  # noqa: E402
    FeatureFusionClassifier, imagery_only_classifier, tabular_only_classifier,
)

DEFAULT_CONFIG = REPO_ROOT / "training" / "config" / "feature_fusion.yaml"
FINAL_RESULTS = REPO_ROOT / "reports" / "final" / "final_results.csv"


def resolve_seed() -> None:
    import random
    import os
    random.seed(0)
    np.random.seed(0)
    os.environ["PYTHONHASHSEED"] = "0"
    torch.manual_seed(0)


def load_config(cfg_path: Path) -> dict:
    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _tables(pop, feat, img, idx, y, with_location: bool):
    """Pre-built torch tensors in pop row order + per-split index slices."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T_tab = torch.from_numpy(feat["X"]).float().to(dev)
    T_img = torch.from_numpy(img["X"]).float().to(dev)
    T_msk = torch.from_numpy(img["mask"]).float().to(dev)
    yt = torch.from_numpy(y).float().to(dev)
    return dict(dev=dev, tab=T_tab, img=T_img, msk=T_msk, y=yt,
                idx=idx, y_np=y, pop=pop, feat=feat, with_location=with_location)


def build_model(mode: str, fusion_type: str, tab_dim: int,
                img_feat_dim: int, cfg: dict, dev: torch.device):
    enc = cfg.get("encoders", {})
    fus = cfg.get("fusion", {})
    if mode == "tabular":
        return tabular_only_classifier(
            tab_dim, embedding_dim=enc.get("tabular_embedding_dim", 256),
            hidden_dim=enc.get("hidden_dim", 512), dropout=enc.get("dropout", 0.2),
        ).to(dev)
    if mode == "imagery":
        return imagery_only_classifier(
            img_feat_dim, embedding_dim=enc.get("image_embedding_dim", 512),
            hidden_dim=enc.get("hidden_dim", 512), dropout=enc.get("dropout", 0.2),
        ).to(dev)
    return FeatureFusionClassifier(
        tabular_dim=tab_dim, image_feature_dim=img_feat_dim,
        tabular_embedding_dim=enc.get("tabular_embedding_dim", 256),
        image_embedding_dim=enc.get("image_embedding_dim", 512),
        fusion_dim=fus.get("fusion_dim", 256),
        hidden_dim=enc.get("hidden_dim", 512), dropout=enc.get("dropout", 0.2),
        fusion_type=fusion_type,
    ).to(dev)


def _param_groups(model: nn.Module, mode: str, cfg: dict):
    tr = cfg.get("training", {})
    lr_head = tr.get("lr_head", 3e-4)
    lr_back = tr.get("lr_backbone", 1e-5)
    wd = tr.get("weight_decay", 1e-4)
    if mode == "fusion":
        backbone = ("tabular_encoder", "image_encoder")
    else:
        backbone = ("encoder",)
    head = [n for n, _ in model.named_parameters()
            if not any(n.startswith(b) for b in backbone)]
    back = [n for n, _ in model.named_parameters()
            if any(n.startswith(b) for b in backbone)]
    groups = [{"params": [p for n, p in model.named_parameters() if n in head],
               "lr": lr_head, "weight_decay": wd}]
    if back:
        groups.append({"params": [p for n, p in model.named_parameters()
                                  if n in back], "lr": lr_back,
                       "weight_decay": wd})
    return groups


@torch.inference_mode()
def _predict(model, mode, T, rows, batch_size, want_emb=False):
    dev = T["dev"]
    model.eval()
    probs = np.zeros(len(rows))
    embs = {}
    for s in range(0, len(rows), batch_size):
        r = rows[s:s + batch_size]
        tb = T["tab"][r].to(dev)
        ib = T["img"][r].to(dev)
        mb = T["msk"][r].to(dev)
        if mode == "tabular":
            out = model(tb)
        elif mode == "imagery":
            out = model(ib, mb)
        else:
            out = model(tb, ib, mb)
        probs[s:s + len(r)] = out["logits"].sigmoid().cpu().numpy()[:, 0]
        if want_emb:
            for k in ("tabular_embedding", "image_embedding",
                      "fused_embedding"):
                if k in out:
                    embs.setdefault(k, []).append(out[k].cpu().numpy())
    if want_emb:
        embs = {k: np.concatenate(v, axis=0) for k, v in embs.items()}
    return probs, embs


def train(model, mode, T, cfg, out_dir, resume: bool):
    tr = cfg.get("training", {})
    epochs = int(tr.get("epochs", 60))
    batch = int(tr.get("batch_size", 128))
    patience = int(tr.get("patience", 15))
    clip = float(tr.get("grad_clip", 1.0))
    use_amp = bool(tr.get("amp", True)) and torch.cuda.is_available()
    eager_sampler = bool(tr.get("weighted_sampler", False))
    seed = int(tr.get("seed", 42))

    torch.manual_seed(seed)

    tr_i = T["idx"]["train"]
    va_i = T["idx"]["val"]
    yt_np = T["y_np"]
    ytr = yt_np[tr_i]
    pos_w = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w]).to(T["dev"]))
    optimizer = torch.optim.AdamW(_param_groups(model, mode, cfg))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    cp = out_dir / "checkpoint.pt"
    if resume and cp.exists():
        model.load_state_dict(torch.load(cp, map_location="cpu"))

    best = {"ba": -1.0, "epoch": 0, "state": None, "val_probs": None}
    history = []
    stale = 0
    weights = None
    if eager_sampler:
        from torch.utils.data import WeightedRandomSampler
        weights = torch.tensor(np.where(yt_np[tr_i] == 1, pos_w, 1.0),
                               dtype=torch.float64)
    from torch.utils.data import BatchSampler, DataLoader, TensorDataset, SequentialSampler

    tr_ds = TensorDataset(T["tab"][tr_i], T["img"][tr_i], T["msk"][tr_i],
                          T["y"][tr_i])
    for epoch in range(epochs):
        model.train()
        if eager_sampler:
            sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                            replacement=True)
        else:
            sampler = SequentialSampler(range(len(tr_ds)))
        loader = DataLoader(tr_ds, batch_size=batch, sampler=sampler,
                            drop_last=False)
        tot_loss = 0.0
        n_batches = 0
        for tb, ib, mb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                if mode == "tabular":
                    out = model(tb)
                elif mode == "imagery":
                    out = model(ib, mb)
                else:
                    out = model(tb, ib, mb)
                loss = criterion(out["logits"].squeeze(-1), yb)
            loss.backward()
            if clip is not None and clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            tot_loss += float(loss.detach().cpu())
            n_batches += 1
        sched.step()

        va_probs, _ = _predict(model, mode, T, va_i, batch)
        m = _metrics_full(yt_np[va_i], (va_probs >= 0.5).astype(int), va_probs)
        history.append({"epoch": epoch + 1, "loss": round(tot_loss / max(n_batches, 1), 4),
                        "val_balanced_accuracy": m["balanced_accuracy"],
                        "val_roc_auc": m["roc_auc"]})
        if m["balanced_accuracy"] > best["ba"]:
            best.update(ba=m["balanced_accuracy"], epoch=epoch + 1,
                        state={k: v.detach().clone().cpu() for k, v in model.state_dict().items()},
                        val_probs=va_probs.copy())
            stale = 0
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), cp)
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best["state"])
    return model, history, best["epoch"], pos_w, best["val_probs"]


def run(args, cfg_path: Path) -> int:
    resolve_seed()
    cfg = load_config(cfg_path)
    mode = args.mode
    fusion_type = getattr(args, "fusion_type", None) or cfg.get("fusion", {}).get("fusion_type", "concat")
    with_loc = bool(getattr(args, "with_location", False))
    smoke = bool(getattr(args, "smoke", False))

    if mode == "tabular":
        tag = "tabular_loc" if with_loc else "tabular"
    elif mode == "imagery":
        tag = "imagery"
    else:
        tag = f"fusion_{fusion_type}"

    out_root = REPORT_DIR / "models" if not smoke \
        else REPO_ROOT / "artifacts" / "feature_fusion" / "smoke"
    out_dir = out_root / tag

    pop, feat, img, idx, y = load_pipeline(with_location=with_loc)
    audit = audit_all(pop, feat, img)
    if not audit["pass"]:
        print(f"[LEAKAGE-FAIL] {tag}: {audit}")
        return 2
    if IMG_CACHE_FILE.exists() and not smoke:
        cache_report = verify_image_cache(pop, img)
        print(f"[{tag}] image-embedding cache verified: {cache_report['consistent']}")

    if smoke:
        epochs_override = 3 if args.epochs is None else args.epochs
        cfg.setdefault("training", {})["epochs"] = epochs_override
        cfg["training"]["patience"] = 1
        cfg["training"]["batch_size"] = 64

    T = _tables(pop, feat, img, idx, y, with_loc)
    tab_dim = feat["X"].shape[1]
    img_feat_dim = img["X"].shape[-1]
    model = build_model(mode, fusion_type, tab_dim, img_feat_dim, cfg, T["dev"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] mode={mode} fusion={fusion_type} loc={with_loc} "
          f"tab_dim={tab_dim} img_feat_dim={img_feat_dim} params={n_params:,} "
          f"device={T['dev']}")

    model, history, best_epoch, pos_w, va_probs = train(
        model, mode, T, cfg, out_dir, resume=bool(args.resume))

    batch = int(cfg.get("training", {}).get("batch_size", 128))
    val_i = T["idx"]["val"]
    test_i = T["idx"]["test"]
    nv = len(val_i)
    comb = np.concatenate([val_i, test_i])
    comb_probs, embs = _predict(model, mode, T, comb, batch,
                                want_emb=(mode == "fusion"))
    val_probs = comb_probs[:nv]
    test_probs = comb_probs[nv:]

    t_val, _ = select_threshold(y[val_i], val_probs)
    m_val = _metrics_full(y[val_i], (val_probs >= t_val).astype(int), val_probs)
    m_test = _metrics_full(y[test_i], (test_probs >= t_val).astype(int),
                           test_probs)
    m_test_fixed = _metrics_full(y[test_i], (test_probs >= 0.5).astype(int),
                                 test_probs)

    row = metrics_row(tag, mode, fusion_type, t_val, m_val, m_test, m_test_fixed)
    print(f"[{tag}] val_ba={row['val_balanced_accuracy']} "
          f"test_ba={row['test_balanced_accuracy']} "
          f"test_auc={row['test_roc_auc']} thr={row['threshold_val']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_py_json(row, out_dir / "metrics.json")

    import pandas as pd
    all_i = tuple(np.concatenate([T["idx"]["val"], T["idx"]["test"]]))
    preds = pd.DataFrame({
        "field_id": pop["field_id"].to_numpy()[list(all_i)],
        "split": pop["split"].to_numpy()[list(all_i)],
        "survey_year": pop["survey_year"].to_numpy()[list(all_i)],
        "observed_mask_avg": T["msk"][list(all_i)].cpu().numpy().mean(axis=1)
        if mode in ("imagery", "fusion") else np.nan,
        "y_true": y[list(all_i)],
        "prob": np.concatenate([val_probs, test_probs]),
        "pred_thr": (np.concatenate([val_probs, test_probs]) >= t_val).astype(int),
    })
    preds.to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    pd.DataFrame({"threshold": [row["threshold_val"]],
                  "fixed_reference": [0.5]}).to_csv(
        out_dir / "threshold.csv", index=False)
    cm_test = np.array(m_test["confusion_matrix"]).reshape(2, 2)
    cm_val = np.array(m_val["confusion_matrix"]).reshape(2, 2)
    pd.DataFrame({"split": ["val", "test"],
                  "tn": [cm_val[0, 0], cm_test[0, 0]],
                  "fp": [cm_val[0, 1], cm_test[0, 1]],
                  "fn": [cm_val[1, 0], cm_test[1, 0]],
                  "tp": [cm_val[1, 1], cm_test[1, 1]]}).to_csv(
        out_dir / "confusion_matrix.csv", index=False)

    if mode == "fusion" and not smoke and embs:
        va_e = {k: v[:nv] for k, v in embs.items()}
        te_e = {k: v[nv:] for k, v in embs.items()}
        np.savez_compressed(
            out_dir / "embeddings.npz",
            val_field_id=pop["field_id"].to_numpy()[val_i],
            val_split=pop["split"].to_numpy()[val_i],
            val_y=y[val_i], val_prob=val_probs,
            test_field_id=pop["field_id"].to_numpy()[test_i],
            test_y=y[test_i], test_prob=test_probs,
            val_tabular_embedding=va_e["tabular_embedding"],
            val_image_embedding=va_e["image_embedding"],
            val_fused_embedding=va_e["fused_embedding"],
            test_tabular_embedding=te_e["tabular_embedding"],
            test_image_embedding=te_e["image_embedding"],
            test_fused_embedding=te_e["fused_embedding"])

    metadata = {
        "model": tag, "mode": mode, "fusion_type": fusion_type,
        "with_location": with_loc,
        "git_revision": git_head_revision(),
        "seed": cfg.get("training", {}).get("seed"),
        "epochs_run": best_epoch,
        "best_val_epoch": best_epoch,
        "positional": {"tabular_dim": tab_dim, "image_feature_dim": img_feat_dim},
        "feature_columns": feat["cols"],
        "imagery_columns": img["cols"],
        "temporal_grid_years": img["audit"]["timesteps"],
        "padded_slots_fraction": img["audit"]["padded_slots_fraction"],
        "split_hash": split_hash(pop),
        "split_sizes": {s: int(len(idx[s])) for s in ("train", "val", "test")},
        "class_weight_pos_pepper": pos_w,
        "preprocessing": {
            "tab_impute_norm": "median+std from TRAIN only",
            "cat_codes": "fit on TRAIN only",
            "imagery_norm": "mean+std of valid TRAIN cells only",},
        "test_untouched_fields": ["median", "std", "categorical_codes",
                                  "imagery_stats", "pos_weight", "threshold",
                                  "early_stopping", "model_selection"],
        "config": cfg,
        "device": str(T["dev"]),
    }
    write_py_json(metadata, out_dir / "run_metadata.json")
    print(f"[{tag}] artifacts -> {out_dir}")
    return 0


def compile_reports(args) -> int:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    hist_frames = []
    pred_frames = []
    meta = []
    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir():
            continue
        mf = d / "metrics.json"
        if not mf.exists():
            continue
        rows.append(json.loads(mf.read_text()))
        hf = d / "training_history.csv"
        if hf.exists():
            hf_ = pd.read_csv(hf)
            hf_["model"] = d.name
            hist_frames.append(hf_)
        pf = d / "predictions.csv"
        if pf.exists():
            pf_ = pd.read_csv(pf)
            pf_["model"] = d.name
            pred_frames.append(pf_[["model", "field_id", "split", "y_true",
                                    "prob", "pred_thr"]])
        if (d / "run_metadata.json").exists():
            meta.append(json.loads((d / "run_metadata.json").read_text()))

    if FINAL_RESULTS.exists():
        o = pd.read_csv(FINAL_RESULTS)
        rename = {"tabular": "original_tabular", "imagery": "original_imagery",
                  "cropfusion": "original_cropfusion"}
        for m, tag in rename.items():
            r = o[o["model"] == m]
            if len(r):
                rows.append({
                    "model": tag, "mode": "legacy", "fusion_type": "n/a",
                    "threshold_val": 0.5,
                    "val_balanced_accuracy": None, "val_roc_auc": None,
                    "test_balanced_accuracy": r["balanced_accuracy"].iloc[0],
                    "test_roc_auc": r["roc_auc"].iloc[0],
                    "test_macro_f1": r["macro_f1"].iloc[0],
                    "test_recall_coconut": r["coconut_recall"].iloc[0],
                    "test_recall_pepper": r["pepper_recall"].iloc[0],
                    "test_confusion_matrix": json.loads(r["confusion_matrix"].iloc[0]),
                    "threshold_val_src": "fixed 0.5",
                })

    order = ["model", "mode", "fusion_type", "threshold_val",
             "val_balanced_accuracy", "test_balanced_accuracy", "test_roc_auc",
             "test_macro_f1", "test_recall_coconut", "test_recall_pepper",
             "test_confusion_matrix"]
    comp = pd.DataFrame(rows)[order]
    comp.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    comp.to_json(REPORT_DIR / "model_comparison.json", indent=2, orient="records")

    hist_all = pd.concat(hist_frames, ignore_index=True)
    hist_all.to_csv(REPORT_DIR / "training_history.csv", index=False)
    pred_all = pd.concat(pred_frames, ignore_index=True)
    pred_all.to_csv(REPORT_DIR / "predictions.csv", index=False)
    frame_rows = []
    for d in sorted(MODELS_DIR.iterdir()):
        cf = d / "confusion_matrix.csv"
        if cf.exists():
            t = pd.read_csv(cf)
            t["model"] = d.name
            frame_rows.append(t)
    if frame_rows:
        pd.concat(frame_rows, ignore_index=True).to_csv(
            REPORT_DIR / "confusion_matrix.csv", index=False)
    pd.DataFrame(rows).to_csv(REPORT_DIR / "metrics.csv", index=False)
    write_py_json(meta, REPORT_DIR / "run_metadata.json")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs = ["model_comparison_bal_acc.png", "test_roc_curves.png",
            "fusion_embedding_pca.png"]
    if not bool(getattr(args, "no_figures", False)):
        _fig_bar(comp, plt, FIG_DIR / figs[0])
        _fig_roc(pred_all, plt, FIG_DIR / figs[1])
        _fig_pca(FIG_DIR / figs[2])

    print(f"compile -> {REPORT_DIR}")
    print(comp.to_string(index=False))
    return 0


def _fig_bar(comp, plt, out: Path) -> None:
    df = comp[comp["mode"] != "legacy"].dropna(subset=["test_balanced_accuracy"])
    df = df.sort_values("test_balanced_accuracy")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df["model"], df["test_balanced_accuracy"], color="#4C72B0")
    df_leg = comp[comp["mode"] == "legacy"]
    for _, r in df_leg.iterrows():
        ax.axvline(r["test_balanced_accuracy"], ls="--", lw=1.2,
                   color="crimson", label=f"{r['model']} {r['test_balanced_accuracy']}" if r["model"] == "original_cropfusion" else None)
    ax.axvline(0.5, color="grey", lw=0.8)
    ax.set_xlabel("test balanced accuracy (val-selected threshold)")
    ax.set_title("Feature-level fusion models vs legacy CropFusion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_roc(pred, plt, out: Path) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for i, m in enumerate(sorted(pred["model"].unique())):
        d = pred[(pred["model"] == m) & (pred["split"] == "test")]
        if not len(d):
            continue
        fpr, tpr, _ = roc_curve(d["y_true"], d["prob"])
        try:
            au = roc_auc_score(d["y_true"], d["prob"])
        except Exception:
            au = float("nan")
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=1.5,
                label=f"{m} (auc={au:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Test ROC — feature-level fusion campaign")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_pca(out: Path) -> None:
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    emb = None
    for d in sorted(MODELS_DIR.iterdir()):
        f = d / "embeddings.npz"
        if f.exists():
            emb = f
            break
    if emb is None:
        return
    z = np.load(emb)
    F = np.concatenate([z["val_fused_embedding"], z["test_fused_embedding"]])
    y = np.concatenate([z["val_y"], z["test_y"]])
    p = np.concatenate([z["val_prob"], z["test_prob"]])
    try:
        red = PCA(n_components=2).fit_transform(F)
    except Exception:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    sc0 = axes[0].scatter(red[:, 0], red[:, 1], c=y, cmap="coolwarm",
                          s=8, alpha=0.7)
    axes[0].set_title("fused embedding by true class (0 coconut / 1 pepper)")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    sc1 = axes[1].scatter(red[:, 0], red[:, 1], c=p, cmap="viridis",
                          s=8, alpha=0.7)
    axes[1].set_title("fused embedding by predicted pepper probability")
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.colorbar(sc0, ax=axes[0]); fig.colorbar(sc1, ax=axes[1])
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["tabular", "imagery", "fusion"],
                    default="fusion")
    ap.add_argument("--fusion-type", choices=["concat", "gated"], default=None)
    ap.add_argument("--with-location", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compile:
        return compile_reports(args)
    if args.output_dir is not None:
        raise SystemExit("--output-dir is managed internally; use --mode/--fusion-type")
    return run(args, args.config)


if __name__ == "__main__":
    raise SystemExit(main())