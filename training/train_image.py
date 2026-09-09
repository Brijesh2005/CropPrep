"""Phase 4 — image (Sentinel-2) branch TRAINING / benchmark (canonical).

Runs the image-only crop classifier benchmark on Kaggle (or a small local
smoke run). Benchmarks EfficientNetV2-S and ConvNeXt-Tiny using:

    TRAIN: Belthangady / Mangalore / Bantwal   (VALID image samples only)
    VAL:   Puttur                              (selection only)
    TEST:  Sullia                              (NEVER used here)

Protocol
--------
1. Build/load the image index (sample_id -> real Sentinel-2 files) and produce
   the exact coverage + leakage reports.
2. Per-channel patch stats computed on a deterministic TRAIN-only subsample.
3. Class-weighted CE with the Phase-3 "cap10" inverse-frequency scheme
   (train labels only).
4. Two-stage transfer learning per architecture:
       STAGE 1 (head): freeze backbone, train classification head.
       STAGE 2 (finetune): unfreeze selected layers, small LR, from the best
       STAGE 1 checkpoint.
   Best validation macro F1 wins per architecture; the architecture with the
   highest validation macro F1 is selected. No test information is touched.
5. Saves everything to the OUTPUT directory (Kaggle: outputs/image/).

Primary metric: macro F1. Full reproducibility block persisted.

Usage:
    python training/train_image.py --repo-root . --image-dir <kaggle-mount> \
        --output models/image
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.image_data import (
    CLASS_IDS, CLASS_NAMES, SentinelPatchDataset, build_base_index,
    build_patch_stats, coverage_report, discover_sentinel_files,
    leakage_check, load_index, make_weight_schemes, match_samples_to_files,
    metrics_for, validate_refs, weights_array, write_index,
)
from training.image_models import ImageClassifier, build_model, model_config, save_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCHS_DEFAULT = ["efficientnet_v2_s", "convnext_tiny"]
KAGGLE_MOUNT_HINT = "crop-yield-forecasting-karnataka-dakshina-kannada"


def resolve_image_dir(args) -> Path | None:
    if args.image_dir:
        p = Path(args.image_dir)
        return p if p.is_dir() else None
    for cand in (Path("/kaggle/input") / KAGGLE_MOUNT_HINT,):
        if cand.is_dir():
            return cand
    return None


def _train_one_stage(ds_tr, ds_va, model: ImageClassifier, device: torch.device,
                     weights: torch.Tensor, lr: float, epochs: int, early_stop: int,
                     freeze_backbone: bool, seed: int, batch_size: int,
                     workers: int) -> tuple[dict, dict[str, Any], torch.Tensor]:
    """Train with early stopping on validation macro F1. Returns
    (best_state_dict, best_val_metrics, per_epoch_log)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.avgpool.parameters():
            p.requires_grad = False
        params = list(model.head.parameters())
    else:
        for p in model.parameters():
            p.requires_grad = True
        params = [
            {"params": model.features.parameters(), "lr": lr},
            {"params": model.head.parameters(), "lr": lr * 10.0},
        ]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    gen = torch.Generator().manual_seed(seed)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=workers,
                       generator=gen, pin_memory=(device.type == "cuda"))
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=workers,
                       pin_memory=(device.type == "cuda"))

    best_f1: float = -1.0
    best_state: dict | None = None
    best_metrics: dict[str, Any] | None = None
    stall = 0
    log: list[dict[str, Any]] = []
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss, n = 0.0, 0
        for batch in dl_tr:
            x, y = batch["image"].to(device), batch["label"].to(device)
            opt.zero_grad()
            _, logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            tot_loss += float(loss) * len(y)
            n += len(y)
        model.eval()
        all_y, all_p = [], []
        with torch.no_grad():
            for batch in dl_va:
                x = batch["image"].to(device)
                _, logits = model(x)
                all_p.extend(logits.argmax(dim=1).cpu().tolist())
                all_y.extend(batch["label"].tolist())
        val = metrics_for(all_y, all_p, CLASS_NAMES)
        log.append({"epoch": ep, "train_loss": round(tot_loss / max(n, 1), 6),
                    "val": val, "seconds": round(time.time() - t0, 2),
                    "stage": "head" if freeze_backbone else "finetune",
                    "lr": lr})
        if val["macro_f1"] > best_f1:
            best_f1 = float(val["macro_f1"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = val
            stall = 0
        else:
            stall += 1
            if stall >= early_stop:
                break
    if best_state is None:
        raise RuntimeError("no improvement ever observed on validation")
    return best_state, best_metrics, log


def main() -> None:
    p = argparse.ArgumentParser(description="Image (Sentinel-2) model benchmark — canonical")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--image-dir", default=None,
                   help="Attached Kaggle dataset directory (required for full training).")
    p.add_argument("--output", default=None, help="Output dir (default repo-root/models/image).")
    p.add_argument("--index-csv", default=None, help="image_index.csv path (default data/image_index.csv).")
    p.add_argument("--archs", default=",".join(ARCHS_DEFAULT))
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs-head", type=int, default=8)
    p.add_argument("--epochs-finetune", type=int, default=8)
    p.add_argument("--lr-head", type=float, default=3e-3)
    p.add_argument("--lr-finetune", type=float, default=1e-4)
    p.add_argument("--early-stop", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight-scheme", default="cap10")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--stats-samples", type=int, default=2500)
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="Optional cap for quick smoke tests (train only).")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    image_dir = resolve_image_dir(args)
    out_dir = Path(args.output).resolve() if args.output else repo_root / "models" / "image"
    out_dir.mkdir(parents=True, exist_ok=True)
    index_csv = Path(args.index_csv).resolve() if args.index_csv else repo_root / "data" / "image_index.csv"
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. image index + coverage + leakage -----------------------------
    if image_dir is None:
        print("No image directory found; looking for an existing image_index.csv ...")
        if not index_csv.exists():
            raise SystemExit(
                "No imagery and no existing image_index.csv. Run on Kaggle with the "
                "dataset attached (--image-dir /kaggle/input/crop-yield-forecasting-...) "
                "or build the index with a manifest-only pass first."
            )
        idx = load_index(index_csv)
        discovered = []
        print(f"Reusing index: {index_csv} ({len(idx)} rows).")
    else:
        print(f"Discovering Sentinel-2 files under: {image_dir}")
        discovered = discover_sentinel_files(image_dir)
        if not discovered:
            raise SystemExit("No raster files (.tif/.tiff/...) found in the image dir.")
        idx = build_base_index(repo_root)
        idx, channels = match_samples_to_files(idx, discovered)
        idx = validate_refs(idx, discovered, args.patch_size)
        ch_counts = Counter()
        for refs in idx["image_refs"]:
            for ch in json.loads(refs):
                ch_counts[ch] += 1
        print(f"Discovered {len(discovered)} raster files; "
              f"reference-matched file counts by channel: {dict(ch_counts)}")
        write_index(idx, index_csv)
        shutil.copy2(index_csv, out_dir / "image_index.csv")

    coverage = coverage_report(idx)
    (out_dir / "coverage_report.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True), encoding="utf-8")
    leak = leakage_check(idx)
    if not leak["passed"]:
        print("WARNING: leakage check has problems:"); print(json.dumps(leak["problems"], indent=1))
        # Do not silently continue on true leakage (duplicates / exact shared patches).
        if (leak["identical_patches_across_splits"] > 0 or leak["duplicate_sample_ids"] > 0
                or leak["duplicate_rows"] > 0):
            raise SystemExit("Refusing to train: real leakage detected. Fix the index first.")
    (out_dir / "leakage_check.json").write_text(
        json.dumps(leak, indent=2, sort_keys=True), encoding="utf-8")
    print("COVERAGE:"); print(json.dumps(coverage, indent=2, sort_keys=True))
    print("LEAKAGE:", json.dumps(leak, indent=2, sort_keys=True))

    valid = idx[idx["sentinel2_status"] == "VALID"]
    tr = valid[valid["split"] == "train"]
    va = valid[valid["split"] == "val"]
    te = valid[valid["split"] == "test"]
    if len(tr) == 0:
        raise SystemExit("No VALID training images — nothing to train on.")
    if len(va) == 0:
        raise SystemExit("No VALID validation images — cannot select models.")

    channels = json.loads(tr.iloc[0]["channels"]) if len(tr) else []
    if not channels:
        raise SystemExit("No channels resolved for training samples.")
    print(f"Channels (model input bands, order fixed): {channels}")

    if args.max_train_samples:
        tr = tr.groupby("crop_label", sort=False).head(args.max_train_samples)
        tr = tr.head(args.max_train_samples).reset_index(drop=True)

    # --- 2. per-channel stats (train only) --------------------------------
    print("Computing per-channel patch stats (train subsample, deterministic) ...")
    stats = build_patch_stats(tr, channels, args.patch_size, args.img_size,
                              args.stats_samples, args.seed)
    (out_dir / "preprocessing.json").write_text(
        json.dumps({"patch_size": args.patch_size, "img_size": args.img_size,
                    "channels": channels, "channels_order": channels,
                    "normalization": "per-channel p2/p98 clip + zscore (train stats)",
                    "stats": stats}, indent=2, sort_keys=True), encoding="utf-8")

    schemes = make_weight_schemes(tr["crop_label"].map(CLASS_IDS).to_numpy())
    if args.weight_scheme not in schemes:
        raise SystemExit(f"unknown weight scheme {args.weight_scheme}")
    scheme_w = schemes[args.weight_scheme]
    w = torch.tensor([scheme_w.get(c, 1.0) for c, _ in sorted(CLASS_IDS.items())],
                     dtype=torch.float32)
    print(f"Weight scheme '{args.weight_scheme}': {scheme_w}")

    ds_tr = SentinelPatchDataset(tr, channels, args.patch_size, args.img_size, stats)
    ds_tr.augment_hflip = True
    ds_tr.augment_vflip = True
    ds_tr.augment_rotate = True
    ds_va = SentinelPatchDataset(va, channels, args.patch_size, args.img_size, stats)
    ds_va.augment_hflip = False
    ds_va.augment_vflip = False
    ds_va.augment_rotate = False
    print(f"TRAIN VALID samples={len(tr)} (dataset {len(ds_tr)}) | "
          f"VAL VALID={len(va)} (dataset {len(ds_va)})")

    # --- 3. benchmark architectures ---------------------------------------
    results: dict[str, Any] = {}
    per_arch_metrics: dict[str, float] = {}
    for arch in archs:
        cfg = dict(
            arch=arch, in_channels=len(channels), num_classes=len(CLASS_NAMES),
            embed_dim=None, pretrained=args.pretrained, img_size=args.img_size,
        )
        print(f"\n=== {arch} (pretrained={args.pretrained}) ===")
        model = build_model(cfg, pretrained=args.pretrained).to(device)

        print(f"  stage head (epochs={args.epochs_head}, lr={args.lr_head}) ...")
        h_state, h_metrics, h_log = _train_one_stage(
            ds_tr, ds_va, model, device, w, args.lr_head, args.epochs_head,
            args.early_stop, freeze_backbone=True, seed=args.seed,
            batch_size=args.batch_size, workers=args.workers)
        print(f"  -> head val macro_f1={h_metrics['macro_f1']} acc={h_metrics['accuracy']}")

        print(f"  stage finetune (epochs={args.epochs_finetune}, lr={args.lr_finetune}) ...")
        model2 = build_model(cfg, pretrained=args.pretrained).to(device)
        model2.load_state_dict(h_state)
        f_state, f_metrics, f_log = _train_one_stage(
            ds_tr, ds_va, model2, device, w, args.lr_finetune, args.epochs_finetune,
            args.early_stop, freeze_backbone=False, seed=args.seed + 1,
            batch_size=args.batch_size, workers=args.workers)
        print(f"  -> finetune val macro_f1={f_metrics['macro_f1']} acc={f_metrics['accuracy']}")

        best_stage = "finetune" if f_metrics["macro_f1"] >= h_metrics["macro_f1"] else "head"
        best_m = f_metrics if best_stage == "finetune" else h_metrics
        best_state = f_state if best_stage == "finetune" else h_state
        final = build_model(cfg, pretrained=args.pretrained).to(device)
        final.load_state_dict(best_state)
        save_model(final, out_dir / f"{arch}.pt")
        results[arch] = {
            "head_validation_metrics": h_metrics,
            "finetune_validation_metrics": f_metrics,
            "selected_stage": best_stage,
            "selected_validation_metrics": best_m,
            "head_log": h_log,
            "finetune_log": f_log,
            "checkpoint": str(out_dir / f"{arch}.pt"),
            "embed_dim": final.embed_dim,
        }
        per_arch_metrics[arch] = best_m["macro_f1"]

    # --- 4. selection (validation only) ------------------------------------
    best_arch = max(per_arch_metrics, key=per_arch_metrics.get)
    best_run = results[best_arch]
    cfg_best = dict(arch=best_arch, in_channels=len(channels), num_classes=len(CLASS_NAMES),
                    embed_dim=int(best_run["embed_dim"]), pretrained=args.pretrained,
                    img_size=args.img_size)
    best_model = build_model(cfg_best, pretrained=args.pretrained).to(device)
    shutil.copy2(out_dir / f"{best_arch}.pt", out_dir / "best_image_model.pt")
    best_model.load_state_dict(
        torch.load(out_dir / "best_image_model.pt", map_location="cpu", weights_only=False)["state_dict"])

    (out_dir / "best_standalone_image.json").write_text(
        json.dumps({
            "selected_architecture": best_arch,
            "selected_stage": best_run["selected_stage"],
            "selection_criterion": "validation macro F1 (test never used)",
            "validation_metrics": best_run["selected_validation_metrics"],
            "embed_dim": int(best_run["embed_dim"]),
            "checkpoint": str(out_dir / "best_image_model.pt"),
            "model_config": model_config(best_model),
        }, indent=2, sort_keys=True), encoding="utf-8")

    (out_dir / "image_config.json").write_text(json.dumps({
        "phase": 4,
        "kaggle_dataset": KAGGLE_MOUNT_HINT,
        "code": "training/train_image.py (canonical)",
        "seed": args.seed,
        "architectures": archs,
        "input_channels": channels,
        "img_size": args.img_size,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "lr_head": args.lr_head,
        "lr_finetune": args.lr_finetune,
        "epochs_head": args.epochs_head,
        "epochs_finetune": args.epochs_finetune,
        "early_stopping": {"maximize": "validation macro F1", "patience": args.early_stop},
        "class_weights": scheme_w,
        "weight_scheme": args.weight_scheme,
        "augmentation_train": ["horizontal flip", "vertical flip", "rotation +/-10deg"],
        "augmentation_val_test": "none (deterministic)",
        "pretrained": args.pretrained,
        "train_val_test_counts": {
            "train_samples_total": int((idx["split"] == "train").sum()),
            "val_samples_total": int((idx["split"] == "val").sum()),
            "test_samples_total": int((idx["split"] == "test").sum()),
            "train_valid_images": int(len(tr)),
            "val_valid_images": int(len(va)),
            "test_valid_images": int(len(te)),
        },
        "ARTIFACTS": [str(out_dir / f) for f in
                      ("best_image_model.pt", "best_standalone_image.json", "image_config.json",
                       "preprocessing.json", "coverage_report.json", "leakage_check.json")],
    }, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== RESULT (validation only) ===")
    for arch, m in sorted(per_arch_metrics.items(), key=lambda kv: -kv[1]):
        print(f"  {arch:>18}  val macro_f1={m:.4f}  stage={results[arch]['selected_stage']}")
    print(f"SELECTED: {best_arch} (val macro_f1={best_run['selected_validation_metrics']['macro_f1']})")
    print(f"Artifacts -> {out_dir}")
    print("NEXT: python training/evaluate_image.py --repo-root . --output <out> --final-test")


if __name__ == "__main__":
    main()