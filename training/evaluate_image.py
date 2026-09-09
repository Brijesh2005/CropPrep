"""Phase 4 — image branch FINAL evaluation (validation + one-time test).

Loads the artifacts written by ``training/train_image.py`` (best checkpoint,
preprocessing, image_config), rebuilds the exact same datasets, and reports:

    VALIDATION RESULT (Puttur)  — reproduced/consistent with the selection run.
    FINAL TEST RESULT (Sullia)  — only when ``--final-test`` is passed; this is
                                  the single, post-selection test evaluation.

The test set is NEVER used for model selection and is only evaluated once.

Usage:
    python training/evaluate_image.py --repo-root . --output models/image
    python training/evaluate_image.py --repo-root . --output models/image --final-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from torch.utils.data import DataLoader

from training.image_data import (
    CLASS_NAMES, SentinelPatchDataset, load_index, metrics_for,
)
from training.image_models import load_model


def _run_eval(ds, model, device, batch_size: int, workers: int) -> dict[str, Any]:
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=(device.type == "cuda"))
        for batch in dl:
            x = batch["image"].to(device)
            _, logits = model(x)
            all_p.extend(logits.argmax(dim=1).cpu().tolist())
            all_y.extend(batch["label"].tolist())
    return metrics_for(all_y, all_p, CLASS_NAMES)


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 4 image branch evaluation")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--image-dir", default=None, help="Only used to (re)build an index if missing.")
    p.add_argument("--output", default=None, help="Artifact dir (default repo-root/models/image).")
    p.add_argument("--final-test", action="store_true",
                   help="Also evaluate the untouched Sullia test split ONCE.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.output).resolve() if args.output else repo_root / "models" / "image"
    if not (out_dir / "image_config.json").exists():
        raise SystemExit(f"No training artifacts under {out_dir}. Run train_image.py first.")

    config = json.loads((out_dir / "image_config.json").read_text(encoding="utf-8"))
    preprocessing = json.loads((out_dir / "preprocessing.json").read_text(encoding="utf-8"))
    model, mcfg = load_model(out_dir / "best_image_model.pt", pretrained_for_weights=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    channels = preprocessing["channels_order"]
    stats = preprocessing["stats"]
    patch_size = int(preprocessing["patch_size"])
    img_size = int(preprocessing["img_size"])

    idx_path = out_dir / "image_index.csv"
    if not idx_path.exists():
        idx_path = repo_root / "data" / "image_index.csv"
    idx = load_index(idx_path)
    valid = idx[idx["sentinel2_status"] == "VALID"]

    print("=" * 70)
    print("VALIDATION RESULT (Puttur)")
    print("=" * 70)
    va = valid[valid["split"] == "val"]
    ds_va = SentinelPatchDataset(va, channels, patch_size, img_size, stats)
    val_metrics = _run_eval(ds_va, model, device, args.batch_size, args.workers)
    print(json.dumps(val_metrics, indent=2, sort_keys=True))

    if args.final_test:
        print("=" * 70)
        print("FINAL TEST RESULT (Sullia) — evaluated once, post-selection")
        print("=" * 70)
        te = valid[valid["split"] == "test"]
        ds_te = SentinelPatchDataset(te, channels, patch_size, img_size, stats)
        test_metrics = _run_eval(ds_te, model, device, args.batch_size, args.workers)
        print(json.dumps(test_metrics, indent=2, sort_keys=True))
        (out_dir / "results_test.json").write_text(
            json.dumps({
                "phase": 4,
                "split": "test (Sullia) — untouched during training/selection",
                "model": mcfg["arch"],
                "metrics": test_metrics,
                "note": "Single post-selection evaluation. Never used for tuning.",
            }, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote {out_dir / 'results_test.json'}")
    else:
        print("(test set untouched; rerun with --final-test only after selection)")


if __name__ == "__main__":
    main()