"""R5.6 — per-sample image statistics exporter (runs on a Kaggle CPU kernel).

Extracts per-sample NDVI/EVI patch statistics for the frozen binary
(coconut vs pepper) corpus using the exact frozen-corpus build path
(DatasetManager -> STAM -> FrozenCorpusLoader -> Preprocessor fitted on
binary train only), then writes a join-able ``image_stats.csv`` plus a short
JSON summary. No model training, no architecture work.

The output table gives every R5.6 image representation:

  Rep A  nearest real frame        ndvi_last_frame_mean / evi_last_frame_mean
  Rep B  temporal mean image       ndvi_mean / evi_mean (real-frame mask)
  Rep C  full temporal statistics  *_mean / *_std / *_min / *_max,
                                   real_frame_count, total_frames,
                                   zero_fill_fraction

Iteration is single-process and per-sample (Dataset.__getitem__ with
try/except) so a single untransformable observation can never crash the run;
it is recorded as ``skipped`` and still accounted in the summary.

Run from repo root on Kaggle::

    python training/kaggle/scripts/r5_6_kaggle_image_stats.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _add_repo_root(repo_root: Path) -> None:
    root = str(repo_root.resolve())
    for entry in list(sys.path):
        if entry == root or entry == "":
            continue
        shadow = Path(entry) / "training"
    if root not in sys.path:
        sys.path.insert(0, root)


_add_repo_root(_REPO_ROOT)

from training.dataset_manager import DatasetManager, load_settings  # noqa: E402
from training.kaggle.frozen_corpus import FrozenCorpusLoader  # noqa: E402
from training.preprocessing import Preprocessor, load_preprocessing_config  # noqa: E402
from training.preprocessing.dataset import CropFusionDataset  # noqa: E402
from training.stam import STAM  # noqa: E402
from training.stam.config import load_stam_config  # noqa: E402

BINARY = ["coconut", "pepper"]

PHASE = "R5.6 image statistics export"

STAT_COLUMNS = [
    "ndvi_mean", "ndvi_std", "ndvi_min", "ndvi_max",
    "evi_mean", "evi_std", "evi_min", "evi_max",
    "ndvi_last_frame_mean", "evi_last_frame_mean",
    "real_frame_count", "total_frames", "zero_fill_fraction",
]


def _record_id(obs: Any) -> str:
    rid = (obs.provenance or {}).get("record_id")
    return str(rid or obs.observation_id)


def _sample_stats(sample: dict[str, Any]) -> dict[str, Any] | None:
    """Per-sample NDVI/EVI statistics from one transform result."""
    try:
        ndvi = sample["ndvi"]
        evi = sample["evi"]
        mask = sample["temporal_mask"]
        cls = int(sample["crop_label"])
    except Exception:
        return None
    flags = np.nonzero(mask.numpy())
    n_real = int(flags[0].size)
    zero_frac = 1.0 - (n_real / max(1, int(mask.numel())))
    row: dict[str, Any] = {
        "crop_label": BINARY[1] if cls == 1 else BINARY[0],
        "real_frame_count": n_real,
        "total_frames": int(mask.numel()),
        "zero_fill_fraction": round(float(zero_frac), 6),
    }
    for stream, key in ((ndvi, "ndvi"), (evi, "evi")):
        v = stream[mask == 1].float()
        if v.numel() == 0:
            row[f"{key}_mean"] = np.nan
            row[f"{key}_std"] = np.nan
            row[f"{key}_min"] = np.nan
            row[f"{key}_max"] = np.nan
        else:
            row[f"{key}_mean"] = round(float(v.mean()), 6)
            row[f"{key}_std"] = round(float(v.std()), 6)
            row[f"{key}_min"] = round(float(v.min()), 6)
            row[f"{key}_max"] = round(float(v.max()), 6)
        if flags[0].size:
            last = stream[flags[0][-1]].float()
            row[f"{key}_last_frame_mean"] = round(float(last.mean()), 6)
        else:
            row[f"{key}_last_frame_mean"] = np.nan
    return row


def _extract_split(pre: Preprocessor, obs_list: list[Any], split: str,
                   extractor: Any) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    # split="val" disables the train augmentation so statistics describe the
    # real patches, not randomly-cropped/flipped ones. The transform output is
    # deterministic regardless of the sample's official split membership.
    ds = CropFusionDataset.build(pre, obs_list, split="val", extractor=extractor)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for i in range(len(ds)):
        obs = obs_list[i]
        rid = _record_id(obs)
        try:
            sample = ds[i]
        except Exception as exc:  # noqa: BLE001
            skipped.append({"record_id": rid, "class": str(obs.crop),
                            "error": str(exc)[:160]})
            continue
        stats = _sample_stats(sample)
        if stats is None:
            skipped.append({"record_id": rid, "class": str(obs.crop),
                            "error": "transform returned unusable tensor set"})
            continue
        rows.append({"record_id": rid, "split": split, **stats})
        if (i + 1) % 500 == 0:
            print(f"    [{split}] {i + 1}/{len(ds)} rows; skipped={len(skipped)}")
    return pd.DataFrame(rows), skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cropfusion-r56-image-stats")
    parser.add_argument("--paths-config",
                        default=str(_REPO_ROOT / "training" / "config" / "paths.yaml"))
    parser.add_argument("--dataset-config",
                        default=str(_REPO_ROOT / "training" / "config" / "dataset.yaml"))
    parser.add_argument("--stam-config",
                        default=str(_REPO_ROOT / "training" / "config" / "stam.yaml"))
    parser.add_argument("--preprocessing-config",
                        default=str(_REPO_ROOT / "training" / "config" / "preprocessing.yaml"))
    parser.add_argument("--frozen-crop-csv",
                        default=str(_REPO_ROOT / "govt_crop_matched_v2" / "crop_supervised_v2.csv"))
    parser.add_argument("--frozen-manifest",
                        default=str(_REPO_ROOT / "training_manifests"
                                    / "crop_supervised_v2.0_manifest.json"))
    parser.add_argument("--output", default="/kaggle/working/r5_6_image_stats")
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    dataset_settings = load_settings(args.dataset_config)
    stam_cfg = load_stam_config(args.stam_config)
    preprocessing_cfg = load_preprocessing_config(args.preprocessing_config)

    manager = DatasetManager(dataset_settings)
    manifests = manager.provider_manifests()
    imagery = manifests.get("kaggle_hub_image", {})
    if not imagery.get("available"):
        print("[FATAL] imagery not available; R5.6 image stats require the Kaggle mount")
        return 1
    manager.ensure_image()
    manager.generate_image_metadata(force=False)
    stam = STAM(manager, stam_cfg)
    stam.initialize()
    extractor = stam.get_patch

    frozen_loader = FrozenCorpusLoader(
        csv_path=args.frozen_crop_csv, manifest_path=args.frozen_manifest)
    frozen_loader.validate()
    train_obs, val_obs, test_obs = frozen_loader.build(stam)
    bin_train = [o for o in train_obs if o.crop in BINARY]
    bin_val = [o for o in val_obs if o.crop in BINARY]
    bin_test = [o for o in test_obs if o.crop in BINARY]
    print(f"binary roster: train={len(bin_train)} val={len(bin_val)} "
          f"test={len(bin_test)}")
    if not bin_train:
        print("[FATAL] zero binary training observations after corpus build — "
              "aborting (imagery index did not resolve sequences)")
        return 1

    preprocessing_cfg.label.declared_classes = BINARY
    preprocessing_cfg.label.excluded_classes = []
    pre = Preprocessor(preprocessing_cfg)
    pre.fit(bin_train, extractor=extractor)
    enc = list(pre.label.crop_encoder.classes_)
    print(f"binary class encoder: {enc}")
    if enc != BINARY:
        print("[FATAL] class encoder mismatch on binary fit")
        return 1

    dfs: list[pd.DataFrame] = []
    all_skips: list[dict[str, Any]] = []
    for split, obs in (("train", bin_train), ("val", bin_val), ("test", bin_test)):
        part, skips = _extract_split(pre, obs, split, extractor)
        dfs.append(part)
        all_skips.extend(skips)
        print(f"  extracted [{split}] n={len(part)} skipped={len(skips)}")

    df = pd.concat(dfs, ignore_index=True)
    counts = df.groupby(["split", "crop_label"]).size().to_dict()
    print("counts:", {str(k): int(v) for k, v in counts.items()})
    print("missing ndvi_mean:", int(df["ndvi_mean"].isna().sum()))

    csv_path = output / "image_stats.csv"
    df.to_csv(csv_path, index=False)
    (output / "image_stats_skipped.json").write_text(
        json.dumps(all_skips, indent=2, default=str), encoding="utf-8")
    summary = {
        "phase": PHASE,
        "n_samples": int(len(df)),
        "n_skipped": len(all_skips),
        "expected_binary_total": 10560,
        "split_counts": {str(k): int(v) for k, v in counts.items()},
        "columns": list(df.columns),
        "missing_ndvi_mean": int(df["ndvi_mean"].isna().sum()),
        "missing_evi_mean": int(df["evi_mean"].isna().sum()),
        "zero_fill_fraction_mean": round(float(df["zero_fill_fraction"].mean()), 6),
        "zero_fill_mean_per_split": {
            s: round(float(g["zero_fill_fraction"].mean()), 6)
            for s, g in df.groupby("split")
        },
        "mean_real_frames_per_split": {
            s: round(float(g["real_frame_count"].mean()), 3)
            for s, g in df.groupby("split")
        },
        "image_stats_csv": str(csv_path),
        "verify": "image_stats_ok" if len(df) == 10560 else "image_stats_MISMATCH",
    }
    (output / "image_stats_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())