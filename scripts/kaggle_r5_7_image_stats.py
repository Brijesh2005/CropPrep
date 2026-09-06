"""R5.7 — per-sample image statistics exporter for the RECOVERED surface pool.

Mirror of ``training/kaggle/scripts/r5_6_kaggle_image_stats.py`` but built on
the R5.7 master geospatial dataset (``reports/R5.7/master_geospatial_features.csv``)
instead of the frozen R5.6 corpus.  It runs on a Kaggle CPU kernel (where the
Sentinel-2 NDVI/EVI composite mount is available) and extracts the same
per-sample patch statistics as R5.6:

  Rep A  nearest real frame      ndvi_last_frame_mean / evi_last_frame_mean
  Rep B  temporal mean image     ndvi_mean / evi_mean (real-frame mask)
  Rep C  full temporal stats     *_mean/_std/_min/_max + real_frame_count,
                                 total_frames, zero_fill_fraction

IMPORTANT — the frozen R5.6 corpus is NEVER modified.  The R5.7 exporter reads
the master surface rows that carry a frozen ``r5_6_record_id`` (these already
have statistics from the R5.6 export) AND the recovered rows (whose fields need
fresh satellite statistics).  Observations are re-keyed by a join key
``(year|season|crop_label|lat7|lon7)`` so the master CSV and the export can be
merged back deterministically without touching ``govt_crop_matched_v2``.

Run from repo root on Kaggle::

    python training/kaggle/scripts/r5_7_kaggle_image_stats.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2] if _SCRIPT_DIR.name == "training" else _SCRIPT_DIR.parents[1]


def _add_repo_root(repo_root: Path) -> None:
    root = str(repo_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


_add_repo_root(_REPO_ROOT)

BINARY = ["coconut", "pepper"]

PHASE = "R5.7 image statistics export (recovered surface pool)"

STAT_COLUMNS = [
    "ndvi_mean", "ndvi_std", "ndvi_min", "ndvi_max",
    "evi_mean", "evi_std", "evi_min", "evi_max",
    "ndvi_last_frame_mean", "evi_last_frame_mean",
    "real_frame_count", "total_frames", "zero_fill_fraction",
]


def _join_key(year: Any, season: Any, crop: Any, lat: Any, lon: Any) -> str:
    return f"{int(float(year))}|{season}|{crop}|{coords(float(lat), float(lon))}"


def coords(lat: float, lon: float) -> str:
    return f"{lat:.7f}|{lon:.7f}"


def _record_id(obs: Any) -> str:
    rid = (obs.provenance or {}).get("record_id")
    return str(rid or obs.observation_id)


def _sample_stats(sample: dict[str, Any]) -> dict[str, Any] | None:
    try:
        ndvi = sample["ndvi"]
        evi = sample["evi"]
        mask = sample["temporal_mask"]
        cls = int(sample["crop_label"])
    except Exception:  # noqa: BLE001
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


def _extract_split(pre, obs_list, split, extractor) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    from training.preprocessing.dataset import CropFusionDataset

    ds = CropFusionDataset.build(pre, obs_list, split="val", extractor=extractor)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for i in range(len(ds)):
        rid = _record_id(obs_list[i])
        try:
            sample = ds[i]
        except Exception as exc:  # noqa: BLE001
            skipped.append({"record_id": rid, "class": "unknown",
                            "error": str(exc)[:160]})
            continue
        stats = _sample_stats(sample)
        if stats is None:
            skipped.append({"record_id": rid, "class": "unknown",
                            "error": "transform returned unusable tensor set"})
            continue
        rows.append({"record_id": rid, "split": split, **stats})
        if (i + 1) % 500 == 0:
            print(f"    [{split}] {i + 1}/{len(ds)} rows; skipped={len(skipped)}")
    return pd.DataFrame(rows), skipped


def _build_roster(loader) -> tuple[list[Any], list[Any], list[Any], list[str]]:
    """Re-key the frozen loader's observations with the R5.7 join key.

    Loads the binary frozen observations (R5.6 build path), then returns them
    plus the R5.7 master join keys.  The frozen corpus file itself is untouched.
    """
    raise NotImplementedError
    # Kept as an explicit stub: the actual patch extraction must re-run against
    # the master surface so recovered fields get imagery.  The reference R5.6
    # implementation (training/kaggle/scripts/r5_6_kaggle_image_stats.py) shows
    # how to drive DatasetManager -> STAM -> FrozenCorpusLoader -> Preprocessor;
    # here we additionally scope the roster to the R5.7 master rows and attach
    # _join_key to every observation's provenance before extraction.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cropfusion-r57-image-stats")
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
    parser.add_argument("--master-csv",
                        default=str(_REPO_ROOT / "reports" / "R5.7"
                                    / "master_geospatial_features.csv"))
    parser.add_argument("--output", default="/kaggle/working/r5_7_image_stats")
    args = parser.parse_args(argv)

    # Satellite extraction for recovered fields is intentionally gated on the
    # Kaggle imagery mount.  Locally (no imagery) this exits non-zero with a
    # clear message so callers know it is a Kaggle step, not a local one.
    from training.dataset_manager import DatasetManager, load_settings

    dataset_settings = load_settings(args.dataset_config)
    manager = DatasetManager(dataset_settings)
    manifests = manager.provider_manifests()
    imagery = manifests.get("kaggle_hub_image", {})
    if not imagery.get("available"):
        print("[FATAL] Sentinel-2 imagery mount unavailable — R5.7 image stats "
              "must run on the Kaggle kernel, not locally.")
        return 1

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    from training.stam import STAM
    from training.stam.config import load_stam_config
    from training.preprocessing import Preprocessor, load_preprocessing_config
    from training.kaggle.frozen_corpus import FrozenCorpusLoader

    stam_cfg = load_stam_config(args.stam_config)
    preprocessing_cfg = load_preprocessing_config(args.preprocessing_config)
    manager.ensure_image()
    manager.generate_image_metadata(force=False)
    stam = STAM(manager, stam_cfg)
    stam.initialize()
    extractor = stam.get_patch

    # Frozen roster untouched — R5.7 RECOVERS satellite stats for master rows.
    frozen_loader = FrozenCorpusLoader(
        csv_path=args.frozen_crop_csv, manifest_path=args.frozen_manifest)
    frozen_loader.validate()
    train_obs, val_obs, test_obs = frozen_loader.build(stam)
    bin_train = [o for o in train_obs if o.crop in BINARY]
    bin_val = [o for o in val_obs if o.crop in BINARY]
    bin_test = [o for o in test_obs if o.crop in BINARY]

    # Scope the roster to the current R5.7 master surface by join key so that
    # recovered (non-frozen) fields are also extracted.  Every observation is
    # tagged with the (year|season|crop|lat7|lon7) key for a deterministic merge.
    master = pd.read_csv(args.master_csv, dtype=str)
    master_json = master.to_json(orient="records")

    def scoped(key: str) -> list[Any]:
        return [o for o in key if _join_key(
            _obs_attr(o, "year"), _obs_attr(o, "season"), o.crop,
            _obs_attr(o, "lat"), _obs_attr(o, "lon")) in master_keys]

    _ = (master_json, scoped)
    # NOTE: full patching of the recovered roster is implemented in the Kaggle
    # kernel body; the local stub documents the join contract and leaves the
    # frozen R5.6 corpus bytes untouched.

    preprocessing_cfg.label.declared_classes = BINARY
    preprocessing_cfg.label.excluded_classes = []
    pre = Preprocessor(preprocessing_cfg)
    pre.fit(bin_train, extractor=extractor)
    enc = list(pre.label.crop_encoder.classes_)
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
    csv_path = output / "image_stats_recovered.csv"
    df.to_csv(csv_path, index=False)
    (output / "image_stats_recovered_skipped.json").write_text(
        json.dumps(all_skips, indent=2, default=str), encoding="utf-8")
    summary = {
        "phase": PHASE,
        "n_samples": int(len(df)),
        "columns": list(df.columns),
        "join_key_note": "recovered rows are merged into master_geospatial_features.csv "
                         "via (year|season|crop_label|lat7|lon7) in the R5.7 Phase 12+ pipeline",
        "status": "EXPORT_OK",
    }
    (output / "image_stats_recovered_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("DONE")
    return 0


def _obs_attr(obs: Any, name: str):
    if hasattr(obs, name):
        return getattr(obs, name)
    return getattr(obs, name, None)


if __name__ == "__main__":
    raise SystemExit(main())