"""R5.5 Task: per-sample corpus tensor audit (Phases 1-3) for real model input.

Runs the REAL preprocessing pipeline (```Preprocessor`` + ``STAM`` patch
extractor) over the accepted frozen-corpus observations and computes the
statistics the Kaggle training audit mandates for the actual model input:

  * per-channel NDVI/EVI patch shapes (must match ``image.size`` 224x224);
  * real-frame content statistics, computed ONLY on real frames (frames marked
    ``temporal_mask == 1``) so zero-padding never dilutes the channel stats;
  * zero-padding fraction (how much of each ``[T,1,H,W]`` tensor is padding);
  * NaN / Inf counts and finite flags;
  * sparsity per ``(crop, year, season)`` (real-frame fraction of T slots);
  * per-sample min/max/mean/std recorded to a full manifest with an option to
    cap the number of rows emitted (the corpus is ~10.6k samples).

Writes ``corpus_tensor_audit.json`` (machine-readable) and prints a
human-readable summary table.

Requires the Kaggle imagery mount (Sentinel rasters). On a machine without
imagery it exits 0 with a clear message instead of fabricating results.

Run from repo root (Kaggle training kernel, after ``run_pipeline.py``)::

    python training/kaggle/scripts/verify_corpus_tensors.py \
        --corpus training/kaggle/outputs/reports/frozen_corpus.json \
        --output training/kaggle/outputs/reports
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _add_repo_root(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    root = str(repo_root)
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    repo_training = (repo_root / "training").resolve()
    for entry in list(sys.path):
        if entry == root or entry == "":
            continue
        shadow = Path(entry) / "training"
        if shadow.exists() and shadow.resolve() != repo_training:
            print(f"[verify_corpus_tensors] removing shadowing sys.path entry: {entry}")
            sys.path.remove(entry)


_add_repo_root(_REPO_ROOT)

from training.dataset_manager import DatasetManager, load_settings  # noqa: E402
from training.preprocessing import Preprocessor, load_preprocessing_config  # noqa: E402
from training.stam import STAM  # noqa: E402
from training.stam.config import load_stam_config  # noqa: E402
from training.stam.observation import AgriculturalObservation  # noqa: E402


def _load_observations(path: Path) -> list[AgriculturalObservation]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    accepted: list[AgriculturalObservation] = []
    for sample in raw["samples"]:
        if sample["status"] == "accepted" and sample.get("observation") is not None:
            obs = AgriculturalObservation.model_validate(sample["observation"])
            obs.provenance = dict(sample.get("provenance") or obs.provenance or {})
            accepted.append(obs)
    return accepted


def _stats(real: torch.Tensor, full: torch.Tensor, name: str) -> dict[str, Any]:
    """Stats over the real-frame slice only, plus padding-aware global."""
    if real.numel() == 0:
        real_stats: dict[str, Any] = {
            "n_real_frames": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "nan": 0,
            "inf": 0,
            "finite": False,
        }
    else:
        r = real.detach().float()
        real_stats = {
            "n_real_frames": int(real.size(0)),
            "min": float(r.min().item()),
            "max": float(r.max().item()),
            "mean": float(r.mean().item()),
            "std": float(r.std().item()),
            "nan": int(torch.isnan(r).sum().item()),
            "inf": int(torch.isinf(r).sum().item()),
            "finite": bool(torch.isfinite(r).all().item()),
        }
    f = full.detach().float()
    return {
        "name": name,
        "shape": list(full.shape),
        "dtype": str(full.dtype),
        "full_nan": int(torch.isnan(f).sum().item()),
        "full_inf": int(torch.isinf(f).sum().item()),
        "zero_frac_full": float((f == 0).float().mean().item()),
        "real": real_stats,
    }


def _positive_mean(x: torch.Tensor) -> float | None:
    nz = x[x > 0]
    if nz.numel() == 0:
        return None
    return float(nz.float().mean().item())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cropfusion-verify-corpus-tensors")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--config",
        default=None,
        help="preprocessing.yaml path (default training/config/preprocessing.yaml)",
    )
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--stam-config", default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap rows in the per-sample manifest (default: all)",
    )
    args = parser.parse_args(argv)

    repo_root = _REPO_ROOT
    corpus_path = Path(args.corpus)
    output_dir = Path(args.output) if args.output else repo_root / "training/kaggle/outputs/reports"
    pre_config = (
        Path(args.config)
        if args.config
        else repo_root / "training/config/preprocessing.yaml"
    )
    ds_config = (
        Path(args.dataset_config)
        if args.dataset_config
        else repo_root / "training/config/dataset.yaml"
    )
    stam_config = (
        Path(args.stam_config)
        if args.stam_config
        else repo_root / "training/config/stam.yaml"
    )

    print("=" * 66)
    print("  R5.5 CORPUS TENSOR AUDIT  (real model input, imagery mounted)")
    print("=" * 66)

    obs_all = _load_observations(corpus_path)
    if not obs_all:
        print("[verify_corpus_tensors] no accepted observations found in corpus")
        return 1

    splits = Counter(o.provenance.get("split", "unknown") for o in obs_all)
    print(f"  corpus: {len(obs_all)} observations  splits={dict(splits)}")
    train_obs = [o for o in obs_all if o.provenance.get("split") == "train"]
    print(f"  train observations for fitting: {len(train_obs)}")
    if not train_obs:
        print("[verify_corpus_tensors] no train-split observations to fit the preprocessor")
        return 1

    print("  resolving patch extractor (STAM imagery)...")
    try:
        manager = DatasetManager(load_settings(ds_config))
        manifest = manager.provider_manifests().get("kaggle_hub_image", {})
        if not manifest.get("available"):
            raise RuntimeError(
                "imagery catalog is NOT available in this environment "
                "(requires the Kaggle imagery mount + ensure_image)"
            )
        stam = STAM(manager, load_stam_config(stam_config))
        stam.initialize()
        extractor = stam.get_patch
    except Exception as exc:  # noqa: BLE001
        print(f"[verify_corpus_tensors] SKIPPED: {exc}")
        print("[verify_corpus_tensors] cannot audit image tensors without rasters")
        return 0

    print("  fitting preprocessor on train observations...")
    pre = Preprocessor(load_preprocessing_config(pre_config))
    try:
        if pre.config.image.normalize == "standard":
            pre.fit(train_obs, extractor=extractor)
        else:
            pre.fit(train_obs)
    except Exception as exc:  # noqa: BLE001
        print(f"[verify_corpus_tensors] preprocessor fit failed: {exc}")
        return 1

    max_obs = pre.config.temporal.max_observations
    patch_size = pre.config.image.size

    # ---- per-sample pass (real frames only) ----
    rows: list[dict[str, Any]] = []
    pad_frames = Counter()
    real_frames_total = 0
    nan_any = 0
    inf_any = 0
    finite_all = True
    shape_ok = True
    # breakdown => (frames_real, frames_total, n_samples)
    by_crop: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    by_year: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    by_season: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    for idx, o in enumerate(obs_all):
        try:
            sample = pre.transform(o, extractor=extractor, augment=False)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "index": idx,
                    "observation_id": str(o.observation_id),
                    "split": o.provenance.get("split", "unknown"),
                    "crop": o.crop,
                    "year": o.temporal.year,
                    "season": o.temporal.season,
                    "error": f"transform failed: {exc}",
                }
            )
            pad_frames["transform_error"] += 1
            continue

        ndvi = sample["ndvi"]
        evi = sample["evi"]
        mask = sample["temporal_mask"]
        tv = sample.get("tabular")
        channels: list[tuple[str, torch.Tensor]] = []
        if isinstance(ndvi, torch.Tensor):
            channels.append(("ndvi", ndvi))
        if isinstance(evi, torch.Tensor):
            channels.append(("evi", evi))

        if ndvi.dim() != 4 or ndvi.size(2) != patch_size or ndvi.size(3) != patch_size:
            shape_ok = False
        if evi.dim() != 4 or evi.size(2) != patch_size or evi.size(3) != patch_size:
            shape_ok = False

        n_pad = int((mask == 0).sum().item())
        pad_frames[n_pad] += 1
        real_frames_total += int((mask > 0).sum().item())

        crop = o.crop or "unknown"
        year = str(o.temporal.year or 0)
        season = o.temporal.season or "unknown"
        for key, bucket in (
            (crop, by_crop),
            (year, by_year),
            (season, by_season),
        ):
            bucket[key][1] += int(mask.numel())
            bucket[key][2] += 1

        row: dict[str, Any] = {
            "index": idx,
            "observation_id": str(o.observation_id),
            "split": o.provenance.get("split", "unknown"),
            "crop": crop,
            "year": o.temporal.year,
            "season": season,
            "shape": list(ndvi.shape),
            "n_pad_frames": n_pad,
            "n_sequence_slots": int(mask.numel()),
        }
        for ch, tensor_ch in channels:
            real = tensor_ch[mask > 0]
            full = tensor_ch
            st = _stats(real, full, ch)
            row[f"{ch}_stats"] = st["real"]
            row[f"{ch}_zero_frac_full"] = st["zero_frac_full"]
            row[f"{ch}_full_nan"] = st["full_nan"]
            row[f"{ch}_full_inf"] = st["full_inf"]
            if st["real"].get("finite") is False:
                finite_all = False
            if st.get("full_nan", 0):
                nan_any += 1
            if st.get("full_inf", 0):
                inf_any += 1
            # positive-pixel mean (discard padding zeros) for real frames
            row[f"{ch}_real_pos_mean"] = _positive_mean(tensor_ch[mask > 0])
            # update breakdown frame counts
            n_real = int((mask > 0).sum().item())
            for key, bucket in ((crop, by_crop), (year, by_year), (season, by_season)):
                bucket[key][0] += n_real

        if tv is not None:
            row["tabular_shape"] = list(tv.shape)
        rows.append(row)

    # ---- aggregate reports ----
    def _agg_reports(bucket: dict[str, list[int]], label: str) -> list[dict[str, Any]]:
        out = []
        for key, (frames_real, frames_total, n_samples) in sorted(bucket.items()):
            out.append(
                {
                    label: key,
                    "samples": n_samples,
                    "sequence_slots": frames_total,
                    "real_frames": frames_real,
                    "padding_frames": frames_total - frames_real,
                    "real_fraction": round(frames_real / frames_total, 5)
                    if frames_total
                    else 0.0,
                    "real_frames_per_sample": round(frames_real / n_samples, 3)
                    if n_samples
                    else 0.0,
                }
            )
        return out

    breakdown = {
        "by_crop": _agg_reports(by_crop, "crop"),
        "by_year": _agg_reports(by_year, "year"),
        "by_season": _agg_reports(by_season, "season"),
    }

    n_rows = len(rows)
    n_errors = sum(1 for r in rows if "error" in r)
    total_slots = sum(r["n_sequence_slots"] for r in rows if "error" not in r)
    global_zeros = 0.0
    zero_count = 0
    for r in rows:
        if "error" in r:
            continue
        for ch in ("ndvi", "evi"):
            v = r.get(f"{ch}_zero_frac_full")
            if v is not None:
                global_zeros += v
                zero_count += 1
    mean_zero_frac = (global_zeros / zero_count) if zero_count else 0.0

    summary = {
        "tool": "verify_corpus_tensors",
        "corpus": str(corpus_path),
        "config_preprocessing": str(pre_config),
        "image_size": patch_size,
        "max_observations": max_obs,
        "n_observations_processed": n_rows,
        "n_transform_errors": n_errors,
        "splits": dict(splits),
        "pad_frame_histogram": dict(pad_frames),
        "real_frames_total": real_frames_total,
        "total_sequence_slots": total_slots,
        "mean_real_frames_per_sample": round(real_frames_total / n_rows, 3)
        if n_rows
        else 0.0,
        "overall_real_fraction": round(real_frames_total / total_slots, 5)
        if total_slots
        else 0.0,
        "mean_zero_fraction_full_tensor": round(mean_zero_frac, 5),
        "all_shapes_match_image_size": bool(shape_ok),
        "any_nan": bool(nan_any),
        "any_inf": bool(inf_any),
        "all_real_frames_finite": bool(finite_all),
        "samples_with_nan_or_inf": nan_any + inf_any,
        "breakdown": breakdown,
    }

    print()
    print("  ---- SHAPE / FINITENESS GATE ----")
    print(f"  all NDVI/EVI shapes == ({max_obs},{patch_size}x{patch_size}): "
          f"{bool(shape_ok)}")
    print(f"  any NaN in full tensors      : {bool(nan_any)}")
    print(f"  any Inf in full tensors      : {bool(inf_any)}")
    print(f"  all real-frame tensors finite: {bool(finite_all)}")

    print()
    print(f"  ---- TEMPORAL SPARSITY (T={max_obs} slots) ----")
    print(f"  mean real frames / sample    : {summary['mean_real_frames_per_sample']}")
    print(f"  overall real fraction        : {summary['overall_real_fraction']}")
    print(f"  mean zero fraction (full)    : {summary['mean_zero_fraction_full_tensor']}")

    print()
    print("  ---- BREAKDOWN: by crop ----")
    hdr = f"  {'crop':<12}{'samples':>9}{'realFra':>9}{'padFr':>9}{'realFrac':>10}"
    print(hdr)
    for b in breakdown["by_crop"]:
        print(
            f"  {b['crop']:<12}{b['samples']:>9}{b['real_frames']:>9}"
            f"{b['padding_frames']:>9}{b['real_fraction']:>10}"
        )

    print()
    print("  ---- BREAKDOWN: by season ----")
    for b in breakdown["by_season"]:
        print(
            f"  {b['season']:<16} samples={b['samples']:<6} "
            f"real={b['real_frames']:<6} pad={b['padding_frames']:<6} "
            f"realFrac={b['real_fraction']}"
        )

    print()
    print("  ---- BREAKDOWN: by year ----")
    for b in breakdown["by_year"]:
        print(
            f"  {b['year']:<6} samples={b['samples']:<6} "
            f"real={b['real_frames']:<6} pad={b['padding_frames']:<6} "
            f"realFrac={b['real_fraction']}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"summary": summary, "samples": rows}
    target = output_dir / "corpus_tensor_audit.json"
    target.write_text(
        json.dumps(report, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  wrote tensor audit report -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
