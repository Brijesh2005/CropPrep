"""R5.2 Task 3/7 verification: what tensors does the model actually receive?

Reconstructs accepted AgriculturalObservations from a corpus.json and runs the
REAL TabularPipeline + LabelPipeline (same code the trainer uses) over them to
verify: feature count/shape, per-feature statistics, missing/constant fill for
DK district-level samples, crop-class distribution, and yield-scale behaviour.

Run from repo root::

    python training/kaggle/scripts/verify_model_input.py \\
        --corpus kaggle_runs/train-dk-bridge/reports/CropPrep/training/kaggle/outputs/reports/corpus.json \\
        --output training/artifacts/input_verification
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

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
            print(f"[verify_model_input] removing shadowing sys.path entry: {entry}")
            sys.path.remove(entry)


_add_repo_root(_REPO_ROOT)

from training.preprocessing import Preprocessor, load_preprocessing_config  # noqa: E402
from training.stam.observation import AgriculturalObservation  # noqa: E402


def _load_observations(path: Path) -> list[AgriculturalObservation]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    accepted = []
    for sample in raw["samples"]:
        if sample["status"] == "accepted" and sample.get("observation") is not None:
            accepted.append(AgriculturalObservation.model_validate(sample["observation"]))
    return accepted


def _fmt_table(data: list[list]) -> str:
    ncols = max(len(r) for r in data)
    widths = [max(len(str(r[i])) if i < len(r) else 0 for r in data) for i in range(ncols)]
    rows = [" | ".join(str(r[i]).ljust(widths[i]) if i < len(r) else " " * widths[i]
                       for i in range(ncols)) for r in data]
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cropfusion-verify-model-input")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default=None, help="preprocessing.yaml path")
    args = parser.parse_args(argv)

    obs = _load_observations(Path(args.corpus))
    print(f"[verify_model_input] accepted observations loaded: {len(obs)}")

    config = (
        Preprocessor.from_config(args.config).config
        if args.config
        else load_preprocessing_config(_REPO_ROOT / "training" / "config" / "preprocessing.yaml")
    )
    pre = Preprocessor(config)
    # No extractor => image tensors unavailable; verify tabular + label paths.
    accepted, _ = pre.filter(obs)
    print(f"[verify_model_input] after quality filter: {len(accepted)}")
    pre.fit(accepted)

    tab = pre.tabular
    print("\n=== TabularPipeline summary ===")
    print("numeric_features:", tab.numeric_features)
    print("categorical_features:", tab.categorical_features)
    print("feature_names:", tab.feature_names)
    print("feature_count:", tab.summary()["feature_count"])
    print("missing_columns:", tab.missing_columns)
    print("dropped_constant:", tab.dropped_constant)
    print("missing_fill:", tab.missing_fill)

    # Transform every accepted observation (no extractor needed for tabular).
    vectors, crops, yields, sources, levels = [], [], [], [], []
    nan_rows = []
    for o in accepted:
        vec = tab.transform(o)
        vectors.append(np.asarray(vec.numpy(), dtype="float64"))
        yields.append(o.yield_value)
        crops.append(o.crop)
        sources.append(str(o.tabular.source_path or "").split("/")[-1])
        levels.append(o.tabular.matched_level)
        if not np.isfinite(vec.numpy()).all():
            nan_rows.append(o.observation_id)

    matrix = np.vstack(vectors)
    print("\n=== Actual tabular tensor (model input) ===")
    print("shape:", matrix.shape)
    print("finite rows:", np.isfinite(matrix).all(axis=1).sum(), "non-finite rows:", len(nan_rows))
    print("min/max per column:")
    for i, name in enumerate(tab.feature_names):
        col = matrix[:, i]
        print(f"  {name:<24} min={col.min():>10.4f} max={col.max():>10.4f} "
              f"std={col.std():>10.4f}")

    # Per-source breakdown: are DK district samples constant-filled?
    print("\n=== Per-source tabular vectors (mean per column) ===")
    by_source: dict[str, list[np.ndarray]] = {}
    for vec, src in zip(vectors, sources):
        by_source.setdefault(src, []).append(vec)
    rows = []
    for src, vecs in by_source.items():
        m = np.mean(np.vstack(vecs), axis=0)
        rows.append([src, len(vecs), np.round(m, 4)])
    print(_fmt_table([["source", "n", "mean_vec"]] + rows))

    # Distinct vectors: how many unique tabular rows?
    uniq = np.unique(matrix, axis=0)
    print(f"\nunique tabular vectors: {len(uniq)} / {len(matrix)}")

    # Label pipeline.
    print("\n=== LabelPipeline summary ===")
    print("num_classes:", pre.label.num_classes)
    print("classes:", pre.label.crop_encoder.classes_ if pre.label.crop_encoder else None)
    yield_scaler = pre.label.yield_scaler
    if yield_scaler is not None:
        print("yield_scaler:", yield_scaler.to_dict())
    if pre.label.yield_scale_stats:
        print("yield_scale_stats:", pre.label.yield_scale_stats)
    if pre.label.warnings:
        print("yield scale warnings:")
        for warning in pre.label.warnings:
            print("  -", warning)
    y = np.asarray([v for v in yields if v is not None], dtype="float64")
    print("yield raw: n=", len(y), "min=", round(float(y.min()), 3),
          "max=", round(float(y.max()), 3), "mean=", round(float(y.mean()), 3))
    print("crop distribution:", dict(Counter(crops)))
    print("yield by matched_level:")
    for lvl in sorted(set(levels)):
        vals = [v for v, l in zip(yields, levels) if l == lvl and v is not None]
        if vals:
            print(f"  {lvl:<8} n={len(vals):>4} min={min(vals):>10.3f} "
                  f"max={max(vals):>10.3f} mean={sum(vals)/len(vals):>10.3f}")

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        report = {
            "observations": len(accepted),
            "tabular": {
                "numeric_features": tab.numeric_features,
                "categorical_features": tab.categorical_features,
                "feature_names": tab.feature_names,
                "feature_count": tab.summary()["feature_count"],
                "missing_columns": tab.missing_columns,
                "dropped_constant": tab.dropped_constant,
                "missing_fill": tab.missing_fill,
                "matrix_shape": list(matrix.shape),
                "unique_vectors": int(len(uniq)),
                "finite_rows": int(np.isfinite(matrix).all(axis=1).sum()),
                "per_source_count": {k: len(v) for k, v in by_source.items()},
            },
            "label": {
                "num_classes": pre.label.num_classes,
                "classes": pre.label.crop_encoder.classes_ if pre.label.crop_encoder else None,
                "yield_scaler": yield_scaler.to_dict() if yield_scaler else None,
                "yield_scale_stats": pre.label.yield_scale_stats,
                "warnings": pre.label.warnings,
                "crop_distribution": dict(Counter(crops)),
                "yield_min": float(y.min()) if len(y) else None,
                "yield_max": float(y.max()) if len(y) else None,
            },
        }
        (out / "input_verification.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"\n[verify_model_input] wrote {out / 'input_verification.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
