"""R5.2 Task 5/7: yield-target scale-consistency diagnostics.

The accepted corpus mixes two yield UNITS into one regression target:
  - data_season (village-level)      -> true kg/ha  (62 .. 73730)
  - DK_Features (district-level)     -> Yield_Proxy_NPP (~0.5 .. 1.5)
Both flow into the SAME StandardScaler (LabelPipeline), which can wreck the
regression target's scale consistency and the loss balance.

This script quantifies the mismatch using the REAL corpus + the REAL
LabelPipeline and measures the impact of two candidate mitigations:
  (a) fit the scaler on ALL yields (current behaviour)
  (b) fit the scaler ONLY on kg/ha (data_season) yields
      (DK proxy targets then scaled by the kg/ha scaler -> extreme negatives)

Run from repo root::

    python training/kaggle/scripts/verify_yield_scale.py \\
        --corpus kaggle_runs/train-dk-bridge/reports/CropPrep/training/kaggle/outputs/reports/corpus.json \\
        --output training/artifacts/input_verification
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
            print(f"[verify_yield_scale] removing shadowing sys.path entry: {entry}")
            sys.path.remove(entry)


_add_repo_root(_REPO_ROOT)

from training.preprocessing import Preprocessor, load_preprocessing_config  # noqa: E402
from training.preprocessing.transforms import StandardScaler  # noqa: E402
from training.stam.observation import AgriculturalObservation  # noqa: E402


def _load_observations(path: Path) -> list[AgriculturalObservation]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    accepted = []
    for sample in raw["samples"]:
        if sample["status"] == "accepted" and sample.get("observation") is not None:
            accepted.append(AgriculturalObservation.model_validate(sample["observation"]))
    return accepted


def _stats(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype="float64")
    return {
        "n": int(len(a)),
        "min": round(float(a.min()), 4),
        "max": round(float(a.max()), 4),
        "mean": round(float(a.mean()), 4),
        "std": round(float(a.std()), 4),
        "q50": round(float(np.median(a)), 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cropfusion-verify-yield-scale")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--config",
        default=None,
        help="preprocessing.yaml path (default training/config/preprocessing.yaml)",
    )
    args = parser.parse_args(argv)

    obs = _load_observations(Path(args.corpus))
    print(f"[verify_yield_scale] accepted observations: {len(obs)}")

    config = (
        Preprocessor.from_config(args.config).config
        if args.config
        else load_preprocessing_config(_REPO_ROOT / "training" / "config" / "preprocessing.yaml")
    )
    pre = Preprocessor(config)
    accepted, _ = pre.filter(obs)
    print(f"[verify_yield_scale] after quality filter: {len(accepted)}")
    pre.fit(accepted)

    # ---- Raw yield units by matched_level ------------------------------- #
    by_level: dict[str, list[float]] = defaultdict(list)
    raw_all = []
    for o in accepted:
        if o.yield_value is None:
            continue
        by_level[o.tabular.matched_level].append(float(o.yield_value))
        raw_all.append(float(o.yield_value))

    print("\n=== Raw yield by matched_level (UNITS) ===")
    level_stats: dict[str, dict] = {}
    for lvl in sorted(by_level):
        st = _stats(by_level[lvl])
        level_stats[lvl] = st
        print(f"  {lvl:<10} {st}")

    raw = np.asarray(raw_all, dtype="float64")
    print(f"\nALL raw yields: n={len(raw)} min={raw.min():.4f} max={raw.max():.4f} "
          f"max/min={raw.max() / max(raw.min(), 1e-9):.2e}")

    # ---- Current scaler (fit on ALL yields) ----------------------------- #
    scaler_all = StandardScaler().fit(raw.reshape(-1, 1))
    print("\n=== Scaler (a): fit on ALL yields (current training behaviour) ===")
    print("  mean:", round(float(scaler_all.mean_[0]), 4),
          " scale:", round(float(scaler_all.scale_[0]), 4))
    print("  scaled stats by level:")
    scaled_levels: dict[str, dict] = {}
    for lvl in sorted(by_level):
        vals = by_level[lvl]
        scaled = scaler_all.transform(np.asarray(vals, dtype="float64").reshape(-1, 1))[:, 0]
        st = _stats([float(v) for v in scaled])
        scaled_levels[lvl] = st
        print(f"    {lvl:<10} {st}")

    # ---- Scaler (b): fit ONLY on kg/ha (data_season) yields ------------- #
    village = by_level.get("village", [])
    print("\n=== Scaler (b): fit ONLY on data_season kg/ha yields ===")
    if len(village) < 2:
        print("  (not enough village yields to fit a scaler)")
        scaler_village = None
    else:
        scaler_village = StandardScaler().fit(np.asarray(village, dtype="float64").reshape(-1, 1))
        print("  mean:", round(float(scaler_village.mean_[0]), 4),
              " scale:", round(float(scaler_village.scale_[0]), 4))
        for lvl in sorted(by_level):
            vals = by_level[lvl]
            scaled = scaler_village.transform(
                np.asarray(vals, dtype="float64").reshape(-1, 1)
            )[:, 0]
            print(f"    {lvl:<10} {_stats([float(v) for v in scaled])}")

    # ---- Loss-balance impact: contribution per level --------------------- #
    # Huber loss is scale-invariant per sample only up to the delta term;
    # the KEY effect is that DK samples cluster at scaled ~ -0.39 (a single
    # standardised point) so the model trivially regresses them to a constant.
    print("\n=== Target-collapse assessment ===")
    print("  distinct raw yields:", len(set(round(float(v), 4) for v in raw)))
    print("  distinct scaled values (a):",
          len(set(round(float(v), 4) for v in scaler_all.transform(raw.reshape(-1, 1))[:, 0])))

    report = {
        "observations": len(accepted),
        "raw_yield_by_level": level_stats,
        "raw_all": _stats(raw_all),
        "scaler_fit_on_all": {
            "mean": float(scaler_all.mean_[0]),
            "scale": float(scaler_all.scale_[0]),
            "scaled_by_level": scaled_levels,
        },
        "scaler_fit_on_village_only": (
            {"mean": float(scaler_village.mean_[0]),
             "scale": float(scaler_village.scale_[0])}
            if scaler_village is not None
            else None
        ),
        "distinct_raw": int(len(set(round(float(v), 4) for v in raw))),
    }

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "yield_scale_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"\n[verify_yield_scale] wrote {out / 'yield_scale_report.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
