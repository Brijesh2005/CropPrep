"""R5.2 Task 6: temporal-split composition diagnostics.

Verifies the actual train/val/test composition produced by the REAL
``split_observations`` (temporal strategy, config from preprocessing.yaml) over
the accepted corpus: year ranges, crop-label support, yield-label support,
constant-tabular/constant-yield detection per split.

The concern: all 74 crop-labeled samples are data_season (2018-2019) -> the
temporal split places them all in train; val/test (2022/2023) are exclusively
DK district-level samples with no crop labels, identical mean-filled tabular
vectors, and constant yields. This makes crop metrics undefined and yield R2
meaningless on val/test.

Run from repo root::

    python training/kaggle/scripts/verify_split_composition.py \\
        --corpus kaggle_runs/train-dk-bridge/reports/CropPrep/training/kaggle/outputs/reports/corpus.json \\
        --output training/artifacts/input_verification
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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
            print(f"[verify_split] removing shadowing sys.path entry: {entry}")
            sys.path.remove(entry)


_add_repo_root(_REPO_ROOT)

from training.preprocessing import (  # noqa: E402
    Preprocessor,
    load_preprocessing_config,
    split_observations,
)
from training.stam.observation import AgriculturalObservation  # noqa: E402


def _load_observations(path: Path) -> list[AgriculturalObservation]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    accepted = []
    for sample in raw["samples"]:
        if sample["status"] == "accepted" and sample.get("observation") is not None:
            accepted.append(AgriculturalObservation.model_validate(sample["observation"]))
    return accepted


def _split_stats(name: str, obs: list[AgriculturalObservation]) -> dict[str, Any]:
    years = [o.temporal.year for o in obs]
    crops = [o.crop for o in obs]
    yields = [o.yield_value for o in obs if o.yield_value is not None]
    levels = [o.tabular.matched_level for o in obs]
    by_level = Counter(levels)
    crop_labeled = sum(1 for c in crops if c is not None)
    stats = {
        "n": len(obs),
        "years": sorted(set(years)),
        "crop_labeled": crop_labeled,
        "yield_present": len(yields),
        "matched_levels": dict(by_level),
        "yield_min": float(min(yields)) if yields else None,
        "yield_max": float(max(yields)) if yields else None,
    }
    if yields:
        y = np.asarray(yields, dtype="float64")
        stats["yield_std"] = round(float(y.std()), 4)
        stats["yield_distinct"] = int(len(set(round(float(v), 4) for v in y)))
        stats["yield_near_constant"] = bool(y.std() < 1e-4)
    # Tabular uniqueness needs the fitted pipeline -> done separately in main().
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cropfusion-verify-split")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    obs = _load_observations(Path(args.corpus))
    print(f"[verify_split] accepted observations: {len(obs)}")

    config = (
        Preprocessor.from_config(args.config).config
        if args.config
        else load_preprocessing_config(_REPO_ROOT / "training" / "config" / "preprocessing.yaml")
    )
    pre = Preprocessor(config)
    accepted, _ = pre.filter(obs)
    print(f"[verify_split] after quality filter: {len(accepted)}")

    split_cfg = config.split
    print(f"\n=== Split config: strategy={split_cfg.strategy} "
          f"ratios=({split_cfg.train_ratio},{split_cfg.val_ratio},{split_cfg.test_ratio}) "
          f"test_years={split_cfg.test_years} val_years={split_cfg.val_years}")

    train, val, test = split_observations(accepted, split_cfg)
    for name, part in (("train", train), ("val", val), ("test", test)):
        st = _split_stats(name, part)
        print(f"\n--- {name} ---")
        for k, v in st.items():
            print(f"  {k}: {v}")

    # Fit on train only (no leakage) -> transform all -> tabular uniqueness per split.
    pre.fit(train)
    for name, part in (("train", train), ("val", val), ("test", test)):
        vectors = []
        for o in part:
            vec = pre.tabular.transform(o)
            vectors.append(np.asarray(vec.numpy(), dtype="float64"))
        m = np.vstack(vectors) if vectors else np.empty((0, 0))
        unique = len(np.unique(m, axis=0)) if m.size else 0
        print(f"\n{name}: tabular unique vectors = {unique} / {len(part)}")

    report = {
        "split_config": {
            "strategy": split_cfg.strategy,
            "train_ratio": split_cfg.train_ratio,
            "val_ratio": split_cfg.val_ratio,
            "test_ratio": split_cfg.test_ratio,
            "test_years": split_cfg.test_years,
            "val_years": split_cfg.val_years,
        },
        "splits": {
            name: _split_stats(name, part)
            for name, part in (("train", train), ("val", val), ("test", test))
        },
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
    }

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "split_composition_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"\n[verify_split] wrote {out / 'split_composition_report.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
