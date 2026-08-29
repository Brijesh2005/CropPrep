"""R5.2 Task 6: split-composition diagnostics over the frozen corpus.

Verifies the actual train/val/test composition carried by the FROZEN corpus:
the taluk-level spatial split recorded in each observation's
``provenance.split`` (train Belthangady+Mangalore+Puttur / val Bantwal /
test Sullia), NOT a re-split of the accepted corpus at verify time.

Why: re-splitting with the temporal strategy at verify time produced the
8601/0/1518 contradiction (all crop-labelled 2018–2019 samples land in train,
val/test are exclusively DK district-level rows), while the frozen corpus ships
5561/2267/2291. The provenance split is authoritative: it is the split the
corpus was built with, and the same one the training loop consumes.

Run from repo root::

    python training/kaggle/scripts/verify_split_composition.py \\
        --corpus training/kaggle/outputs/reports/corpus.json \\
        --output training/artifacts/input_verification
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
# R5.3: ensure ``import training`` resolves to THIS repository before any
# package import. Running the script from the repo root puts the script's own
# directory (training/kaggle/scripts) on sys.path[0], NOT the repo root, which
# previously raised ``ModuleNotFoundError: No module named 'training'`` on
# Kaggle only (where CWD is also not on sys.path).
import sys  # noqa: E402

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.stam.observation import AgriculturalObservation  # noqa: E402


def _load_observations(path: Path) -> list[AgriculturalObservation]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    accepted = []
    for sample in raw["samples"]:
        if sample["status"] == "accepted" and sample.get("observation") is not None:
            accepted.append(AgriculturalObservation.model_validate(sample["observation"]))
    return accepted


def _split_from_provenance(
    observations: list[AgriculturalObservation],
) -> tuple[list[AgriculturalObservation], list[AgriculturalObservation], list[AgriculturalObservation]]:
    """Split by the frozen taluk split recorded in ``provenance.split``.

    This is the authoritative split (see module docstring). Unknown splits are
    reported separately — they must never be silently merged into train.
    """
    train: list[AgriculturalObservation] = []
    val: list[AgriculturalObservation] = []
    test: list[AgriculturalObservation] = []
    unknown: list[AgriculturalObservation] = []
    for obs in observations:
        split = obs.provenance.get("split", "unknown")
        if split == "train":
            train.append(obs)
        elif split == "val":
            val.append(obs)
        elif split == "test":
            test.append(obs)
        else:
            unknown.append(obs)
    return train, val, test, unknown


def _sequence_counts(obs: AgriculturalObservation) -> dict[str, Any]:
    """Real (paired or single-index) imagery dates for one observation.

    R5.3: the frozen corpus carries exactly one survey date per record, so
    every observation resolves to exactly one real temporal slot; the other
    ``T - 1`` slots are zero-filled padding. This helper quantifies that per
    split so the verifier reports imagery availability from the SAME frozen
    corpus / provenance the training loop consumed.
    """
    pairs = getattr(getattr(obs, "sequence", None), "pairs", None) or []
    ndvi = sum(1 for p in pairs if p.ndvi is not None)
    evi = sum(1 for p in pairs if p.evi is not None)
    return {
        "real_ndvi": ndvi,
        "real_evi": evi,
        "lights": ndvi + evi,
        "has_imagery": (ndvi + evi) > 0,
    }


def _imagery_stats(name: str, obs: list[AgriculturalObservation]) -> dict[str, Any]:
    ndvi = sum(_sequence_counts(o)["real_ndvi"] for o in obs)
    evi = sum(_sequence_counts(o)["real_evi"] for o in obs)
    no_imagery = sum(1 for o in obs if not _sequence_counts(o)["has_imagery"])
    return {
        "samples": len(obs),
        "real_ndvi_slots": ndvi,
        "real_evi_slots": evi,
        "real_slots_per_sample": round((ndvi + evi) / (2 * max(len(obs), 1)), 4),
        "samples_without_imagery": no_imagery,
    }


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
        "imagery": _imagery_stats(name, obs),
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

    # The authoritative split is the frozen taluk split in provenance. Splitting
    # again at verify time (temporal strategy) gives the invalid 8601/0/1518
    # composition and leaks across the still-served 2025 images.
    train, val, test, unknown = _split_from_provenance(obs)
    print(f"[verify_split] frozen provenance split: "
          f"train={len(train)} val={len(val)} test={len(test)}")
    if unknown:
        print(f"[verify_split] WARNING: {len(unknown)} observations have an "
              f"unknown provenance.split — these never enter any split")
        print(f"[verify_split] FATAL: unknown-split samples found (no silent "
              f"train assignment)")
        for u in unknown:
            print(f"  - record_id={u.provenance.get('record_id')} "
                  f"taluk={u.location.admin.taluk} split={u.provenance.get('split')}")
        return 2

    for name, part in (("train", train), ("val", val), ("test", test)):
        st = _split_stats(name, part)
        print(f"\n--- {name} ---")
        for k, v in st.items():
            print(f"  {k}: {v}")

    # Tabular uniqueness per split (fit on train only — no leakage).
    from training.preprocessing import Preprocessor, load_preprocessing_config

    pre = Preprocessor(
        load_preprocessing_config(
            args.config or (_REPO_ROOT / "training" / "config" / "preprocessing.yaml")
        )
    )
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
        "split_source": "provenance.split (frozen taluk-level spatial split)",
        "splits": {
            name: _split_stats(name, part)
            for name, part in (("train", train), ("val", val), ("test", test))
        },
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_unknown": len(unknown),
        "no_val0_contradiction": len(val) > 0 and len(test) > 0,
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
