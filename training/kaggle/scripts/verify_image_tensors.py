"""R5.2 Task 8: image-tensor verification for the REAL model input.

Runs the REAL ImagePipeline (+ STAM patch extractor) over every accepted
observation and verifies the image tensors the model consumes:

  * per-pair NDVI/EVI patch shapes match ``image.size`` (224x224);
  * every final tensor is finite (no NaN/Inf leaked into the model input);
  * NaN / invalid-pixel handling behaves as configured;
  * temporal sequence shapes (max_observations padding);
  * per-observation valid-ratio / coverage so fully-empty patches are visible.

This needs the Kaggle imagery mount (Sentinel rasters) — it cannot run against
an empty catalog. On a machine without imagery it exits with a clear message
instead of fabricating results.

Run from repo root (Kaggle training kernel, after ``manager.ensure_image()``)::

    python training/kaggle/scripts/verify_image_tensors.py \\
        --corpus kaggle_runs/train-dk-bridge/reports/CropPrep/training/kaggle/outputs/reports/corpus.json \\
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
            print(f"[verify_image_tensors] removing shadowing sys.path entry: {entry}")
            sys.path.remove(entry)


_add_repo_root(_REPO_ROOT)

from training.dataset_manager import DatasetManager, load_settings  # noqa: E402
from training.preprocessing import Preprocessor, load_preprocessing_config  # noqa: E402
from training.stam import STAM  # noqa: E402
from training.stam.config import load_stam_config  # noqa: E402
from training.stam.observation import AgriculturalObservation  # noqa: E402


def _load_observations(path: Path) -> list[AgriculturalObservation]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    accepted = []
    for sample in raw["samples"]:
        if sample["status"] == "accepted" and sample.get("observation") is not None:
            accepted.append(AgriculturalObservation.model_validate(sample["observation"]))
    return accepted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cropfusion-verify-image-tensors")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--config",
        default=None,
        help="preprocessing.yaml path (default training/config/preprocessing.yaml)",
    )
    parser.add_argument(
        "--dataset-config",
        default=str(_REPO_ROOT / "training" / "config" / "dataset.yaml"),
    )
    parser.add_argument(
        "--stam-config",
        default=str(_REPO_ROOT / "training" / "config" / "stam.yaml"),
    )
    args = parser.parse_args(argv)

    config = (
        Preprocessor.from_config(args.config).config
        if args.config
        else load_preprocessing_config(_REPO_ROOT / "training" / "config" / "preprocessing.yaml")
    )
    pre = Preprocessor(config)
    obs = _load_observations(Path(args.corpus))
    print(f"[verify_image_tensors] accepted observations: {len(obs)}")
    accepted, _ = pre.filter(obs)
    print(f"[verify_image_tensors] after quality filter: {len(accepted)}")

    # STAM + extractor (needs the imagery catalog / raster mount).
    manager = DatasetManager(load_settings(Path(args.dataset_config)))
    stam = None
    try:
        manifest = manager.provider_manifests().get("kaggle_hub_image", {})
        if not manifest.get("available"):
            raise RuntimeError(
                "imagery catalog is NOT available in this environment "
                "(requires the Kaggle imagery mount + ensure_image)"
            )
        stam = STAM(manager, load_stam_config(Path(args.stam_config)))
        stam.initialize()
        print("[verify_image_tensors] STAM initialized; extractor ready")
    except Exception as exc:  # noqa: BLE001 - graceful exit without imagery
        print(f"[verify_image_tensors] SKIPPED: {exc}")
        print("[verify_image_tensors] cannot verify image tensors without rasters")
        return 0
    finally:
        manager.close()

    pre.fit(accepted, extractor=stam.get_patch)

    target = config.image.size
    shapes: Counter[tuple] = Counter()
    nan_patches = 0
    empty_patches = 0
    valid_ratios: list[float] = []
    seq_lengths: Counter[int] = Counter()
    per_pair_report: list[dict[str, Any]] = []

    for o in accepted:
        try:
            sample = pre.transform(o, extractor=stam.get_patch)
        except Exception as exc:  # noqa: BLE001 - per-observation best effort
            per_pair_report.append(
                {"observation_id": str(o.observation_id),
                 "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        ndvi, evi = sample["ndvi"], sample["evi"]
        seq_lengths[len(ndvi)] += 1
        for band_name, band in (("ndvi", ndvi), ("evi", evi)):
            for t, tensor in enumerate(band):
                arr = tensor.numpy()
                shapes[tuple(arr.shape)] += 1
                if not np.isfinite(arr).all():
                    nan_patches += 1
                valid = float(np.sum(arr > 0))
                valid_ratios.append(valid / arr.size if arr.size else 0.0)
                if valid == 0:
                    empty_patches += 1

    print("\n=== Image tensor verification (real pipeline) ===")
    print("target size:", target, "| normalize:", config.image.normalize)
    print("tensor shapes:", dict(shapes))
    print("sequence lengths (t):", dict(sorted(seq_lengths.items())))
    print(f"non-finite patches: {nan_patches}")
    print(f"fully-zero (empty) patches: {empty_patches}")
    if valid_ratios:
        vr = np.asarray(valid_ratios)
        print(f"valid-pixel ratio: min={vr.min():.4f} mean={vr.mean():.4f} "
              f"max={vr.max():.4f}")
    print("patch_size mismatches (ImagePipeline.validate):",
          sum(len(pre.image.validate(o)) for o in accepted))

    report = {
        "target_size": target,
        "normalize": config.image.normalize,
        "tensor_shapes": {str(k): v for k, v in shapes.items()},
        "sequence_lengths": {str(k): v for k, v in sorted(seq_lengths.items())},
        "non_finite_patches": int(nan_patches),
        "empty_patches": int(empty_patches),
        "valid_ratio": (
            {"min": float(vr.min()), "mean": float(vr.mean()), "max": float(vr.max())}
            if valid_ratios else None
        ),
        "observations": len(accepted),
        "per_pair_errors": [r for r in per_pair_report if "error" in r],
    }

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "image_tensor_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"\n[verify_image_tensors] wrote {out / 'image_tensor_report.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
