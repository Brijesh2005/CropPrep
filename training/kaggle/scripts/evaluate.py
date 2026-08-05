"""Kaggle evaluation runner — evaluate a trained checkpoint (orchestration only).

Reloads a trained checkpoint, rebuilds the observation set through the Dataset
Manager + STAM, splits a hold-out test set with the preprocessing pipeline, and
runs the :class:`Evaluator`. The evaluation *engine* lives in
``training/training`` — this script only wires configs + data + model.

Run on Kaggle::

    !python training/kaggle/scripts/evaluate.py \\
        --checkpoint /kaggle/working/runs/run-name/checkpoint_epoch0100.pt

Run on a research machine::

    python training/kaggle/scripts/evaluate.py \\
        --repo-root . --checkpoint runs/run-name/checkpoint_epoch0100.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _add_repo_root(repo_root: Path) -> None:
    import sys

    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _build_manager(repo_root: Path, dataset_config: Path):
    from training.dataset_manager import DatasetManager, load_settings

    return DatasetManager(load_settings(dataset_config))


def _ensure_data(manager) -> None:
    manager.tabular_names()
    manager.ensure_image()


def _build_test_observations(manager, stam_config_path: Path, locations: str):
    from training.stam import STAM

    stam = STAM.from_config(manager, config_path=str(stam_config_path))
    stam.initialize()
    observations = []
    for lon, lat in _load_locations(locations):
        observations.append(stam.build_observation(lon, lat))
    print(f"[evaluate] observations built: {len(observations)}")
    return observations


def _load_locations(locations: str) -> list[tuple[float, float]]:
    import csv

    if "," in locations and ":" not in locations and len(locations.split(",")) == 2:
        lon, lat = locations.split(",")
        return [(float(lon), float(lat))]
    path = Path(locations)
    points: list[tuple[float, float]] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2 and row[0] and row[1]:
                try:
                    points.append((float(row[0]), float(row[1])))
                except ValueError:
                    continue
    if not points:
        raise ValueError(f"no valid lon/lat rows in {path}")
    return points


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cropfusion-evaluate", description="Evaluate a trained checkpoint"
    )
    parser.add_argument("--repo-root", default=str(_REPO_ROOT))
    parser.add_argument(
        "--dataset-config",
        default=str(_REPO_ROOT / "training" / "config" / "dataset.yaml"),
    )
    parser.add_argument(
        "--training-config",
        default=str(_REPO_ROOT / "training" / "config" / "training.yaml"),
    )
    parser.add_argument(
        "--stam-config",
        default=str(_REPO_ROOT / "training" / "config" / "validation.yaml"),
    )
    parser.add_argument(
        "--locations",
        default=str(_REPO_ROOT / "training" / "kaggle" / "locations.csv"),
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the .pt checkpoint")
    parser.add_argument("--output", default=None, help="Write the eval report JSON here")
    args = parser.parse_args(argv)

    _add_repo_root(Path(args.repo_root))

    from training.training.config import load_training_config
    from training.preprocessing import Preprocessor

    training = load_training_config(Path(args.training_config))

    manager = _build_manager(Path(args.repo_root), Path(args.dataset_config))
    try:
        _ensure_data(manager)
        observations = _build_test_observations(
            manager, Path(args.stam_config), args.locations
        )

        from training.models.factory import ModelFactory
        from training.preprocessing import build_dataloader, split_observations
        from training.training.evaluator import Evaluator

        model = ModelFactory.from_checkpoint(Path(args.checkpoint))

        preprocessor = Preprocessor().fit(observations)
        _, _, test = split_observations(observations, preprocessor.config.split)
        test_loader = build_dataloader(test, preprocessor.config, split="test")

        evaluator = Evaluator(
            model=model,
            metrics_config=training.metrics,
            input_map=_default_input_map,
        )
        result = evaluator.evaluate(test_loader)

        payload = {"metrics": result.metrics, "artifacts": {}}
        if hasattr(result, "to_dict"):
            payload = result.to_dict()
        print(json.dumps(payload, indent=2, default=str))
        if args.output:
            Path(args.output).write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
        return 0
    finally:
        manager.close()


def _default_input_map(batch: dict):
    inputs = {k: batch[k] for k in ("tabular", "ndvi", "evi", "temporal_mask") if k in batch}
    targets = {}
    if "crop_label" in batch:
        targets["crop"] = batch["crop_label"]
    if "yield_label" in batch:
        targets["yield"] = batch["yield_label"]
    return inputs, targets


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
