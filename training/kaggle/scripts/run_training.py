"""Kaggle training runner — end-to-end model training orchestration.

This script drives a complete CropFusion training run on Kaggle / a GPU
machine. It is **orchestration only** — the actual training engine lives in
``training/training`` (Experiment + Trainer + Evaluator) and the data assembly
lives in the Dataset Manager + STAM. This script:

1. Loads ``training/config/{dataset,training,model,validation}.yaml``.
2. Builds the :class:`DatasetManager` and materialises both data sources
   (tabular Git CSVs + Kaggle imagery).
3. Builds the observation set with :class:`STAM` (sole data access path).
4. Runs :func:`run_experiment` and writes the experiment report.

Run on Kaggle::

    !python training/kaggle/scripts/run_training.py \\
        --locations training/kaggle/locations.csv --epochs 100

Run on a research machine::

    python training/kaggle/scripts/run_training.py \\
        --repo-root . --locations training/kaggle/locations.csv
"""

from __future__ import annotations

import argparse
import csv
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

    settings = load_settings(dataset_config)
    return DatasetManager(settings)


def _ensure_data(manager) -> None:
    names = manager.tabular_names()
    print(f"[run_training] tabular datasets available: {len(names)}")
    manager.ensure_image()
    print("[run_training] imagery data ready")


def _load_locations(locations: str) -> list[tuple[float, float]]:
    """Read ``lon,lat`` rows from a CSV (header optional) or a "lon,lat" pair."""
    if locations.startswith("[") or "," in locations and ":" not in locations:
        lon, lat = locations.split(",")
        return [(float(lon), float(lat))]
    path = Path(locations)
    if not path.exists():
        raise FileNotFoundError(f"locations file not found: {path}")
    points: list[tuple[float, float]] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2 or not row[0] or not row[1]:
                continue
            try:
                points.append((float(row[0]), float(row[1])))
            except ValueError:
                continue
    if not points:
        raise ValueError(f"no valid lon/lat rows in {path}")
    return points


def _build_observations(manager, stam_config_path: Path, locations: str):
    from training.stam import STAM

    stam = STAM.from_config(manager, config_path=str(stam_config_path))
    stam.initialize()

    observations = []
    for lon, lat in _load_locations(locations):
        obs = stam.build_observation(lon, lat)
        observations.append(obs)
        print(
            f"[run_training] observation ({lon:.4f}, {lat:.4f}) "
            f"season={obs.temporal.season} score={obs.quality.overall_score}"
        )
    print(f"[run_training] observations built: {len(observations)}")
    return observations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cropfusion-run-training",
        description="End-to-end model training orchestration",
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
        "--model-config",
        default=str(_REPO_ROOT / "training" / "config" / "model.yaml"),
    )
    parser.add_argument(
        "--stam-config",
        default=str(_REPO_ROOT / "training" / "config" / "validation.yaml"),
    )
    parser.add_argument(
        "--locations",
        default=str(_REPO_ROOT / "training" / "kaggle" / "locations.csv"),
        help="CSV of lon,lat rows (or a single 'lon,lat' pair)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output", default=None, help="Write the report JSON here")
    args = parser.parse_args(argv)

    _add_repo_root(Path(args.repo_root))

    from training.training.config import load_training_config
    from training.models.config import load_model_config

    training = load_training_config(Path(args.training_config))
    model = load_model_config(Path(args.model_config))
    if args.epochs is not None:
        training.train.epochs = args.epochs

    manager = _build_manager(Path(args.repo_root), Path(args.dataset_config))
    try:
        _ensure_data(manager)
        observations = _build_observations(
            manager, Path(args.stam_config), args.locations
        )

        from training.training.experiment import run_experiment

        report = run_experiment(
            training_config=training,
            observations=observations,
            model_config=model,
            run_name=args.run_name,
        )
        payload = report.to_dict()
        print(json.dumps(payload, indent=2, default=str))
        if args.output:
            Path(args.output).write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
        return 0
    finally:
        manager.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
