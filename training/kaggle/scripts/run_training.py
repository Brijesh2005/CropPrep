"""Kaggle training runner — pipeline orchestration (R2.1, no training yet).

This script initialises the complete CropFusion training pipeline and verifies
that every component is wired and ready. It performs **no training** — model
construction, data assembly and the training loop arrive in a later phase.

It does:

1. Initialise environment (runtime / system / GPU / dependencies) + logging.
2. Load every configuration file (dataset / training / model / validation /
   kaggle / paths / logging) with ``KAGGLE_*`` / platform env overrides.
3. Initialise providers + the :class:`DatasetManager`.
4. Initialise STAM, preprocessing, trainer, evaluator and exporter component
   readiness (constructor descriptors — no model / no data required).
5. Create the workspace (logs / outputs / checkpoints / cache / configs),
   checkpoint manager + training cache.
6. Generate the orchestration report (config, providers, components,
   validation) as JSON.

Run on Kaggle::

    !python training/kaggle/scripts/run_training.py

Run on a research machine::

    python training/kaggle/scripts/run_training.py --repo-root .
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _add_repo_root(repo_root: Path) -> None:
    """Force the repository root to the front of ``sys.path``.

    Called before any ``training.*`` import so ``import training`` always
    resolves to THIS repository — a stale ``/kaggle/working/training`` folder
    or a working-directory entry must not shadow the real package.
    """
    import sys

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
            print(f"[run_training] removing shadowing sys.path entry: {entry}")
            sys.path.remove(entry)


_add_repo_root(_REPO_ROOT)

from training.kaggle.config import (
    load_kaggle_config,
    load_logging_config,
    load_paths_config,
    WorkspaceLayout,
)
from training.kaggle.environment import EnvironmentManager
from training.kaggle.logging import TrainingLogger
from training.kaggle.validation import TrainingValidator
from training.kaggle.workspace import WorkspaceManager


def _component_descriptor(cls: type) -> dict[str, Any]:
    """Constructor signature + module for an orchestration component."""
    sig = inspect.signature(cls.__init__)
    params = [
        name
        for name, param in sig.parameters.items()
        if name != "self" and param.default is inspect.Parameter.empty
    ]
    return {
        "class": f"{cls.__module__}.{cls.__name__}",
        "required_init_args": params,
        "instantiated": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cropfusion-run-training",
        description="Training pipeline orchestration readiness check (no training)",
    )
    parser.add_argument("--repo-root", default=str(_REPO_ROOT))
    parser.add_argument(
        "--paths-config",
        default=str(_REPO_ROOT / "training" / "config" / "paths.yaml"),
    )
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
        "--validation-config",
        default=str(_REPO_ROOT / "training" / "config" / "validation.yaml"),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the orchestration report JSON here (default: workspace outputs)",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    _add_repo_root(repo_root)

    # 1. Configuration.
    paths = load_paths_config(Path(args.paths_config))
    kaggle_cfg = load_kaggle_config()
    logging_cfg = load_logging_config()

    # 2. Environment + logging + workspace.
    environment = EnvironmentManager(repo_root)
    env_report = environment.report()
    layout = WorkspaceLayout.resolve(paths, repo_root=repo_root)
    logger = TrainingLogger(logging_cfg, log_dir=layout.logs).setup()
    workspace = WorkspaceManager(layout)
    workspace.create()
    logger.log_experiment(
        "orchestration_start",
        repo_root=str(repo_root),
        python=env_report["system"].get("python_version"),
        gpu=env_report["gpu"].get("available"),
    )

    report: dict[str, Any] = {
        "environment": env_report,
        "configuration": _config_report(args, paths, kaggle_cfg),
        "workspace": workspace.report(),
    }

    # 3. Providers + Dataset Manager.
    from training.dataset_manager import DatasetManager, load_settings

    settings = load_settings(Path(args.dataset_config))
    manager = DatasetManager(settings)
    try:
        manifests = manager.provider_manifests()
        report["dataset_manager"] = {
            "providers": manifests,
            "tabular_datasets": manager.tabular_names(),
        }
    finally:
        manager.close()

    # 4. Pipeline components (constructor descriptors — no model/data).
    from training.training import Evaluator, Trainer, TrainingConfig
    from training.training.config import load_training_config as load_training_cfg
    from training.models.config import load_model_config as load_model_cfg
    from training.preprocessing import Preprocessor

    training_cfg = load_training_cfg(Path(args.training_config))
    model_cfg = load_model_cfg(Path(args.model_config))
    components = {
        "training_config": {
            "name": training_cfg.name,
            "device": training_cfg.general.device,
            "epochs": training_cfg.train.epochs,
            "checkpoint_dir": training_cfg.checkpoint.directory,
        },
        "model_config": {"name": model_cfg.name},
        "preprocessor": _component_descriptor(Preprocessor),
        "trainer": _component_descriptor(Trainer),
        "evaluator": _component_descriptor(Evaluator),
        "stam": _component_descriptor(_stam_class()),
    }
    components["trainer"]["config_loaded"] = isinstance(
        training_cfg, TrainingConfig
    )
    report["components"] = components

    # 5. Validation (config / python / gpu / deps / folders / disk).
    validator = TrainingValidator(paths, layout, env_report)
    validation = validator.validate(provider_manifests=manifests)
    report["validation"] = validation.to_dict()
    logger.log_experiment(
        "orchestration_complete",
        passed=validation.passed,
        severity_summary=validation.by_severity(),
    )

    output = Path(args.output) if args.output else workspace.output_path("reports")
    output.mkdir(parents=True, exist_ok=True)
    target = output / "orchestration.json"
    target.write_text(
        json.dumps(report, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[run_training] wrote orchestration report -> {target}")
    print(
        f"[run_training] validation passed={validation.passed} "
        f"({validation.by_severity()})"
    )
    return 0


def _stam_class():
    from training.stam import STAM

    return STAM


def _config_report(args, paths, kaggle_cfg) -> dict[str, Any]:
    return {
        "paths": paths.model_dump(),
        "kaggle": kaggle_cfg.model_dump(),
        "dataset_config": args.dataset_config,
        "training_config": args.training_config,
        "model_config": args.model_config,
        "validation_config": args.validation_config,
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
