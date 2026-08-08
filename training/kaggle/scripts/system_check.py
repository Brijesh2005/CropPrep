"""Kaggle system check — full infrastructure validation (R2.1).

Runs the Training Validator over the Kaggle environment + workspace and
generates every report. Used by ``notebooks/system_check.ipynb`` and CI-style
readiness gates. Pure infrastructure — no training logic.

Run on Kaggle::

    !python training/kaggle/scripts/system_check.py

Run on a research machine::

    python training/kaggle/scripts/system_check.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _add_repo_root(repo_root: Path) -> None:
    """Force the repository root to the front of ``sys.path``.

    Called before any ``training.*`` import so ``import training`` always
    resolves to THIS repository — a stale ``/kaggle/working/training`` folder
    or a working-directory entry must not shadow the real package.
    """
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
            print(f"[system_check] removing shadowing sys.path entry: {entry}")
            sys.path.remove(entry)


_add_repo_root(_REPO_ROOT)

from training.kaggle.config import (
    load_kaggle_config,
    load_logging_config,
    load_paths_config,
    WorkspaceLayout,
)
from training.kaggle.environment import EnvironmentManager
from training.kaggle import reports
from training.kaggle.logging import TrainingLogger
from training.kaggle.validation import TrainingValidator
from training.kaggle.workspace import WorkspaceManager


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cropfusion-system-check",
        description="Kaggle infrastructure validation + reports",
    )
    parser.add_argument("--repo-root", default=str(_REPO_ROOT))
    parser.add_argument(
        "--paths-config",
        default=str(_REPO_ROOT / "training" / "config" / "paths.yaml"),
    )
    parser.add_argument(
        "--dataset-config",
        default=str(_REPO_ROOT / "training" / "config" / "dataset.yaml"),
        help="Path to the Dataset Manager YAML config (provider manifests)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write reports + validation JSON here (default: workspace outputs)",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    _add_repo_root(repo_root)

    paths = load_paths_config(Path(args.paths_config))
    kaggle_cfg = load_kaggle_config()
    logging_cfg = load_logging_config()

    environment = EnvironmentManager(repo_root)
    env_report = environment.report()
    layout = WorkspaceLayout.resolve(paths, repo_root=repo_root)
    logger = TrainingLogger(logging_cfg, log_dir=layout.logs).setup()
    workspace = WorkspaceManager(layout)
    workspace.create()

    provider_manifests = _provider_manifests(repo_root, args)
    validator = TrainingValidator(paths, layout, env_report)
    validation = validator.validate(provider_manifests=provider_manifests)

    output = Path(args.output) if args.output else workspace.output_path("reports")
    report_set = {
        "environment": reports.environment_report(env_report),
        "gpu": reports.gpu_report(env_report),
        "dependency": reports.dependency_report(env_report),
        "storage": reports.storage_report(workspace, env_report),
        "workspace": reports.workspace_report(workspace),
        "configuration": reports.configuration_report(
            paths, layout, extra={"kaggle": kaggle_cfg.model_dump()}
        ),
        "validation": validation.to_dict(),
    }
    written = reports.write_reports(report_set, output)
    for name, path in written.items():
        print(f"[system_check] wrote {name} report -> {path}")

    logger.log_system(
        "system_check_complete",
        passed=validation.passed,
        severity_summary=validation.by_severity(),
    )
    print(f"[system_check] validation passed={validation.passed}")
    for issue in validation.issues:
        print(f"  [{issue.severity.value:8s}] {issue.code}: {issue.message}")
    return 0 if validation.passed else 1


def _provider_manifests(repo_root: Path, args) -> dict | None:
    """Optional Dataset Manager provider manifests (skipped when unavailable)."""
    try:
        from training.dataset_manager import DatasetManager, load_settings

        settings = load_settings(Path(args.dataset_config))
        manager = DatasetManager(settings)
        try:
            return manager.provider_manifests()
        finally:
            manager.close()
    except Exception:  # noqa: BLE001 - system check must never crash
        return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
