"""Kaggle bootstrap — environment + data-source readiness (orchestration only).

This script prepares a Kaggle Notebook (or GPU research machine) to run the
CropFusion Training Platform. It performs **no model training or dataset
implementation** — it only:

1. Verifies the Kaggle runtime, Python version, CUDA and GPU.
2. Loads and validates the platform configuration (paths / kaggle / logging)
   with ``KAGGLE_*`` environment overrides and ``extends`` inheritance.
3. Initializes the Training Logger (startup / system / experiment logs).
4. Creates the Kaggle workspace (logs, outputs, checkpoints, cache, configs).
5. Optionally ``pip install -e`` the first-party editable packages.
6. Builds the :class:`DatasetManager` from ``training/config/dataset.yaml``
   and verifies both provider manifests (tabular + image readiness).
7. Verifies repository integrity + tabular datasets (+ imagery with
   ``--ensure-data``).
8. Generates the environment, startup, configuration and workspace reports.

Run on Kaggle::

    !python training/kaggle/scripts/bootstrap.py --ensure-data

Run on a research machine::

    python training/kaggle/scripts/bootstrap.py \
        --repo-root . --dataset-config training/config/dataset.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from training.kaggle.config import (
    load_kaggle_config,
    load_logging_config,
    load_paths_config,
    WorkspaceLayout,
)
from training.kaggle.environment import EnvironmentManager
from training.kaggle.logging import TrainingLogger
from training.kaggle import reports
from training.kaggle.workspace import WorkspaceManager

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _prepend_to_path(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    print(f"[bootstrap] repository root on sys.path: {repo_root}")


def _install_editable(repo_root: Path, packages: list[str], skip: bool = False) -> None:
    if skip or not packages:
        print("[bootstrap] skipping editable install")
        return
    for package in packages:
        target = repo_root / package
        if not target.is_dir():
            print(f"[bootstrap] WARNING: editable package dir missing: {target}")
            continue
        print(f"[bootstrap] pip install -e {target}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-e", str(target)]
        )


def _verify_repo_integrity(repo_root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "repo_root": str(repo_root),
        "is_git_repo": (repo_root / ".git").exists(),
        "has_training": (repo_root / "training").is_dir(),
        "has_config": (repo_root / "training" / "config").is_dir(),
        "has_kaggle": (repo_root / "training" / "kaggle").is_dir(),
    }
    report["integrity_ok"] = all(
        report[k]
        for k in ("is_git_repo", "has_training", "has_config", "has_kaggle")
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cropfusion-bootstrap",
        description="Kaggle / GPU environment bootstrap (orchestration only)",
    )
    parser.add_argument(
        "--kaggle-config",
        default=str(_REPO_ROOT / "training" / "config" / "kaggle.yaml"),
        help="Path to training/config/kaggle.yaml",
    )
    parser.add_argument(
        "--paths-config",
        default=str(_REPO_ROOT / "training" / "config" / "paths.yaml"),
        help="Path to training/config/paths.yaml (workspace + registry)",
    )
    parser.add_argument(
        "--logging-config",
        default=str(_REPO_ROOT / "training" / "config" / "logging.yaml"),
        help="Path to training/config/logging.yaml",
    )
    parser.add_argument(
        "--dataset-config",
        default=str(_REPO_ROOT / "training" / "config" / "dataset.yaml"),
        help="Path to the Dataset Manager YAML config",
    )
    parser.add_argument(
        "--repo-root",
        default=str(_REPO_ROOT),
        help="Repository root (where training/ lives)",
    )
    parser.add_argument(
        "--skip-install", action="store_true", help="Do not pip install editable packages"
    )
    parser.add_argument(
        "--ensure-data", action="store_true", help="Materialise the Kaggle imagery dataset"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the bootstrap report JSON here (default: workspace outputs)",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    _prepend_to_path(repo_root)

    # 1-2. Configuration (paths + kaggle + logging) with env overrides.
    paths = load_paths_config(Path(args.paths_config))
    kaggle_cfg = load_kaggle_config(Path(args.kaggle_config))
    logging_cfg = load_logging_config(Path(args.logging_config))

    # 3. Environment capability report (runtime / system / GPU / deps).
    environment = EnvironmentManager(repo_root)
    env_report = environment.report()

    # 4. Training Logger + workspace.
    layout = WorkspaceLayout.resolve(paths, repo_root=repo_root)
    logger = TrainingLogger(logging_cfg, log_dir=layout.logs).setup()
    workspace = WorkspaceManager(layout)
    workspace.create()

    logger.log_startup(
        "bootstrap_start",
        repo_root=str(repo_root),
        is_kaggle=env_report["runtime"].get("is_kaggle"),
        python=env_report["system"].get("python_version"),
        gpu=env_report["gpu"].get("available"),
    )

    # 5. Editable installs.
    _install_editable(
        repo_root,
        kaggle_cfg.install.editable_packages,
        skip=args.skip_install,
    )

    # 6-7. Dataset Manager providers + data verification.
    from training.dataset_manager import DatasetManager, load_settings

    settings = load_settings(Path(args.dataset_config))
    manager = DatasetManager(settings)
    try:
        manifests = manager.provider_manifests()
        print("[bootstrap] provider manifests:")
        print(json.dumps(manifests, indent=2, default=str))
        image_ready = manifests["kaggle_hub_image"]["available"]
        tabular_ready = manifests["git_repository_tabular"]["available"]
        tabular_names = manager.tabular_names()
        logger.log_startup(
            "providers_verified",
            tabular_ready=tabular_ready,
            image_ready=image_ready,
            tabular_datasets=len(tabular_names),
        )
        if args.ensure_data:
            print("[bootstrap] ensuring imagery data (download-or-reuse)...")
            path = manager.ensure_image()
            print(f"[bootstrap] imagery materialised at {path}")
    finally:
        manager.close()

    # 8. Reports.
    integrity = _verify_repo_integrity(repo_root)
    report_set = {
        "environment": reports.environment_report(env_report),
        "gpu": reports.gpu_report(env_report),
        "dependency": reports.dependency_report(env_report),
        "storage": reports.storage_report(workspace, env_report),
        "workspace": reports.workspace_report(workspace),
        "configuration": reports.configuration_report(
            paths, layout, extra={"kaggle": kaggle_cfg.model_dump()}
        ),
        "bootstrap": {
            "generated_at": reports.now(),
            "repo_integrity": integrity,
            "tabular_datasets": len(tabular_names),
            "providers": manifests,
        },
    }
    output = Path(args.output) if args.output else workspace.output_path("reports")
    written = reports.write_reports(report_set, output)
    for name, path in written.items():
        print(f"[bootstrap] wrote {name} report -> {path}")

    logger.log_startup(
        "bootstrap_complete",
        ready=bool(tabular_ready and integrity["integrity_ok"]),
        report_dir=str(output),
    )
    print(
        f"[bootstrap] tabular={tabular_ready} image={image_ready} "
        f"repo_integrity={integrity['integrity_ok']} "
        f"-> {'READY' if tabular_ready and integrity['integrity_ok'] else 'NOT READY'}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
