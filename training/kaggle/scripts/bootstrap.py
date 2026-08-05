"""Kaggle bootstrap — environment + data-source readiness (orchestration only).

This script prepares a Kaggle Notebook (or GPU research machine) to run the
CropFusion Training Platform. It performs **no model training or dataset
implementation** — it only:

1. Reads ``training/config/kaggle.yaml`` (runtime paths, install list).
2. Adds the repository root to ``sys.path``.
3. Optionally ``pip install -e`` the first-party editable packages.
4. Builds the :class:`DatasetManager` from ``training/config/dataset.yaml``
   and reports both provider manifests (tabular + image readiness).
5. Optionally materialises the Kaggle imagery dataset (``--ensure-data``).

Run on Kaggle::

    !python training/kaggle/scripts/bootstrap.py --ensure-data

Run on a research machine::

    python training/kaggle/scripts/bootstrap.py \
        --repo-root . --dataset-config training/config/dataset.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_kaggle_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"kaggle config not found: {config_path}")
    with config_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def _prepend_to_path(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    for entry in (str(repo_root),):
        if entry not in sys.path:
            sys.path.insert(0, entry)
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


def _report_providers(manager) -> dict:
    manifests = manager.provider_manifests()
    print("[bootstrap] provider manifests:")
    print(json.dumps(manifests, indent=2, default=str))
    return manifests


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
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    kaggle_config = _load_kaggle_config(Path(args.kaggle_config))

    _prepend_to_path(repo_root)
    _install_editable(
        repo_root,
        kaggle_config.get("install", {}).get("editable_packages", []),
        skip=args.skip_install,
    )

    from training.dataset_manager import DatasetManager, load_settings

    settings = load_settings(Path(args.dataset_config))
    manager = DatasetManager(settings)
    try:
        manifests = _report_providers(manager)
        image_ready = manifests["kaggle_hub_image"]["available"]
        tabular_ready = manifests["git_repository_tabular"]["available"]
        print(
            f"[bootstrap] tabular={tabular_ready} image={image_ready} "
            f"-> {'READY' if tabular_ready else 'tabular missing'}"
        )
        if args.ensure_data:
            print("[bootstrap] ensuring imagery data (download-or-reuse)...")
            path = manager.ensure_image()
            print(f"[bootstrap] imagery materialised at {path}")
        return 0
    finally:
        manager.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
