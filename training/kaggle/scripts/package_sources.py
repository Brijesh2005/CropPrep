"""Kaggle release-sources producer (runs in the train kernel, after training).

The export kernel only sees the trained checkpoint, so it cannot reconstruct
the fitted scaler, label encoder or dataset snapshot that the Prediction
Platform's ``cropfusion_release/`` contract requires. This script runs in the
*training* environment -- where the corpus, the dataset manager and the
preprocessing configuration live -- re-resolves the corpus, re-fits the
preprocessor (identical config + observations => identical fitted state) and
persists the sources the export stage consumes::

    training/artifacts/release_sources/
        preprocess/scaler.pkl          <- sklearn StandardScaler (numeric feats)
        preprocess/label_encoder.pkl   <- sklearn LabelEncoder (crop classes)
        metadata/metadata.db           <- dataset metadata snapshot (best effort)
        metadata/historical_context.parquet
        metadata/location_index.parquet
        metadata/village_metadata.parquet
        reports/metrics.json           <- checkpoint metrics
        sources.json                   <- feature_order + versions + file list

The scaler and label encoder are the core -- always produced, hard errors if
they fail. The dataset snapshots are best-effort: imagery-dependent steps that
fail (e.g. a degraded catalog) are recorded in ``sources.json`` under
``warnings`` and skipped rather than aborting the run.

Run on Kaggle after training::

    !python training/kaggle/scripts/package_sources.py
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from datetime import UTC, datetime
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
            print(f"[package_sources] removing shadowing sys.path entry: {entry}")
            sys.path.remove(entry)
    os.chdir(repo_root)


def _to_sklearn_scaler(fitted_scaler: Any, feature_names: list[str]) -> Any:
    """Copy the fitted training scaler into a plain sklearn StandardScaler.

    A pure ``sklearn.preprocessing.StandardScaler`` is used on purpose: the
    Prediction Platform unpickles ``scaler.pkl`` and must never import the
    training package (``application/inference_package/release/loader.py``).
    ``feature_names`` is attached as an extra attribute so ``build_release``
    can recover ``feature_order`` even without ``sources.json``.
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.mean_ = np.asarray(fitted_scaler.mean_, dtype="float64")
    scaler.scale_ = np.asarray(fitted_scaler.scale_, dtype="float64")
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(feature_names)
    scaler.feature_names = list(feature_names)
    return scaler


def _to_sklearn_label_encoder(crop_encoder: Any) -> Any:
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    encoder.classes_ = np.asarray(crop_encoder.classes_, dtype=object)
    return encoder


def _persist_pipeline(preprocessor: Any, out_dir: Path) -> dict[str, Any]:
    """Write scaler.pkl + label_encoder.pkl; return the feature metadata."""
    tabular = getattr(preprocessor, "tabular", None)
    label = getattr(preprocessor, "label", None)
    if tabular is None or tabular.scaler is None:
        raise RuntimeError("fitted preprocessor has no tabular scaler")
    if label is None or label.crop_encoder is None:
        raise RuntimeError("fitted preprocessor has no crop label encoder")

    feature_order = list(tabular.numeric_features)

    (out_dir / "preprocess").mkdir(parents=True, exist_ok=True)
    with (out_dir / "preprocess" / "scaler.pkl").open("wb") as fh:
        pickle.dump(
            _to_sklearn_scaler(tabular.scaler, feature_order),
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    with (out_dir / "preprocess" / "label_encoder.pkl").open("wb") as fh:
        pickle.dump(
            _to_sklearn_label_encoder(label.crop_encoder),
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return {
        "feature_order": feature_order,
        "num_features": len(feature_order),
        "num_classes": int(label.num_classes),
        "crop_classes": list(label.crop_encoder.classes_),
        "yield_scale_stats": getattr(label, "yield_scale_stats", None),
        "warnings": list(getattr(label, "warnings", [])),
    }


def _persist_dataset_sources(manager: Any, out_dir: Path, warnings: list[str]) -> None:
    """Snapshot metadata.db / historical_context / location_index (best effort)."""
    from training.inference.dataset_sources import (
        DatasetSourceError,
        persist_dataset_sources,
    )

    try:
        sources = persist_dataset_sources(manager, out_dir / "metadata")
        for _name, path in sources.to_dict().items():
            print(f"[package_sources] dataset source -> {path}")
    except DatasetSourceError as exc:
        warnings.append(f"dataset sources skipped: {exc}")
        print(f"[package_sources] WARNING {exc}")


def _persist_village_metadata(out_dir: Path, warnings: list[str]) -> None:
    """Derive village_metadata.parquet from the staged location index."""
    import pandas as pd

    loc_path = out_dir / "metadata" / "location_index.parquet"
    if not loc_path.exists():
        warnings.append("village_metadata skipped: no location_index.parquet")
        return
    index = pd.read_parquet(loc_path)
    if "village" not in index.columns:
        warnings.append("village_metadata skipped: location_index has no 'village'")
        return
    village_cols = [c for c in ("village", "district", "taluk", "lon", "lat")
                    if c in index.columns]
    frame = index[village_cols].drop_duplicates(subset=["village", "district"])
    frame = frame.reset_index(drop=True)
    frame.to_parquet(out_dir / "metadata" / "village_metadata.parquet", index=False)
    print(f"[package_sources] village_metadata -> {frame.shape}")


def _latest_checkpoint_metrics(layout: Any) -> dict[str, Any]:
    checkpoints = sorted(layout.checkpoints.glob("checkpoint_*.pt"))
    if not checkpoints:
        return {}
    from training.models.checkpoint import CheckpointManager

    state = CheckpointManager.load(checkpoints[-1])
    return dict(state.get("metrics", {}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cropfusion-package-sources",
        description="Persist the train-side artefacts for the release package.",
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
        "--preprocessing-config",
        default=str(_REPO_ROOT / "training" / "config" / "preprocessing.yaml"),
    )
    parser.add_argument(
        "--stam-config",
        default=str(_REPO_ROOT / "training" / "config" / "stam.yaml"),
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="cap the number of (location, year, season) cells resolved",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output dir (default: training/artifacts/release_sources)",
    )
    args = parser.parse_args(argv)

    _add_repo_root(Path(args.repo_root))
    repo_root = Path(args.repo_root).resolve()

    from training.dataset_manager import DatasetManager, load_settings
    from training.kaggle.config import (
        WorkspaceLayout,
        load_paths_config,
    )
    from training.kaggle.workspace import WorkspaceManager
    from training.preprocessing import Preprocessor
    from training.stam import STAM
    from training.stam.config import load_stam_config
    from training.stam.observation_resolver import ObservationResolver

    paths = load_paths_config(Path(args.paths_config))
    layout = WorkspaceLayout.resolve(paths, repo_root=repo_root)
    workspace = WorkspaceManager(layout)
    workspace.create()

    out_dir = (
        Path(args.output).resolve()
        if args.output
        else (repo_root / "training" / "artifacts" / "release_sources").resolve()
    )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    dataset_version: str | None = None
    preprocessor: Any | None = None

    dataset_settings = load_settings(Path(args.dataset_config))
    stam_cfg = load_stam_config(Path(args.stam_config))
    manager = DatasetManager(dataset_settings)
    try:
        current = getattr(manager, "current_version", None)
        if callable(current):
            try:
                info = current()
                dataset_version = str(getattr(info, "version", "") or "") or None
            except Exception:  # noqa: S110 - versions are best-effort
                pass

        image_manifest = manager.provider_manifests().get("kaggle_hub_image", {})
        if image_manifest.get("available") is True:
            manager.ensure_image()
            manager.generate_image_metadata()

        # Corpus (mirrors run_pipeline.py: DatasetManager -> STAM -> resolver).
        stam = STAM(manager, stam_cfg)
        stam.initialize()
        resolver = ObservationResolver(stam)
        plan = resolver.plan()
        if args.max_cells and plan.total > args.max_cells:
            plan = plan.model_copy(update={"cells": plan.cells[: args.max_cells]})
        corpus = resolver.resolve(plan)
        accepted = corpus.accepted_observations()
        print(f"[package_sources] corpus accepted={len(accepted)} "
              f"total={corpus.total}")

        # Re-fit the preprocessor (same config/observations as training).
        if not accepted:
            warnings.append("corpus has no accepted observations; preprocessor not fit")
        else:
            preprocessor = Preprocessor.from_config(args.preprocessing_config)
            accepted, _ = preprocessor.filter(accepted)
            if not accepted:
                warnings.append("no accepted observations after quality filter")
            else:
                try:
                    preprocessor.fit(accepted, extractor=stam.get_patch)
                except Exception as exc:  # noqa: BLE001 - image fit may need the catalog
                    warnings.append(f"full fit failed ({exc}); fitting tabular+label")
                    preprocessor.tabular.fit(accepted)
                    preprocessor.label.fit(accepted)
                    preprocessor.fitted = True

        # Dataset snapshot + village metadata (best effort).
        _persist_dataset_sources(manager, out_dir, warnings)
    finally:
        manager.close()

    _persist_village_metadata(out_dir, warnings)

    # Core pipelines (hard requirement).
    if preprocessor is None:
        raise SystemExit(
            "release sources FAILED: no fitted preprocessor (see warnings above)"
        )
    pipeline_meta = _persist_pipeline(preprocessor, out_dir)

    # Metrics from the latest checkpoint.
    metrics = _latest_checkpoint_metrics(workspace.layout)
    model_version = str(metrics.get("model_version") or "") or None
    metrics_dir = out_dir / "reports"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "metrics.json").write_text(
        json.dumps(
            {"model_version": model_version or "unknown",
             "dataset_version": dataset_version or "1.0.0",
             "metrics": metrics},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    sources_meta: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_version": model_version,
        "dataset_version": dataset_version,
        "metrics": metrics,
        "pipeline": pipeline_meta,
        "files": sorted(
            str(p.relative_to(out_dir))
            for p in out_dir.rglob("*") if p.is_file()
        ),
        "warnings": warnings,
    }
    (out_dir / "sources.json").write_text(
        json.dumps(sources_meta, indent=2, default=str), encoding="utf-8"
    )

    print(json.dumps(sources_meta, indent=2, default=str))
    print(f"[package_sources] release sources -> {out_dir}")
    if warnings:
        print("[package_sources] warnings:")
        for warning in warnings:
            print("  -", warning)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
