"""Kaggle release-package builder (R6 -> Prediction Platform contract).

Assembles the application's ``cropfusion_release/`` directory tree from the
exported model artifacts plus the train-side sources. This is the artifact
the Prediction Platform's ``ReleasePackageLoader`` (and nothing else) reads::

    cropfusion_release-v<version>/
        model/cropfusion.pt            <- TorchScript export (renamed)
        preprocess/scaler.pkl          <- fitted tabular scaler (sources)
        preprocess/label_encoder.pkl   <- fitted crop label encoder (sources)
        metadata/metadata.db           <- dataset metadata snapshot (sources)
        metadata/historical_context.parquet
        metadata/location_index.parquet
        metadata/village_metadata.parquet
        configs/model.yaml             <- model config (+ feature_order/input_dim)
        configs/inference.yaml         <- generated inference settings
        version/manifest.json          <- {"format": "cropfusion_release", ...}
        version/checksum.json          <- sha256 per file
        reports/metrics.json           <- training/evaluation metrics

The model files come from the export kernel; ``--sources-dir`` is the
``release_sources/`` directory written by ``package_sources.py`` in the train
kernel (uploaded to the checkpoint dataset and mounted here). Files that the
sources cannot provide are reported; without ``--allow-partial`` a missing
required file fails the build, mirroring the app's own validation order.

Run on Kaggle::

    !python training/kaggle/scripts/build_release.py \
        --torchscript training/artifacts/releases/model.torchscript.pt \
        --model-config training/artifacts/releases/model.yaml \
        --sources-dir /kaggle/input/cropfusion-checkpoints/release_sources \
        --output training/artifacts/releases
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Relative files the Prediction Platform loader requires (in the order the
#: app validates them). Kept local on purpose: the training platform must never
#: import ``application.*``.
REQUIRED_RELEASE_FILES: tuple[str, ...] = (
    "model/cropfusion.pt",
    "metadata/metadata.db",
    "metadata/historical_context.parquet",
    "metadata/location_index.parquet",
    "metadata/village_metadata.parquet",
    "preprocess/scaler.pkl",
    "preprocess/label_encoder.pkl",
    "configs/model.yaml",
    "configs/inference.yaml",
    "version/manifest.json",
    "version/checksum.json",
    "reports/metrics.json",
)

SOURCE_ALIASES: tuple[tuple[str, str], ...] = (
    ("preprocess/scaler.pkl", "preprocess/scaler.pkl"),
    ("preprocess/label_encoder.pkl", "preprocess/label_encoder.pkl"),
    ("metadata/metadata.db", "metadata/metadata.db"),
    ("metadata/historical_context.parquet", "metadata/historical_context.parquet"),
    ("metadata/location_index.parquet", "metadata/location_index.parquet"),
    ("metadata/village_metadata.parquet", "metadata/village_metadata.parquet"),
    ("reports/metrics.json", "reports/metrics.json"),
)


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
            print(f"[build_release] removing shadowing sys.path entry: {entry}")
            sys.path.remove(entry)
    os.chdir(repo_root)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return path


def _copy(source_dir: Path, rel: str, dest_dir: Path) -> bool:
    src = source_dir / rel
    if not src.exists():
        return False
    target = dest_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    return True


def _resolve_version(model_config: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return explicit
    return str(model_config.get("version") or "1.0.0")


def _infer_feature_order(
    sources_dir: Path, sources_meta: dict[str, Any]
) -> list[str] | None:
    order = sources_meta.get("feature_order")
    if isinstance(order, list) and order:
        return [str(name) for name in order]
    scaler_path = sources_dir / "preprocess" / "scaler.pkl"
    if scaler_path.exists():
        try:
            import pickle

            with scaler_path.open("rb") as fh:
                scaler = pickle.load(fh)  # noqa: S301 - our own train-side scaler
            names = getattr(scaler, "feature_names", None)
            if isinstance(names, list) and names:
                return [str(name) for name in names]
        except Exception:  # noqa: S110 - best effort, feature_order is optional
            pass
    return None


def _write_inference_config(
    dest_dir: Path,
    *,
    model_name: str,
    version: str,
    dataset_version: str,
    feature_order: list[str] | None,
) -> Path:
    payload: dict[str, Any] = {
        "model": model_name,
        "model_version": version,
        "dataset_version": dataset_version,
        "device": "auto",
        "precision": "float32",
        "max_service_radius_km": 50.0,
    }
    if feature_order:
        payload["feature_order"] = feature_order
        payload["input_dim"] = len(feature_order)
    path = dest_dir / "configs" / "inference.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def build_release(
    *,
    torchscript: Path,
    model_config_path: Path,
    sources_dir: Path | None,
    output_dir: Path,
    version: str | None = None,
    dataset_version: str | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Assemble a release package and return its report dict."""
    for label, path in (("--torchscript", torchscript), ("--model-config", model_config_path)):
        if not path.exists():
            raise SystemExit(f"missing required input: {label} = {path}")

    import yaml

    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    if not isinstance(model_config, dict):
        raise SystemExit(f"model config is not a YAML mapping: {model_config_path}")

    model_name = str(model_config.get("name") or "cropfusion")
    release_version = _resolve_version(model_config, version)

    sources = sources_dir.resolve() if sources_dir is not None else None
    sources_meta = _read_json(sources / "sources.json") if sources else {}
    dataset_version = dataset_version or str(
        sources_meta.get("dataset_version") or "1.0.0"
    )

    dest = (output_dir / f"cropfusion_release-v{release_version}").resolve()
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}

    model_target = dest / "model" / "cropfusion.pt"
    model_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(torchscript, model_target)
    files["model/cropfusion.pt"] = model_target

    for rel, _alias in SOURCE_ALIASES:
        if sources is not None and _copy(sources, rel, dest):
            files[rel] = dest / rel

    feature_order = _infer_feature_order(
        sources, sources_meta
    ) if sources else sources_meta.get("feature_order")
    model_config_payload = dict(model_config)
    if feature_order:
        # The app's FeatureBuilder reads feature_order from configs/model.yaml.
        # Top-level extras are fine for the primary TorchScript path (the dict
        # is consumed raw); the degraded state_dict fallback rebuilds the model
        # via ModelConfig (extra=forbid) and would reject them -- acceptable
        # because the TorchScript export is always present in this stage.
        model_config_payload["feature_order"] = [str(name) for name in feature_order]
        model_config_payload["input_dim"] = len(feature_order)

    model_yaml = dest / "configs" / "model.yaml"
    model_yaml.parent.mkdir(parents=True, exist_ok=True)
    model_yaml.write_text(
        yaml.safe_dump(model_config_payload, sort_keys=False), encoding="utf-8"
    )
    files["configs/model.yaml"] = model_yaml

    inference_yaml = dest / "configs" / "inference.yaml"
    if inference_yaml.exists():
        files["configs/inference.yaml"] = inference_yaml
    else:
        files["configs/inference.yaml"] = _write_inference_config(
            dest,
            model_name=model_name,
            version=release_version,
            dataset_version=dataset_version,
            feature_order=(
                [str(name) for name in feature_order] if feature_order else None
            ),
        )

    metrics_path = dest / "reports" / "metrics.json"
    if metrics_path.exists():
        files["reports/metrics.json"] = metrics_path
    else:
        metrics_payload = dict(sources_meta.get("metrics", {}))
        metrics_payload.setdefault("model_version", release_version)
        files["reports/metrics.json"] = _write_json(metrics_path, metrics_payload)

    checksums = {
        rel: _sha256(path) for rel, path in sorted(files.items())
    }
    files["version/checksum.json"] = _write_json(
        dest / "version" / "checksum.json", {"files": checksums}
    )

    manifest: dict[str, Any] = {
        "format": "cropfusion_release",
        "schema_version": 1,
        "package_name": "cropfusion",
        "model_version": release_version,
        "dataset_version": dataset_version,
        "released_at": datetime.now(UTC).isoformat(),
        "files": sorted(files),
    }
    files["version/manifest.json"] = _write_json(
        dest / "version" / "manifest.json", manifest
    )

    missing = [rel for rel in REQUIRED_RELEASE_FILES if not (dest / rel).exists()]
    valid = not missing

    report: dict[str, Any] = {
        "release_dir": str(dest),
        "version": release_version,
        "model_version": release_version,
        "dataset_version": dataset_version,
        "files": sorted(files),
        "missing_required": missing,
        "valid": valid,
    }
    if not valid and not allow_partial:
        raise SystemExit(
            "release package incomplete; missing required files: "
            + ", ".join(missing)
        )

    print(json.dumps(report, indent=2, default=str))
    print(
        f"[build_release] release package -> {dest} "
        f"({len(files)} files, valid={valid})"
    )
    if not valid:
        print("[build_release] WARNING missing required files: " + ", ".join(missing))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cropfusion-build-release",
        description="Assemble the Prediction Platform release package.",
    )
    parser.add_argument("--repo-root", default=str(_REPO_ROOT))
    parser.add_argument("--torchscript", required=True, help="exported model.torchscript.pt")
    parser.add_argument("--model-config", required=True, help="exported model.yaml")
    parser.add_argument(
        "--sources-dir",
        default=None,
        help="train-side release_sources/ (scaler, encoder, metadata, metrics)",
    )
    parser.add_argument(
        "--output",
        default=str(_REPO_ROOT / "training" / "artifacts" / "releases"),
    )
    parser.add_argument("--version", default=None, help="release model version")
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="exit 0 even when train-side files are missing (report only)",
    )
    args = parser.parse_args(argv)

    _add_repo_root(Path(args.repo_root))

    build_release(
        torchscript=Path(args.torchscript),
        model_config_path=Path(args.model_config),
        sources_dir=Path(args.sources_dir) if args.sources_dir else None,
        output_dir=Path(args.output),
        version=args.version,
        dataset_version=args.dataset_version,
        allow_partial=args.allow_partial,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
