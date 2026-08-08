"""Release packager (Phase R6).

:class:`ReleasePackager` turns the flat Phase R5 inference package into the
canonical :mod:`cropfusion_release-<version> <training.runtime.layout>`
release layout. It moves the R5 artefacts into their sub-directories,
generates the derived artefacts the runtime requires
(``metadata/feature_lookup.parquet``, ``model/model_metadata.json``,
``preprocess/preprocess_metadata.json``, a release ``README.md``) and writes
the release manifest + checksums.

Only exported artefacts are consumed — the Kaggle catalog, the dataset manager
and the training loop are never touched.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from shared.utils.hash import sha256_file
from shared.versioning import SemanticVersion

from .exceptions import ReleasePackagingError
from .layout import (
    REQUIRED_RELEASE_FILES,
    ReleaseLayout,
    ReleaseManifest,
    release_dir_name,
)

#: Flat (R5) artefacts that are copied straight across.
FLAT_MODEL_FILES: dict[str, str] = {
    "cropfusion.pt": "model/cropfusion.pt",
    "cropfusion.torchscript.pt": "model/cropfusion.torchscript.pt",
    "cropfusion.onnx": "model/cropfusion.onnx",
}
FLAT_PREPROCESS_FILES: dict[str, str] = {
    "feature_scalers.pkl": "preprocess/feature_scalers.pkl",
    "label_encoder.pkl": "preprocess/label_encoder.pkl",
}
FLAT_METADATA_FILES: dict[str, str] = {
    "metadata.db": "metadata/metadata.db",
    "historical_context.parquet": "metadata/historical_context.parquet",
    "location_index.parquet": "metadata/location_index.parquet",
}
FLAT_CONFIG_FILES: dict[str, str] = {
    "model_config.yaml": "configs/model_config.yaml",
    "training_config.yaml": "configs/training_config.yaml",
}
FLAT_VERSION_FILES: dict[str, str] = {
    "model_version.json": "version/model_version.json",
    "dataset_version.json": "version/dataset_version.json",
    "checksums.json": "version/original_checksums.json",
    "manifest.json": "version/original_manifest.json",
}

#: Required flat sources for a release build.
REQUIRED_FLAT_SOURCES: tuple[str, ...] = (
    "cropfusion.pt",
    "feature_scalers.pkl",
    "label_encoder.pkl",
    "metadata.db",
    "historical_context.parquet",
    "location_index.parquet",
    "model_config.yaml",
    "training_config.yaml",
    "metrics.json",
    "manifest.json",
    "checksums.json",
    "model_version.json",
    "dataset_version.json",
)


@dataclass
class ReleaseReport:
    """Result of a release build."""

    target_dir: Path
    version: str
    files: dict[str, Path] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    checksums: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_dir": str(self.target_dir),
            "version": self.version,
            "files": {name: str(path) for name, path in self.files.items()},
            "manifest": self.manifest,
            "checksums": self.checksums,
        }


class ReleasePackager:
    """Build a :mod:`cropfusion_release` from a Phase R5 inference package.

    Args:
        package_name: Override for the release package name (defaults to the
            value in the source manifest).
    """

    def __init__(self, package_name: str | None = None) -> None:
        self.package_name = package_name

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def build(
        self,
        inference_package_dir: str | Path,
        *,
        releases_root: str | Path | None = None,
        target_dir: str | Path | None = None,
        version: str | None = None,
    ) -> ReleaseReport:
        """Build a release package from a flat inference package.

        Args:
            inference_package_dir: The Phase R5 package output directory.
            releases_root: Directory under which the release is written
                (``releases/<cropfusion_release-vX.Y.Z>`` by default). Ignored
                when ``target_dir`` is given.
            target_dir: Explicit destination for the release package
                directory (takes precedence over ``releases_root``).
            version: Release version override (defaults to the source
                manifest's ``package_version``).

        Raises:
            ReleasePackagingError: When the source package is incomplete or a
                required artefact cannot be produced.
        """
        source = Path(inference_package_dir)
        if not source.is_dir():
            raise ReleasePackagingError(
                "inference package directory does not exist",
                detail=str(source),
            )
        missing = [name for name in REQUIRED_FLAT_SOURCES if not (source / name).exists()]
        if missing:
            raise ReleasePackagingError(
                "inference package is missing required artefacts",
                detail={"package_dir": str(source), "missing": missing},
            )

        manifest_payload = _load_json(source / "manifest.json")
        version = version or self._resolve_version(source, manifest_payload)
        try:
            SemanticVersion.from_string(version)
        except Exception as exc:  # InvalidVersionError
            raise ReleasePackagingError(
                f"release version is not valid semver: {version!r}", detail=version
            ) from exc

        if target_dir is not None:
            dest = Path(target_dir)
        else:
            dest = Path(releases_root or "releases") / release_dir_name(version)
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)

        files: dict[str, Path] = {}
        for flat_name, rel in {**FLAT_MODEL_FILES, **FLAT_PREPROCESS_FILES,
                               **FLAT_METADATA_FILES, **FLAT_CONFIG_FILES,
                               **FLAT_VERSION_FILES}.items():
            src_path = source / flat_name
            if not src_path.exists():
                continue
            target_path = dest / rel
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, target_path)
            files[rel] = target_path

        # -- Reports ------------------------------------------------------- #
        metrics_path = dest / "reports" / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "metrics.json", metrics_path)
        files["reports/metrics.json"] = metrics_path

        # -- Derived artefacts --------------------------------------------- #
        model_cfg = _load_model_config(dest / "configs" / "model_config.yaml")
        model_meta = self._build_model_metadata(
            dest, manifest_payload, model_cfg
        )
        files["model/model_metadata.json"] = _write_json(
            dest / "model" / "model_metadata.json", model_meta
        )
        pre_meta = self._build_preprocess_metadata(
            dest / "preprocess" / "feature_scalers.pkl",
            dest / "preprocess" / "label_encoder.pkl",
            model_cfg,
        )
        files["preprocess/preprocess_metadata.json"] = _write_json(
            dest / "preprocess" / "preprocess_metadata.json", pre_meta
        )
        feature_lookup = self._build_feature_lookup(pre_meta)
        lookup_path = dest / "metadata" / "feature_lookup.parquet"
        feature_lookup.to_parquet(lookup_path, index=False)
        files["metadata/feature_lookup.parquet"] = lookup_path

        # -- README -------------------------------------------------------- #
        files["README.md"] = _write_text(dest / "README.md", self._readme(version))

        # -- Checksums + manifest ------------------------------------------ #
        checksums = {
            rel: sha256_file(path)
            for rel, path in sorted(files.items())
        }
        files["version/checksums.json"] = _write_json(
            dest / "version" / "checksums.json", checksums
        )

        package_name = self.package_name or str(
            manifest_payload.get("package_name", "cropfusion")
        )
        manifest = self._manifest(
            package_name=package_name,
            version=version,
            manifest_payload=manifest_payload,
            files=files,
            checksums=checksums,
        )
        files["version/manifest.json"] = _write_json(
            dest / "version" / "manifest.json", manifest.model_dump()
        )

        release_version = self._release_version(
            package_name, version, manifest_payload
        )
        files["version/release_version.json"] = _write_json(
            dest / "version" / "release_version.json", release_version
        )

        layout = ReleaseLayout(dest)
        valid, errors = layout.is_valid_structure()
        if not valid:
            raise ReleasePackagingError(
                "built release is missing required files",
                detail={"target_dir": str(dest), "missing": errors},
            )
        return ReleaseReport(
            target_dir=dest,
            version=version,
            files=files,
            manifest=manifest.model_dump(),
            checksums=checksums,
        )

    # ------------------------------------------------------------------ #
    # Version resolution
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_version(source: Path, manifest_payload: Mapping[str, Any]) -> str:
        manifest_version = str(manifest_payload.get("package_version") or "")
        if manifest_version:
            return manifest_version
        return "1.0.0"

    # ------------------------------------------------------------------ #
    # Derived artefacts
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_model_metadata(
        dest: Path,
        manifest_payload: Mapping[str, Any],
        model_cfg: Any,
    ) -> dict[str, Any]:
        name = getattr(model_cfg, "name", None) or manifest_payload.get("package_name")
        return {
            "model_name": name,
            "model_version": (
                manifest_payload.get("model_version")
                or getattr(model_cfg, "version", None)
                or "1.0.0"
            ),
            "architecture_version": getattr(
                model_cfg, "architecture_version", None
            ),
            "model_fingerprint": manifest_payload.get("model_fingerprint"),
            "formats": [
                fmt
                for fmt, rel in (
                    ("pytorch", "model/cropfusion.pt"),
                    ("torchscript", "model/cropfusion.torchscript.pt"),
                    ("onnx", "model/cropfusion.onnx"),
                )
                if (dest / rel).exists()
            ],
            "uses_tabular": bool(getattr(model_cfg, "uses_tabular", True)),
            "uses_image": bool(getattr(model_cfg, "uses_image", False)),
            "parameter_count": None,
        }

    @staticmethod
    def _build_preprocess_metadata(
        scalers_path: Path,
        encoder_path: Path,
        model_cfg: Any,
    ) -> dict[str, Any]:
        scalers = _load_pickle(scalers_path) if scalers_path.exists() else None
        encoder = _load_pickle(encoder_path) if encoder_path.exists() else None

        feature_names = list(getattr(scalers, "feature_names", None) or [])
        numeric = list(getattr(scalers, "numeric_features", None) or [])
        categorical = list(getattr(scalers, "categorical_features", None) or [])
        if not feature_names and numeric:
            feature_names = list(numeric)
        if not feature_names and getattr(model_cfg, "tabular", None) is not None:
            feature_names = list(range(model_cfg.tabular.numeric_dim or 0))

        if not categorical and getattr(scalers, "encoder", None) is not None:
            encoder_obj = getattr(scalers, "encoder", None)
            categories = getattr(encoder_obj, "categories_", None) or []
            if categories and not categorical:
                n = len(categories)
                categorical = feature_names[-n:] if feature_names else list(range(n))

        num_classes = int(getattr(encoder, "num_classes", 0) or 0)

        feature_types = []
        for name in feature_names:
            if name in categorical or name in set(getattr(
                scalers, "_categorical_columns", None
            ) or set()):
                feature_types.append("categorical")
            else:
                feature_types.append("numeric")

        return {
            "feature_names": feature_names,
            "feature_types": feature_types,
            "numeric_features": numeric,
            "categorical_features": categorical,
            "num_features": len(feature_names),
            "num_classes": num_classes,
            "fitted": bool(getattr(scalers, "fitted", False))
            and bool(getattr(encoder, "fitted", False)),
        }

    @staticmethod
    def _build_feature_lookup(pre_meta: Mapping[str, Any]):
        """Derive ``feature_lookup.parquet`` from the preprocessing metadata."""
        import pandas as pd

        names = list(pre_meta.get("feature_names", []))
        types = list(pre_meta.get("feature_types", []))
        if types and len(types) != len(names):
            types = []
        rows = []
        for index, name in enumerate(names):
            rows.append(
                {
                    "feature_index": index,
                    "feature_name": str(name),
                    "feature_type": types[index] if types else "unknown",
                    "feature_group": (
                        "tabular"
                        if index < len(names)
                        else "other"
                    ),
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "feature_index",
                "feature_name",
                "feature_type",
                "feature_group",
            ],
        )

    # ------------------------------------------------------------------ #
    # Manifest / version files
    # ------------------------------------------------------------------ #

    def _manifest(
        self,
        *,
        package_name: str,
        version: str,
        manifest_payload: Mapping[str, Any],
        files: Mapping[str, Path],
        checksums: Mapping[str, str],
    ) -> ReleaseManifest:
        required = list(REQUIRED_RELEASE_FILES)
        # ``manifest.json`` / ``checksums.json`` cannot checksum themselves.
        self_referential = {"version/manifest.json", "version/checksums.json"}
        missing = [rel for rel in required if rel not in checksums
                   and rel not in self_referential]
        if missing:
            raise ReleasePackagingError(
                "release would be missing required files",
                detail={"missing": missing},
            )
        return ReleaseManifest(
            package_name=package_name,
            package_version=version,
            model_version=str(
                manifest_payload.get("model_version") or version
            ),
            dataset_version=str(
                manifest_payload.get("dataset_version") or "1.0.0"
            ),
            release_version=version,
            created_at=datetime.now(timezone.utc).isoformat(),
            git_commit=manifest_payload.get("git_commit"),
            model_fingerprint=str(
                manifest_payload.get("model_fingerprint") or ""
            ),
            dataset_fingerprint=str(
                manifest_payload.get("dataset_fingerprint") or ""
            ),
            training_fingerprint=manifest_payload.get("training_fingerprint"),
            formats=[
                fmt
                for fmt, rel in (
                    ("pytorch", "model/cropfusion.pt"),
                    ("torchscript", "model/cropfusion.torchscript.pt"),
                    ("onnx", "model/cropfusion.onnx"),
                )
                if rel in checksums
            ],
            required_files=required,
            files=sorted(checksums),
        )

    @staticmethod
    def _release_version(
        package_name: str, version: str, manifest_payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "kind": "release",
            "name": f"{package_name}-release",
            "version": version,
            "status": "ready",
            "checksum": manifest_payload.get("model_fingerprint"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": "Release package assembled from the inference package.",
        }

    def _readme(self, version: str) -> str:
        return f"""# cropfusion release v{version}

This directory is a **release package** produced by the CropFusion training
platform and consumed by the Phase R6 runtime. It contains only exported
artefacts — the Kaggle catalog, the dataset manager and the training loop are
**not** required.

## Layout

| directory | content |
| --- | --- |
| `model/` | exported model artefacts (`cropfusion.pt`, optional TorchScript / ONNX) |
| `preprocess/` | fitted feature scaler + label encoder pipelines |
| `metadata/` | predict-only metadata snapshot + feature lookup |
| `configs/` | resolved model / training configuration |
| `reports/` | evaluation metrics + validation reports |
| `version/` | manifest, checksums and version files |

## Usage

Activate and load with `training.runtime.InferenceRuntime`:

```python
from training.runtime import RuntimeConfig, InferenceRuntime

runtime = InferenceRuntime(RuntimeConfig(general={{"releases_root": "releases"}}))
runtime.start(version="{version}")
print(runtime.health().to_dict())
```

## Integrity

Validate the release with `training.runtime.validation.ReleaseValidator` or
`ReleaseManager.validate(version)`.
"""


def _load_model_config(path: Path) -> Any:
    import yaml

    from training.models import ModelConfig

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ModelConfig.model_validate(raw)


def _load_pickle(path: Path) -> Any:
    import pickle

    with open(path, "rb") as fh:
        return pickle.load(fh)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReleasePackagingError(
            "malformed JSON artefact", detail=str(path)
        ) from exc
    return raw if isinstance(raw, dict) else {}


def _write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, default=_json_default), encoding="utf-8"
    )
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError(f"not JSON serializable: {type(value)}")
