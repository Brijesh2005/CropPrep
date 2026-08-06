"""Inference package generator (Phase R5).

:class:`InferencePackageBuilder` assembles a complete, self-contained,
**predict-only** inference package that is fully independent of the Kaggle /
Sentinel training catalog:

====================================================  ==========================
artifact                                             content
====================================================  ==========================
``cropfusion.pt``                                     PyTorch model export
``feature_scalers.pkl``                               fitted tabular pipeline
``label_encoder.pkl``                                 fitted label pipeline
``metadata.db``                                       dataset metadata snapshot
``historical_context.parquet``                        season availability
``location_index.parquet``                            spatial index records
``model_config.yaml``                                 architecture configuration
``training_config.yaml``                              training configuration
``metrics.json``                                      evaluation metrics
``dataset_version.json``                              dataset semver + checksum
``model_version.json``                                model semver + checksum
``README.md``                                         package documentation
``LICENSE``                                           license text
``checksums.json``                                    SHA-256 of every file
``manifest.json``                                     package manifest
====================================================  ==========================

The optional TorchScript / ONNX exports are written alongside when enabled in
the exporter configuration and are included in the checksums + manifest.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import InferenceConfig
from .dataset_sources import DatasetSources
from .exceptions import PackageBuildError
from .exporter import BUNDLE_FORMAT_FILES, ModelExporter
from .validate import InferencePackageValidator
from .versioning import (
    build_version_files,
    content_sha256,
    file_sha256,
    git_commit,
    model_fingerprint,
    resolve_versions,
)

#: The 14 required artefacts (the manifest is the 15th file).
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "cropfusion.pt",
    "feature_scalers.pkl",
    "label_encoder.pkl",
    "metadata.db",
    "historical_context.parquet",
    "location_index.parquet",
    "model_config.yaml",
    "training_config.yaml",
    "metrics.json",
    "dataset_version.json",
    "model_version.json",
    "README.md",
    "LICENSE",
    "checksums.json",
)


@dataclass
class BuildReport:
    """Result of an inference-package build."""

    output_dir: Path
    files: dict[str, Path]
    manifest: dict[str, Any]
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        validation = self.validation
        if hasattr(validation, "to_dict"):
            validation = validation.to_dict()
        return {
            "output_dir": str(self.output_dir),
            "files": {name: str(path) for name, path in self.files.items()},
            "manifest": self.manifest,
            "validation": validation,
        }


class InferencePackageBuilder:
    """Assemble the inference package from trained artefacts.

    Args:
        model: A trained :class:`CropFusionModel`.
        preprocessor: A fitted :class:`Preprocessor` (provides the tabular
            scaler and label encoder pipelines).
        config: Validated :class:`InferenceConfig` (``None`` = defaults).
    """

    def __init__(
        self,
        model: Any,
        preprocessor: Any,
        config: InferenceConfig | None = None,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.config = config or InferenceConfig()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def build(
        self,
        dataset_sources: DatasetSources,
        *,
        metrics: Mapping[str, Any] | None = None,
        training_config: Any = None,
        dataset_manager: Any = None,
        sample_batch: Mapping[str, Any] | None = None,
    ) -> BuildReport:
        """Build the package and return a :class:`BuildReport`.

        Args:
            dataset_sources: Staged dataset artefacts (from
                :func:`persist_dataset_sources` or a custom :class:`DatasetSources`).
            metrics: Evaluation metrics (``EvaluationOutcome.to_dict()``) written
                to ``metrics.json``.
            training_config: The ``TrainingConfig`` used for training (written
                to ``training_config.yaml``).
            dataset_manager: Optional ``DatasetManager`` used to resolve the
                dataset version for ``dataset_version.json``.
            sample_batch: Optional batch dict for export example inputs.

        Raises:
            PackageBuildError: When a required artefact cannot be produced.
        """
        out_dir = Path(self.config.general.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        files: dict[str, Path] = {}
        self._training_config = training_config

        # -- Model exports + model_config.yaml + metrics.json ------------ #
        exporter = ModelExporter(self.model, sample_batch)
        bundle = exporter.export_bundle(
            out_dir,
            config=self.config,
            metrics=metrics,
            training_config=training_config,
            versions=None,
            dataset_fingerprint=None,
            git_commit_sha=git_commit(),
        )
        for key, path in bundle.items():
            files[key] = path

        # -- Preprocessor pipelines -------------------------------------- #
        files["feature_scalers"] = self._export_feature_scalers(out_dir)
        files["label_encoder"] = self._export_label_encoder(out_dir)

        # -- Dataset snapshot --------------------------------------------- #
        files["metadata_db"] = self._copy_source(
            dataset_sources.metadata_db, out_dir / "metadata.db"
        )
        files["historical_context"] = self._copy_source(
            dataset_sources.historical_context, out_dir / "historical_context.parquet"
        )
        files["location_index"] = self._copy_source(
            dataset_sources.location_index, out_dir / "location_index.parquet"
        )

        # -- Training config ---------------------------------------------- #
        files["training_config"] = self._export_training_config(
            out_dir, training_config
        )

        # -- Versions ------------------------------------------------------ #
        dataset_version = self._dataset_version(dataset_manager)
        resolved = resolve_versions(
            package_version=self.config.general.version,
            model_version=self.config.general.model_version,
            dataset_version=self.config.general.dataset_version or dataset_version,
            model_config_version=getattr(self.model.config, "version", "1.0.0"),
        )
        model_fp = model_fingerprint(self.model)
        dataset_fp = content_sha256(
            {
                "metadata_db": file_sha256(files["metadata_db"]),
                "historical_context": file_sha256(files["historical_context"]),
                "location_index": file_sha256(files["location_index"]),
            }
        )
        dataset_version_payload, model_version_payload = build_version_files(
            resolved,
            model_fingerprint=model_fp,
            dataset_fingerprint=dataset_fp,
            training_fingerprint=self._training_fingerprint(training_config),
            git_commit_sha=git_commit(),
            package_name=self.config.general.package_name,
            extra={"output_dir": str(out_dir)},
        )
        files["dataset_version"] = _write_json(
            out_dir / "dataset_version.json", dataset_version_payload
        )
        files["model_version"] = _write_json(
            out_dir / "model_version.json", model_version_payload
        )

        # -- Docs ----------------------------------------------------------- #
        if self.config.package.docs:
            files["readme"] = _write_text(
                out_dir / "README.md", self._render_readme(resolved)
            )
            files["license"] = _write_text(
                out_dir / "LICENSE", self._render_license()
            )

        # -- Checksums + manifest ------------------------------------------ #
        # ``checksums.json`` and ``manifest.json`` are self-referential and are
        # therefore not listed inside their own checksum map.
        checksums = {
            path.name: file_sha256(path)
            for name, path in files.items()
            if path.exists() and name not in ("checksums", "manifest")
        }
        files["checksums"] = _write_json(out_dir / "checksums.json", checksums)

        manifest = self._build_manifest(
            resolved,
            model_fingerprint=model_fp,
            dataset_fingerprint=dataset_fp,
            files=files,
            checksums=checksums,
        )
        files["manifest"] = _write_json(out_dir / "manifest.json", manifest)

        report = BuildReport(
            output_dir=out_dir, files=files, manifest=manifest
        )

        if self.config.validation.verify_checksums or self.config.validation.strict:
            report.validation = InferencePackageValidator(
                self.config
            ).validate_package(out_dir)
        return report

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _export_feature_scalers(self, out_dir: Path) -> Path:
        pipeline = getattr(self.preprocessor, "tabular", None)
        if pipeline is None:
            raise PackageBuildError(
                "preprocessor has no fitted tabular pipeline"
            )
        return _pickle_pipeline(pipeline, out_dir / "feature_scalers.pkl")

    def _export_label_encoder(self, out_dir: Path) -> Path:
        pipeline = getattr(self.preprocessor, "label", None)
        if pipeline is None:
            raise PackageBuildError("preprocessor has no fitted label pipeline")
        return _pickle_pipeline(pipeline, out_dir / "label_encoder.pkl")

    @staticmethod
    def _copy_source(source: Path, destination: Path) -> Path:
        import shutil

        if not Path(source).exists():
            raise PackageBuildError(
                f"dataset source artefact missing: {source}"
            )
        shutil.copy2(source, destination)
        return destination

    @staticmethod
    def _export_training_config(out_dir: Path, training_config: Any) -> Path:
        path = out_dir / "training_config.yaml"
        if training_config is None:
            path.write_text("# training_config.yaml (not provided)\n", encoding="utf-8")
            return path
        if hasattr(training_config, "to_yaml"):
            path.write_text(training_config.to_yaml(), encoding="utf-8")
        elif isinstance(training_config, Mapping):
            path.write_text(
                yaml.safe_dump(dict(training_config), sort_keys=False),
                encoding="utf-8",
            )
        else:
            raise PackageBuildError(
                "training_config must be a TrainingConfig or a mapping",
                detail=type(training_config).__name__,
            )
        return path

    @staticmethod
    def _dataset_version(dataset_manager: Any) -> str:
        if dataset_manager is None:
            return "1.0.0"
        current = getattr(dataset_manager, "current_version", None)
        if callable(current):
            try:
                version_info = current()
                value = getattr(version_info, "version", None)
                if value is not None:
                    return str(value)
            except Exception:
                pass
        return "1.0.0"

    @staticmethod
    def _training_fingerprint(training_config: Any) -> str | None:
        if training_config is None:
            return None
        if hasattr(training_config, "model_dump"):
            return content_sha256(training_config.model_dump())
        if isinstance(training_config, Mapping):
            return content_sha256(dict(training_config))
        return None

    def _build_manifest(
        self,
        resolved: Any,
        *,
        model_fingerprint: str,
        dataset_fingerprint: str,
        files: Mapping[str, Path],
        checksums: Mapping[str, str],
    ) -> dict[str, Any]:
        required = [name for name in REQUIRED_ARTIFACTS if name != "checksums.json"]
        missing = [name for name in required if name not in checksums]
        if missing:
            raise PackageBuildError(
                f"missing required artefacts: {missing}",
                detail={"required": REQUIRED_ARTIFACTS, "missing": missing},
            )
        return {
            "manifest_version": 1,
            "package_name": self.config.general.package_name,
            "package_version": str(resolved.package_version),
            "model_version": str(resolved.model_version),
            "dataset_version": str(resolved.dataset_version),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "model_fingerprint": model_fingerprint,
            "dataset_fingerprint": dataset_fingerprint,
            "training_fingerprint": self._training_fingerprint(
                getattr(self, "_training_config", None)
            ),
            "formats": [
                name
                for name, filename in BUNDLE_FORMAT_FILES.items()
                if filename in checksums
            ],
            "required_files": list(REQUIRED_ARTIFACTS),
            "files": dict(checksums),
        }

    def _render_readme(self, resolved: Any) -> str:
        formats = ", ".join(self.config.exporter.formats)
        return f"""# {self.config.general.package_name} inference package

{self.config.package.description}

## Contents

| artifact | description |
| --- | --- |
| `cropfusion.pt` | PyTorch model export ({formats}) |
| `feature_scalers.pkl` | Fitted tabular scaler / encoder pipeline |
| `label_encoder.pkl` | Fitted crop + yield label encoders |
| `metadata.db` | Dataset metadata snapshot (predict-only) |
| `historical_context.parquet` | Season availability context |
| `location_index.parquet` | Spatial index of registered locations |
| `model_config.yaml` | Architecture configuration |
| `training_config.yaml` | Training configuration |
| `metrics.json` | Evaluation metrics |
| `dataset_version.json` | Dataset version + checksum |
| `model_version.json` | Model version + checksum |
| `checksums.json` | SHA-256 of every artifact |
| `manifest.json` | Package manifest |

## Versioning

- Package version: {resolved.package_version}
- Model version: {resolved.model_version}
- Dataset version: {resolved.dataset_version}

## Usage

This package is **predict-only**: it consumes the Phase-4 batch contract
(`tabular`, `ndvi`, `evi`, `temporal_mask`) and never touches the Kaggle /
Sentinel catalog. The PyTorch model can be restored with
`training.inference.exporter.load_pytorch_model`; the ONNX / TorchScript
artifacts expose a tensor-only forward.

## Integrity

Verify the package with
`training.inference.validate.InferencePackageValidator.validate_package`.
"""

    def _render_license(self) -> str:
        identifier = self.config.package.license
        return (
            f"{identifier} license\n\n"
            "This inference package is distributed under the SPDX identifier "
            f"'{identifier}'. See the upstream license terms for the exact "
            "text before redistribution.\n"
        )


def _pickle_pipeline(pipeline: Any, path: Path) -> Path:
    with open(path, "wb") as fh:
        pickle.dump(pipeline, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _write_json(path: Path, data: Any) -> Path:
    path.write_text(
        json.dumps(data, indent=2, default=_json_default), encoding="utf-8"
    )
    return path


def _write_text(path: Path, text: str) -> Path:
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
