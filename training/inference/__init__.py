"""CropFusion inference package framework (Phase R5).

Produces the self-contained, predict-only inference package that ships the
trained model to the application platform without any dependency on the
Kaggle / Sentinel training catalog:

* :class:`ModelExporter` — PyTorch / TorchScript / ONNX model exports plus the
  ``model_config.yaml``, ``metrics.json``, ``metadata.json`` and
  ``checksums.json`` sidecars.
* :func:`persist_dataset_sources` — snapshots ``metadata.db``,
  ``historical_context.parquet`` and ``location_index.parquet`` from the
  Dataset Manager.
* :class:`InferencePackageBuilder` — assembles the 14 required artefacts plus
  the ``manifest.json`` (15 files).
* :class:`InferencePackageValidator` — file integrity, manifest consistency,
  model / config / dataset compatibility and a smoke test.
* Versioning — semver resolution + content fingerprints for the model / dataset
  / training run, stamped into ``dataset_version.json`` / ``model_version.json``
  and the manifest.
"""

from __future__ import annotations

from .config import (
    ExporterConfig,
    GeneralConfig,
    InferenceConfig,
    PackageConfig,
    ValidationConfig,
    load_inference_config,
    save_inference_template,
)
from .dataset_sources import DatasetSources, persist_dataset_sources
from .exceptions import (
    DatasetSourceError,
    ExportError,
    InferenceConfigurationError,
    InferenceError,
    PackageBuildError,
    PackageValidationError,
    VersioningError,
)
from .exporter import BUNDLE_FORMAT_FILES, ModelExporter, load_pytorch_model
from .package_builder import (
    REQUIRED_ARTIFACTS,
    BuildReport,
    InferencePackageBuilder,
)
from .reports import (
    generate_export_reports,
    generate_inference_package_reports,
)
from .validate import InferencePackageValidator, ValidationResult
from .versioning import (
    ResolvedVersions,
    build_version_files,
    bump_semver,
    content_sha256,
    file_sha256,
    git_commit,
    model_fingerprint,
    resolve_versions,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "InferenceConfig",
    "GeneralConfig",
    "ExporterConfig",
    "PackageConfig",
    "ValidationConfig",
    "load_inference_config",
    "save_inference_template",
    # Exporter
    "ModelExporter",
    "load_pytorch_model",
    "BUNDLE_FORMAT_FILES",
    # Dataset sources
    "DatasetSources",
    "persist_dataset_sources",
    # Package builder
    "InferencePackageBuilder",
    "BuildReport",
    "REQUIRED_ARTIFACTS",
    # Validation
    "InferencePackageValidator",
    "ValidationResult",
    # Versioning
    "ResolvedVersions",
    "resolve_versions",
    "build_version_files",
    "bump_semver",
    "content_sha256",
    "file_sha256",
    "git_commit",
    "model_fingerprint",
    # Reports
    "generate_export_reports",
    "generate_inference_package_reports",
    # Exceptions
    "InferenceError",
    "InferenceConfigurationError",
    "ExportError",
    "PackageBuildError",
    "PackageValidationError",
    "VersioningError",
    "DatasetSourceError",
]
