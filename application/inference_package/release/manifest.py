"""Expected artifact manifest for an exported ``cropfusion_release/`` package.

This is the R6 (Prediction Platform, final phase) contract. It supersedes the
flat layout described in ``application/inference_package/manifest.py`` (that
file is left untouched — it documented an earlier, pre-R6 draft). The
Prediction Platform **only** reads this directory tree; it never touches
Kaggle, GeoTIFFs, or the Training Platform's internal dataset/model objects.

    cropfusion_release/
        model/
            cropfusion.pt
        metadata/
            metadata.db
            historical_context.parquet
            location_index.parquet
            village_metadata.parquet
        preprocess/
            scaler.pkl
            label_encoder.pkl
        configs/
            model.yaml
            inference.yaml
        version/
            manifest.json
            checksum.json
        reports/
            metrics.json
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FileKind = Literal["model", "db", "data", "serialized", "config", "version", "report"]


@dataclass(frozen=True, slots=True)
class ReleaseArtifact:
    """One expected file inside ``cropfusion_release/``, relative path included."""

    rel_path: str
    kind: FileKind
    required: bool = True
    description: str = ""


RELEASE_PACKAGE_FILES: tuple[ReleaseArtifact, ...] = (
    ReleaseArtifact("model/cropfusion.pt", "model", description="Exported CropFusion weights"),
    ReleaseArtifact("metadata/metadata.db", "db", description="SQLite inference metadata store"),
    ReleaseArtifact(
        "metadata/historical_context.parquet",
        "data",
        description="Historical climatology / seasonality context per location",
    ),
    ReleaseArtifact(
        "metadata/location_index.parquet",
        "data",
        description="Index of known lon/lat -> village/taluk/district",
    ),
    ReleaseArtifact(
        "metadata/village_metadata.parquet",
        "data",
        description="Per-village static metadata (soil class, elevation, etc.)",
    ),
    ReleaseArtifact("preprocess/scaler.pkl", "serialized", description="Fitted feature scaler"),
    ReleaseArtifact("preprocess/label_encoder.pkl", "serialized", description="Fitted crop label encoder"),
    ReleaseArtifact("configs/model.yaml", "config", description="Model architecture configuration"),
    ReleaseArtifact("configs/inference.yaml", "config", description="Inference-time configuration"),
    ReleaseArtifact("version/manifest.json", "version", description="Package file manifest"),
    ReleaseArtifact("version/checksum.json", "version", description="Per-file checksums (sha256)"),
    ReleaseArtifact("reports/metrics.json", "report", description="Evaluation metrics recorded at export time"),
)

#: Minimum manifest schema version this loader understands. ``version/manifest.json``
#: must declare a ``schema_version`` <= this value and a ``format`` we recognise.
SUPPORTED_MANIFEST_SCHEMA_VERSION = 1
SUPPORTED_PACKAGE_FORMAT = "cropfusion_release"

__all__ = [
    "FileKind",
    "RELEASE_PACKAGE_FILES",
    "ReleaseArtifact",
    "SUPPORTED_MANIFEST_SCHEMA_VERSION",
    "SUPPORTED_PACKAGE_FORMAT",
]
