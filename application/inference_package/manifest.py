"""Expected artifact manifest for an exported inference package.

The Prediction Platform **consumes** an inference package produced by the
Training Platform export pipeline. It never generates one. This manifest is
the shared contract between the two: the exporter must emit these files, the
inference validator (``application.inference.validation``) checks them, and
the model loader (``application.inference.loaders``) reads them.

Files land in ``application/inference_package/``; model weights live beside
them under ``application/models/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FileKind = Literal["db", "data", "serialized", "config", "metadata"]


@dataclass(frozen=True, slots=True)
class ExpectedArtifact:
    """One expected file in an inference package."""

    filename: str
    kind: FileKind
    required: bool = True
    description: str = ""


INFERENCE_PACKAGE_FILES: tuple[ExpectedArtifact, ...] = (
    ExpectedArtifact(
        "metadata.db",
        "db",
        description="SQLite store for inference metadata / history lookups",
    ),
    ExpectedArtifact(
        "historical_context.parquet",
        "data",
        description="Long-run historical context (climatology / seasonality)",
    ),
    ExpectedArtifact(
        "location_index.parquet",
        "data",
        description="Index of known locations for reverse geocoding",
    ),
    ExpectedArtifact(
        "feature_scalers.pkl",
        "serialized",
        description="Fitted feature scalers used by preprocessing",
    ),
    ExpectedArtifact(
        "label_encoder.pkl",
        "serialized",
        description="Fitted crop-label encoder",
    ),
    ExpectedArtifact(
        "model_config.yaml",
        "config",
        description="Model architecture configuration",
    ),
    ExpectedArtifact(
        "dataset_version.json",
        "metadata",
        description="Dataset version used to train the model",
    ),
    ExpectedArtifact(
        "model_version.json",
        "metadata",
        description="Model version / checksum / status",
    ),
    ExpectedArtifact(
        "metrics.json",
        "metadata",
        description="Evaluation metrics recorded at export time",
    ),
    ExpectedArtifact(
        "README.md",
        "metadata",
        required=False,
        description="Human-readable package description",
    ),
)

#: The real model weights are NOT inside the package; they live in
#: ``application/models`` as ``cropfusion.pt`` (and future ``cropfusion_*.pt``).
MODEL_WEIGHTS_RELATIVE_DIR = "models"
MODEL_WEIGHTS_DEFAULT_NAME = "cropfusion.pt"
MODEL_WEIGHTS_FUTURE_PATTERN = "cropfusion_{version}.pt"

__all__ = [
    "ExpectedArtifact",
    "INFERENCE_PACKAGE_FILES",
    "MODEL_WEIGHTS_DEFAULT_NAME",
    "MODEL_WEIGHTS_FUTURE_PATTERN",
    "MODEL_WEIGHTS_RELATIVE_DIR",
]
