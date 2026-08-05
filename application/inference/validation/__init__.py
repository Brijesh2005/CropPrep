"""Inference package validation port (architecture contract).

Validates that an inference package directory (``application/inference_package``)
contains every artifact a future engine needs, before the loader touches it.
Reuses the ``shared.validation`` result types for a consistent report shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from shared.validation import ValidationResult


class InferencePackageValidator(ABC):
    """Port for validating an exported inference package."""

    @abstractmethod
    def validate(self, package_dir: Path) -> ValidationResult:
        """Check that ``package_dir`` holds the expected artifact set.

        Expected artifacts are listed in ``application/inference_package``
        (metadata.db, historical_context.parquet, location_index.parquet,
        feature_scalers.pkl, label_encoder.pkl, model_config.yaml,
        dataset_version.json, model_version.json, metrics.json).
        """


__all__ = ["InferencePackageValidator"]
