"""Exception hierarchy for the AI model package.

Every failure raises :class:`ModelError` (or a subclass) with a stable
machine-readable ``code`` (``MDL-<AREA>-<NNN>``), mirroring the convention
used by the other CropFusion packages (``PP-*``, ``ST-*``).
"""

from __future__ import annotations

from typing import Any

from shared.exceptions import CropFusionError


class ModelError(CropFusionError):
    """Base class for all AI model errors."""

    code: str = "MDL-ERROR"


class ModelConfigurationError(ModelError):
    """Raised when the model configuration is invalid or inconsistent."""

    code = "MDL-CONFIG-001"


class ModelInputError(ModelError):
    """Raised when an input tensor fails validation (shape / dtype / value)."""

    code = "MDL-INPUT-001"


class ShapeMismatchError(ModelError):
    """Raised when tensor shapes do not match the expected contract."""

    code = "MDL-SHAPE-001"


class MissingDependencyError(ModelError):
    """Raised when an optional dependency (onnx, tensorrt) is unavailable."""

    code = "MDL-DEP-001"


class CheckpointError(ModelError):
    """Raised when a checkpoint cannot be saved, loaded or resumed."""

    code = "MDL-CKPT-001"


class ExportError(ModelError):
    """Raised when model export (TorchScript / ONNX / TensorRT) fails."""

    code = "MDL-EXPORT-001"
