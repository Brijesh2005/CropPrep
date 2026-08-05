"""Model, training, checkpoint and inference related shared exceptions."""

from __future__ import annotations

from .base import CropFusionError


class ModelError(CropFusionError):
    """Base class for model-layer failures."""

    code = "CF-MODEL-001"


class TrainingError(ModelError):
    """Raised when a training run fails."""

    code = "CF-MODEL-002"


class CheckpointError(ModelError):
    """Raised when a checkpoint cannot be written, read or resumed."""

    code = "CF-MODEL-003"


class InferenceError(ModelError):
    """Raised when model inference fails."""

    code = "CF-MODEL-004"
