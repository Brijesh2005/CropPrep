"""Exception hierarchy for the training package.

Every failure raises :class:`TrainingError` (or a subclass) with a stable
machine-readable ``code`` (``TR-<AREA>-<NNN>``), mirroring the convention used
by the other CropFusion packages (``MDL-*``, ``PP-*``, ``ST-*``).
"""

from __future__ import annotations

from typing import Any

from shared.exceptions import CropFusionError


class TrainingError(CropFusionError):
    """Base class for all training errors."""

    code: str = "TR-ERROR"


class TrainingConfigurationError(TrainingError):
    """Raised when the training configuration is invalid or inconsistent."""

    code = "TR-CONFIG-001"


class OptimizerBuildError(TrainingError):
    """Raised when an optimizer cannot be built from its configuration."""

    code = "TR-OPT-001"


class SchedulerBuildError(TrainingError):
    """Raised when a scheduler cannot be built from its configuration."""

    code = "TR-SCHED-001"


class LossBuildError(TrainingError):
    """Raised when a loss / weight strategy cannot be built."""

    code = "TR-LOSS-001"


class MetricError(TrainingError):
    """Raised when a metric cannot be computed."""

    code = "TR-METRIC-001"


class CheckpointError(TrainingError):
    """Raised when a training checkpoint cannot be saved, loaded or resumed."""

    code = "TR-CKPT-001"


class TrainingRunError(TrainingError):
    """Raised when a training run fails."""

    code = "TR-RUN-001"


class ValidationError(TrainingError):
    """Raised when a validation / cross-validation strategy fails."""

    code = "TR-VAL-001"


class BenchmarkError(TrainingError):
    """Raised when benchmarking fails."""

    code = "TR-BENCH-001"


class VisualizationError(TrainingError):
    """Raised when a visualization cannot be produced."""

    code = "TR-VIZ-001"


class AblationError(TrainingError):
    """Raised when an ablation sweep cannot be produced."""

    code = "TR-ABL-001"
