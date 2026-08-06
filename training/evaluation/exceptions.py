"""Exception hierarchy for the evaluation package.

Every failure raises :class:`EvaluationError` (or a subclass) with a stable
machine-readable ``code`` (``EV-<AREA>-<NNN>``), mirroring the convention used
by the other CropFusion packages (``TR-*``, ``MXAI-*``, ``MOD-*``).
"""

from __future__ import annotations

from shared.exceptions import CropFusionError


class EvaluationError(CropFusionError):
    """Base class for all evaluation errors."""

    code: str = "EV-ERROR"


class EvaluationConfigurationError(EvaluationError):
    """Raised when the evaluation configuration is invalid or inconsistent."""

    code = "EV-CONFIG-001"


class MetricComputationError(EvaluationError):
    """Raised when a metric cannot be computed."""

    code = "EV-METRIC-001"


class ComparisonError(EvaluationError):
    """Raised when a comparison table cannot be built."""

    code = "EV-COMP-001"


class AblationStudyError(EvaluationError):
    """Raised when an ablation study cannot be produced."""

    code = "EV-ABL-001"


class ErrorAnalysisError(EvaluationError):
    """Raised when error analysis fails."""

    code = "EV-ERR-001"


class EvaluationReportError(EvaluationError):
    """Raised when an evaluation report cannot be generated."""

    code = "EV-RPT-001"
