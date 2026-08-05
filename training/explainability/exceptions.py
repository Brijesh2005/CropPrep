"""Exception hierarchy for the explainability package.

Every failure raises :class:`ExplainabilityError` (or a subclass) with a stable
machine-readable ``code`` (``MXAI-<AREA>-<NNN>``), mirroring the convention used
by the other CropFusion packages (``MDL-*``, ``PP-*``, ``TR-*``).
"""

from __future__ import annotations

from typing import Any


class ExplainabilityError(Exception):
    """Base class for all explainability errors."""

    code: str = "MXAI-ERROR"

    def __init__(self, message: str, *, detail: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def __str__(self) -> str:
        text = f"{self.code}: {self.message}"
        if self.detail is not None:
            text += f" (detail={self.detail!r})"
        return text


class ExplainabilityConfigurationError(ExplainabilityError):
    """Raised when the explainability configuration is invalid."""

    code = "MXAI-CONFIG-001"


class ShapError(ExplainabilityError):
    """Raised when SHAP attribution fails."""

    code = "MXAI-SHAP-001"


class CamError(ExplainabilityError):
    """Raised when a CAM (GradCAM) explanation fails."""

    code = "MXAI-CAM-001"


class AttentionError(ExplainabilityError):
    """Raised when attention extraction fails."""

    code = "MXAI-ATTN-001"


class CounterfactualError(ExplainabilityError):
    """Raised when a counterfactual perturbation fails."""

    code = "MXAI-CF-001"


class UncertaintyError(ExplainabilityError):
    """Raised when confidence / uncertainty estimation fails."""

    code = "MXAI-UNC-001"


class AttributionError(ExplainabilityError):
    """Raised when an attribution method (e.g. integrated gradients) fails."""

    code = "MXAI-ATTR-001"


class ExportError(ExplainabilityError):
    """Raised when an explanation cannot be exported."""

    code = "MXAI-EXPORT-001"


class ReportError(ExplainabilityError):
    """Raised when an explanation report cannot be generated."""

    code = "MXAI-REPORT-001"


class VisualizationError(ExplainabilityError):
    """Raised when a visualization cannot be produced."""

    code = "MXAI-VIZ-001"
