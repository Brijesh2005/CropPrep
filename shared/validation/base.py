"""Validation framework core: Validator port + result models.

The shared validation framework provides a uniform way to validate different
kinds of artefacts (CSV files, images/rasters, metadata records, config
mappings, schemas and versions).  Each validator returns a
:class:`ValidationResult` describing whether it passed and any issues found.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..enums import FAILING_SEVERITY, Severity


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A single issue found during validation."""

    code: str
    severity: Severity
    message: str
    path: str | None = None
    detail: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "path": self.path,
            "detail": self.detail,
        }


@dataclass(slots=True)
class ValidationResult:
    """Aggregated outcome of a validation run."""

    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    target: str | None = None
    validated_at: datetime = field(default_factory=datetime.now)

    @property
    def failing_issues(self) -> list[ValidationIssue]:
        """Issues with ERROR / CRITICAL severity."""
        return [i for i in self.issues if i.severity in FAILING_SEVERITY]

    def by_severity(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for issue in self.issues:
            key = issue.severity.value
            out[key] = out.get(key, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "target": self.target,
            "validated_at": self.validated_at.isoformat(),
            "by_severity": self.by_severity(),
            "issues": [i.to_dict() for i in self.issues],
        }


class Validator(ABC):
    """Port: validate a target artefact and report issues."""

    #: Stable name of this validator.
    name: str = "validator"

    @abstractmethod
    def validate(self, target: Any, **context: Any) -> ValidationResult:
        """Validate ``target`` and return a :class:`ValidationResult`."""
