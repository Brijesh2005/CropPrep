"""Validation report schemas (generic)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..enums import Severity


@dataclass(slots=True)
class ValidationIssueSchema:
    """A single issue found during validation."""

    severity: Severity
    code: str
    category: str
    message: str
    path: str | None = None
    detail: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "category": self.category,
            "message": self.message,
            "path": self.path,
            "detail": self.detail,
        }


@dataclass(slots=True)
class ValidationReportSchema:
    """Aggregated outcome of a validation run."""

    root: str
    passed: bool
    issues: list[ValidationIssueSchema] = field(default_factory=list)
    files_scanned: int = 0
    validated_at: datetime = field(default_factory=datetime.now)

    def by_severity(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for issue in self.issues:
            key = issue.severity.value
            out[key] = out.get(key, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "passed": self.passed,
            "files_scanned": self.files_scanned,
            "validated_at": self.validated_at.isoformat(),
            "by_severity": self.by_severity(),
            "issues": [i.to_dict() for i in self.issues],
        }
