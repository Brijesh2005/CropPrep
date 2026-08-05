"""Concrete validators: CSV / Image / Metadata / Config / Schema / Version."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..enums import Severity
from ..versioning import SemanticVersion
from ..utils import count_lines_fast, is_geotiff_path, is_csv_path
from .base import Validator, ValidationIssue, ValidationResult


class CsvValidator(Validator):
    """Validate a CSV file: presence, header, rows and size."""

    name = "csv"

    def __init__(self, *, require_rows: bool = True) -> None:
        self.require_rows = require_rows

    def validate(self, target: Any, **context: Any) -> ValidationResult:
        issues: list[ValidationIssue] = []
        path = Path(target)
        if not path.exists():
            issues.append(
                ValidationIssue(
                    code="CSV-001",
                    severity=Severity.ERROR,
                    message="CSV file does not exist",
                    path=str(path),
                )
            )
            return ValidationResult(passed=False, issues=issues, target="csv")

        if path.suffix.lower() not in {".csv", ".txt"}:
            issues.append(
                ValidationIssue(
                    code="CSV-002",
                    severity=Severity.WARNING,
                    message="File does not look like a CSV (unexpected extension)",
                    path=str(path),
                    detail=path.suffix,
                )
            )

        if path.stat().st_size == 0:
            issues.append(
                ValidationIssue(
                    code="CSV-003",
                    severity=Severity.ERROR,
                    message="CSV file is empty",
                    path=str(path),
                )
            )
            return ValidationResult(passed=False, issues=issues, target="csv")

        if self.require_rows:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                header = fh.readline().strip()
            if not header:
                issues.append(
                    ValidationIssue(
                        code="CSV-004",
                        severity=Severity.WARNING,
                        message="CSV file has no header row",
                        path=str(path),
                    )
                )
            rows = count_lines_fast(path) - 1
            if rows <= 0:
                issues.append(
                    ValidationIssue(
                        code="CSV-005",
                        severity=Severity.WARNING,
                        message="CSV file has no data rows",
                        path=str(path),
                    )
                )

        passed = all(i.severity in {Severity.INFO, Severity.WARNING} for i in issues)
        return ValidationResult(passed=passed, issues=issues, target="csv")


class ImageValidator(Validator):
    """Validate a raster file: exists, non-empty and GeoTIFF magic."""

    name = "image"

    def validate(self, target: Any, **context: Any) -> ValidationResult:
        issues: list[ValidationIssue] = []
        path = Path(target)
        if not path.exists():
            issues.append(
                ValidationIssue(
                    code="IMG-001",
                    severity=Severity.ERROR,
                    message="Raster file does not exist",
                    path=str(path),
                )
            )
            return ValidationResult(passed=False, issues=issues, target="image")

        if path.stat().st_size == 0:
            issues.append(
                ValidationIssue(
                    code="IMG-002",
                    severity=Severity.ERROR,
                    message="Raster file is empty",
                    path=str(path),
                )
            )
            return ValidationResult(passed=False, issues=issues, target="image")

        if path.suffix.lower() not in {".tif", ".tiff"}:
            issues.append(
                ValidationIssue(
                    code="IMG-003",
                    severity=Severity.WARNING,
                    message="Raster has an unexpected extension",
                    path=str(path),
                    detail=path.suffix,
                )
            )

        if not is_geotiff_path(path):
            issues.append(
                ValidationIssue(
                    code="IMG-004",
                    severity=Severity.ERROR,
                    message="File is not a valid TIFF/GeoTIFF (magic bytes mismatch)",
                    path=str(path),
                )
            )

        passed = all(i.severity in {Severity.INFO, Severity.WARNING} for i in issues)
        return ValidationResult(passed=passed, issues=issues, target="image")


class MetadataValidator(Validator):
    """Validate a metadata record mapping: required keys present."""

    name = "metadata"

    REQUIRED_KEYS = ("relative_path", "category")

    def validate(self, target: Any, **context: Any) -> ValidationResult:
        issues: list[ValidationIssue] = []
        data = target
        if not isinstance(data, dict):
            issues.append(
                ValidationIssue(
                    code="META-001",
                    severity=Severity.ERROR,
                    message="Metadata must be a mapping",
                )
            )
            return ValidationResult(passed=False, issues=issues, target="metadata")

        for key in self.REQUIRED_KEYS:
            if key not in data or data[key] in (None, ""):
                issues.append(
                    ValidationIssue(
                        code="META-002",
                        severity=Severity.ERROR,
                        message=f"Metadata is missing required key: {key}",
                        path=key,
                    )
                )

        if "index_type" in data and data["index_type"] not in {"NDVI", "EVI", "NONE"}:
            issues.append(
                ValidationIssue(
                    code="META-003",
                    severity=Severity.WARNING,
                    message="Metadata has an unknown index_type",
                    path="index_type",
                    detail=data["index_type"],
                )
            )

        passed = all(i.severity in {Severity.INFO, Severity.WARNING} for i in issues)
        return ValidationResult(passed=passed, issues=issues, target="metadata")


class ConfigValidator(Validator):
    """Validate a config mapping: must be a mapping with no empty root."""

    name = "config"

    def validate(self, target: Any, **context: Any) -> ValidationResult:
        issues: list[ValidationIssue] = []
        if not isinstance(target, dict):
            issues.append(
                ValidationIssue(
                    code="CONF-001",
                    severity=Severity.ERROR,
                    message="Configuration root must be a mapping",
                    detail=type(target).__name__,
                )
            )
            return ValidationResult(passed=False, issues=issues, target="config")
        passed = all(i.severity in {Severity.INFO, Severity.WARNING} for i in issues)
        return ValidationResult(passed=passed, issues=issues, target="config")


class SchemaValidator(Validator):
    """Validate a mapping against a schema (list of required key names)."""

    name = "schema"

    def __init__(self, required: list[str]) -> None:
        self.required = list(required)

    def validate(self, target: Any, **context: Any) -> ValidationResult:
        issues: list[ValidationIssue] = []
        if not isinstance(target, dict):
            issues.append(
                ValidationIssue(
                    code="SCHEMA-001",
                    severity=Severity.ERROR,
                    message="Schema target must be a mapping",
                )
            )
            return ValidationResult(passed=False, issues=issues, target="schema")

        for key in self.required:
            if key not in target:
                issues.append(
                    ValidationIssue(
                        code="SCHEMA-002",
                        severity=Severity.ERROR,
                        message=f"Missing required schema key: {key}",
                        path=key,
                    )
                )
        for key in target:
            if key in {"path", "root"} and target[key] is not None:
                if not os.path.exists(str(target[key])):
                    issues.append(
                        ValidationIssue(
                            code="SCHEMA-003",
                            severity=Severity.WARNING,
                            message=f"Referenced path does not exist: {key}",
                            path=key,
                            detail=str(target[key]),
                        )
                    )
        passed = all(i.severity in {Severity.INFO, Severity.WARNING} for i in issues)
        return ValidationResult(passed=passed, issues=issues, target="schema")


class VersionValidator(Validator):
    """Validate a version string against semantic versioning."""

    name = "version"

    def validate(self, target: Any, **context: Any) -> ValidationResult:
        issues: list[ValidationIssue] = []
        try:
            SemanticVersion.from_string(str(target))
        except Exception as exc:  # noqa: BLE001 - converted to an issue
            issues.append(
                ValidationIssue(
                    code="VER-001",
                    severity=Severity.ERROR,
                    message="Version is not valid semantic versioning",
                    detail=str(exc),
                )
            )
        passed = not issues
        return ValidationResult(passed=passed, issues=issues, target="version")
