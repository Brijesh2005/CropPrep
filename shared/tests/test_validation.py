"""Tests for the shared validation framework."""

from __future__ import annotations

import csv

from shared.enums import Severity
from shared.exceptions import ValidationNotSupportedError
from shared.validation import (
    ConfigValidator,
    CsvValidator,
    MetadataValidator,
    ValidationIssue,
    ValidationResult,
    default_registry,
    validate,
)


def _write_csv(tmp_path, name: str, rows: list[list[str]]) -> None:
    with open(tmp_path / name, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def test_default_registry_has_builtins() -> None:
    assert default_registry.names() == ["config", "csv", "image", "metadata", "version"]


def test_csv_validator_ok(tmp_path) -> None:
    _write_csv(tmp_path, "ok.csv", [["a", "b"], ["1", "2"], ["3", "4"]])
    result = validate(tmp_path / "ok.csv", "csv")
    assert result.passed
    assert result.issues == []


def test_csv_validator_missing_file_fails(tmp_path) -> None:
    result = validate(tmp_path / "missing.csv", "csv")
    assert not result.passed
    assert result.issues[0].code == "CSV-001"
    assert result.issues[0].severity is Severity.ERROR


def test_csv_validator_missing_header_warns(tmp_path) -> None:
    _write_csv(tmp_path, "nohdr.csv", [[], ["1", "2"], ["3", "4"]])
    result = validate(tmp_path / "nohdr.csv", "csv")
    assert result.passed
    assert any(i.code == "CSV-004" for i in result.issues)


def test_metadata_validator_mapping(tmp_path) -> None:
    record = {"relative_path": "raw/2020/x.tif", "category": "geotiff", "index_type": "NDVI"}
    result = validate(record, "metadata")
    assert result.passed


def test_metadata_validator_missing_key(tmp_path) -> None:
    result = validate({"category": "csv"}, "metadata")
    assert not result.passed
    assert result.issues[0].code == "META-002"


def test_config_validator_mapping() -> None:
    result = validate({"dataset_root": "/data", "scan": {"workers": 4}}, "config")
    assert result.passed


def test_config_validator_not_a_mapping() -> None:
    result = validate("not-a-mapping", "config")
    assert not result.passed
    assert result.issues[0].code == "CONF-001"


def test_result_helpers() -> None:
    passed = ValidationResult(passed=True)
    assert passed.passed
    assert passed.failing_issues == []
    failed = ValidationResult(
        passed=False,
        issues=[ValidationIssue(code="X", severity=Severity.ERROR, message="boom")],
    )
    assert len(failed.failing_issues) == 1
    assert failed.by_severity() == {"error": 1}


def test_validate_unknown_name_raises() -> None:
    import pytest

    with pytest.raises(ValidationNotSupportedError):
        validate({"a": 1}, "nope")


def test_validators_are_registered_instances() -> None:
    assert isinstance(default_registry.get("csv"), CsvValidator)
    assert isinstance(default_registry.get("config"), ConfigValidator)
    assert isinstance(default_registry.get("metadata"), MetadataValidator)
