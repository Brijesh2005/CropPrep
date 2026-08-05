"""Tests for the shared enums."""

from __future__ import annotations

import enum

from shared.enums import (
    CropType,
    DatasetStatus,
    FAILING_SEVERITY,
    FileCategory,
    IndexType,
    ModelStatus,
    Resolution,
    Season,
    Severity,
    TrainingStage,
    ValidationStatus,
)


def _is_str_enum(cls: type[enum.Enum]) -> bool:
    return issubclass(cls, str) and issubclass(cls, enum.Enum)


def test_enums_are_str_enums() -> None:
    for cls in (IndexType, Resolution, FileCategory, Severity, DatasetStatus,
                ValidationStatus, CropType, Season, ModelStatus, TrainingStage):
        assert _is_str_enum(cls)


def test_index_type_values() -> None:
    assert IndexType.NDVI.value == "NDVI"
    assert IndexType.EVI.value == "EVI"
    assert IndexType.NONE.value == "NONE"


def test_resolution_values() -> None:
    assert Resolution.R10M.value == "R10m"
    assert Resolution.R20M.value == "R20m"


def test_failing_severity() -> None:
    assert Severity.ERROR in FAILING_SEVERITY
    assert Severity.CRITICAL in FAILING_SEVERITY
    assert Severity.WARNING not in FAILING_SEVERITY


def test_roundtrip_via_value() -> None:
    assert IndexType(IndexType.NDVI.value) is IndexType.NDVI
    assert Season(Season.KHARIF.value) is Season.KHARIF
    assert DatasetStatus(DatasetStatus.READY.value) is DatasetStatus.READY
