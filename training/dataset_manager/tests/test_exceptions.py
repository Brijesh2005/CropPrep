"""Unit tests for the exception hierarchy."""

from __future__ import annotations

import pytest

from training.dataset_manager.exceptions import (
    CacheError,
    CorruptedDatasetError,
    DatasetManagerError,
    DatasetNotFoundError,
    DownloadFailedError,
    InvalidConfigurationError,
    InvalidMetadataError,
    RegistryError,
    ScannerError,
    UnsupportedFormatError,
    ValidationFailedError,
)


@pytest.mark.parametrize(
    "exception, code",
    [
        (DatasetManagerError, "DM-ERROR"),
        (InvalidConfigurationError, "DM-CONFIG-001"),
        (DatasetNotFoundError, "DM-FIND-001"),
        (CorruptedDatasetError, "DM-CORRUPT-001"),
        (InvalidMetadataError, "DM-META-001"),
        (ValidationFailedError, "DM-VALID-001"),
        (DownloadFailedError, "DM-DL-001"),
        (CacheError, "DM-CACHE-001"),
        (RegistryError, "DM-REG-001"),
        (UnsupportedFormatError, "DM-FMT-001"),
        (ScannerError, "DM-SCAN-001"),
    ],
)
def test_error_codes(exception, code):
    assert exception("boom").code == code


def test_all_errors_are_catchable_as_base():
    with pytest.raises(DatasetManagerError):
        raise DownloadFailedError("nope")


def test_error_message_and_detail():
    error = CorruptedDatasetError("bad tiff", detail="path/to/file.tif")
    assert "bad tiff" in str(error)
    assert "path/to/file.tif" in str(error)
    assert error.detail == "path/to/file.tif"


def test_base_error_default_code():
    assert DatasetManagerError("x").code == "DM-ERROR"
