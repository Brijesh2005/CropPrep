"""Tests for the shared exception hierarchy."""

from __future__ import annotations

import pytest

from shared.exceptions import (
    AuthenticationError,
    CacheError,
    ConfigurationError,
    CropFusionError,
    IntegrityError,
    InvalidVersionError,
    NotFoundError,
    PredictionError,
    SerializationError,
    ValidationFailedError,
    VersionError,
)


def test_base_error_str_without_detail() -> None:
    err = CropFusionError("boom")
    assert str(err) == "CF-ERROR: boom"
    assert err.code == "CF-ERROR"
    assert err.detail is None
    assert err.suggested_resolution is None


def test_base_error_str_with_detail() -> None:
    err = CropFusionError("boom", detail={"path": "/x"})
    assert str(err) == "CF-ERROR: boom (detail={'path': '/x'})"


def test_base_error_suggested_resolution() -> None:
    err = ConfigurationError("bad", suggested_resolution="fix the yaml")
    assert err.suggested_resolution == "fix the yaml"


def test_domain_codes_unique_and_stable() -> None:
    seen: set[str] = set()
    for cls in (ConfigurationError, NotFoundError, IntegrityError, CacheError,
                SerializationError, ValidationFailedError, VersionError,
                InvalidVersionError, PredictionError, AuthenticationError):
        code = cls.code
        assert code.startswith("CF-")
        assert code not in seen
        seen.add(code)


def test_all_errors_share_base() -> None:
    for cls in (ConfigurationError, NotFoundError, IntegrityError, CacheError,
                SerializationError, ValidationFailedError, VersionError,
                InvalidVersionError, PredictionError, AuthenticationError):
        assert issubclass(cls, CropFusionError)


def test_invalid_version_raises() -> None:
    from shared.versioning import SemanticVersion

    with pytest.raises(InvalidVersionError):
        SemanticVersion.from_string("not-a-version")
