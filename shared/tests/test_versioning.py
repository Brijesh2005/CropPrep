"""Tests for the shared versioning helpers."""

from __future__ import annotations

import pytest

from shared.exceptions import InvalidVersionError
from shared.versioning import (
    ApplicationVersion,
    DatasetVersion,
    InferenceVersion,
    ModelVersion,
    SemanticVersion,
    VersionProvider,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1.2.3", (1, 2, 3)),
        ("0.0.0", (0, 0, 0)),
        ("10.20.30", (10, 20, 30)),
        (" 2.0.1 ", (2, 0, 1)),
    ],
)
def test_parse_semver(raw: str, expected) -> None:
    v = SemanticVersion.from_string(raw)
    assert (v.major, v.minor, v.patch) == expected
    assert str(v) == raw.strip()


@pytest.mark.parametrize("bad", ["1.2", "1.2.3.4", "abc", "", "1.2.x", "v2.0.1", "01.2.3"])
def test_parse_invalid(bad: str) -> None:
    with pytest.raises(InvalidVersionError):
        SemanticVersion.from_string(bad)


def test_bump() -> None:
    v = SemanticVersion.from_string("1.2.3")
    assert str(v.bump("major")) == "2.0.0"
    assert str(v.bump("minor")) == "1.3.0"
    assert str(v.bump("patch")) == "1.2.4"


def test_bump_unknown_part_raises() -> None:
    with pytest.raises(InvalidVersionError):
        SemanticVersion.from_string("1.2.3").bump("everything")


def test_comparison() -> None:
    a = SemanticVersion.from_string("1.2.3")
    b = SemanticVersion.from_string("2.0.0")
    assert a < b
    assert b > a
    assert a <= SemanticVersion.from_string("1.2.3")
    assert SemanticVersion.from_string("1.2.3") == a
    assert a != b


def test_version_info_kind_tags() -> None:
    assert DatasetVersion("ds", "1.0.0").kind == "dataset"
    assert ModelVersion("m", "1.0.0").kind == "model"
    assert InferenceVersion("i", "1.0.0").kind == "inference"
    assert ApplicationVersion("app", "1.0.0").kind == "application"


def test_version_info_repr() -> None:
    v = DatasetVersion("ds-1", "1.0.0")
    assert v.name == "ds-1"
    assert v.version == "1.0.0"
    assert "ds-1" in repr(v)


def test_version_provider_is_abstract_port() -> None:
    assert VersionProvider.__abstractmethods__ == {"current", "list", "bump"}


class _FakeProvider(VersionProvider):
    def __init__(self) -> None:
        self._version = "1.0.0"

    def current(self, name: str) -> str | None:
        return self._version

    def list(self, name: str) -> list[str]:
        return [self._version]

    def bump(self, name: str, part: str = "patch", *, message: str = "") -> str:
        self._version = str(SemanticVersion.from_string(self._version).bump(part))
        return self._version


def test_version_provider_implementation() -> None:
    provider = _FakeProvider()
    assert provider.current("ds") == "1.0.0"
    assert provider.bump("ds", "minor") == "1.1.0"
    assert provider.list("ds") == ["1.1.0"]
