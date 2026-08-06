"""Tests for the tabular / location feature builder."""

from __future__ import annotations

from training.feature_engineering.tabular import TabularFeatureBuilder


def test_build_contains_location_and_labels(accepted):
    if not accepted:
        import pytest

        pytest.skip("no accepted observations in fixture corpus")
    row = TabularFeatureBuilder().build(accepted[0])
    assert row["tab.lon"] == accepted[0].location.lon
    assert row["tab.lat"] == accepted[0].location.lat
    assert "tab.distance_km" in row
    assert "tab.year" in row
    assert "tab.crop" in row
    assert "tab.yield_value" in row
    assert row["tab.crop"] == accepted[0].crop


def test_build_without_prefix(accepted):
    if not accepted:
        import pytest

        pytest.skip("no accepted observations in fixture corpus")
    row = TabularFeatureBuilder().build(accepted[0], prefix="")
    assert "lon" in row and "crop" in row
    assert not any(k.startswith("tab.") for k in row)


def test_build_excludes_labels_when_disabled(accepted):
    if not accepted:
        import pytest

        pytest.skip("no accepted observations in fixture corpus")
    builder = TabularFeatureBuilder(_cfg(include_labels=False))
    row = builder.build(accepted[0])
    assert "tab.crop" not in row
    assert "tab.yield_value" not in row


def test_build_excludes_location_when_disabled(accepted):
    if not accepted:
        import pytest

        pytest.skip("no accepted observations in fixture corpus")
    builder = TabularFeatureBuilder(_cfg(include_location=False))
    row = builder.build(accepted[0])
    assert "tab.lon" not in row
    assert "tab.distance_km" not in row
    assert "tab.crop" in row


def test_fields_are_json_safe(accepted):
    if not accepted:
        import pytest

        pytest.skip("no accepted observations in fixture corpus")
    row = TabularFeatureBuilder().build(accepted[0])
    for value in row.values():
        assert isinstance(value, (str, int, float, bool, type(None), list))


def _cfg(**kwargs):
    from training.feature_engineering.config import TabularFeatureConfig

    return TabularFeatureConfig(**kwargs)
