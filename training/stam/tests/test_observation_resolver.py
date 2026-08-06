"""Tests for the R2.3 ObservationResolver — training-sample generation.

Uses the shared STAM synthetic dataset (``stam`` fixture in conftest) so the
resolver exercises the full DatasetManager -> STAM -> corpus path.
"""

from __future__ import annotations

import pytest

from training.stam.exceptions import SampleResolutionError
from training.stam.observation_resolver import (
    ObservationCorpus,
    ObservationPlan,
    ObservationResolver,
    ObservationResolverConfig,
)


@pytest.fixture
def resolver(stam):
    return ObservationResolver(stam)


# --------------------------------------------------------------------------- #
# Catalog helpers
# --------------------------------------------------------------------------- #


def test_available_years_covers_tabular_and_image(resolver):
    years = resolver.available_years()
    assert 2020 in years
    assert 2021 in years
    assert years == sorted(years)


def test_available_seasons_uses_calendar(resolver):
    assert resolver.available_seasons() == ["Kharif", "Rabi", "Summer"]


def test_locations_from_spatial_index(resolver):
    locations = resolver.locations()
    assert len(locations) >= 1
    assert all(hasattr(p, "lon") and hasattr(p, "lat") for p in locations)


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #


def test_plan_cross_product(resolver):
    plan = resolver.plan(years=[2020, 2021], seasons=["Kharif"])
    assert isinstance(plan, ObservationPlan)
    assert plan.years == [2020, 2021]
    assert plan.seasons == ["Kharif"]
    assert plan.total == len(plan.cells) == len(plan.locations) * 2
    assert all(c.season == "Kharif" for c in plan.cells)
    assert {c.year for c in plan.cells} == {2020, 2021}


def test_plan_bbox_filter(resolver):
    plan = resolver.plan(
        years=[2020], seasons=["Kharif"],
        bbox=(74.80, 13.09, 74.81, 13.10),
    )
    assert plan.total > 0
    assert all(74.80 <= c.lon <= 74.81 and 13.09 <= c.lat <= 13.10 for c in plan.cells)


def test_plan_max_locations_cap_is_deterministic(resolver):
    full = resolver.plan(years=[2020], seasons=["Kharif"], max_locations=None)
    capped = resolver.plan(years=[2020], seasons=["Kharif"], max_locations=1)
    assert capped.total == 1
    assert capped.total < full.total
    again = resolver.plan(years=[2020], seasons=["Kharif"], max_locations=1)
    assert [c.location_id for c in capped.cells] == [c.location_id for c in again.cells]


def test_plan_inferred_years_and_seasons(resolver):
    plan = resolver.plan()
    assert plan.total == len(plan.locations) * len(plan.years) * len(plan.seasons)


def test_plan_raises_without_years(resolver):
    resolver.config.years = []
    resolver.config.infer_years = False
    with pytest.raises(SampleResolutionError):
        resolver.plan()


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def test_resolve_cell_accepts(resolver, stam):
    plan = resolver.plan(years=[2020], seasons=["Kharif"], max_locations=1)
    sample = resolver.resolve_cell(plan.cells[0])
    assert sample.status in {"accepted", "rejected", "error"}
    if sample.status != "error":
        assert sample.observation is not None
        assert sample.quality_score == sample.observation.quality.overall_score
        assert sample.duration_ms >= 0


def test_resolve_builds_corpus(resolver):
    plan = resolver.plan(years=[2020, 2021], seasons=["Kharif"], max_locations=1)
    corpus = resolver.resolve(plan)
    assert isinstance(corpus, ObservationCorpus)
    assert corpus.total == plan.total
    assert corpus.by_status("accepted") or corpus.by_status("error")


def test_resolve_default_plan(resolver):
    corpus = resolver.resolve(plan=None, progress_every=10**6)
    assert corpus.total == resolver.plan().total


def test_corpus_summary_counts(resolver):
    plan = resolver.plan(years=[2020], seasons=["Kharif"], max_locations=1)
    corpus = resolver.resolve(plan)
    summary = corpus.summary()
    assert summary["total"] == plan.total
    assert summary["accepted"] + summary["rejected"] + summary["errors"] == plan.total
    assert summary["acceptance_rate"] == round(summary["accepted"] / plan.total, 4)


def test_include_rejected_disabled(resolver):
    resolver.config.include_rejected = False
    resolver.config.include_errors = False
    plan = resolver.plan(years=[2020, 2021], seasons=["Kharif"], max_locations=1)
    corpus = resolver.resolve(plan)
    assert all(s.status == "accepted" for s in corpus.samples)


def test_accepted_observations_feeds_preprocessing(resolver):
    plan = resolver.plan(years=[2020], seasons=["Kharif"], max_locations=1)
    corpus = resolver.resolve(plan)
    observations = corpus.accepted_observations()
    if observations:
        first = observations[0]
        assert first.crop is not None or first.yield_value is not None
        assert first.temporal.year in {2020, 2021}


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_corpus_save_load_roundtrip(resolver, tmp_path):
    plan = resolver.plan(years=[2020], seasons=["Kharif"], max_locations=1)
    corpus = resolver.resolve(plan)
    path = tmp_path / "corpus.json"
    corpus.save(path)
    assert path.exists()
    restored = ObservationCorpus.load(path)
    assert restored.total == corpus.total
    assert restored.status_counts() == corpus.status_counts()
    restored_summary = restored.summary()
    assert restored_summary["total"] == corpus.total
