"""Tests for leakage-free data splitting."""

from __future__ import annotations

from training.preprocessing.config import SplitConfig
from training.preprocessing.dataset import split_observations


class _Temporal:
    def __init__(self, year):
        self.year = year


class _Admin:
    def __init__(self, village):
        self.village = village


class _Loc:
    def __init__(self, village):
        self.admin = _Admin(village)


class _Tab:
    def __init__(self, fields):
        self.fields = fields


class _Obs:
    def __init__(self, year=2020, village="A", crop="Rice"):
        self.temporal = _Temporal(year)
        self.location = _Loc(village)
        self.tabular = _Tab({"village": village})
        self.crop = crop


def _observations():
    return [
        _Obs(year=2018, village="A"), _Obs(year=2018, village="B"),
        _Obs(year=2019, village="A"), _Obs(year=2019, village="B"),
        _Obs(year=2020, village="A"), _Obs(year=2020, village="B"),
        _Obs(year=2021, village="A"), _Obs(year=2021, village="B"),
    ]


def test_random_split_ratios():
    obs = _observations()
    train, val, test = split_observations(obs, SplitConfig(strategy="random", seed=1))
    assert len(train) + len(val) + len(test) == len(obs)
    assert len(test) > 0


def test_temporal_split_no_leakage():
    obs = _observations()
    train, val, test = split_observations(
        obs, SplitConfig(strategy="temporal", test_years=[2021], val_years=[2020])
    )
    assert {o.temporal.year for o in test} == {2021}
    assert {o.temporal.year for o in val} == {2020}
    assert {o.temporal.year for o in train} == {2018, 2019}


def test_temporal_split_recent_years_test():
    obs = _observations()
    train, _, test = split_observations(
        obs, SplitConfig(strategy="temporal", train_ratio=0.6, val_ratio=0.2,
                         test_ratio=0.2)
    )
    # The most recent year(s) go to test; no train sample shares test years.
    test_years = {o.temporal.year for o in test}
    train_years = {o.temporal.year for o in train}
    assert test_years.isdisjoint(train_years)


def test_temporal_split_two_years_falls_back_to_observation_split():
    # 2 distinct years (< min_years_for_temporal=3): must not raise, must fall
    # back to an observation-level split with no validation set, and must keep
    # every observation (a whole-year assignment would leave train empty).
    obs = [
        _Obs(year=2018, village="A"), _Obs(year=2018, village="B"),
        _Obs(year=2019, village="A"), _Obs(year=2019, village="B"),
    ]
    train, val, test = split_observations(
        obs, SplitConfig(strategy="temporal", seed=7)
    )
    assert val == []
    assert len(train) > 0 and len(test) > 0
    assert len(train) + len(test) == len(obs)
    # Observation-level split: train spans both years rather than a single year.
    assert {o.temporal.year for o in train} == {2018, 2019}


def test_temporal_split_single_year_falls_back():
    # 1 distinct year: must not raise and must produce non-empty train/test.
    obs = [
        _Obs(year=2018, village="A"), _Obs(year=2018, village="B"),
        _Obs(year=2018, village="C"), _Obs(year=2018, village="D"),
    ]
    train, val, test = split_observations(
        obs, SplitConfig(strategy="temporal", seed=7)
    )
    assert val == []
    assert len(train) > 0 and len(test) > 0
    assert len(train) + len(test) == len(obs)


def test_temporal_split_many_years_no_regression():
    # 5 distinct years: whole-year temporal holdout behaves exactly as before.
    obs = [
        _Obs(year=year, village=village)
        for year in range(2016, 2021)
        for village in ("A", "B")
    ]
    train, val, test = split_observations(
        obs, SplitConfig(strategy="temporal", seed=42)
    )
    train_years = {o.temporal.year for o in train}
    val_years = {o.temporal.year for o in val}
    test_years = {o.temporal.year for o in test}
    assert train_years and val_years and test_years
    assert train_years.isdisjoint(val_years)
    assert train_years.isdisjoint(test_years)
    assert val_years.isdisjoint(test_years)
    assert train_years | val_years | test_years == set(range(2016, 2021))
    # Whole-year assignment: all observations of a year land in one split.
    for year in range(2016, 2021):
        per_split = (
            sum(1 for o in train if o.temporal.year == year),
            sum(1 for o in val if o.temporal.year == year),
            sum(1 for o in test if o.temporal.year == year),
        )
        assert per_split.count(0) == 2


def test_temporal_split_min_years_threshold_configurable():
    # Raising the threshold above the year count forces the whole-year path;
    # lowering it below keeps the fallback off for that corpus size.
    obs = [
        _Obs(year=2018, village="A"), _Obs(year=2018, village="B"),
        _Obs(year=2019, village="A"), _Obs(year=2019, village="B"),
    ]
    train, val, test = split_observations(
        obs, SplitConfig(strategy="temporal", min_years_for_temporal=2, seed=7)
    )
    train_years = {o.temporal.year for o in train}
    test_years = {o.temporal.year for o in test}
    # Whole-year path: disjoint single years, no mixed-year train set.
    assert train_years.isdisjoint(test_years)
    assert len(train_years) == 1 and len(test_years) == 1


def test_spatial_split_no_village_leakage():
    obs = _observations()
    train, val, test = split_observations(
        obs, SplitConfig(strategy="spatial", seed=7)
    )
    train_villages = {o.location.admin.village for o in train}
    test_villages = {o.location.admin.village for o in test}
    assert train_villages.isdisjoint(test_villages)


def test_group_split():
    obs = _observations()
    train, _, test = split_observations(
        obs, SplitConfig(strategy="group", seed=3)
    )
    train_villages = {o.location.admin.village for o in train}
    test_villages = {o.location.admin.village for o in test}
    assert train_villages.isdisjoint(test_villages)


def test_stratified_preserves_classes():
    obs = [_Obs(crop="Rice") for _ in range(6)] + [_Obs(crop="Coconut") for _ in range(4)]
    train, val, test = split_observations(
        obs, SplitConfig(strategy="stratified", train_ratio=0.7, val_ratio=0.15,
                         test_ratio=0.15, seed=1)
    )
    assert len(train) == 7  # 70% of 10
    assert len(test) >= 1
    assert len(train) + len(val) + len(test) == len(obs)
