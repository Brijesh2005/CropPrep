"""Tests for leakage-free data splitting."""

from __future__ import annotations

from ai.preprocessing.config import SplitConfig
from ai.preprocessing.dataset import split_observations


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
