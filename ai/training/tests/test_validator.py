"""Validator tests: validation loop and cross-validation fold generators."""

from __future__ import annotations

import pytest

from ai.training import (
    Validator,
    build_fold_generator,
    cross_validation_splits,
)
from ai.training.config import ValidationConfig


def _loader():
    import torch

    class _Fake:
        def __len__(self) -> int:
            return 2

        def __iter__(self):
            return iter(
                [
                    {"tabular": torch.randn(4, 5),
                     "crop_label": torch.randint(0, 3, (4,)),
                     "yield_label": torch.randn(4, 1)}
                    for _ in range(2)
                ]
            )

    return _Fake()


def test_validator_runs_and_returns_metrics(tabular_model, train_config):
    from ai.training import MultiTaskLoss

    loss_module = MultiTaskLoss(train_config.loss)
    validator = Validator(tabular_model, loss_module)
    result = validator.validate(_loader())
    assert "val_loss" in result.metrics
    assert "crop/accuracy" in result.metrics
    assert "yield/rmse" in result.metrics
    assert result.samples == 8


def test_fold_generators_partition_all_observations():
    observations = [f"obs_{i}" for i in range(20)]
    for strategy, kwargs in [
        ("kfold", {"k_folds": 4}),
        ("stratified_kfold", {"k_folds": 4}),
        ("temporal", {"k_folds": 4}),
    ]:
        config = ValidationConfig(strategy=strategy, **kwargs)
        splits = cross_validation_splits(observations, config)
        assert len(splits) == 4
        for train, val in splits:
            assert set(train).isdisjoint(set(val))
            assert len(train) + len(val) == 20


def test_fold_union_covers_everything():
    observations = [f"obs_{i}" for i in range(12)]
    splits = cross_validation_splits(
        observations, ValidationConfig(strategy="kfold", k_folds=3)
    )
    covered = set()
    for _, val in splits:
        covered.update(val)
    assert covered == set(observations)


def test_spatial_folds_keep_groups_together():
    class Obs:
        def __init__(self, name: str, village: str) -> None:
            self.name = name
            self.location = type("L", (), {"admin": type("A", (), {"village": village})()})()

    observations = [
        Obs("a", "v1"), Obs("b", "v1"), Obs("c", "v1"),
        Obs("d", "v2"), Obs("e", "v2"),
        Obs("f", "v3"), Obs("g", "v3"),
    ]
    config = ValidationConfig(strategy="spatial", k_folds=3)
    splits = cross_validation_splits(observations, config)
    for train, val in splits:
        train_villages = {o.location.admin.village for o in train}
        val_villages = {o.location.admin.village for o in val}
        assert train_villages.isdisjoint(val_villages)
