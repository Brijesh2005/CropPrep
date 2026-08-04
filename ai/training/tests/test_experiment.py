"""Experiment integration tests over the real STAM -> preprocessing chain."""

from __future__ import annotations

import pytest

from ai.training import Experiment
from ai.training.config import TrainingConfig


def test_experiment_holdout_full_pipeline(
    preprocessor, stam_chain, derived_model_config, integration_training_config
):
    report = Experiment(
        integration_training_config,
        _observations(preprocessor, stam_chain),
        preprocessor=preprocessor,
        extractor=stam_chain.get_patch,
        model_config=derived_model_config,
    ).run()
    assert report.evaluation is not None
    assert "crop/accuracy" in report.evaluation.metrics
    assert "yield/rmse" in report.evaluation.metrics
    assert report.artifacts.get("dashboard") is not None
    assert (report.run_dir / "config.yaml").exists()


def test_experiment_cross_validation(
    preprocessor, stam_chain, derived_model_config, tmp_path
):
    config = TrainingConfig(
        name="cv",
        general={"device": "cpu", "seed": 42,
                 "output_dir": str(tmp_path / "out")},
        train={"epochs": 1, "early_stopping_patience": 3},
        validation={"strategy": "kfold", "k_folds": 2},
        checkpoint={"directory": str(tmp_path / "ckpt")},
        logging={"console": False},
        visualization={"enabled": False},
    )
    observations = _observations(preprocessor, stam_chain)
    report = Experiment(
        config, observations, preprocessor=preprocessor,
        extractor=stam_chain.get_patch, model_config=derived_model_config,
    ).run()
    assert report.config_snapshot["validation_strategy"] == "kfold"
    assert (report.run_dir / "cross_validation.json").exists()


def _observations(preprocessor, stam_chain) -> list:
    """Re-derive accepted observations from the fitted preprocessor chain."""
    from services.spatial_alignment.tests.conftest import _build_synthetic_dataset  # noqa

    obs = []
    for lon, lat in [(74.801, 13.099), (74.802, 13.098), (74.803, 13.097)]:
        obs.append(stam_chain.build_observation(lon, lat, year=2020, season="Kharif"))
    obs.append(stam_chain.build_observation(74.802, 13.098, year=2021, season="Kharif"))
    accepted, _ = preprocessor.filter(obs)
    return accepted
