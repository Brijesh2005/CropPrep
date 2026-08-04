"""Ablation tests: variant configs build and the runner compares results."""

from __future__ import annotations

import pytest

from ai.models import ModelConfig
from ai.training import (
    ABLATION_VARIANTS,
    AblationRunner,
    build_variant_config,
)
from ai.training.config import TrainingConfig


@pytest.mark.parametrize("variant", sorted(ABLATION_VARIANTS))
def test_variant_configs_build_and_forward(variant, full_config):
    import torch

    variant_config = build_variant_config(full_config, variant)
    from ai.models import ModelFactory

    model = ModelFactory.create(variant_config)
    model.eval()
    with torch.no_grad():
        batch = model.sample_batch(batch_size=2, seq_len=3)
        out = model(batch)
    assert out.crop_logits is not None
    assert out.yield_pred is not None


def test_variant_unknown_raises(full_config):
    with pytest.raises(Exception):
        build_variant_config(full_config, "not_a_variant")


def test_ablation_runner_comparison(
    preprocessor, stam_chain, derived_model_config, tmp_path
):
    observations = _accepted(preprocessor, stam_chain)
    config = TrainingConfig(
        name="ablation",
        general={"device": "cpu", "seed": 42,
                 "output_dir": str(tmp_path / "out")},
        train={"epochs": 1, "early_stopping_patience": 3},
        logging={"console": False},
        visualization={"enabled": False},
    )
    runner = AblationRunner(
        config,
        derived_model_config,
        preprocessor=preprocessor,
        observations=observations,
        extractor=stam_chain.get_patch,
    )
    report = runner.run(variants=["full", "only_tabular"])
    assert set(report.results) == {"full", "only_tabular"}
    assert all(data["multi_task_score"] is not None
               for data in report.results.values())
    assert (runner.output_dir / "ablation_report.json").exists()
    assert (runner.output_dir / "ablation_comparison.csv").exists()


def _accepted(preprocessor, stam_chain) -> list:
    obs = []
    for lon, lat in [(74.801, 13.099), (74.802, 13.098), (74.803, 13.097)]:
        obs.append(stam_chain.build_observation(lon, lat, year=2020, season="Kharif"))
    obs.append(stam_chain.build_observation(74.802, 13.098, year=2021, season="Kharif"))
    accepted, _ = preprocessor.filter(obs)
    return accepted
