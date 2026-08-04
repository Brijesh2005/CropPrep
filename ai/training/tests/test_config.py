"""Training configuration tests: validation, YAML round-trip, env overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai.training import TrainingConfig, load_training_config, save_training_template
from ai.training.exceptions import TrainingConfigurationError


def test_defaults_valid():
    cfg = TrainingConfig()
    assert cfg.train.epochs == 100
    assert cfg.optimizer.name == "adamw"
    assert cfg.scheduler.name == "cosine"


def test_optimizer_name_validated_at_build_time():
    # The optimizer name is validated by the factory, not the config schema.
    cfg = TrainingConfig(optimizer={"name": "nope"})
    assert cfg.optimizer.name == "nope"
    with pytest.raises(Exception):
        from ai.training import build_optimizer

        build_optimizer(
            _dummy_model(), cfg.optimizer
        )


def _dummy_model():
    import torch.nn as nn

    return nn.Linear(2, 2)


def test_rejects_bad_amp_dtype():
    with pytest.raises(Exception):
        TrainingConfig(general={"amp_dtype": "float8"})


def test_yaml_round_trip(tmp_path: Path):
    path = tmp_path / "training.yaml"
    cfg = TrainingConfig(name="roundtrip", train={"epochs": 7})
    cfg.save(path)
    loaded = load_training_config(path)
    assert loaded.name == "roundtrip"
    assert loaded.train.epochs == 7


def test_template_written(tmp_path: Path):
    path = save_training_template(tmp_path / "template.yaml")
    assert path.exists()
    loaded = load_training_config(path)
    assert loaded.train.epochs == 100


def test_env_overrides(tmp_path: Path):
    path = save_training_template(tmp_path / "template.yaml")
    cfg = load_training_config(
        env={
            "TRN_CONFIG_FILE": str(path),
            "TRN_TRAIN__EPOCHS": "12",
            "TRN_OPTIMIZER__LR": "0.0005",
            "TRN_LOSS__WEIGHTING_MODE": "gradnorm",
        }
    )
    assert cfg.train.epochs == 12
    assert cfg.optimizer.lr == 0.0005
    assert cfg.loss.weighting_mode == "gradnorm"


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(TrainingConfigurationError):
        load_training_config(tmp_path / "nope.yaml")
