"""Configuration tests: validation, YAML round-trip, template, derivation."""

from __future__ import annotations

from pathlib import Path

import pytest

from training.models import ModelConfig, load_model_config, save_model_template
from training.models.exceptions import ModelConfigurationError


def test_defaults_valid():
    cfg = ModelConfig()
    assert cfg.uses_tabular or cfg.uses_image


def test_requires_modality():
    with pytest.raises(Exception):  # pydantic ValidationError
        ModelConfig(tabular={"numeric_dim": 0, "categorical_cardinalities": []},
                    image_encoder={"backbone": None})


def test_requires_head():
    with pytest.raises(Exception):
        ModelConfig(heads={"crop": None, "yield_prediction": None})


def test_rejects_bad_fusion_method():
    with pytest.raises(Exception):
        ModelConfig(image_fusion={"method": "magic"})


def test_rejects_bad_activation():
    with pytest.raises(Exception):
        ModelConfig(tabular={"activation": "swishy"})


def test_rejects_heads_not_dividing_dim():
    with pytest.raises(Exception):
        ModelConfig(tabular={"numeric_dim": 2, "embedding_dim": 64, "num_heads": 7})


def test_yaml_round_trip(tmp_path: Path):
    path = tmp_path / "model.yaml"
    cfg = ModelConfig(name="roundtrip", version="9.9")
    cfg.save(path)
    loaded = load_model_config(path)
    assert loaded.name == "roundtrip"
    assert loaded.version == "9.9"
    assert loaded.model_dump() == cfg.model_dump()


def test_template_written(tmp_path: Path):
    path = save_model_template(tmp_path / "template.yaml")
    assert path.exists()
    loaded = load_model_config(path)
    assert loaded.uses_image  # default backbone present


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(ModelConfigurationError):
        load_model_config(tmp_path / "nope.yaml")


def test_env_overrides(tmp_path: Path):
    path = save_model_template(tmp_path / "template.yaml")
    cfg = load_model_config(
        env={
            "MODEL_CONFIG_FILE": str(path),
            "MODEL_SHARED_ENCODER__OUT_DIM": "768",
            "MODEL_HEADS__CROP__NUM_CLASSES": "10",
        }
    )
    assert cfg.shared_encoder.out_dim == 768
    assert cfg.heads.crop.num_classes == 10


def test_derivation_from_preprocessor_ordinal(preprocessor_ordinal):
    cfg = ModelConfig.from_preprocessor(preprocessor_ordinal)
    # rainfall_mm numeric + village/district categorical (2 slots)
    assert cfg.tabular.numeric_dim == 1
    assert len(cfg.tabular.categorical_cardinalities) == 2
    assert cfg.tabular_feature_dim == 1 + 2
    assert cfg.heads.crop.num_classes >= 1
    assert cfg.image_encoder.input_size == 16


def test_derivation_from_preprocessor_onehot(preprocessor_onehot):
    cfg = ModelConfig.from_preprocessor(preprocessor_onehot)
    # one-hot -> whole vector consumed as continuous
    assert cfg.tabular.categorical_cardinalities == []
    assert cfg.tabular.numeric_dim == len(preprocessor_onehot.tabular.feature_names)
    assert cfg.tabular_feature_dim == len(preprocessor_onehot.tabular.feature_names)


def test_factory_requires_fitted_preprocessor():
    from training.models import ModelFactory
    from training.preprocessing import Preprocessor

    with pytest.raises(ModelConfigurationError):
        ModelFactory.from_preprocessor(Preprocessor())
