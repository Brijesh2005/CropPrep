"""ModelFactory tests: construction, config files, freezing, loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from training.models import CheckpointManager, ModelConfig, ModelFactory


def test_create_from_config(config):
    model = ModelFactory.create(config)
    assert model.output_names == ["crop", "yield"]


def test_create_from_dict(config):
    model = ModelFactory.create(config.model_dump())
    assert model.output_names == ["crop", "yield"]


def test_from_config_file(config, tmp_path: Path):
    path = config.save(tmp_path / "model.yaml")
    model = ModelFactory.from_config_file(path)
    assert model.config.name == config.name


def test_save_config(config, tmp_path: Path):
    from training.models import load_model_config

    path = ModelFactory.save_config(config, tmp_path / "saved.yaml")
    reloaded = load_model_config(path)
    assert reloaded.name == config.name


def test_save_config_from_dict(config, tmp_path: Path):
    from training.models import load_model_config

    path = ModelFactory.save_config(config.model_dump(), tmp_path / "dict.yaml")
    reloaded = load_model_config(path)
    assert reloaded.name == config.name


def test_from_preprocessor_with_config_path(preprocessor_ordinal, tmp_path: Path):
    from training.models import save_model_template

    path = save_model_template(tmp_path / "base.yaml")
    model = ModelFactory.from_preprocessor(
        preprocessor_ordinal,
        config_path=path,
        image_encoder={"backbone": "mobilenetv3_small_050", "input_size": 16},
    )
    assert model.use_tabular and model.use_image
    assert model.config.heads.crop.num_classes == 2


def test_freeze_layers(model):
    frozen = ModelFactory.freeze_layers(model, [r"^ndvi_encoder\.backbone\."])
    assert len(frozen) > 0
    for name, param in model.named_parameters():
        if name in frozen:
            assert param.requires_grad is False


def test_freeze_backbone(model):
    frozen = ModelFactory.freeze_backbone(model)
    assert len(frozen) > 0
    ndvi_params = [
        p for n, p in model.named_parameters() if n.startswith("ndvi_encoder.backbone")
    ]
    assert all(p.requires_grad is False for p in ndvi_params)


def test_freeze_backbone_requires_image():
    cfg = ModelConfig(
        tabular={"numeric_dim": 3, "categorical_cardinalities": [2]},
        image_encoder={"backbone": None},
        shared_encoder={"d_model": 32, "depth": 1, "num_heads": 4, "ff_dim": 64,
                        "out_dim": 64},
        heads={"crop": {"num_classes": 2}, "yield_prediction": {}},
    )
    tab_only = ModelFactory.create(cfg)
    with pytest.raises(Exception):
        ModelFactory.freeze_backbone(tab_only)


def test_load_pretrained_roundtrip(model, config, tmp_path: Path):
    path = CheckpointManager(tmp_path).save(model)
    clone = ModelFactory.create(config)
    report = ModelFactory.load_pretrained(clone, path)
    assert report.success
    batch = model.sample_batch(batch_size=2, seq_len=4)
    model.eval()
    clone.eval()
    with torch.no_grad():
        a = model(batch)
        b = clone(batch)
    assert torch.allclose(a.crop_logits, b.crop_logits, atol=1e-6)


def test_load_backbone_partial(model, config, tmp_path: Path):
    path = CheckpointManager(tmp_path).save(model)
    clone = ModelFactory.create(config)
    report = ModelFactory.load_backbone(clone, path)
    assert report.loaded_keys > 0
    assert report.unexpected_keys == []  # every provided backbone key matched


def test_from_checkpoint(model, config, tmp_path: Path):
    path = CheckpointManager(tmp_path).save(model, epoch=3, metrics={"loss": 1.0})
    restored = ModelFactory.from_checkpoint(path)
    assert restored.output_names == model.output_names


def test_from_preprocessor_ordinal(preprocessor_ordinal, stam_chain):
    model = ModelFactory.from_preprocessor(
        preprocessor_ordinal,
        image_encoder={"backbone": "mobilenetv3_small_050", "input_size": 16},
    )
    assert model.use_tabular and model.use_image
    assert model.config.tabular.numeric_dim == 1
    assert len(model.config.tabular.categorical_cardinalities) == 2
