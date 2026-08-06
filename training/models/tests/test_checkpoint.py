"""Checkpoint tests: save/load, resume, partial loading, pruning."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from training.models import CheckpointManager, ModelFactory
from training.models.exceptions import CheckpointError


def _manager(tmp_path: Path, keep_last: int | None = 3):
    return CheckpointManager(tmp_path / "ckpt", keep_last=keep_last)


def test_save_and_load_state(model, tmp_path: Path):
    ckpt = _manager(tmp_path)
    path = ckpt.save(model, epoch=1, step=100, metrics={"loss": 0.5})
    state = ckpt.load(path)
    assert state["epoch"] == 1
    assert state["step"] == 100
    assert state["metrics"]["loss"] == 0.5
    assert "model_state_dict" in state
    assert state["model_config"]["name"] == model.config.name


def test_save_returns_path_in_directory(model, tmp_path: Path):
    ckpt = _manager(tmp_path)
    path = ckpt.save(model, epoch=42)
    assert path.name == "checkpoint_epoch0042.pt"
    assert path.exists()


def test_load_state_into_model(model, config, tmp_path: Path):
    ckpt = _manager(tmp_path)
    path = ckpt.save(model)
    clone = ModelFactory.create(config)
    report = ckpt.load_state_into(clone, path)
    assert report.success
    model.eval()
    clone.eval()
    batch = model.sample_batch(batch_size=2, seq_len=4)
    with torch.no_grad():
        assert torch.allclose(
            model(batch).crop_logits, clone(batch).crop_logits, atol=1e-6
        )


def test_resume_restores_metadata_and_state(model, config, tmp_path: Path):
    ckpt = _manager(tmp_path)
    path = ckpt.save(model, epoch=7, metrics={"loss": 0.2}, extra={"note": "hi"})
    clone = ModelFactory.create(config)
    resume = ckpt.resume(path, model=clone)
    assert resume.epoch == 7
    assert resume.metrics["loss"] == 0.2
    assert resume.extra["note"] == "hi"
    assert resume.model is clone
    assert resume.model_config is not None


def test_partial_load_only_ndvi(model, config, tmp_path: Path):
    ckpt = _manager(tmp_path)
    path = ckpt.save(model)
    clone = ModelFactory.create(config)
    report = ckpt.partial_load(clone, path, include=[r"^ndvi_encoder\."])
    assert report.loaded_keys > 0
    # all provided keys matched the clone; everything else is "missing"
    assert report.unexpected_keys == []
    assert any(key.startswith("tab_encoder") for key in report.missing_keys)


def test_partial_load_exclude_backbone(model, config, tmp_path: Path):
    ckpt = _manager(tmp_path)
    path = ckpt.save(model)
    clone = ModelFactory.create(config)
    report = ckpt.partial_load(clone, path, exclude=[r"backbone\."])
    assert all("backbone" not in key for key in report.unexpected_keys)


def test_keep_last_prunes(model, tmp_path: Path):
    ckpt = _manager(tmp_path, keep_last=2)
    for epoch in range(1, 5):
        ckpt.save(model, epoch=epoch)
    checkpoints = list((tmp_path / "ckpt").glob("checkpoint_*.pt"))
    assert len(checkpoints) == 2


def test_keep_all_when_none(model, tmp_path: Path):
    ckpt = _manager(tmp_path, keep_last=None)
    for epoch in range(1, 4):
        ckpt.save(model, epoch=epoch)
    assert len(list((tmp_path / "ckpt").glob("checkpoint_*.pt"))) == 3


def test_load_missing_file_raises(tmp_path: Path):
    ckpt = _manager(tmp_path)
    with pytest.raises(CheckpointError):
        ckpt.load(tmp_path / "does_not_exist.pt")
