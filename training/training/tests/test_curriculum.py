"""Curriculum training tests: stage scheduling, freezing, resume, callback."""

from __future__ import annotations

import pytest

from training.models import ModelFactory
from training.training.config import TrainingConfig
from training.training.curriculum import (
    Curriculum,
    CurriculumCallback,
    CURRICULUM_STAGES,
    STAGE_ORDER,
    stage_for,
)
from training.training.exceptions import TrainingRunError

from training.training.tests.conftest import (
    make_fake_loader,
    small_full_config,
    small_tabular_config,
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_curriculum_config_defaults():
    cfg = TrainingConfig().curriculum
    assert cfg.enabled is False
    assert cfg.start_stage == 1
    assert cfg.epochs_per_stage is None


def test_curriculum_config_validation():
    with pytest.raises(Exception):
        TrainingConfig(curriculum={"start_stage": 0})
    with pytest.raises(Exception):
        TrainingConfig(curriculum={"start_stage": 6})


def test_curriculum_config_env_override():
    from training.training.config import load_training_config

    loaded = load_training_config(env={"TRN_CURRICULUM__ENABLED": "true",
                                       "TRN_CURRICULUM__START_STAGE": "3"})
    assert loaded.curriculum.enabled is True
    assert loaded.curriculum.start_stage == 3
    # Defaults untouched when nothing is overridden.
    assert TrainingConfig().curriculum.enabled is False


# --------------------------------------------------------------------------- #
# Stage registry
# --------------------------------------------------------------------------- #


def test_stage_registry_order():
    assert STAGE_ORDER == ("tabular", "image", "temporal", "fusion", "finetune")
    assert [s.number for s in CURRICULUM_STAGES] == [1, 2, 3, 4, 5]


def test_stage_for_lookup():
    assert stage_for(1).name == "tabular"
    assert stage_for("fusion").number == 4
    assert stage_for(5).name == "finetune"
    with pytest.raises(TrainingRunError):
        stage_for(9)
    with pytest.raises(TrainingRunError):
        stage_for("bogus")


# --------------------------------------------------------------------------- #
# Scheduling
# --------------------------------------------------------------------------- #


@pytest.fixture
def full_model():
    return ModelFactory.create(small_full_config())


def test_active_stages_filters_missing_modules():
    tabular_model = ModelFactory.create(small_tabular_config())
    cur = Curriculum(tabular_model, num_epochs=5)
    names = [s.name for s in cur.active_stages()]
    # tabular-only model: image / temporal / fusion components do not exist.
    assert names == ["tabular", "finetune"]


def test_active_stages_full_model(full_model):
    cur = Curriculum(full_model, num_epochs=5)
    assert [s.name for s in cur.active_stages()] == list(STAGE_ORDER)


def test_stage_epochs_even_split(full_model):
    cur = Curriculum(full_model, num_epochs=10)
    assert cur.stage_epochs() == [2, 2, 2, 2, 2]
    cur = Curriculum(full_model, num_epochs=7)
    assert cur.stage_epochs() == [2, 2, 1, 1, 1]


def test_stage_epochs_override(full_model):
    config = TrainingConfig(
        curriculum={
            "enabled": True,
            "epochs_per_stage": {"tabular": 3, "finetune": 4},
        }
    )
    cur = Curriculum(full_model, config.curriculum, num_epochs=10)
    assert cur.stage_epochs() == [3, 1, 1, 1, 4]


def test_stage_at_mapping(full_model):
    config = TrainingConfig(curriculum={"enabled": True})
    cur = Curriculum(full_model, config.curriculum, num_epochs=5)
    expected = ["tabular", "image", "temporal", "fusion", "finetune"]
    assert [cur.stage_at(e).name for e in range(5)] == expected


def test_stage_at_disabled_curriculum_is_finetune(full_model):
    cur = Curriculum(full_model, num_epochs=5)
    assert cur.stage_at(0).name == "finetune"


def test_start_stage_skips_earlier(full_model):
    config = TrainingConfig(curriculum={"enabled": True, "start_stage": 3})
    cur = Curriculum(full_model, config.curriculum, num_epochs=3)
    assert [s.name for s in cur.active_stages()] == ["temporal", "fusion", "finetune"]
    assert cur.stage_at(0).name == "temporal"
    assert cur.stage_at(2).name == "finetune"


# --------------------------------------------------------------------------- #
# Freezing
# --------------------------------------------------------------------------- #


def test_stage1_freezes_all_but_tabular(full_model):
    cur = Curriculum(full_model, num_epochs=5)
    info = cur.apply_stage(stage_for(1))
    assert info["stage"] == "tabular"
    assert "heads" in info["frozen"]
    assert "fusion_engine" in info["frozen"]
    assert "ndvi_encoder" in info["frozen"]
    for name, param in full_model.named_parameters():
        top = name.split(".")[0]
        assert param.requires_grad == (top == "tab_encoder")


def test_stage4_freezes_only_fusion_trainable(full_model):
    cur = Curriculum(full_model, num_epochs=5)
    info = cur.apply_stage(stage_for(4))
    assert info["stage"] == "fusion"
    assert "fusion_engine" in info["trainable"]
    assert not full_model.tab_encoder.blocks[0].self_attn.in_proj_weight.requires_grad
    assert full_model.fusion_engine.shared_encoder.blocks[0].linear1.weight.requires_grad


def test_stage5_unfreezes_everything(full_model):
    cur = Curriculum(full_model, num_epochs=5)
    cur.apply_stage(stage_for(1))
    assert not full_model.fusion_engine.shared_encoder.blocks[0].linear1.weight.requires_grad
    cur.apply_stage(stage_for(5))
    assert all(p.requires_grad for p in full_model.parameters())


def test_apply_eval_mode_keeps_frozen_in_eval(full_model):
    cur = Curriculum(full_model, num_epochs=5)
    cur.apply_stage(stage_for(1))
    full_model.train()  # would flatten everything back to train mode
    cur.apply_eval_mode()
    assert full_model.tab_encoder.training is True
    assert full_model.fusion_engine.training is False
    assert full_model.heads.training is False


def test_apply_eval_mode_stage5_trains_all(full_model):
    cur = Curriculum(full_model, num_epochs=5)
    cur.apply_stage(stage_for(5))
    full_model.eval()
    cur.apply_eval_mode()
    assert full_model.training is True


# --------------------------------------------------------------------------- #
# Callback
# --------------------------------------------------------------------------- #


def test_curriculum_callback_applies_stage(full_model):
    config = TrainingConfig(curriculum={"enabled": True})
    cur = Curriculum(full_model, config.curriculum, num_epochs=5)
    cb = CurriculumCallback(cur)
    assert cb.all_ranks is True  # DDP consistency
    cb.on_epoch_begin(0)
    assert cb.current_stage.name == "tabular"
    assert len(cb.stages_log) == 1
    assert cb.stages_log[0]["stage"] == "tabular"
    assert not full_model.fusion_engine.shared_encoder.blocks[0].linear1.weight.requires_grad
    cb.on_model_train_mode()
    assert full_model.tab_encoder.training is True
    assert full_model.fusion_engine.training is False


def test_curriculum_callback_end_to_end(full_model):
    """A full curriculum run via the base Trainer works (tabular -> finetune)."""
    import copy

    from training.training.trainer import Trainer

    config = TrainingConfig(
        name="curriculum_e2e",
        general={"device": "cpu", "seed": 42, "reports": False,
                 "output_dir": "artifacts/training"},
        train={"epochs": 5, "early_stopping_patience": 3},
        curriculum={"enabled": True},
        checkpoint={"save_best": False, "save_latest": False},
        logging={"console": False},
    )
    model = ModelFactory.create(small_full_config())
    cur = Curriculum(model, config.curriculum, num_epochs=5)
    cb = CurriculumCallback(cur)
    trainer = Trainer(
        model, make_fake_loader(n=16, batch_size=8, feature_dim=4,
                                multimodal=True), config, callbacks=[cb],
    )
    result = trainer.train()
    assert len(result.history) == 5
    stages = [log["stage"] for log in cb.stages_log]
    assert stages == ["tabular", "image", "temporal", "fusion", "finetune"]
