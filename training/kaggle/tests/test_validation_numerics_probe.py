"""Regression: validation_numerics_probe.py must load the training config.

TR: the R5.4 Kaggle run failed at validation_numerics_probe.py:299 with
``NameError: name 'load_training_cfg' is not defined`` — the probe imported
the canonical ``load_training_config`` but called the alias that
``run_pipeline`` imports as ``load_training_cfg``. These tests prove the
probe module (a) exposes the alias via the canonical loader and (b) can load
the shipped ``training/config/training.yaml`` exactly as line 301 does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_KAGGLE = Path(__file__).resolve().parents[1]  # training/kaggle
_TRAINING_CONFIG = Path(__file__).resolve().parents[2] / "config" / "training.yaml"


def _probe_module():
    """Import the probe (repo root is auto-inserted by the module itself)."""
    import sys

    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import training.kaggle.scripts.validation_numerics_probe as probe

    return probe


def test_probe_exposes_load_training_cfg_alias():
    """Regression: importing the probe must not raise NameError, and the
    ``load_training_cfg`` it calls at line 301 must resolve to the canonical
    ``load_training_config`` loader."""
    from training.training.config import load_training_config

    probe = _probe_module()
    assert callable(probe.load_training_cfg)
    assert probe.load_training_cfg is load_training_config


def test_probe_loads_shipped_training_yaml():
    """The exact call from validation_numerics_probe.py's config block must
    load the committed training.yaml and carry the R5.4 settings."""
    assert _TRAINING_CONFIG.exists(), f"missing {_TRAINING_CONFIG}"
    probe = _probe_module()

    cfg = probe.load_training_cfg(Path(_TRAINING_CONFIG))
    assert cfg.train.early_stopping_metric == "crop/macro_f1"
    assert cfg.train.early_stopping_mode == "max"
    assert cfg.optimizer.backbone_lr_multiplier == pytest.approx(0.3)
    assert cfg.scheduler.name == "warmup_cosine"
    assert cfg.fine_tuning.enabled is True


def test_probe_sibling_config_loaders():
    """The rest of the probe's configuration block uses the same loaders as
    run_pipeline, so their imports must resolve on the same code path. Note:
    model.yaml ships num_classes=0 (derived from the preprocessor at runtime),
    so we only assert the canonical loaders resolve, not their contents."""
    from training.models.config import ModelConfig
    from training.preprocessing.config import PreprocessingConfig, load_preprocessing_config

    probe = _probe_module()

    model_cfg = probe.load_model_cfg(Path(_REPO_KAGGLE.parent / "config" / "model.yaml"))
    assert isinstance(model_cfg, ModelConfig)
    assert model_cfg.image_encoder.backbone  # backbone name present in model.yaml

    pre_cfg = load_preprocessing_config(Path(_REPO_KAGGLE.parent / "config" / "preprocessing.yaml"))
    assert isinstance(pre_cfg, PreprocessingConfig)
    assert pre_cfg.augmentation.brightness_jitter == 0.0