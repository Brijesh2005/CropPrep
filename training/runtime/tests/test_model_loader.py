"""Model loader tests (Phase R6)."""

from __future__ import annotations

import shutil

import pytest

from training.models import ModelExporter
from training.runtime import ReleaseLayout, RuntimeConfig
from training.runtime.exceptions import ModelLoadError, ModelWarmupError
from training.runtime.model_loader import ModelLoader
from training.runtime.tests.conftest import clone_release


def _cloned(release_env, tmp_path, name="clone_release"):
    target = tmp_path / name
    clone_release(release_env.release_path, target)
    return target


def _loader(release_path, config=None):
    return ModelLoader(ReleaseLayout(release_path), config)


def test_pytorch_load_and_health(release_env):
    loader = _loader(release_env.release_path)
    model = loader.load(backend="pytorch")
    assert model is not None
    health = loader.health()
    assert health.backend == "pytorch"
    assert health.loaded is True
    assert health.config_loaded is True
    assert health.metadata_loaded is True
    assert health.warmup_ok is False
    assert health.parameter_count > 0
    assert health.model_version == "1.0.0"


def test_auto_backend_prefers_pytorch(release_env):
    loader = _loader(release_env.release_path)
    loader.load()
    assert loader.health().backend == "pytorch"


def test_load_config(release_env):
    loader = _loader(release_env.release_path)
    loader.load(backend="pytorch")
    cfg = loader.load_config()
    assert cfg.tabular_feature_dim == 4
    assert cfg.uses_image is True
    assert cfg.uses_tabular is True


def test_load_metadata(release_env):
    loader = _loader(release_env.release_path)
    loader.load(backend="pytorch")
    metadata = loader.load_metadata()
    assert metadata["model_version"] == "1.0.0"
    assert "model_fingerprint" in metadata
    assert metadata["formats"] == ["pytorch"]


def test_warmup_runs_forward_pass(release_env):
    loader = _loader(release_env.release_path)
    loader.load(backend="pytorch")
    assert loader.warmup(steps=1, batch_size=2) is True
    assert loader.health().warmup_ok is True


def test_warmup_zero_steps(release_env):
    loader = _loader(release_env.release_path)
    loader.load(backend="pytorch")
    assert loader.warmup(steps=0) is True
    assert loader.health().warmup_ok is True


def test_warmup_before_load():
    loader = _loader("unused")
    with pytest.raises(ModelWarmupError):
        loader.warmup()


def test_unknown_backend(release_env):
    loader = _loader(release_env.release_path)
    with pytest.raises(ModelLoadError):
        loader.load(backend="tensorrt")


def test_missing_requested_backend(release_env):
    loader = _loader(release_env.release_path)
    with pytest.raises(ModelLoadError):
        loader.load(backend="onnx")


def test_no_backend_available(tmp_path):
    target = tmp_path / "empty"
    target.mkdir()
    loader = ModelLoader(ReleaseLayout(target))
    with pytest.raises(ModelLoadError):
        loader.load(backend="auto")


def test_corrupt_pytorch_file(release_env, tmp_path):
    target = _cloned(release_env, tmp_path)
    (target / "model" / "cropfusion.pt").write_bytes(b"garbage")
    loader = _loader(target)
    with pytest.raises(ModelLoadError):
        loader.load(backend="pytorch")


def test_missing_model_config(release_env, tmp_path):
    target = _cloned(release_env, tmp_path)
    (target / "configs" / "model_config.yaml").unlink()
    loader = _loader(target)
    with pytest.raises(ModelLoadError):
        loader.load(backend="pytorch")


def test_torchscript_backend(release_env, tmp_path):
    target = _cloned(release_env, tmp_path)
    out = target / "model" / "cropfusion.torchscript.pt"
    ModelExporter(release_env.model).export_torchscript(out)
    loader = _loader(target)
    model = loader.load(backend="torchscript")
    assert loader.health().backend == "torchscript"
    assert model is not None
    assert loader.warmup(steps=1, batch_size=2) is True


def test_onnx_backend(release_env, tmp_path):
    target = _cloned(release_env, tmp_path)
    out = target / "model" / "cropfusion.onnx"
    ModelExporter(release_env.model).export_onnx(out)
    loader = _loader(target)
    session = loader.load(backend="onnx")
    assert loader.health().backend == "onnx"
    assert session is not None
    assert loader.warmup(steps=1, batch_size=2) is True


def test_unload(release_env):
    loader = _loader(release_env.release_path)
    loader.load(backend="pytorch")
    loader.unload()
    health = loader.health()
    assert health.loaded is False
    assert health.backend is None


def test_configurable_warmup_batch(release_env):
    config = RuntimeConfig(model={"warmup_steps": 2, "warmup_batch_size": 3})
    loader = _loader(release_env.release_path, config)
    loader.load(backend="pytorch")
    assert loader.warmup() is True
    assert loader.health().warmup_ok is True
