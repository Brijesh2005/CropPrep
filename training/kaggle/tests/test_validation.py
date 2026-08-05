"""R2.1 Training Validator tests."""

from __future__ import annotations

from pathlib import Path

from training.kaggle.config import (
    ConfigRegistry,
    EnvironmentRequirements,
    PathsConfig,
    WorkspaceConfig,
    WorkspaceLayout,
)
from training.kaggle.validation import TrainingValidator


def _ready_env() -> dict:
    return {
        "system": {"python_version": "3.12.9"},
        "gpu": {"available": True, "cuda_available": True, "device_count": 1},
        "dependencies": {
            "numpy": {"installed": True},
            "torch": {"installed": True},
        },
        "runtime": {},
    }


def _write_config_files(tmp_path: Path) -> None:
    for name in (
        "dataset",
        "training",
        "kaggle",
        "model",
        "logging",
        "paths",
        "validation",
    ):
        (tmp_path / f"{name}.yaml").write_text(f"{name}: ok\n", encoding="utf-8")


def _make_paths(tmp_path: Path) -> PathsConfig:
    return PathsConfig(
        workspace=WorkspaceConfig(
            root="ws", logs_dir="logs", outputs_dir="outputs",
            checkpoints_dir="checkpoints", cache_dir="cache", configs_dir="configs",
        ),
        config=ConfigRegistry(
            dataset="dataset.yaml",
            training="training.yaml",
            kaggle="kaggle.yaml",
            model="model.yaml",
            logging="logging.yaml",
            paths="paths.yaml",
            validation="validation.yaml",
        ),
        environment=EnvironmentRequirements(
            min_python="3.0",
            require_gpu=False,
            min_free_gb=0.0,
            required_dependencies=["numpy"],
            gpu_dependencies=["torch"],
        ),
    )


def _make_validator(tmp_path: Path, env: dict | None = None):
    paths = _make_paths(tmp_path)
    layout = WorkspaceLayout.resolve(paths, repo_root=tmp_path)
    for directory in (layout.logs, layout.outputs, layout.checkpoints, layout.cache, layout.configs):
        directory.mkdir(parents=True, exist_ok=True)
    return TrainingValidator(paths, layout, env or _ready_env())


def test_validate_passes_when_ready(tmp_path: Path) -> None:
    _write_config_files(tmp_path)
    validator = _make_validator(tmp_path)
    result = validator.validate(provider_manifests={"tabular": {"available": True}})
    assert result.passed is True
    assert result.by_severity()["info"] >= 6


def test_missing_config_fails(tmp_path: Path) -> None:
    validator = _make_validator(tmp_path)  # no config files written
    result = validator.validate()
    assert result.passed is False
    codes = {issue.code for issue in result.issues}
    assert "CONFIG_MISSING" in codes


def test_gpu_required_fails_without_gpu(tmp_path: Path) -> None:
    _write_config_files(tmp_path)
    env = _ready_env()
    env["gpu"] = {"available": False, "cuda_available": False, "device_count": 0}
    validator = _make_validator(tmp_path, env)
    validator.paths.environment.require_gpu = True
    result = validator.validate()
    assert result.passed is False
    assert "GPU_UNAVAILABLE" in {i.code for i in result.issues}


def test_gpu_not_required_passes_without_gpu(tmp_path: Path) -> None:
    _write_config_files(tmp_path)
    paths = _make_paths(tmp_path)
    paths.environment.require_gpu = False
    env = _ready_env()
    env["gpu"] = {"available": False, "cuda_available": False, "device_count": 0}
    layout = WorkspaceLayout.resolve(paths, repo_root=tmp_path)
    for d in (layout.logs, layout.outputs, layout.checkpoints, layout.cache, layout.configs):
        d.mkdir(parents=True, exist_ok=True)
    result = TrainingValidator(paths, layout, env).validate()
    assert result.passed is True


def test_missing_dependency_fails(tmp_path: Path) -> None:
    _write_config_files(tmp_path)
    env = _ready_env()
    env["dependencies"] = {"numpy": {"installed": False}, "torch": {"installed": True}}
    validator = _make_validator(tmp_path, env)
    result = validator.validate()
    assert "DEPENDENCY_MISSING" in {i.code for i in result.issues}


def test_missing_provider_warns(tmp_path: Path) -> None:
    _write_config_files(tmp_path)
    validator = _make_validator(tmp_path)
    result = validator.validate(provider_manifests={"image": {"available": False}})
    assert "PROVIDER_UNAVAILABLE" in {i.code for i in result.issues}
    assert result.passed is True  # warning only


def test_report_dict(tmp_path: Path) -> None:
    _write_config_files(tmp_path)
    validator = _make_validator(tmp_path)
    report = validator.report(provider_manifests={"tabular": {"available": True}})
    assert report["passed"] is True
    assert "by_severity" in report
    assert "issues" in report


def test_validation_result_uses_shared_types(tmp_path: Path) -> None:
    from shared.validation import ValidationResult

    _write_config_files(tmp_path)
    result = _make_validator(tmp_path).validate()
    assert isinstance(result, ValidationResult)
