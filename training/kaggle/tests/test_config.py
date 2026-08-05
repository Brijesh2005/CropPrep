"""R2.1 config layer tests: loading, inheritance and env overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from training.kaggle.config import (
    ConfigRegistry,
    EnvironmentRequirements,
    KaggleConfig,
    LoggingConfig,
    PathsConfig,
    WorkspaceConfig,
    WorkspaceLayout,
    load_kaggle_config,
    load_logging_config,
    load_paths_config,
)


def test_load_real_paths_config() -> None:
    paths = load_paths_config()
    assert isinstance(paths, PathsConfig)
    assert paths.workspace.root == "training/kaggle"
    assert paths.config.dataset == "training/config/dataset.yaml"
    assert paths.environment.require_gpu is True


def test_env_override_wins(tmp_path: Path) -> None:
    target = tmp_path / "paths.yaml"
    target.write_text("workspace:\n  root: training/kaggle\n", encoding="utf-8")
    paths = load_paths_config(
        target, env={"KAGGLE_WORKSPACE__ROOT": "training/kaggle-other"}
    )
    assert paths.workspace.root == "training/kaggle-other"
    assert paths.workspace.logs_dir == "logs"  # default retained


def test_extends_inheritance(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    parent.write_text(
        "workspace:\n  root: base/root\n  logs_dir: parent-logs\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "extends: parent.yaml\n"
        "workspace:\n  root: child/root\n  outputs_dir: child-out\n",
        encoding="utf-8",
    )
    paths = load_paths_config(child)
    assert paths.workspace.root == "child/root"  # child wins
    assert paths.workspace.logs_dir == "parent-logs"  # inherited
    assert paths.workspace.outputs_dir == "child-out"
    assert paths.workspace.checkpoints_dir == "checkpoints"  # default retained


def test_extends_chain(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("workspace:\n  root: base\n", encoding="utf-8")
    mid = tmp_path / "mid.yaml"
    mid.write_text("extends: base.yaml\nworkspace:\n  logs_dir: mid-logs\n", encoding="utf-8")
    top = tmp_path / "top.yaml"
    top.write_text("extends: mid.yaml\nworkspace:\n  root: top\n", encoding="utf-8")
    paths = load_paths_config(top)
    assert paths.workspace.root == "top"
    assert paths.workspace.logs_dir == "mid-logs"


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_paths_config(tmp_path / "nope.yaml")


def test_load_kaggle_config() -> None:
    cfg = load_kaggle_config()
    assert isinstance(cfg, KaggleConfig)
    assert cfg.kaggle.dataset_handle.startswith("shathanandabhatn")
    assert "training/dataset_manager" in cfg.install.editable_packages


def test_load_logging_config() -> None:
    cfg = load_logging_config()
    assert isinstance(cfg, LoggingConfig)
    assert cfg.json_format is True
    assert cfg.level == "INFO"


def test_workspace_layout_resolution(tmp_path: Path) -> None:
    paths = PathsConfig(
        workspace=WorkspaceConfig(root="ws", logs_dir="lg", outputs_dir="out")
    )
    layout = WorkspaceLayout.resolve(paths, repo_root=tmp_path)
    assert layout.repo_root == tmp_path.resolve()
    assert layout.root == tmp_path.resolve() / "ws"
    assert layout.logs == layout.root / "lg"
    assert layout.outputs == layout.root / "out"
    # defaults retained for unset dirs
    assert layout.checkpoints == layout.root / "checkpoints"
    assert layout.cache == layout.root / "cache"


def test_workspace_layout_overrides(tmp_path: Path) -> None:
    paths = PathsConfig()
    layout = WorkspaceLayout.resolve(paths, repo_root=tmp_path, logs=str(tmp_path / "x"))
    assert layout.logs == tmp_path.resolve() / "x"


def test_invalid_paths_rejected() -> None:
    with pytest.raises(ValidationError):
        PathsConfig(workspace=WorkspaceConfig(root=123))  # type: ignore[arg-type]


def test_config_registry_snapshot() -> None:
    registry = ConfigRegistry()
    assert "validation" in registry.snapshot()


def test_environment_requirements_defaults() -> None:
    req = EnvironmentRequirements()
    assert req.min_python == "3.10"
    assert "rasterio" in req.required_dependencies
