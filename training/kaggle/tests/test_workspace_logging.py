"""R2.1 Workspace Manager + Training Logger tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.kaggle.config import LoggingConfig, PathsConfig, WorkspaceLayout
from training.kaggle.logging import TrainingLogger
from training.kaggle.workspace import WorkspaceManager


@pytest.fixture()
def workspace(tmp_path: Path) -> WorkspaceManager:
    paths = PathsConfig()
    layout = WorkspaceLayout.resolve(paths, repo_root=tmp_path)
    return WorkspaceManager(layout)


def test_create_makes_directories(workspace: WorkspaceManager) -> None:
    created = workspace.create()
    for key in ("logs", "outputs", "checkpoints", "cache", "configs"):
        assert Path(created[key]).is_dir()
    assert workspace.checkpoints.metadata_file.exists()


def test_ensure_true_after_create(workspace: WorkspaceManager) -> None:
    assert workspace.ensure() is True


def test_output_path(workspace: WorkspaceManager) -> None:
    target = workspace.output_path("reports", "x.json")
    assert target == workspace.layout.outputs / "reports" / "x.json"


def test_run_output_creates(workspace: WorkspaceManager) -> None:
    target = workspace.run_output("run-1")
    assert target.is_dir()
    assert target.name == "run-1"


def test_configs_dir(workspace: WorkspaceManager) -> None:
    assert workspace.configs_dir().is_dir()


def test_clean_cache(workspace: WorkspaceManager) -> None:
    workspace.create()
    workspace.cache.set("metadata", "k", 1)
    assert workspace.cache.stats()["total"] == 1
    removed = workspace.clean_cache()
    assert removed >= 1
    assert workspace.cache.stats()["total"] == 0


def test_resolve_resume(workspace: WorkspaceManager) -> None:
    workspace.create()
    assert workspace.resolve_resume() is None
    workspace.checkpoints.register("run-a", epoch=1, resume=True)
    assert workspace.resolve_resume("run-a")["epoch"] == 1


def test_report(workspace: WorkspaceManager) -> None:
    workspace.create()
    workspace.cache.set("validation", "k", 1)
    report = workspace.report()
    assert "layout" in report
    assert report["cache"]["total"] == 1
    assert report["checkpoints"]["count"] == 0


def test_logger_writes_all_files(tmp_path: Path) -> None:
    logger = TrainingLogger(LoggingConfig(dir=None), log_dir=tmp_path).setup()
    logger.log_startup("startup_event", repo_root="x")
    logger.log_system("system_event")
    logger.log_experiment("experiment_event", epoch=1)
    for name in ("startup", "system", "experiment"):
        path = tmp_path / f"{name}.log"
        assert path.exists()
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        assert lines, f"no lines in {name}.log"
        payload = json.loads(lines[0])
        assert payload["logger"].endswith(f".{name}")


def test_logger_idempotent(tmp_path: Path) -> None:
    logger = TrainingLogger(LoggingConfig(dir=None), log_dir=tmp_path)
    logger.setup()
    assert logger.configured
    assert logger.log_files()["startup"] == str(tmp_path / "startup.log")


def test_logger_unknown_child(tmp_path: Path) -> None:
    logger = TrainingLogger(log_dir=tmp_path)
    with pytest.raises(KeyError):
        logger.child("bogus")
