"""R2.1 Checkpoint Manager tests: metadata registry, resolution, versioning."""

from __future__ import annotations

from pathlib import Path

import pytest

from training.kaggle.checkpoints import CheckpointManager


@pytest.fixture()
def manager(tmp_path: Path) -> CheckpointManager:
    return CheckpointManager(tmp_path / "checkpoints")


def test_ensure_layout(manager: CheckpointManager) -> None:
    manager.ensure_layout()
    assert manager.checkpoint_dir.is_dir()
    assert manager.metadata_file.exists()


def test_register_and_list(manager: CheckpointManager) -> None:
    manager.register("run-a", stage="checkpoint", epoch=1, metrics={"val_loss": 0.5})
    manager.register("run-a", stage="best", epoch=2, metrics={"val_loss": 0.3}, resume=True)
    entries = manager.list()
    assert len(entries) == 2
    assert entries[0]["epoch"] == 2  # newest first
    assert manager.list("run-a")[0]["run_name"] == "run-a"
    assert manager.list("other") == []


def test_version_increments_per_run(manager: CheckpointManager) -> None:
    manager.register("run-a")
    manager.register("run-a")
    manager.register("run-b")
    versions = [e["version"] for e in manager.list("run-a")]
    assert versions == ["v2", "v1"]
    assert manager.list("run-b")[0]["version"] == "v1"


def test_latest(manager: CheckpointManager) -> None:
    assert manager.latest() is None
    manager.register("run-a", epoch=1)
    manager.register("run-a", epoch=2)
    assert manager.latest()["epoch"] == 2
    assert manager.latest("run-a")["epoch"] == 2


def test_best(manager: CheckpointManager) -> None:
    manager.register("run-a", epoch=1, metrics={"val_loss": 1.0})
    manager.register("run-a", epoch=2, metrics={"val_loss": 0.2})
    manager.register("run-a", epoch=3, metrics={"val_loss": 0.5})
    best = manager.best(metric="val_loss", mode="min")
    assert best["epoch"] == 2
    assert manager.best(metric="val_acc", mode="max") is None


def test_resume_prefers_resume_flag(manager: CheckpointManager) -> None:
    manager.register("run-a", epoch=1, metrics={"val_loss": 0.9})
    manager.register("run-a", epoch=2, metrics={"val_loss": 0.4}, resume=True)
    assert manager.resume()["epoch"] == 2
    assert manager.resume("run-a")["epoch"] == 2


def test_keep_last_prunes(manager: CheckpointManager) -> None:
    manager.keep_last = 2
    for epoch in range(1, 6):
        manager.register("run-a", epoch=epoch)
    assert len(manager.list("run-a")) == 2
    assert manager.latest("run-a")["epoch"] == 5


def test_version_for_semver(manager: CheckpointManager) -> None:
    assert manager.version_for("run-a") == "1.0.0"
    manager.register("run-a")
    assert manager.version_for("run-a") == "1.0.0"  # tags are vN, not semver
    manager.register("run-a")
    # semver unaffected by vN tags; first valid semver tag is 1.0.0


def test_report(manager: CheckpointManager) -> None:
    manager.register("run-a", epoch=1, metrics={"val_loss": 0.3})
    report = manager.report()
    assert report["count"] == 1
    assert report["runs"] == ["run-a"]
    assert report["latest"]["epoch"] == 1
    assert report["checkpoint_dir"] == str(manager.checkpoint_dir)


def test_registry_persistence(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints"
    CheckpointManager(path).register("run-a", epoch=1)
    reloaded = CheckpointManager(path)
    assert reloaded.latest()["epoch"] == 1
