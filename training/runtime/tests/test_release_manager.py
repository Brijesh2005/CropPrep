"""Release manager tests (Phase R6)."""

from __future__ import annotations

import pytest

from training.runtime import ReleaseManager, RuntimeConfig
from training.runtime.exceptions import (
    ReleaseNotFoundError,
    ReleaseRollbackError,
    ReleaseValidationError,
)


@pytest.fixture
def manager(releases_root):
    config = RuntimeConfig(general={"releases_root": str(releases_root)})
    return ReleaseManager(config).initialize()


def test_discover_single(manager):
    versions = manager.versions()
    assert versions == ["1.0.0"]


def test_get(manager):
    info = manager.get("1.0.0")
    assert info.version == "1.0.0"
    assert info.manifest.package_version == "1.0.0"
    assert info.manifest.model_version == "1.0.0"
    assert info.model_version == "1.0.0"
    assert info.formats == ["pytorch"]
    assert info.valid is True


def test_get_missing(manager):
    with pytest.raises(ReleaseNotFoundError):
        manager.get("9.9.9")


def test_latest(manager):
    assert manager.latest().version == "1.0.0"


def test_release_path(manager):
    assert manager.release_path("1.0.0").name == "cropfusion_release-v1.0.0"


def test_activate_and_state(manager):
    assert manager.current_version() is None
    info = manager.activate("1.0.0")
    assert info.version == "1.0.0"
    assert manager.current_version() == "1.0.0"
    assert manager.active().version == "1.0.0"
    assert manager.state().history == []


def test_state_persists(releases_root):
    config = RuntimeConfig(general={"releases_root": str(releases_root)})
    manager = ReleaseManager(config).initialize()
    manager.activate("1.0.0")
    # A fresh manager reads the persisted state.
    other = ReleaseManager(config).initialize()
    assert other.current_version() == "1.0.0"


def test_activate_unknown_version(manager):
    with pytest.raises(ReleaseNotFoundError):
        manager.activate("9.9.9")


def test_rollback_requires_previous(manager):
    with pytest.raises(ReleaseRollbackError):
        manager.rollback()


def test_rollback_flow(releases_root):
    config = RuntimeConfig(general={"releases_root": str(releases_root)})
    manager = ReleaseManager(config).initialize()
    manager.activate("1.0.0")

    # Build a second release on top of the first.
    second = releases_root / "cropfusion_release-v1.1.0"
    clone = releases_root / "_clone"
    import shutil

    shutil.copytree(manager.release_path("1.0.0"), clone)
    shutil.copytree(clone, second)
    shutil.rmtree(clone)
    _bump_release_version(second, "1.1.0")

    assert manager.versions() == ["1.1.0", "1.0.0"]
    manager.activate("1.1.0")
    assert manager.current_version() == "1.1.0"
    assert manager.state().history == ["1.0.0"]

    rolled_back = manager.rollback()
    assert rolled_back.version == "1.0.0"
    assert manager.current_version() == "1.0.0"
    assert manager.state().history == []


def test_rollback_after_restart(releases_root):
    config = RuntimeConfig(general={"releases_root": str(releases_root)})
    manager = ReleaseManager(config).initialize()
    manager.activate("1.0.0")

    # Build a second release and activate it, so history is recorded.
    second = releases_root / "cropfusion_release-v1.1.0"
    import shutil

    shutil.copytree(manager.release_path("1.0.0"), second)
    _bump_release_version(second, "1.1.0")
    manager.activate("1.1.0")
    assert manager.state().history == ["1.0.0"]

    # Restart persists the active version *and* history, so rollback works.
    again = ReleaseManager(config).initialize()
    assert again.current_version() == "1.1.0"
    assert again.state().history == ["1.0.0"]
    rolled_back = again.rollback()
    assert rolled_back.version == "1.0.0"
    assert again.current_version() == "1.0.0"


def test_validate_active(manager):
    result = manager.validate("1.0.0")
    assert result.valid is True
    assert result.checks["integrity"] is True
    assert result.checks["manifest"] is True
    assert result.checks["versions"] is True
    assert result.checks["dependencies"] is True
    assert result.checks["compatibility"] is True
    assert result.checks["smoke_test"] is True


def test_validate_missing(manager):
    with pytest.raises(ReleaseNotFoundError):
        manager.validate("9.9.9")


def test_validate_raises_strict_on_bad_release(releases_root, tmp_path):
    # A release that fails validation (missing file) must raise in strict mode.
    import shutil

    from training.runtime import ReleaseLayout

    bad = tmp_path / "cropfusion_release-v0.0.1"
    shutil.copytree(releases_root / "cropfusion_release-v1.0.0", bad)
    (bad / "model" / "cropfusion.pt").unlink()

    config = RuntimeConfig(general={"releases_root": str(tmp_path)})
    manager = ReleaseManager(config).initialize()
    with pytest.raises(ReleaseValidationError):
        manager.validate("0.0.1")


def test_activate_fails_strict(releases_root, tmp_path):
    import shutil

    bad = tmp_path / "cropfusion_release-v0.0.1"
    shutil.copytree(releases_root / "cropfusion_release-v1.0.0", bad)
    (bad / "configs" / "model_config.yaml").unlink()

    config = RuntimeConfig(general={"releases_root": str(tmp_path)})
    manager = ReleaseManager(config).initialize()
    with pytest.raises(ReleaseValidationError):
        manager.activate("0.0.1")


def test_status(releases_root, tmp_path):
    # A fresh root with no activation history.
    fresh = tmp_path / "status_root"
    manager = ReleaseManager(
        RuntimeConfig(general={"releases_root": str(fresh)})
    ).initialize()
    status = manager.status()
    assert status["releases_root"] == str(manager.releases_root)
    assert status["active"] is None
    assert status["releases"] == []


def test_status_with_releases(releases_root):
    manager = ReleaseManager(
        RuntimeConfig(general={"releases_root": str(releases_root)})
    ).initialize()
    status = manager.status()
    assert [r["version"] for r in status["releases"]] == ["1.0.0"]


def test_empty_root(tmp_path):
    manager = ReleaseManager(
        RuntimeConfig(general={"releases_root": str(tmp_path / "empty")})
    ).initialize()
    assert manager.versions() == []
    with pytest.raises(ReleaseNotFoundError):
        manager.latest()


def _bump_release_version(release_dir, version):
    import json

    manifest_path = release_dir / "version" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["package_version"] = version
    manifest["release_version"] = version
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    release_version_path = release_dir / "version" / "release_version.json"
    payload = json.loads(release_version_path.read_text(encoding="utf-8"))
    payload["version"] = version
    release_version_path.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    from training.runtime.tests.conftest import refresh_checksums

    refresh_checksums(release_dir)
