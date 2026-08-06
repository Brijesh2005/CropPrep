"""Inference runtime tests (Phase R6)."""

from __future__ import annotations

import json

import pytest

from training.runtime import (
    STATUS_DEGRADED,
    STATUS_LOADING,
    STATUS_NOT_READY,
    STATUS_READY,
    InferenceRuntime,
    ReleasePackager,
    RuntimeConfig,
)
from training.runtime.exceptions import (
    MemoryLimitError,
    ReleaseValidationError,
    RuntimeEnvironmentError,
)


def _config(releases_root, **sections):
    return RuntimeConfig(general={"releases_root": str(releases_root)}, **sections)


def _second_release(release_env):
    target = release_env.root / "cropfusion_release-v1.1.0"
    ReleasePackager().build(
        release_env.flat_dir, target_dir=target, version="1.1.0"
    )
    return target


# --------------------------------------------------------------------- #
# Start / load
# --------------------------------------------------------------------- #


def test_start_and_health(release_env):
    runtime = InferenceRuntime(_config(release_env.root))
    runtime.start("1.0.0")
    health = runtime.health()
    assert health.status == STATUS_READY
    assert health.ready is True
    assert health.release_ready is True
    assert health.model_ready is True
    assert health.preprocess_ready is True
    assert health.metadata_ready is True
    assert health.warmup_ok is True
    assert health.version == "1.0.0"
    assert health.model_version == "1.0.0"
    assert health.backend == "pytorch"
    assert health.startup_time_ms is not None
    assert health.uptime_seconds >= 0
    assert health.checks["release_ready"] is True
    runtime.shutdown()


def test_start_defaults_to_latest(release_env):
    runtime = InferenceRuntime(_config(release_env.root))
    runtime.start()
    assert runtime.health().version == "1.0.0"
    runtime.shutdown()


def test_start_persists_active_version(release_env):
    runtime = InferenceRuntime(_config(release_env.root))
    runtime.start("1.0.0")
    assert runtime.manager.current_version() == "1.0.0"
    runtime.shutdown()


def test_health_before_start(release_env):
    runtime = InferenceRuntime(_config(release_env.root))
    health = runtime.health()
    assert health.status == STATUS_NOT_READY
    assert health.ready is False
    runtime.shutdown()


def test_health_after_shutdown(release_env):
    runtime = InferenceRuntime(_config(release_env.root))
    runtime.start("1.0.0")
    runtime.shutdown()
    health = runtime.health()
    assert health.status == STATUS_NOT_READY
    assert health.ready is False
    assert health.model_ready is False


def test_health_disabled_raises(release_env):
    runtime = InferenceRuntime(
        _config(release_env.root, health={"enabled": False})
    )
    with pytest.raises(RuntimeEnvironmentError):
        runtime.health()


def test_load_second_release(release_env):
    _second_release(release_env)
    runtime = InferenceRuntime(_config(release_env.root))
    runtime.start("1.1.0")
    health = runtime.health()
    assert health.version == "1.1.0"
    assert health.model_version == "1.0.0"  # underlying model is unchanged
    runtime.shutdown()


# --------------------------------------------------------------------- #
# Validation on start
# --------------------------------------------------------------------- #


def test_start_strict_failure(release_env, tmp_path):
    bad = tmp_path / "cropfusion_release-v2.0.0"
    import shutil

    shutil.copytree(release_env.release_path, bad)
    (bad / "model" / "cropfusion.pt").unlink()

    config = _config(tmp_path)
    runtime = InferenceRuntime(config)
    with pytest.raises(ReleaseValidationError):
        runtime.start("2.0.0")
    runtime.shutdown()


def test_start_with_validation_disabled(release_env, tmp_path):
    bad = tmp_path / "cropfusion_release-v2.0.0"
    import shutil

    shutil.copytree(release_env.release_path, bad)
    (bad / "model" / "cropfusion.pt").unlink()

    config = _config(tmp_path, validation={"strict": False})
    runtime = InferenceRuntime(config)
    # validate=False skips the battery entirely -> load fails on the missing file.
    with pytest.raises(Exception):
        runtime.start("2.0.0", validate=False)
    runtime.shutdown()


# --------------------------------------------------------------------- #
# Reload / hot reload
# --------------------------------------------------------------------- #


def test_reload_release(release_env):
    runtime = InferenceRuntime(_config(release_env.root))
    runtime.start("1.0.0")
    info = runtime.reload_release("1.0.0")
    assert info.version == "1.0.0"
    assert runtime.health().ready is True
    runtime.shutdown()


def test_poll_reload_no_change(release_env):
    runtime = InferenceRuntime(_config(release_env.root))
    runtime.start("1.0.0")
    assert runtime.poll_reload() is False
    runtime.shutdown()


def test_poll_reload_detects_change(release_env):
    runtime = InferenceRuntime(_config(release_env.root))
    runtime.start("1.0.0")
    checksums_path = (
        release_env.release_path / "version" / "checksums.json"
    )
    checksums_path.write_text(
        checksums_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    assert runtime.poll_reload() is True
    runtime.shutdown()


def test_poll_reload_auto_reloads(release_env):
    config = _config(
        release_env.root,
        hot_reload={"enabled": True, "auto_reload": True},
    )
    runtime = InferenceRuntime(config)
    runtime.start("1.0.0")
    checksums_path = (
        release_env.release_path / "version" / "checksums.json"
    )
    checksums_path.write_text(
        checksums_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    assert runtime.poll_reload() is True
    assert runtime.status()["hot_reload"]["reloads"] == 1
    assert runtime.health().ready is True
    runtime.shutdown()


def test_poll_reload_without_auto(release_env):
    config = _config(
        release_env.root,
        hot_reload={"enabled": True, "auto_reload": False},
    )
    runtime = InferenceRuntime(config)
    runtime.start("1.0.0")
    checksums_path = (
        release_env.release_path / "version" / "checksums.json"
    )
    checksums_path.write_text(
        checksums_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    assert runtime.poll_reload() is True
    assert runtime.status()["hot_reload"]["reloads"] == 1
    assert runtime.health().ready is True
    runtime.shutdown()


def test_hot_reload_max_reloads(release_env):
    config = _config(
        release_env.root,
        hot_reload={"enabled": True, "auto_reload": True, "max_reloads": 1},
    )
    runtime = InferenceRuntime(config)
    runtime.start("1.0.0")
    checksums_path = (
        release_env.release_path / "version" / "checksums.json"
    )
    checksums_path.write_text(
        checksums_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    assert runtime.poll_reload() is True  # reload #1
    checksums_path.write_text(
        checksums_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    assert runtime.poll_reload() is True  # detected but capped
    assert runtime.status()["hot_reload"]["reloads"] == 2
    runtime.shutdown()


def test_hot_reload_thread(release_env):
    config = _config(
        release_env.root,
        hot_reload={"enabled": True, "poll_interval_seconds": 0.01},
    )
    runtime = InferenceRuntime(config)
    runtime.start("1.0.0")
    thread = runtime.start_hot_reload(interval=0.01)
    assert thread.is_alive()
    runtime.stop_hot_reload()
    thread.join(timeout=5)
    assert not thread.is_alive()
    runtime.shutdown()


# --------------------------------------------------------------------- #
# Rollback
# --------------------------------------------------------------------- #


def test_rollback_switches_version(release_env):
    _second_release(release_env)
    runtime = InferenceRuntime(_config(release_env.root))
    runtime.start("1.0.0")
    runtime.load_release("1.1.0")
    assert runtime.health().version == "1.1.0"
    info = runtime.rollback()
    assert info.version == "1.0.0"
    assert runtime.health().version == "1.0.0"
    runtime.shutdown()


# --------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------- #


def test_memory_limit_fails_start(release_env):
    config = _config(release_env.root, memory={"limit_mb": 1})
    runtime = InferenceRuntime(config)
    with pytest.raises(MemoryLimitError):
        runtime.start("1.0.0")
    runtime.shutdown()


# --------------------------------------------------------------------- #
# Status / validation
# --------------------------------------------------------------------- #


def test_status(release_env):
    runtime = InferenceRuntime(_config(release_env.root))
    runtime.start("1.0.0")
    status = runtime.status()
    assert status["started"] is True
    assert status["ready"] is True
    assert status["active"] == "1.0.0"
    assert status["releases"]["active"] == "1.0.0"
    assert status["hot_reload"]["enabled"] is False
    assert "memory" in status
    runtime.shutdown()


def test_validate_runtime(release_env):
    runtime = InferenceRuntime(_config(release_env.root))
    runtime.start("1.0.0")
    result = runtime.validate()
    assert result.valid is True
    assert runtime._last_validation is result
    runtime.shutdown()


def test_optional_preprocess_degraded(release_env, tmp_path):
    import shutil

    from training.runtime.tests.conftest import refresh_checksums

    degraded = tmp_path / "cropfusion_release-v1.2.0"
    shutil.copytree(release_env.release_path, degraded)
    manifest_path = degraded / "version" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["package_version"] = "1.2.0"
    manifest["release_version"] = "1.2.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    release_version_path = degraded / "version" / "release_version.json"
    payload = json.loads(release_version_path.read_text(encoding="utf-8"))
    payload["version"] = "1.2.0"
    release_version_path.write_text(json.dumps(payload), encoding="utf-8")
    refresh_checksums(degraded)

    config = _config(
        tmp_path,
        preprocess={"required": False},
        validation={"verify_checksums": False, "smoke_test": False},
    )
    runtime = InferenceRuntime(config)
    runtime.start("1.2.0")
    health = runtime.health()
    assert health.status == STATUS_DEGRADED
    assert health.ready is True
    assert health.preprocess_ready is False
    runtime.shutdown()
