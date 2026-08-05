"""Tests for the command-line interface (runs subcommands end-to-end)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.dataset_manager import cli


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch, tmp_path: Path):
    """Point every CLI invocation at an isolated dataset root."""
    dataset_root = tmp_path / "datasets"
    monkeypatch.setenv("DM_DATASET_ROOT", str(dataset_root))
    return dataset_root


def _run(*argv: str) -> tuple[int, str]:
    """Run the CLI and return (exit_code, merged stdout)."""
    import contextlib
    import io

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = cli.main(list(argv))
    return code, stdout.getvalue()


def test_no_command_fails():
    with pytest.raises(SystemExit):
        _run()


def test_info(tmp_path: Path):
    code, out = _run("info")
    assert code == 0
    assert "dataset_root" in out


def test_config_template(tmp_path: Path):
    dest = tmp_path / "dm.yaml"
    code, out = _run("config-template", str(dest))
    assert code == 0
    assert dest.exists()


def test_scan_and_summary(synthetic_dataset: Path, tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    code, out = _run("scan")
    assert code == 0
    assert "geotiff" in out

    code, out = _run("summary", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["summary"]["csv_count"] == 1


def test_validate_empty_root_is_failure(tmp_path: Path):
    code, out = _run("validate", "--json")
    payload = json.loads(out)
    assert code == 0
    assert payload["passed"] is False
    codes = {i["code"] for i in payload["issues"]}
    assert "V-STRUCT-006" in codes


def test_metadata_and_versions(synthetic_dataset: Path, tmp_path: Path):
    code, out = _run("metadata", "--json")
    assert code == 0
    assert json.loads(out)["records_written"] == 4

    code, out = _run("bump-version", "minor", "--message", "baseline", "--json")
    assert code == 0
    assert json.loads(out)["version"]["version"] == "0.1.0"

    code, out = _run("versions", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["current"] == "0.1.0"


def test_images_filter(synthetic_dataset: Path, tmp_path: Path):
    code, out = _run("images", "--index", "NDVI", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["count"] == 2
    assert all("NDVI" in p for p in payload["files"])


def test_csvs(synthetic_dataset: Path, tmp_path: Path):
    code, out = _run("csvs", "--json")
    assert code == 0
    assert json.loads(out)["count"] == 1


def test_error_boundary_returns_nonzero(tmp_path: Path, monkeypatch):
    # A missing config file raises InvalidConfigurationError at startup.
    monkeypatch.setenv("DM_CONFIG_FILE", str(tmp_path / "missing.yaml"))
    code, out = _run("info", "--json")
    assert code == 1
    assert '"ok": false' in out
