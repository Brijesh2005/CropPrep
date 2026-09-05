"""R5.5 diagnostic prep tests — notebook + launcher + module wiring.

Validate the classifier-collapse diagnostic deploy WITHOUT touching the real
deployment directory or calling the Kaggle CLI (username resolution mocked).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _ensure_sys_path():
    inserted = False
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
        inserted = True
    try:
        yield
    finally:
        if inserted:
            sys.path.remove(str(_REPO_ROOT))


@pytest.fixture
def launcher():
    from scripts import kaggle_r5_5_diagnostic as launcher

    return launcher


@pytest.fixture
def notebook():
    nb = json.loads(
        (
            _REPO_ROOT
            / "training"
            / "kaggle"
            / "notebooks"
            / "R5_5_diagnose_collapse.ipynb"
        ).read_text(encoding="utf-8")
    )
    return nb


def _nb_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


@pytest.fixture
def isolated_prepare(tmp_path, launcher):
    note = tmp_path / "nb_src.ipynb"
    note.write_text(
        json.dumps({"cells": [_nb_cell("print('setup')")]}),
        encoding="utf-8",
    )
    deploy = tmp_path / "deploy"
    deploy.mkdir()

    patched = patch.multiple(
        "scripts.kaggle_r5_5_diagnostic",
        NOTEBOOK_SRC=note,
        DEPLOY_DIR=deploy,
    )
    patched.start()
    yield tmp_path, note, deploy
    patched.stop()


def test_notebook_runs_diagnostic_driver(notebook):
    sources = [json.dumps("".join(c.get("source", []))) for c in notebook["cells"]]
    assert any("diagnose_collapse_kaggle.py" in s for s in sources)
    assert any("diagnostic_r5_5.json" in s for s in sources)
    # The imagery window must be pinned with the same defaults as R5.3.
    assert any("ST_IMAGERY__WINDOW_DAYS" in s and "'180'" in s for s in sources)
    # No shell-ism leaks into Python cells.
    assert not any("$?" in s and "echo" not in s for s in sources)


def test_notebook_has_no_invalid_python(notebook):
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        if src.startswith("!") or src.strip().startswith("%"):
            continue
        compile(src, "<cell>", "exec")


def test_prepare_copies_notebook_and_metadata(isolated_prepare, launcher):
    with patch.object(launcher, "_get_kaggle_username", return_value="user"):
        rc = launcher.cmd_prepare(launcher.argparse.Namespace())

    _, _, deploy = isolated_prepare
    assert rc == 0
    assert (deploy / "nb_src.ipynb").exists()
    metadata = json.loads((deploy / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "user/r5-5-classifier-collapse-diagnostic"
    assert metadata["enable_gpu"] is True
    assert metadata["kernel_type"] == "notebook"
    assert len(metadata["dataset_sources"]) == 1


def test_prepare_requires_username(isolated_prepare, launcher):
    args = launcher.argparse.Namespace()
    with patch.object(launcher, "_get_kaggle_username", return_value=None):
        rc = launcher.cmd_prepare(args)
    assert rc == 1


def test_kernel_slug_stable(launcher):
    assert launcher.KERNEL_SLUG == "r5-5-classifier-collapse-diagnostic"


def test_diagnostic_module_help_parses():
    import sys as _sys

    _sys.path.insert(0, str(_REPO_ROOT / "training" / "kaggle" / "scripts"))
    from diagnose_collapse_kaggle import BINARY_CLASSES, main

    assert BINARY_CLASSES == ["coconut", "pepper"]
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_driver_uses_preprocessing_config_not_preprocessor():
    src = (
        _REPO_ROOT
        / "training"
        / "kaggle"
        / "scripts"
        / "diagnose_collapse_kaggle.py"
    ).read_text(encoding="utf-8")
    # The main flow must load a PreprocessingConfig (NOT a Preprocessor via
    # Preprocessor.from_config) so `Preprocessor(cfg)` doesn't double-wrap and
    # build_dataloader(config=cfg) receives a valid config object.
    assert (
        "preprocessing_cfg = load_preprocessing_config(args.preprocessing_config)"
        in src
    )
    assert "pre = Preprocessor(preprocessing_cfg)" in src
    # The binary/tiny preprocessor reuse the same loader output directly and
    # must NOT re-wrap the Preprocessor it receives.
    assert "pre_bin = bin_cfg" in src
    assert "pre_tiny = tiny_cfg" in src


def test_preprocessing_config_load_roundtrip():
    from training.preprocessing import Preprocessor, load_preprocessing_config
    from training.preprocessing.dataloader import build_dataloader

    cfg = load_preprocessing_config(_REPO_ROOT / "training/config/preprocessing.yaml")
    cfg.label.declared_classes = ["coconut", "pepper", "coffee", "cardamom"]
    cfg.label.excluded_classes = []
    pre = Preprocessor(cfg)
    assert pre.config is cfg
    assert build_dataloader.__doc__ is not None


def test_run_config_clamps_warmup():
    from training.training.config import load_training_config

    training_cfg = load_training_config(_REPO_ROOT / "training/config/training.yaml")
    warmup_raw = training_cfg.scheduler.warmup_epochs
    assert warmup_raw >= 2, "test assumes warmup >= 2"
    # With epochs <= warmup, _run_config must clamp so warmup < epochs
    sys.path.insert(0, str(_REPO_ROOT / "training" / "kaggle" / "scripts"))
    from diagnose_collapse_kaggle import _run_config

    cfg2 = _run_config(training_cfg, epochs=2)
    assert cfg2.train.epochs == 2
    assert cfg2.scheduler.warmup_epochs == 1, "must clamp to epochs-1"

    cfg1 = _run_config(training_cfg, epochs=1)
    assert cfg1.train.epochs == 1
    assert cfg1.scheduler.warmup_epochs == 0

    cfg_big = _run_config(training_cfg, epochs=10)
    assert cfg_big.scheduler.warmup_epochs == warmup_raw, "unclamped for large epochs"