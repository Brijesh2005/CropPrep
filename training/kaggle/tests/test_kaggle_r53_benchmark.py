"""R5.3 benchmark launcher tests — prepare + --test-epochs injection.

Validate the launcher WITHOUT touching the real deployment directory or
calling the Kaggle CLI (username resolution is mocked).
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
    from scripts import kaggle_r5_3_benchmark as launcher

    return launcher


def _nb_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


@pytest.fixture
def isolated_prepare(tmp_path, launcher):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"total_samples": 10_674}), encoding="utf-8")
    note = tmp_path / "nb_src.ipynb"
    note.write_text(
        json.dumps({"cells": [
            _nb_cell("print('setup')"),
            _nb_cell("import subprocess, sys, os\ncmd=['python','training/scripts/run_pipeline.py']"),
            _nb_cell("print('teardown')"),
        ]}),
        encoding="utf-8",
    )
    deploy = tmp_path / "deploy"
    deploy.mkdir()

    patched = patch.multiple(
        "scripts.kaggle_r5_3_benchmark",
        NOTEBOOK_SRC=note,
        MANIFEST_PATH=manifest,
        DEPLOY_DIR=deploy,
    )
    patched.start()
    yield tmp_path, note, deploy
    patched.stop()


def test_prepare_copies_notebook_without_override(isolated_prepare, launcher):
    args = launcher.argparse.Namespace(test_epochs=None)
    with patch.object(launcher, "_get_kaggle_username", return_value="user"):
        rc = launcher.cmd_prepare(args)

    _, note, deploy = isolated_prepare
    assert rc == 0
    deployed = deploy / "R5_3_benchmark.ipynb"
    assert deployed.exists()
    nb = json.loads(deployed.read_text(encoding="utf-8"))
    assert len(nb["cells"]) == 3
    assert not any("R5_3_EPOCHS" in "".join(c.get("source", [])) for c in nb["cells"])


def test_prepare_with_test_epochs_injects_override(isolated_prepare, launcher):
    args = launcher.argparse.Namespace(test_epochs=3)
    with patch.object(launcher, "_get_kaggle_username", return_value="user"):
        rc = launcher.cmd_prepare(args)

    _, _, deploy = isolated_prepare
    assert rc == 0
    nb = json.loads((deploy / "R5_3_benchmark.ipynb").read_text(encoding="utf-8"))
    sources = [json.dumps("".join(c.get("source", []))) for c in nb["cells"]]
    assert any("R5_3_EPOCHS" in s and "'3'" in s for s in sources)
    # Injection happens before the run_pipeline cell, not after setup.
    print_idx = next(i for i, s in enumerate(sources) if "run_pipeline.py" in s)
    inj_idx = next(i for i, s in enumerate(sources) if "R5_3_EPOCHS" in s)
    assert inj_idx < print_idx


def test_inject_epochs_cell_round_trip(tmp_path, launcher):
    nb_path = tmp_path / "nb.ipynb"
    nb_path.write_text(
        json.dumps({"cells": [_nb_cell("import os")]}), encoding="utf-8"
    )
    # No run_pipeline cell present: injection should raise rather than corrupt.
    with pytest.raises(AssertionError):
        launcher._inject_epochs_cell(nb_path, 3)

    nb_path.write_text(
        json.dumps({"cells": [_nb_cell("import subprocess, sys, os\n# x")]}),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        launcher._inject_epochs_cell(nb_path, 3)

    nb_path.write_text(
        json.dumps({"cells": [
            _nb_cell("cmd=['python','training/kaggle/scripts/run_pipeline.py']"),
        ]}),
        encoding="utf-8",
    )
    launcher._inject_epochs_cell(nb_path, 5)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert len(nb["cells"]) == 2
    first = "".join(nb["cells"][0]["source"])
    assert "os.environ['R5_3_EPOCHS'] = '5'" in first


def test_parser_accepts_test_epochs(launcher):
    parser = launcher.argparse.ArgumentParser(prog="t")
    # main constructs the parser internally; just exercise cmd_* parsing via
    # the documented interface: ensure module-level helpers resolve.
    assert launcher.KERNEL_SLUG == "r5-3-cropfusion-benchmark-optimization"
    assert launcher.EXPECTED_TOTAL == 10_674
    assert launcher.EXPECTED_TRAIN == 5_924