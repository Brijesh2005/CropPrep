"""R5.4 launcher tests — deployment prep and the --test-epochs override cell.

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
    from scripts import kaggle_r5_4 as launcher

    return launcher


@pytest.fixture
def isolated_prepare(tmp_path, launcher):
    """Point the launcher's file-system constants at a throwaway dir."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"total_samples": 10_674, "train_samples": 6_116,
                    "validation_samples": 2_267, "test_samples": 2_291}),
        encoding="utf-8",
    )
    note = tmp_path / "nb_src.ipynb"
    note.write_text(
        json.dumps({"cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": ["print('train')"]},
        ]}),
        encoding="utf-8",
    )
    deploy = tmp_path / "deploy"
    deploy.mkdir()

    patched = patch.multiple(
        "scripts.kaggle_r5_4",
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

    tmp_path, note, deploy = isolated_prepare
    assert rc == 0
    deployed = deploy / "R5_4_train.ipynb"
    assert deployed.exists()
    nb = json.loads(deployed.read_text(encoding="utf-8"))
    assert len(nb["cells"]) == len(note_load(note)["cells"])
    assert not any("R5_4_EPOCHS" in "".join(c.get("source", [])) for c in nb["cells"])


def test_prepare_with_test_epochs_injects_override(isolated_prepare, launcher):
    args = launcher.argparse.Namespace(test_epochs=3)
    with patch.object(launcher, "_get_kaggle_username", return_value="user"):
        rc = launcher.cmd_prepare(args)

    _, _, deploy = isolated_prepare
    assert rc == 0
    nb = json.loads((deploy / "R5_4_train.ipynb").read_text(encoding="utf-8"))
    sources = [json.dumps("".join(c.get("source", []))) for c in nb["cells"]]
    assert any("R5_4_EPOCHS" in s and "'3'" in s for s in sources)
    # Prefixed cell must run before the training cell.
    assert "R5_4_EPOCHS" in sources[0]


def test_parser_accepts_test_epochs_on_prepare_and_full(launcher):
    parser = launcher.build_parser()

    args = parser.parse_args(["prepare", "--test-epochs", "3"])
    assert args.command == "prepare"
    assert args.test_epochs == 3

    args = parser.parse_args(["full", "--test-epochs", "3", "--confirm"])
    assert args.command == "full"
    assert args.test_epochs == 3
    assert args.confirm is True

    args = parser.parse_args(["prepare"])
    assert args.test_epochs is None


def test_inject_epochs_cell_round_trip(tmp_path, launcher):
    nb_path = tmp_path / "nb.ipynb"
    nb_path.write_text(
        json.dumps({"cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": ["print('train')"]},
        ]}),
        encoding="utf-8",
    )
    launcher._inject_epochs_cell(nb_path, 3)

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert len(nb["cells"]) == 2
    first = "".join(nb["cells"][0]["source"])
    assert "os.environ['R5_4_EPOCHS'] = '3'" in first


def note_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))