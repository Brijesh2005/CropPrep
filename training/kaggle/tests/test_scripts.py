"""R2.1 script-level tests: module import + pure helpers (no full run)."""

from __future__ import annotations

from pathlib import Path

import pytest

from training.kaggle import scripts as _scripts_pkg  # noqa: F401 (namespace check)


def test_bootstrap_imports() -> None:
    from training.kaggle.scripts import bootstrap

    assert hasattr(bootstrap, "main")
    assert callable(bootstrap._verify_repo_integrity)


def test_bootstrap_repo_integrity(tmp_path: Path) -> None:
    from training.kaggle.scripts.bootstrap import _verify_repo_integrity

    report = _verify_repo_integrity(tmp_path)
    assert report["is_git_repo"] is False
    assert report["integrity_ok"] is False

    (tmp_path / ".git").mkdir()
    (tmp_path / "training").mkdir()
    (tmp_path / "training" / "config").mkdir()
    (tmp_path / "training" / "kaggle").mkdir()
    report = _verify_repo_integrity(tmp_path)
    assert report["integrity_ok"] is True


def test_run_training_imports() -> None:
    from training.kaggle.scripts import run_training

    assert callable(run_training.main)


def test_run_training_component_descriptor() -> None:
    from training.training import Trainer

    from training.kaggle.scripts.run_training import _component_descriptor

    descriptor = _component_descriptor(Trainer)
    assert descriptor["class"].endswith("Trainer")
    assert "model" in descriptor["required_init_args"]
    assert descriptor["instantiated"] is False


def test_system_check_imports() -> None:
    from training.kaggle.scripts import system_check

    assert callable(system_check.main)


def test_notebooks_are_valid_json() -> None:
    import json

    root = Path(__file__).resolve().parents[1] / "notebooks"
    notebooks = ["train", "evaluate", "export", "system_check"]
    for name in notebooks:
        path = root / f"{name}.ipynb"
        assert path.exists(), f"missing notebook {path}"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["nbformat"] == 4
        assert any("bootstrap.py" in "".join(c.get("source", [])) for c in data["cells"])


def test_docs_exist() -> None:
    docs = Path(__file__).resolve().parents[1] / "docs"
    for name in ("SETUP", "KAGGLE", "BOOTSTRAP", "WORKSPACE", "CONFIGURATION"):
        assert (docs / f"{name}.md").exists(), f"missing docs/{name}.md"


def test_verify_multimodal_shape_formatter() -> None:
    """Regression: the tensor trace formatter must not format a raw list with
    a width spec (``TypeError: unsupported format string passed to
    list.__format__``) — stringify before aligning."""
    from training.kaggle.scripts.verify_multimodal_tensors import _shape_str

    shape = [4, 8, 1, 224, 224]
    rendered = _shape_str(shape)
    assert rendered == "[4, 8, 1, 224, 224]"
    assert "224" in rendered
    # The exact render the trace uses must not raise.
    line = f"  {'ndvi_encoder':30s} shape={rendered:26s}"
    assert line.startswith("  ndvi_encoder")
    with pytest.raises(TypeError):
        f"{list(shape):30s}"  # the old, broken pattern


def test_verify_split_uses_frozen_provenance_split() -> None:
    """Regression: verify_split_composition must consume the frozen
    provenance.split (raw train=5924/val=2459/test=2291 under R5.4 Option B),
    never re-split the accepted corpus temporally into the invalid
    8601/0/1518 composition."""
    import sys
    from unittest.mock import Mock

    import training.kaggle.scripts.verify_split_composition as mod

    obs = []
    for i, split in enumerate(("train", "val", "test")):
        o = Mock()
        o.provenance = {"split": split, "record_id": f"r{i}"}
        o.location = Mock()
        o.location.admin = Mock()
        o.location.admin.taluk = split
        obs.append(o)

    train, val, test, unknown = mod._split_from_provenance(obs)
    assert [len(train), len(val), len(test), len(unknown)] == [1, 1, 1, 0]

    obs.append(Mock(provenance={"split": "who_knows"}, location=Mock(admin=Mock(taluk="X"))))
    train, val, test, unknown = mod._split_from_provenance(obs)
    assert len(unknown) == 1
    assert len(train) == 1  # no silent train assignment

    assert mod.__doc__ and "provenance.split" in mod.__doc__
