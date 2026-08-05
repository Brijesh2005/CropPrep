"""Root-level pytest fixtures and path bootstrap for the enterprise QA suite.

The CropFusion repo is organised as three platform roots (``training``,
``application``, ``shared``). The ``application/tests/`` tree exercises the
application together with the training packages, so the import path must
expose every component root.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]  # <repo root>
_APPLICATION_ROOT = ROOT / "application"
for _path in (
    _APPLICATION_ROOT / "backend",
    _APPLICATION_ROOT / "gis",
    ROOT,
):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ.setdefault("BACKEND_ENVIRONMENT", "test")
os.environ.setdefault("BACKEND_MODEL__WARMUP", "false")
os.environ.setdefault("BACKEND_DATASET__VALIDATE_ON_STARTUP", "false")
os.environ.setdefault("BACKEND_RATE_LIMIT__ENABLED", "false")
os.environ.setdefault("BACKEND_LOG__JSON_LOGS", "false")


def pytest_collection_modifyitems(config, items) -> None:
    """Auto-apply markers based on the test's directory."""
    for item in items:
        path = Path(str(item.fspath)).as_posix()
        if "/tests/unit/" in path:
            item.add_marker("unit")
        elif "/tests/integration/" in path:
            item.add_marker("integration")
        elif "/tests/system/" in path:
            item.add_marker("system")
        elif "/tests/backend/" in path:
            item.add_marker("backend")
        elif "/tests/database/" in path:
            item.add_marker("database")
        elif "/tests/gis/" in path:
            item.add_marker("gis")
        elif "/tests/ai/" in path:
            item.add_marker("ai")
        elif "/tests/api/" in path:
            item.add_marker("api")
        elif "/tests/security/" in path:
            item.add_marker("security")
        elif "/tests/performance/" in path:
            item.add_marker("performance")
        elif "/tests/smoke/" in path:
            item.add_marker("smoke")
        elif "/tests/e2e/" in path:
            item.add_marker("e2e")


@pytest.fixture
def tmp_out(tmp_path: Path) -> Path:
    """A clean output directory for any artifact-generating test."""
    out = tmp_path / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    return out
