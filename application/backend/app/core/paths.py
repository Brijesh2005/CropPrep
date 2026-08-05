"""Filesystem path bootstrap.

Adds the application/backend root, the application root and the repository root
to ``sys.path`` so the app can import ``app.*``, ``database.*``, ``gis.*`` and
the CropFusion training / shared packages (``training.*``, ``shared.*``)
regardless of the working directory.

NOTE: the repository root (not ``training/`` itself) is added so that
``import training.models`` resolves to ``<repo>/training/models``. Adding the
``training`` directory directly would make ``import training`` resolve to the
``training/training`` sub-package instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ``paths.py`` lives in ``application/backend/app/core/``:
# parents[0]=core, [1]=app, [2]=application/backend, [3]=application,
# [4]=<repo root>.
BACKEND_ROOT = Path(__file__).resolve().parents[2]  # application/backend/
APPLICATION_ROOT = Path(__file__).resolve().parents[3]  # application/
REPO_ROOT = Path(__file__).resolve().parents[4]  # <repo root>

for _path in (
    BACKEND_ROOT,
    APPLICATION_ROOT,
    REPO_ROOT,
):
    _str = str(_path)
    if _str not in sys.path:
        sys.path.insert(0, _str)
