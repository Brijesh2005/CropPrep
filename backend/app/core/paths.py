"""Filesystem path bootstrap.

Adds the backend root and the repository root to ``sys.path`` so the app can
import ``app.*`` and the shared CropFusion packages (``services``, ``ai``)
regardless of the working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ``paths.py`` lives in ``backend/app/core/``: parents[0]=core, [1]=app,
# [2]=backend, [3]=<repo root>.
BACKEND_ROOT = Path(__file__).resolve().parents[2]  # backend/
REPO_ROOT = Path(__file__).resolve().parents[3]  # <repo root>

for _path in (BACKEND_ROOT, REPO_ROOT):
    _str = str(_path)
    if _str not in sys.path:
        sys.path.insert(0, _str)
