"""Launcher for the CropFusion backend (no manual PYTHONPATH needed).

Sets up the module search path (repo root for ``shared``, plus the
``application/`` packages) and the default settings file, then starts uvicorn::

    D:\\CropPrep\\.venv\\Scripts\\python.exe run.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent
_REPO = _BACKEND.parent.parent
_APPLICATION = _REPO / "application"

for _path in (_REPO, _BACKEND, _APPLICATION):
    _value = str(_path)
    if _value not in sys.path:
        sys.path.insert(0, _value)

os.environ.setdefault(
    "BACKEND_CONFIG_FILE", str(_BACKEND / "config" / "settings.yaml")
)

import uvicorn


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
