"""CropFusion backend — a domain-driven modular monolith.

The backend exposes the Phase 2–7 CropFusion stack (Dataset Manager, STAM,
preprocessing, model, explainability) through a modular FastAPI monolith whose
modules are isolated and extractable into independent microservices.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Path bootstrap must run before any ``app.*`` submodule import.
_BACKEND_ROOT = Path(__file__).resolve().parents[1]  # backend/
_REPO_ROOT = Path(__file__).resolve().parents[2]  # <repo root>
for _path in (_BACKEND_ROOT, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

__version__ = "0.1.0"
