#!/usr/bin/env python
"""Executable entry point for the Dataset Manager CLI.

Run from anywhere on the filesystem::

    python services/dataset_manager/manage_dataset.py download
    python services/dataset_manager/manage_dataset.py validate --json
    python services/dataset_manager/manage_dataset.py summary

The script inserts the repository root into ``sys.path`` so that the
``services`` package is importable regardless of the current working
directory.
"""

from __future__ import annotations

import os
import sys

# Make `services` (and thus `services.dataset_manager`) importable no matter
# where this script is invoked from. parents: [dataset_manager, services, repo].
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from services.dataset_manager.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
