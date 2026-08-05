"""Kaggle runtime detection for the Training Platform (R2.1).

Identifies whether the process runs inside a Kaggle notebook/kernel, where the
input/working directories are, and whether internet is expected. Pure
infrastructure — never touches training code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

#: Env vars that indicate an active Kaggle kernel.
_KAGGLE_ENV_MARKERS = ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_KERNEL_INTEGRATIONS")


def detect_runtime() -> dict[str, Any]:
    """Return a Kaggle-runtime report.

    Flags ``is_kaggle`` when ``/kaggle`` exists or a ``KAGGLE_*`` marker is
    present. Input / working dirs fall back to the standard locations when the
    process is not on Kaggle.
    """
    on_kaggle = Path("/kaggle").exists() or any(
        os.environ.get(key) for key in _KAGGLE_ENV_MARKERS
    )
    input_dir = Path("/kaggle/input")
    working_dir = Path("/kaggle/working")
    return {
        "is_kaggle": bool(on_kaggle),
        "input_dir": str(input_dir) if input_dir.exists() else None,
        "working_dir": str(working_dir) if working_dir.exists() else None,
        "kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        "kernel_id": os.environ.get("KAGGLE_KERNEL_RUN_ID"),
        "internet": os.environ.get("KAGGLE_KERNEL_INTERNET") != "off",
        "gpu_requested": os.environ.get("KAGGLE_KERNEL_GPU") == "true",
    }
