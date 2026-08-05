"""Host system detection for the Training Platform (R2.1).

Reports CPU, RAM, disk space, Python and platform identity. Uses ``psutil``
when available and falls back to stdlib probes (Windows ``ctypes`` RAM query,
``shutil.disk_usage``, ``platform``). Pure infrastructure — no training logic.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


def detect_system(target_path: str | Path | None = None) -> dict[str, Any]:
    """Return a host capability report.

    Args:
        target_path: Optional path to measure free disk space against
            (defaults to the current working directory).
    """
    target = Path(target_path or os.getcwd()).resolve()
    ram = _detect_ram()
    disk = _detect_disk(target)
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cpu_physical": _cpu_physical(),
        "ram_total_bytes": ram["total"],
        "ram_total_gb": ram["total_gb"],
        "ram_available_gb": ram["available_gb"],
        "disk": disk,
    }


def detect_disk(target: str | Path) -> dict[str, Any]:
    """Free / total disk space for the filesystem holding ``target``."""
    return _detect_disk(Path(target))


def _detect_disk(target: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(target)
        free_gb = round(usage.free / (1024 ** 3), 2)
        total_gb = round(usage.total / (1024 ** 3), 2)
        used_gb = round(usage.used / (1024 ** 3), 2)
        return {
            "path": str(target),
            "total_gb": total_gb,
            "used_gb": used_gb,
            "free_gb": free_gb,
        }
    except OSError:  # pragma: no cover - platform dependent
        return {"path": str(target), "total_gb": None, "used_gb": None, "free_gb": None}


def _detect_ram() -> dict[str, Any]:
    """RAM total/available using psutil (preferred) or a stdlib fallback."""
    if importlib.util.find_spec("psutil") is not None:
        import psutil

        vm = psutil.virtual_memory()
        total_gb = round(vm.total / (1024 ** 3), 2)
        available_gb = round(vm.available / (1024 ** 3), 2)
        return {"total": vm.total, "total_gb": total_gb, "available_gb": available_gb}

    total = _windows_ram_total() or 0
    total_gb = round(total / (1024 ** 3), 2) if total else None
    return {"total": total, "total_gb": total_gb, "available_gb": None}


def _windows_ram_total() -> int | None:  # pragma: no cover - Windows only
    if platform.system() != "Windows":
        return None
    try:
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong),
                ("avail_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_ext", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.length = ctypes.sizeof(_MemoryStatus)
        kernel32 = ctypes.windll.kernel32
        if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_phys)
    except (AttributeError, OSError):  # pragma: no cover
        return None
    return None


def _cpu_physical() -> int | None:
    if importlib.util.find_spec("psutil") is not None:
        import psutil

        return psutil.cpu_count(logical=False)
    return os.cpu_count()
