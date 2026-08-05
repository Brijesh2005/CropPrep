"""GPU / CUDA detection for the Training Platform (R2.1).

Pure infrastructure: reports GPU presence, CUDA availability, device details
and (when ``nvidia-smi`` is on PATH) driver + VRAM info. Never touches models
or training code.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any


def detect_gpu() -> dict[str, Any]:
    """Return a capability report for the available GPUs.

    Uses PyTorch's CUDA runtime when installed; falls back to ``nvidia-smi``
    parsing for driver-level details. Never raises — missing torch / no GPU
    produce an ``available=False`` report.
    """
    report: dict[str, Any] = {
        "available": False,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "driver": None,
        "cuda_version": None,
        "backend": None,
    }

    torch_ok = _try_import_torch()
    smi = _try_nvidia_smi()

    if torch_ok:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if cuda_available else 0
        report["cuda_available"] = cuda_available
        report["device_count"] = count
        report["cuda_version"] = getattr(torch.version, "cuda", None)
        report["backend"] = f"torch {torch.__version__}"
        if cuda_available:
            for index in range(count):
                props = torch.cuda.get_device_properties(index)
                report["devices"].append(
                    {
                        "index": index,
                        "name": props.name,
                        "total_memory_mb": round(props.total_memory / (1024 ** 2), 1),
                        "capability": f"{props.major}.{props.minor}",
                        "is_available": bool(torch.cuda.is_available()),
                    }
                )
            report["available"] = True

    if smi is not None:
        report["driver"] = smi.get("driver")
        report["cuda_version"] = smi.get("cuda_version") or report["cuda_version"]
        if not report["devices"]:
            report["devices"] = smi["devices"]
            report["device_count"] = len(smi["devices"])
            report["available"] = bool(smi["devices"])
        if report["backend"] is None:
            report["backend"] = "nvidia-smi"

    if not report["backend"]:
        report["backend"] = "none"
    return report


def _try_import_torch() -> bool:
    import importlib.util

    return importlib.util.find_spec("torch") is not None


def _try_nvidia_smi() -> dict[str, Any] | None:
    """Parse ``nvidia-smi --query-gpu=...`` output; None when unavailable."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        header = (
            "index,name,memory.total,memory.free,memory.used,"
            "compute_cap,driver_version"
        )
        query = (
            "index,name,memory.total,memory.free,memory.used,"
            "compute_cap,driver_version"
        )
        out = subprocess.run(
            [binary, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode != 0:
            return None
        rows = [r.strip() for r in out.stdout.strip().splitlines() if r.strip()]
        devices = []
        for row in rows:
            parts = [p.strip() for p in row.split(",")]
            if len(parts) != 7:
                continue
            idx, name, total, free, used, cap, driver = parts
            devices.append(
                {
                    "index": int(idx),
                    "name": name,
                    "total_memory_mb": round(float(total) / 1024, 1),
                    "free_memory_mb": round(float(free) / 1024, 1),
                    "used_memory_mb": round(float(used) / 1024, 1),
                    "capability": cap,
                    "is_available": True,
                }
            )
        driver = devices[0]["capability"] if devices else None
        _ = driver
        # driver_version is a query column; recover it from a second query.
        driver_version = _smi_driver_version(binary)
        return {
            "driver": driver_version,
            "cuda_version": None,
            "devices": devices,
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _smi_driver_version(binary: str) -> str | None:
    try:
        out = subprocess.run(
            [binary, "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip().splitlines()[0].strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
