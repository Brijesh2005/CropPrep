"""Dependency detection for the Training Platform (R2.1).

Probes whether required Python packages are importable and reports their
versions. Handles the common distribution-name → import-name aliases
(``scikit-learn`` → ``sklearn``, ``opencv`` → ``cv2``, ``gdal`` →
``osgeo``, ``PyYAML`` → ``yaml``, ``Pillow`` → ``PIL``). Pure infrastructure.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from typing import Any

#: Distribution name -> import module name for the packages CropFusion checks.
ALIASES = {
    "scikit-learn": "sklearn",
    "scikit_learn": "sklearn",
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "opencv": "cv2",
    "gdal": "osgeo",
    "PyYAML": "yaml",
    "pyyaml": "yaml",
    "Pillow": "PIL",
    "pillow": "PIL",
    "torch": "torch",
    "timm": "timm",
    "kagglehub": "kagglehub",
}

#: Default probes used by :func:`detect_dependencies` when none are given.
DEFAULT_PROBES = [
    "numpy",
    "pandas",
    "torch",
    "scikit-learn",
    "rasterio",
    "gdal",
    "tensorflow",
    "opencv-python",
    "timm",
    "kagglehub",
    "yaml",
]


def detect_dependencies(
    requirements: list[str] | None = None,
) -> dict[str, Any]:
    """Probe each package and return ``{name: {installed, version, import}}``."""
    probes = requirements or DEFAULT_PROBES
    result: dict[str, Any] = {}
    for name in probes:
        import_name = ALIASES.get(name, name)
        result[name] = {
            "installed": find_spec(import_name) is not None,
            "version": _version_of(name, import_name),
            "import": import_name,
        }
    return result


def _version_of(distribution: str, import_name: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        pass
    try:
        module = __import__(import_name)
        return str(getattr(module, "__version__", None))
    except Exception:  # noqa: BLE001 - probing must never raise
        return None
