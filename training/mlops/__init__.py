"""CropFusion MLOps toolkit.

Model registry, promotion gates, experiment tracking, monitoring scheduler
and release reporting. Runs standalone (filesystem registry + quality QA
gates) and as the ``admin`` container's scheduler in the compose stack.

CLI: ``cropfusion-mlops`` (see :mod:`training.mlops.cli`).
"""

from .config import MLOpsSettings, load_settings
from .registry import ModelRegistry, ModelRecord

__version__ = "1.0.0"

__all__ = ["MLOpsSettings", "load_settings", "ModelRegistry", "ModelRecord", "__version__"]
