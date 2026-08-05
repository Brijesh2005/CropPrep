"""Inference package — the consumed artifact set of the Prediction Platform.

R1.4 prepares the contract only: the package directory layout, the expected
files and the README that documents them. The actual artifacts are produced
by the Training Platform export pipeline and shipped into this directory
during deployment (see ``application/docker/Dockerfile.inference.standalone``).
"""

from __future__ import annotations

from .manifest import (
    INFERENCE_PACKAGE_FILES,
    MODEL_WEIGHTS_DEFAULT_NAME,
    MODEL_WEIGHTS_FUTURE_PATTERN,
    MODEL_WEIGHTS_RELATIVE_DIR,
    ExpectedArtifact,
)

__all__ = [
    "ExpectedArtifact",
    "INFERENCE_PACKAGE_FILES",
    "MODEL_WEIGHTS_DEFAULT_NAME",
    "MODEL_WEIGHTS_FUTURE_PATTERN",
    "MODEL_WEIGHTS_RELATIVE_DIR",
]
