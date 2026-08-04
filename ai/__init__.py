"""CropFusion AI module.

``ai`` hosts the machine-learning side of the platform:

* :mod:`ai.preprocessing` — the preprocessing / feature-engineering pipeline
  (Phase 4) that converts :class:`~services.spatial_alignment.
  observation.AgriculturalObservation` samples into AI-ready tensors.
* :mod:`ai.models` — the multimodal neural architecture (Phase 5):
  TabTransformer + dual timm encoders + temporal transformer + cross-modal
  attention + adaptive gated fusion + multi-task heads, plus configuration,
  factory, checkpointing and export.

Every sample consumed here is produced by STAM — no AI component reads the
raw datasets directly.
"""

from __future__ import annotations

__version__ = "0.1.0"
