"""EVI image-sequence encoder.

Wraps the shared :class:`~ai.models.backbone.TimmImageEncoder` with the
configuration resolved for the EVI modality. Like
:class:`~ai.models.ndvi_encoder.NdviEncoder`, the EVI encoder is an
independent module with its own weights.
"""

from __future__ import annotations

from .backbone import TimmImageEncoder


class EviEncoder(TimmImageEncoder):
    """EVI encoder — consumes ``[B, T, 1, H, W]`` EVI patches.

    See :class:`~ai.models.backbone.TimmImageEncoder` for parameters and
    behaviour; this subclass only fixes the semantic role of the encoder so
    the architecture can name and swap each modality independently.
    """
