"""NDVI image-sequence encoder.

Wraps the shared :class:`~ai.models.backbone.TimmImageEncoder` with the
configuration resolved for the NDVI modality. The NDVI and EVI encoders are
separate modules with independent weights so each vegetation index learns its
own spatio-temporal representation.
"""

from __future__ import annotations

from .backbone import TimmImageEncoder


class NdviEncoder(TimmImageEncoder):
    """NDVI encoder — consumes ``[B, T, 1, H, W]`` NDVI patches.

    See :class:`~ai.models.backbone.TimmImageEncoder` for parameters and
    behaviour; this subclass only fixes the semantic role of the encoder so
    the architecture can name and swap each modality independently.
    """
