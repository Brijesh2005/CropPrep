"""Bridge from the generated corpus to the Phase 4 preprocessing datasets.

The R2.3 corpus (:class:`~training.stam.observation_resolver.ObservationCorpus`)
feeds the existing preprocessing stack
(:func:`~training.preprocessing.dataset.split_observations` +
:class:`~training.preprocessing.dataset.CropFusionDataset`) through this thin
glue layer: accepted observations are split leakage-free and wrapped in
train/val/test PyTorch datasets.
"""

from __future__ import annotations

from typing import Any

from .logger import get_logger
from .utils import observations_from_corpus

logger = get_logger("dataset")


def build_cropfusion_datasets(
    corpus: Any,
    preprocessor: Any,
    split_config: Any | None = None,
    *,
    extractor: Any | None = None,
    split_observations: Any | None = None,
    cropfusion_dataset: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Split a corpus's accepted observations and build train/val/test datasets.

    Args:
        corpus: An :class:`ObservationCorpus` (accepted observations used) or
            a list of :class:`AgriculturalObservation`.
        preprocessor: A fitted
            :class:`~training.preprocessing.master_pipeline.Preprocessor`.
        split_config: Optional
            :class:`~training.preprocessing.config.SplitConfig` (leakage-free
            ``temporal`` strategy by default).
        extractor: Patch extractor (e.g. ``stam.get_patch``) forwarded to the
            datasets.
        split_observations / cropfusion_dataset: Injectable functions/classes
            (defaults to the preprocessing implementations).

    Returns:
        The ``(train, val, test)`` :class:`CropFusionDataset` trio.
    """
    if split_observations is None:
        from training.preprocessing import split_observations as _split
    else:
        _split = split_observations
    if cropfusion_dataset is None:
        from training.preprocessing import CropFusionDataset as _Dataset
    else:
        _Dataset = cropfusion_dataset

    observations = observations_from_corpus(corpus)
    if not observations:
        logger.warning("No accepted observations to split")
        return _Dataset.build(preprocessor, [], split="train"), \
            _Dataset.build(preprocessor, [], split="val"), \
            _Dataset.build(preprocessor, [], split="test")

    train, val, test = _split(observations, split_config)
    logger.info(
        "Preprocessing datasets built",
        extra={"train": len(train), "val": len(val), "test": len(test)},
    )
    return (
        _Dataset.build(preprocessor, train, split="train", extractor=extractor),
        _Dataset.build(preprocessor, val, split="val", extractor=extractor),
        _Dataset.build(preprocessor, test, split="test", extractor=extractor),
    )
