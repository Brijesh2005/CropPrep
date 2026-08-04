"""PyTorch DataLoader builder + custom collate.

`build_dataloader` wires a :class:`~ai.preprocessing.dataset.CropFusionDataset`
into a :class:`torch.utils.data.DataLoader` with configurable batch size,
workers, pin memory, persistent workers, prefetching and a custom collate that
stacks the AI-ready sample dicts into batched tensors.
"""

from __future__ import annotations

from typing import Any, Iterable

from .config import DataloaderConfig, PreprocessingConfig
from .dataset import CropFusionDataset
from .logger import get_logger

logger = get_logger("dataloader")


def collate_samples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a batch of AI-ready sample dicts into batched tensors."""
    import torch

    out: dict[str, Any] = {
        "observation_id": [sample["observation_id"] for sample in batch],
        "metadata": [sample["metadata"] for sample in batch],
    }
    for key in ("tabular", "ndvi", "evi", "temporal_mask"):
        out[key] = torch.stack([sample[key] for sample in batch], dim=0)
    out["crop_label"] = torch.stack([sample["crop_label"] for sample in batch])
    out["yield_label"] = torch.stack([sample["yield_label"] for sample in batch])
    return out


def build_dataloader(
    dataset: Any,
    config: PreprocessingConfig | DataloaderConfig | None = None,
    *,
    split: str = "train",
    batch_size: int | None = None,
    workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    shuffle: bool | None = None,
    collate_fn: Any = collate_samples,
    drop_last: bool = False,
) -> Any:
    """Build a :class:`torch.utils.data.DataLoader` for a split.

    Args:
        dataset: A :class:`CropFusionDataset` (or raw torch Dataset).
        config: Preprocessing or Dataloader configuration.
        split: ``"train"`` (shuffled) / ``"val"`` / ``"test"``.
        batch_size / workers / pin_memory / persistent_workers /
            prefetch_factor: Explicit overrides (else from config).
        shuffle: Explicit shuffle override (train defaults to config).
        collate_fn: Custom collate (defaults to :func:`collate_samples`).
        drop_last: Drop the final incomplete batch.
    """
    import torch

    if isinstance(config, PreprocessingConfig):
        loader_config = config.dataloader
    else:
        loader_config = config or DataloaderConfig()

    torch_dataset = (
        dataset.torch_dataset if isinstance(dataset, CropFusionDataset) else dataset
    )
    num_workers = loader_config.workers if workers is None else workers
    use_persistent = (
        (loader_config.persistent_workers if persistent_workers is None else persistent_workers)
        and num_workers > 0
    )
    prefetch = loader_config.prefetch_factor if prefetch_factor is None else prefetch_factor

    loader = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=batch_size or loader_config.batch_size,
        shuffle=(
            (loader_config.shuffle_train and split == "train")
            if shuffle is None
            else shuffle
        ),
        num_workers=num_workers,
        pin_memory=loader_config.pin_memory if pin_memory is None else pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    logger.info(
        "DataLoader built",
        extra={"split": split, "batch_size": loader.batch_size,
               "workers": num_workers},
    )
    return loader
