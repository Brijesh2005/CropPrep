"""PyTorch dataset + leakage-free data splitting.

* :func:`split_observations` — random / stratified / temporal / spatial /
  group splits. Temporal splits assign whole years and spatial/group splits
  assign whole groups so no spatial or temporal leakage occurs.
* :class:`CropFusionDataset` — lazy PyTorch dataset: each ``__getitem__``
  calls the preprocessor (and patch extractor) on demand, applies
  augmentation only for the training split, and returns the AI-ready sample
  dict consumed by the DataLoader collate function.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Sequence

from .config import PreprocessingConfig, SplitConfig
from .logger import get_logger
from .master_pipeline import Preprocessor
from .statistics import DatasetStatistics

logger = get_logger("dataset")


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def split_observations(
    observations: Sequence[Any],
    config: SplitConfig | None = None,
) -> tuple[list[Any], list[Any], list[Any]]:
    """Split accepted observations into ``(train, val, test)`` without leakage.

    Strategies:

    * ``random`` — seeded shuffle, ratio split.
    * ``stratified`` — ratio split within each crop class (preserves class
      balance).
    * ``temporal`` — whole years are assigned to splits (test = most recent
      years, then val). Prevents temporal leakage.
    * ``spatial`` / ``group`` — whole groups (villages) are assigned to
      splits. Prevents spatial leakage.
    """
    items = list(observations)
    strategy = config.strategy if config else "temporal"
    cfg = config or SplitConfig()

    if strategy == "random":
        return _ratio_split(items, cfg, cfg.seed)
    if strategy == "stratified":
        return _stratified_split(items, cfg)
    if strategy == "temporal":
        return _temporal_split(items, cfg)
    if strategy in {"spatial", "group"}:
        return _group_split(items, cfg)
    raise ValueError(f"Unknown split strategy: {strategy}")


def _ratio_split(
    items: list[Any], cfg: SplitConfig, seed: int
) -> tuple[list[Any], list[Any], list[Any]]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_n = int(round(total * cfg.train_ratio))
    val_n = int(round(total * cfg.val_ratio))
    train = shuffled[:train_n]
    val = shuffled[train_n: train_n + val_n]
    test = shuffled[train_n + val_n:]
    return train, val, test


def _stratified_split(
    items: list[Any], cfg: SplitConfig
) -> tuple[list[Any], list[Any], list[Any]]:
    by_class: dict[str, list[Any]] = defaultdict(list)
    for obs in items:
        by_class[str(getattr(obs, "crop", None) or "unknown")].append(obs)
    train, val, test = [], [], []
    for seed_offset, group in enumerate(by_class.values()):
        t, v, s = _ratio_split(group, cfg, cfg.seed + seed_offset)
        train.extend(t)
        val.extend(v)
        test.extend(s)
    return train, val, test


def _temporal_split(
    items: list[Any], cfg: SplitConfig
) -> tuple[list[Any], list[Any], list[Any]]:
    by_year: dict[int, list[Any]] = defaultdict(list)
    for obs in items:
        by_year[int(getattr(obs, "temporal").year)].append(obs)
    years = sorted(by_year)

    # Small-corpus fallback: with fewer than ``min_years_for_temporal`` distinct
    # years a whole-year assignment would force every year out of train (the
    # ``max(1, ...)`` floor on both test and val). Use an observation-level
    # split instead, keep everything for train/test, and drop the separate val
    # set so the holdout never sees empty train/test splits. Explicit
    # ``test_years`` / ``val_years`` are honored as-is (the user chose them).
    explicit = bool(cfg.test_years) or bool(cfg.val_years)
    if items and len(years) < cfg.min_years_for_temporal and not explicit:
        logger.warning(
            "temporal split fallback: %d distinct year(s) < "
            "min_years_for_temporal=%d — using an observation-level split "
            "with no validation set",
            len(years),
            cfg.min_years_for_temporal,
        )
        rng = random.Random(cfg.seed)
        shuffled = list(items)
        rng.shuffle(shuffled)
        total = len(shuffled)
        n_test = min(max(1, int(round(total * cfg.test_ratio))), total - 1)
        return shuffled[: total - n_test], [], shuffled[total - n_test:]

    test_years = set(cfg.test_years) if cfg.test_years else _last_fraction(years, cfg.test_ratio)
    remaining = [y for y in years if y not in test_years]
    val_years = set(cfg.val_years) if cfg.val_years else _last_fraction(
        remaining, cfg.val_ratio / (cfg.train_ratio + cfg.val_ratio)
    )
    train_years = [y for y in remaining if y not in val_years]

    train = [obs for y in train_years for obs in by_year[y]]
    val = [obs for y in sorted(val_years) for obs in by_year[y]]
    test = [obs for y in sorted(test_years) for obs in by_year[y]]
    return train, val, test


def _group_split(
    items: list[Any], cfg: SplitConfig
) -> tuple[list[Any], list[Any], list[Any]]:
    def _group(obs: Any) -> str:
        admin = getattr(getattr(obs, "location", None), "admin", None)
        if admin is not None and getattr(admin, "village", None):
            return f"village:{admin.village}"
        fields = getattr(getattr(obs, "tabular", None), "fields", {}) or {}
        return f"{cfg.group_column}:{fields.get(cfg.group_column, 'unknown')}"

    by_group: dict[str, list[Any]] = defaultdict(list)
    for obs in items:
        by_group[_group(obs)].append(obs)

    groups = sorted(by_group)
    rng = random.Random(cfg.seed)
    rng.shuffle(groups)
    total = len(groups)
    train_n = int(round(total * cfg.train_ratio))
    val_n = int(round(total * cfg.val_ratio))
    train_groups = set(groups[:train_n])
    val_groups = set(groups[train_n: train_n + val_n])
    test_groups = set(groups[train_n + val_n:])

    train = [obs for g in train_groups for obs in by_group[g]]
    val = [obs for g in val_groups for obs in by_group[g]]
    test = [obs for g in test_groups for obs in by_group[g]]
    return train, val, test


def _last_fraction(years: list[int], fraction: float) -> set[int]:
    if not years:
        return set()
    count = int(round(len(years) * fraction))
    # Never assign away the ONLY remaining year(s): keep at least one year for
    # train so a whole-year split can never starve it.
    count = min(max(count, 1), len(years) - 1)
    if count <= 0:
        return set()
    return set(years[-count:])


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class CropFusionDataset:
    """Lazy PyTorch-compatible dataset over STAM observations.

    Args:
        preprocessor: Fitted :class:`Preprocessor`.
        observations: Accepted observations (train/val/test subset).
        split: ``"train"`` (augmentation applied) | ``"val"`` | ``"test"``.
        extractor: ``callable(path, lon, lat, size=...) -> RasterPatch``.
        config: Full preprocessing config (used for statistics/augmentation).
    """

    def __init__(
        self,
        preprocessor: Preprocessor,
        observations: Sequence[Any],
        *,
        split: str = "train",
        extractor: Any | None = None,
        config: PreprocessingConfig | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.observations = list(observations)
        self.split = split
        self.extractor = extractor
        self.config = config or preprocessor.config

    # -- torch integration --------------------------------------------------- #

    @property
    def torch_dataset(self) -> Any:
        """Adapt to :class:`torch.utils.data.Dataset`."""
        import torch

        observations = self.observations
        preprocessor = self.preprocessor
        extractor = self.extractor
        split = self.split

        class _Adapter(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return len(observations)

            def __getitem__(self, index: int) -> dict[str, Any]:
                return preprocessor.transform(
                    observations[index],
                    extractor=extractor,
                    augment=split == "train",
                )

        return _Adapter()

    # -- Public API ----------------------------------------------------------- #

    @classmethod
    def build(
        cls,
        preprocessor: Preprocessor,
        observations: Sequence[Any],
        *,
        split: str = "train",
        extractor: Any | None = None,
    ) -> "CropFusionDataset":
        """Build a dataset from accepted observations (preprocessor must be fitted)."""
        return cls(preprocessor, observations, split=split, extractor=extractor)

    def statistics(self, output_dir: str | None = None) -> dict[str, Any]:
        """Compute dataset statistics and (optionally) save a report."""
        report = DatasetStatistics.summarize(self.observations)
        if output_dir is not None:
            report.save(output_dir)
        return report.to_dict()

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.preprocessor.transform(
            self.observations[index],
            extractor=self.extractor,
            augment=self.split == "train",
        )
