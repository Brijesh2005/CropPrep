"""Configuration for the preprocessing / feature-engineering pipeline.

Settings resolve env (``PRE_``) > YAML (``PRE_CONFIG_FILE``) > defaults, and
every field is validated by pydantic. Each stage (tabular / image / temporal /
label / split / augmentation / dataloader / quality) has its own section so
the whole pipeline is configurable without code changes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field

from training.dataset_manager.config import _apply_case_insensitive, _parse_env, deep_merge
from .exceptions import ConfigurationError

ENV_PREFIX = "PRE_"


class TabularConfig(BaseModel):
    """Tabular feature processing settings."""

    model_config = ConfigDict(extra="forbid")

    #: Numerical scaling: standard | minmax | robust | none.
    scaler: str = Field(default="standard", pattern="^(standard|minmax|robust|none)$")
    #: Missing-value strategy: mean | median | zero | drop | none.
    handle_missing: str = Field(default="mean", pattern="^(mean|median|zero|drop|none)$")
    #: Outlier handling: iqr | zscore | none.
    outlier_method: str = Field(default="iqr", pattern="^(iqr|zscore|none)$")
    #: Z-score threshold for outlier flagging (used with zscore).
    outlier_threshold: float = Field(default=3.0, ge=0.0)
    #: Categorical encoding: onehot | ordinal | none.
    categorical_encoding: str = Field(default="onehot", pattern="^(onehot|ordinal|none)$")
    #: Drop columns with a single unique value.
    drop_constant_features: bool = True
    #: Drop one of any pair of features with |corr| above this (None disables).
    max_correlation: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Explicit numeric feature columns (auto-inferred when empty).
    numeric_features: list[str] = Field(default_factory=list)
    #: Explicit categorical feature columns (auto-inferred when empty).
    categorical_features: list[str] = Field(default_factory=list)
    #: Field keys to drop from features (labels, identifiers, context).
    exclude_columns: list[str] = Field(default_factory=list)


class ImageConfig(BaseModel):
    """Image (NDVI/EVI patch) processing settings."""

    model_config = ConfigDict(extra="forbid")

    #: Output patch edge in pixels (128 / 224 / 256).
    size: int = Field(default=128, ge=8, le=2048)
    #: Normalization: minmax | standard | identity.
    normalize: str = Field(default="minmax", pattern="^(minmax|standard|identity)$")
    #: Physical NDVI range used for minmax normalization.
    ndvi_range: tuple[float, float] = (-1.0, 1.0)
    #: Physical EVI range used for minmax normalization.
    evi_range: tuple[float, float] = (-1.0, 1.0)
    #: NaN handling: zero | mean | drop.
    nan_policy: str = Field(default="zero", pattern="^(zero|mean|drop)$")
    #: Invalid (masked) pixel handling.
    invalid_policy: str = Field(default="zero", pattern="^(zero|drop)$")
    #: Clip values to the configured range after normalization.
    clip: bool = True
    #: Resize patches whose native size differs from ``size``.
    resize: bool = True
    #: Pad edges to the target size (mirror of STAM patch generator).
    pad: bool = True


class TemporalConfig(BaseModel):
    """Temporal sequence processing settings."""

    model_config = ConfigDict(extra="forbid")

    #: Sequences are truncated/padded to this many observations.
    max_observations: int = Field(default=8, ge=1)
    #: Samples with fewer observations are rejected.
    min_observations: int = Field(default=1, ge=0)
    #: Value used for padded positions.
    pad_value: float = 0.0
    #: Which end to pad: right | left.
    pad_mode: str = Field(default="right", pattern="^(right|left)$")
    #: Which end to truncate: tail | head.
    truncation: str = Field(default="tail", pattern="^(tail|head)$")
    #: Produce a 1/0 validity mask alongside the sequences.
    mask_padding: bool = True
    #: Sort observation dates ascending (normalises out-of-order input).
    sort_by_date: bool = True
    #: Drop duplicate dates before building the sequence.
    drop_duplicate_dates: bool = True


class LabelConfig(BaseModel):
    """Label processing settings."""

    model_config = ConfigDict(extra="forbid")

    #: Crop label encoding: label | onehot.
    crop_encoding: str = Field(default="label", pattern="^(label|onehot)$")
    #: Yield task: regression (classification is a future extension).
    yield_task: str = Field(default="regression", pattern="^(regression|classification)$")
    #: Yield scaling: standard | minmax | none.
    yield_scaler: str = Field(default="standard", pattern="^(standard|minmax|none)$")


class SplitConfig(BaseModel):
    """Data-splitting settings (no spatial/temporal leakage)."""

    model_config = ConfigDict(extra="forbid")

    #: random | stratified | spatial | temporal | group.
    strategy: str = Field(
        default="temporal", pattern="^(random|stratified|spatial|temporal|group)$"
    )
    train_ratio: float = Field(default=0.7, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    seed: int = 42
    #: Attribute used for group / spatial splits (e.g. "village").
    group_column: str = "village"
    #: Attribute used for temporal splits (e.g. "year").
    temporal_column: str = "year"
    #: Explicit test years (temporal strategy) — overrides ratios.
    test_years: list[int] = Field(default_factory=list)
    #: Explicit validation years (temporal strategy).
    val_years: list[int] = Field(default_factory=list)


class AugmentationConfig(BaseModel):
    """Image augmentation (training only)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    flip_horizontal: bool = False
    flip_vertical: bool = False
    #: Rotations to sample from, in degrees.
    rotation_degrees: list[float] = Field(default_factory=list)
    random_crop: bool = False
    crop_fraction: float = Field(default=0.9, gt=0.0, le=1.0)
    brightness_jitter: float = Field(default=0.0, ge=0.0)
    contrast_jitter: float = Field(default=0.0, ge=0.0)
    noise_std: float = Field(default=0.0, ge=0.0)


class DataloaderConfig(BaseModel):
    """PyTorch DataLoader settings."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=32, ge=1)
    workers: int = Field(default=0, ge=0)
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = Field(default=None, ge=1)
    shuffle_train: bool = True


class QualityConfig(BaseModel):
    """Quality-filter thresholds for accepting observations."""

    model_config = ConfigDict(extra="forbid")

    #: Observations below this quality score are rejected.
    min_quality_score: float = Field(default=40.0, ge=0.0, le=100.0)
    require_valid_coordinates: bool = True
    require_crop_label: bool = False
    require_yield_label: bool = False
    min_observations: int = Field(default=1, ge=0)
    #: Reject observations that lack full NDVI+EVI pairing.
    reject_unpaired: bool = False


class PreprocessingConfig(BaseModel):
    """Root preprocessing configuration."""

    model_config = ConfigDict(extra="forbid")

    tabular: TabularConfig = Field(default_factory=TabularConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    label: LabelConfig = Field(default_factory=LabelConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    dataloader: DataloaderConfig = Field(default_factory=DataloaderConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    #: Where fitted artifacts (scalers, encoders, stats) are persisted.
    output_dir: Path = Field(default=Path("artifacts/preprocessing"))


def load_preprocessing_config(
    config_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> PreprocessingConfig:
    """Load and validate preprocessing settings (env > YAML > defaults)."""
    env_map = dict(os.environ if env is None else env)
    if config_path is None:
        env_config = env_map.get("PRE_CONFIG_FILE")
        config_path = env_config or None

    data: dict[str, Any] = {}
    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.exists():
            raise ConfigurationError(
                f"Preprocessing config file not found: {config_file}",
                detail=str(config_file),
            )
        try:
            raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Malformed preprocessing YAML: {exc}", detail=str(config_file)
            ) from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ConfigurationError("Preprocessing config root must be a mapping")
        data = raw

    merged = deep_merge(data, _parse_env(env_map, prefix=ENV_PREFIX))
    merged = _apply_case_insensitive(merged, PreprocessingConfig)
    try:
        return PreprocessingConfig.model_validate(merged)
    except Exception as exc:  # pydantic.ValidationError
        raise ConfigurationError(f"Invalid preprocessing configuration: {exc}") from exc


def save_preprocessing_template(path: str | Path) -> Path:
    """Write an annotated YAML template of the default configuration."""
    template = {
        "tabular": {"scaler": "standard", "handle_missing": "mean",
                    "outlier_method": "iqr", "outlier_threshold": 3.0,
                    "categorical_encoding": "onehot",
                    "drop_constant_features": True, "max_correlation": None,
                    "numeric_features": [], "categorical_features": [],
                    "exclude_columns": []},
        "image": {"size": 128, "normalize": "minmax", "ndvi_range": [-1.0, 1.0],
                  "evi_range": [-1.0, 1.0], "nan_policy": "zero",
                  "invalid_policy": "zero", "clip": True, "resize": True,
                  "pad": True},
        "temporal": {"max_observations": 8, "min_observations": 1,
                     "pad_value": 0.0, "pad_mode": "right",
                     "truncation": "tail", "mask_padding": True,
                     "sort_by_date": True, "drop_duplicate_dates": True},
        "label": {"crop_encoding": "label", "yield_task": "regression",
                  "yield_scaler": "standard"},
        "split": {"strategy": "temporal", "train_ratio": 0.7, "val_ratio": 0.15,
                  "test_ratio": 0.15, "seed": 42, "group_column": "village",
                  "temporal_column": "year", "test_years": [], "val_years": []},
        "augmentation": {"enabled": False, "flip_horizontal": False,
                         "flip_vertical": False, "rotation_degrees": [],
                         "random_crop": False, "crop_fraction": 0.9,
                         "brightness_jitter": 0.0, "contrast_jitter": 0.0,
                         "noise_std": 0.0},
        "dataloader": {"batch_size": 32, "workers": 0, "pin_memory": False,
                       "persistent_workers": False, "prefetch_factor": None,
                       "shuffle_train": True},
        "quality": {"min_quality_score": 40.0, "require_valid_coordinates": True,
                    "require_crop_label": False, "require_yield_label": False,
                    "min_observations": 1, "reject_unpaired": False},
        "output_dir": "artifacts/preprocessing",
    }
    out = Path(path)
    out.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    return out
