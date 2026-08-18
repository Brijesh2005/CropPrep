"""Supervised data contract for CropFusion (R5.2.2).

Separates the mixed corpus into three explicit, non-overlapping datasets:

1. **Crop Recommendation Dataset** — observations with valid crop labels only.
2. **Physical Yield Dataset** — observations with kg/ha yield only.
3. **Auxiliary / Unlabeled Dataset** — NPP-proxy observations for future
   self-supervised or representation-learning tasks.

No dataset silently inherits labels from another. No metrics are reported
for undefined tasks. No mixed yield units are permitted.

Usage::

    from training.preprocessing.supervised_contract import (
        SupervisedDataContract,
        build_contract,
    )

    contract = build_contract(observations)
    crop_obs = contract.crop_dataset.observations
    yield_obs = contract.yield_dataset.observations
    aux_obs  = contract.auxiliary_dataset.observations
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .data_contract import infer_yield_unit, _KG_HA_SOURCES, _NPP_SOURCES


# --------------------------------------------------------------------------- #
# Dataset descriptors
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DatasetDescriptor:
    """Metadata for one supervised or auxiliary dataset partition."""

    name: str
    observations: list[Any] = field(default_factory=list)
    sample_count: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    class_distribution: dict[str, float] = field(default_factory=dict)
    years: list[int] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    source_datasets: list[str] = field(default_factory=list)
    modalities: dict[str, bool] = field(default_factory=dict)
    yield_unit: str | None = None
    yield_source: str | None = None
    yield_stats: dict[str, float] = field(default_factory=dict)
    split_feasibility: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "sample_count": self.sample_count,
            "class_counts": dict(self.class_counts),
            "class_distribution": dict(self.class_distribution),
            "years": self.years,
            "locations": self.locations,
            "source_datasets": self.source_datasets,
            "modalities": dict(self.modalities),
        }
        if self.yield_unit is not None:
            d["yield_unit"] = self.yield_unit
        if self.yield_source is not None:
            d["yield_source"] = self.yield_source
        if self.yield_stats:
            d["yield_stats"] = dict(self.yield_stats)
        if self.split_feasibility:
            d["split_feasibility"] = self.split_feasibility
        if self.notes:
            d["notes"] = list(self.notes)
        return d


@dataclass(frozen=True)
class SupervisedDataContract:
    """The complete R5.2.2 supervised data contract.

    Attributes:
        crop_dataset: Observations with valid crop labels (kg/ha yield).
        yield_dataset: Observations with kg/ha yield (may overlap with crop).
        auxiliary_dataset: NPP-proxy observations (no crop labels, no kg/ha).
        crop_split: Train/val/test split for crop task.
        yield_split: Train/val/test split for yield task.
        leakage_report: Leakage check results.
        crop_split_feasibility: Whether class-wise split is possible.
        temporal_crop_generalization: Whether temporal crop eval is supported.
        overall_validity: Whether the contract passes all checks.
        errors: Hard violations.
        warnings: Degradations.
    """

    crop_dataset: DatasetDescriptor
    yield_dataset: DatasetDescriptor
    auxiliary_dataset: DatasetDescriptor
    crop_split: dict[str, list[Any]] = field(default_factory=dict)
    yield_split: dict[str, list[Any]] = field(default_factory=dict)
    leakage_report: dict[str, Any] = field(default_factory=dict)
    crop_split_feasibility: str = ""
    temporal_crop_generalization: str = ""
    overall_validity: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop_dataset": self.crop_dataset.to_dict(),
            "yield_dataset": self.yield_dataset.to_dict(),
            "auxiliary_dataset": self.auxiliary_dataset.to_dict(),
            "crop_split": {
                name: len(obs) for name, obs in self.crop_split.items()
            },
            "yield_split": {
                name: len(obs) for name, obs in self.yield_split.items()
            },
            "leakage_report": self.leakage_report,
            "crop_split_feasibility": self.crop_split_feasibility,
            "temporal_crop_generalization": self.temporal_crop_generalization,
            "overall_validity": self.overall_validity,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _obs_id(o: Any) -> str:
    """Stable identifier for an observation."""
    oid = getattr(o, "observation_id", None)
    if oid is not None:
        return str(oid)
    parts = [
        str(getattr(getattr(o, "location", None), "lon", "")),
        str(getattr(getattr(o, "location", None), "lat", "")),
        str(getattr(getattr(o, "temporal", None), "year", "")),
        str(getattr(o, "crop", "")),
    ]
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]


def _location_key(o: Any) -> str:
    """Grouping key for location (village or district)."""
    loc = getattr(o, "location", None)
    if loc is None:
        return "unknown"
    admin = getattr(loc, "admin", None)
    if admin is not None:
        village = getattr(admin, "village", None)
        if village:
            return f"village:{village}"
        district = getattr(admin, "district", None)
        if district:
            return f"district:{district}"
    return "unknown"


def _year(o: Any) -> int:
    t = getattr(o, "temporal", None)
    if t is not None:
        return int(getattr(t, "year", 0))
    return 0


def _source_file(o: Any) -> str:
    tab = getattr(o, "tabular", None)
    if tab is not None:
        sp = getattr(tab, "source_path", None)
        if sp:
            return str(Path(sp).name)
    return "unknown"


def _has_images(o: Any) -> bool:
    return bool(getattr(o, "has_paired_images", False))


def _has_tabular(o: Any) -> bool:
    tab = getattr(o, "tabular", None)
    if tab is not None:
        ml = getattr(tab, "matched_level", "none")
        return ml != "none"
    return False


def _has_temporal(o: Any) -> bool:
    """Check observation has temporal identity (year/month/day)."""
    t = getattr(o, "temporal", None)
    if t is None:
        return False
    return getattr(t, "year", None) is not None


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def _classify_observation(o: Any) -> dict[str, Any]:
    """Classify one observation into crop / yield / auxiliary buckets.

    Returns dict with:
      - crop_eligible: has valid crop label (not None, not -1)
      - yield_kg_ha_eligible: has physical kg/ha yield
      - auxiliary: NPP proxy or unlabeled
      - crop_label: crop name or None
      - yield_unit: inferred unit or None
      - yield_value: float or None
    """
    crop = getattr(o, "crop", None)
    crop_eligible = (
        crop is not None
        and str(crop) != "-1"
        and str(crop).lower() not in ("unknown", "none", "")
    )

    yield_value = getattr(o, "yield_value", None)
    yield_unit = None
    if yield_value is not None:
        yield_value = float(yield_value)
        yield_unit = infer_yield_unit(
            getattr(getattr(o, "tabular", None), "source_path", None),
            getattr(getattr(o, "tabular", None), "matched_level", None),
            yield_value,
        )

    yield_kg_ha_eligible = (
        yield_unit == "kg/ha"
        and yield_value is not None
        and yield_value > 0
    )

    auxiliary = (
        not crop_eligible
        and not yield_kg_ha_eligible
        and yield_unit == "npp_proxy"
    )

    return {
        "crop_eligible": crop_eligible,
        "yield_kg_ha_eligible": yield_kg_ha_eligible,
        "auxiliary": auxiliary,
        "crop_label": str(crop) if crop_eligible else None,
        "yield_unit": yield_unit,
        "yield_value": yield_value,
    }


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def _split_crop_dataset(
    observations: list[Any],
    *,
    min_samples_per_class: int = 2,
) -> tuple[dict[str, list[Any]], str, list[str]]:
    """Attempt a stratified split of the crop dataset.

    Returns (split_dict, feasibility_note, warnings).
    """
    warnings: list[str] = []

    if not observations:
        return {"train": [], "val": [], "test": []}, "EMPTY_CROP_DATASET", warnings

    by_class: dict[str, list[Any]] = {}
    for o in observations:
        label = str(getattr(o, "crop", "unknown"))
        by_class.setdefault(label, []).append(o)

    # Check class-wise feasibility
    rare_classes = {k: v for k, v in by_class.items() if len(v) < min_samples_per_class}
    sufficient_classes = {k: v for k, v in by_class.items() if len(v) >= min_samples_per_class}

    if rare_classes:
        for cls, count in rare_classes.items():
            warnings.append(
                f"Class '{cls}' has only {count} sample(s) — "
                f"cannot split into train/val/test (min={min_samples_per_class}). "
                f"Keeping all samples in train only."
            )

    if not sufficient_classes:
        return (
            {"train": list(observations), "val": [], "test": []},
            "CROP_DATA_INSUFFICIENT_FOR_CLASS_WISE_GENERALIZATION",
            warnings,
        )

    # Group-aware split by location for sufficient classes
    train, val, test = [], [], []

    for cls_label, cls_obs in sufficient_classes.items():
        # Group by location
        by_loc: dict[str, list[Any]] = {}
        for o in cls_obs:
            loc = _location_key(o)
            by_loc.setdefault(loc, []).append(o)

        locs = sorted(by_loc.keys())
        n_locs = len(locs)

        # Assign locations to splits
        if n_locs >= 3:
            # Leave-one-group-out: last location = test, second-last = val
            test_locs = {locs[-1]}
            val_locs = {locs[-2]} if n_locs >= 4 else set()
            train_locs = set(locs[:-2]) if n_locs >= 4 else set(locs[:-1])
        elif n_locs == 2:
            train_locs = {locs[0]}
            val_locs = set()
            test_locs = {locs[1]}
        else:
            # Single location — random 70/15/15 split
            import random
            rng = random.Random(42)
            shuffled = list(cls_obs)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_train = max(1, int(n * 0.7))
            n_val = max(0, int(n * 0.15))
            train.extend(shuffled[:n_train])
            val.extend(shuffled[n_train:n_train + n_val])
            test.extend(shuffled[n_train + n_val:])
            continue

        for loc, loc_obs in by_loc.items():
            if loc in train_locs:
                train.extend(loc_obs)
            elif loc in val_locs:
                val.extend(loc_obs)
            elif loc in test_locs:
                test.extend(loc_obs)

    # Add rare-class samples to train only
    for cls_label, cls_obs in rare_classes.items():
        train.extend(cls_obs)

    split = {"train": train, "val": val, "test": test}
    feasibility = (
        f"PARTIAL: {len(sufficient_classes)} sufficient classes split by location; "
        f"{len(rare_classes)} rare classes kept in train only"
        if rare_classes
        else "COMPLETE: all classes have sufficient samples"
    )
    return split, feasibility, warnings


def _split_yield_dataset(
    observations: list[Any],
    *,
    test_ratio: float = 0.2,
    val_ratio: float = 0.15,
) -> dict[str, list[Any]]:
    """Temporal split for yield (kg/ha) observations."""
    import random

    by_year: dict[int, list[Any]] = {}
    for o in observations:
        by_year.setdefault(_year(o), []).append(o)

    years = sorted(by_year.keys())

    if len(years) < 3:
        # Fallback: random split
        rng = random.Random(42)
        shuffled = list(observations)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        return {
            "train": shuffled[:n - n_test - n_val],
            "val": shuffled[n - n_test - n_val:n - n_test],
            "test": shuffled[n - n_test:],
        }

    # Temporal: most recent = test, then val, rest = train
    test_years = {years[-1]}
    val_years = {years[-2]} if len(years) >= 4 else {years[-2]}
    train_years = set(years[:-2])

    return {
        "train": [o for y in train_years for o in by_year[y]],
        "val": [o for y in val_years for o in by_year[y]],
        "test": [o for y in test_years for o in by_year[y]],
    }


def _compute_yield_stats(observations: list[Any]) -> dict[str, float]:
    """Compute yield statistics for kg/ha observations."""
    import torch

    values = [
        float(getattr(o, "yield_value", 0))
        for o in observations
        if getattr(o, "yield_value", None) is not None
    ]
    if not values:
        return {}
    t = torch.tensor(values)
    return {
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()),
        "count": len(values),
    }


# --------------------------------------------------------------------------- #
# Leakage check
# --------------------------------------------------------------------------- #


def _check_leakage(
    train: list[Any], val: list[Any], test: list[Any]
) -> dict[str, Any]:
    """Check for data leakage between splits."""
    train_ids = {_obs_id(o) for o in train}
    val_ids = {_obs_id(o) for o in val}
    test_ids = {_obs_id(o) for o in test}

    train_locs = {_location_key(o) for o in train}
    val_locs = {_location_key(o) for o in val}
    test_locs = {_location_key(o) for o in test}

    train_years = {_year(o) for o in train}
    val_years = {_year(o) for o in val}
    test_years = {_year(o) for o in test}

    id_overlap_tv = train_ids & val_ids
    id_overlap_tt = train_ids & test_ids
    id_overlap_vt = val_ids & test_ids

    loc_overlap_tv = train_locs & val_locs
    loc_overlap_tt = train_locs & test_locs
    loc_overlap_vt = val_locs & test_locs

    year_overlap_tv = train_years & val_years
    year_overlap_tt = train_years & test_years
    year_overlap_vt = val_years & test_years

    issues: list[str] = []
    if id_overlap_tv or id_overlap_tt or id_overlap_vt:
        issues.append(
            f"ID overlap: train/val={len(id_overlap_tv)}, "
            f"train/test={len(id_overlap_tt)}, val/test={len(id_overlap_vt)}"
        )
    if loc_overlap_tv:
        issues.append(f"Location leakage train/val: {loc_overlap_tv}")
    if loc_overlap_tt:
        issues.append(f"Location leakage train/test: {loc_overlap_tt}")
    if loc_overlap_vt:
        issues.append(f"Location leakage val/test: {loc_overlap_vt}")

    return {
        "id_overlap": {
            "train_val": len(id_overlap_tv),
            "train_test": len(id_overlap_tt),
            "val_test": len(id_overlap_vt),
        },
        "location_overlap": {
            "train_val": sorted(loc_overlap_tv),
            "train_test": sorted(loc_overlap_tt),
            "val_test": sorted(loc_overlap_vt),
        },
        "year_overlap": {
            "train_val": sorted(year_overlap_tv),
            "train_test": sorted(year_overlap_tt),
            "val_test": sorted(year_overlap_vt),
        },
        "issues": issues,
        "passed": len(issues) == 0,
    }


# --------------------------------------------------------------------------- #
# Build contract
# --------------------------------------------------------------------------- #


def build_contract(
    observations: list[Any],
    *,
    crop_head_enabled: bool = True,
    yield_head_enabled: bool = True,
) -> SupervisedDataContract:
    """Build the R5.2.2 supervised data contract.

    Classifies every observation, builds per-task datasets, attempts splits,
    runs leakage checks, and produces the final contract.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not observations:
        errors.append("corpus is EMPTY")
        return SupervisedDataContract(
            crop_dataset=DatasetDescriptor(name="crop", sample_count=0),
            yield_dataset=DatasetDescriptor(name="yield_kg_ha", sample_count=0),
            auxiliary_dataset=DatasetDescriptor(name="auxiliary", sample_count=0),
            overall_validity=False,
            errors=errors,
        )

    # Classify every observation
    classifications = [_classify_observation(o) for o in observations]

    # --- Crop dataset --- #
    crop_obs = [
        o for o, c in zip(observations, classifications)
        if c["crop_eligible"]
    ]
    crop_labels = [c["crop_label"] for c in classifications if c["crop_eligible"]]
    crop_class_counts = dict(Counter(crop_labels))
    crop_total = len(crop_obs)
    crop_class_dist = {k: round(v / crop_total, 4) for k, v in crop_class_counts.items()} if crop_total > 0 else {}

    crop_years = sorted(set(_year(o) for o in crop_obs))
    crop_locations = sorted(set(_location_key(o) for o in crop_obs))
    crop_sources = sorted(set(_source_file(o) for o in crop_obs))

    crop_modalities = {
        "tabular": any(_has_tabular(o) for o in crop_obs),
        "image": any(_has_images(o) for o in crop_obs),
        "temporal": any(_has_temporal(o) for o in crop_obs),
    }

    crop_descriptor = DatasetDescriptor(
        name="crop",
        observations=crop_obs,
        sample_count=crop_total,
        class_counts=crop_class_counts,
        class_distribution=crop_class_dist,
        years=crop_years,
        locations=crop_locations,
        source_datasets=crop_sources,
        modalities=crop_modalities,
        yield_unit="kg/ha",
        notes=[],
    )

    # --- Yield dataset (kg/ha only) --- #
    yield_obs = [
        o for o, c in zip(observations, classifications)
        if c["yield_kg_ha_eligible"]
    ]
    yield_stats = _compute_yield_stats(yield_obs)
    yield_years = sorted(set(_year(o) for o in yield_obs))
    yield_locations = sorted(set(_location_key(o) for o in yield_obs))
    yield_sources = sorted(set(_source_file(o) for o in yield_obs))

    yield_descriptor = DatasetDescriptor(
        name="yield_kg_ha",
        observations=yield_obs,
        sample_count=len(yield_obs),
        years=yield_years,
        locations=yield_locations,
        source_datasets=yield_sources,
        modalities={
            "tabular": any(_has_tabular(o) for o in yield_obs),
            "image": any(_has_images(o) for o in yield_obs),
            "temporal": any(_has_temporal(o) for o in yield_obs),
        },
        yield_unit="kg/ha",
        yield_source=", ".join(yield_sources),
        yield_stats=yield_stats,
    )

    # --- Auxiliary dataset --- #
    aux_obs = [
        o for o, c in zip(observations, classifications)
        if c["auxiliary"]
    ]
    aux_years = sorted(set(_year(o) for o in aux_obs))
    aux_locations = sorted(set(_location_key(o) for o in aux_obs))
    aux_sources = sorted(set(_source_file(o) for o in aux_obs))

    aux_descriptor = DatasetDescriptor(
        name="auxiliary",
        observations=aux_obs,
        sample_count=len(aux_obs),
        years=aux_years,
        locations=aux_locations,
        source_datasets=aux_sources,
        modalities={
            "tabular": any(_has_tabular(o) for o in aux_obs),
            "image": any(_has_images(o) for o in aux_obs),
            "temporal": any(_has_temporal(o) for o in aux_obs),
        },
        notes=[
            "NPP-proxy observations. NOT used for supervised crop or yield loss.",
            "Available for: self-supervised pretraining, representation learning.",
            f"supervised_crop=false, supervised_yield_kg_ha=false, auxiliary=true",
        ],
    )

    # --- Crop split analysis --- #
    crop_split, crop_feasibility, crop_split_warnings = _split_crop_dataset(crop_obs)
    warnings.extend(crop_split_warnings)
    crop_descriptor.split_feasibility = crop_feasibility

    # --- Yield split --- #
    yield_split = _split_yield_dataset(yield_obs)

    # Validate yield splits
    for split_name, split_obs in yield_split.items():
        if split_obs:
            sy = [_compute_yield_stats(split_obs)]
            if sy and sy[0]:
                vals = set(round(sy[0]["mean"], 2))
                if len(split_obs) > 1 and sy[0]["std"] < 0.01:
                    warnings.append(
                        f"Yield split '{split_name}' has near-constant target "
                        f"(std={sy[0]['std']:.4f}) — regression metrics unreliable"
                    )

    # --- Leakage checks --- #
    # For crop split
    crop_leakage = _check_leakage(
        crop_split.get("train", []),
        crop_split.get("val", []),
        crop_split.get("test", []),
    )
    if not crop_leakage["passed"]:
        warnings.extend([f"crop leakage: {i}" for i in crop_leakage["issues"]])

    # For yield split
    yield_leakage = _check_leakage(
        yield_split.get("train", []),
        yield_split.get("val", []),
        yield_split.get("test", []),
    )
    if not yield_leakage["passed"]:
        warnings.extend([f"yield leakage: {i}" for i in yield_leakage["issues"]])

    # --- Temporal crop generalization --- #
    all_crop_years = sorted(set(_year(o) for o in crop_obs))
    if len(all_crop_years) <= 1:
        temporal_note = (
            "TEMPORAL CROP GENERALIZATION UNSUPPORTED: "
            f"crop labels span only {all_crop_years} — "
            "cannot evaluate temporal generalization"
        )
    else:
        val_test_years = set()
        for o in crop_split.get("val", []) + crop_split.get("test", []):
            val_test_years.add(_year(o))
        if not val_test_years:
            temporal_note = (
                "TEMPORAL CROP GENERALIZATION UNSUPPORTED: "
                "no crop-labeled observations in val/test — "
                "cannot evaluate temporal generalization"
            )
        else:
            temporal_note = f"Supported: crop labels available in years {all_crop_years}"

    # --- Errors --- #
    if crop_head_enabled and crop_total == 0:
        errors.append(
            "crop classifier enabled but NO observations have valid crop labels"
        )
    if yield_head_enabled and len(yield_obs) == 0:
        errors.append(
            "yield regressor enabled but NO observations have kg/ha yield"
        )
    if yield_head_enabled and len(aux_obs) > 0:
        n_npp = len(aux_obs)
        warnings.append(
            f"{n_npp} NPP-proxy observations excluded from yield loss — "
            "they use a different physical unit"
        )

    # Overlap analysis
    crop_ids = {_obs_id(o) for o in crop_obs}
    yield_ids = {_obs_id(o) for o in yield_obs}
    aux_ids = {_obs_id(o) for o in aux_obs}

    overlap_cy = crop_ids & yield_ids
    overlap_ca = crop_ids & aux_ids
    overlap_ya = yield_ids & aux_ids

    if overlap_cy:
        warnings.append(
            f"{len(overlap_cy)} observations in both crop and yield datasets "
            "(allowed: crop-labeled village observations have kg/ha yield)"
        )
    if overlap_ca:
        errors.append(
            f"{len(overlap_ca)} observations in both crop and auxiliary datasets "
            "— a crop-labeled observation cannot be auxiliary"
        )
    if overlap_ya:
        errors.append(
            f"{len(overlap_ya)} observations in both yield and auxiliary datasets "
            "— a kg/ha observation cannot be auxiliary (NPP proxy)"
        )

    # --- Overall validity --- #
    overall_valid = (
        len(errors) == 0
        and crop_total > 0
        and len(yield_obs) > 0
        and crop_feasibility != "CROP_DATA_INSUFFICIENT_FOR_CLASS_WISE_GENERALIZATION"
    )

    return SupervisedDataContract(
        crop_dataset=crop_descriptor,
        yield_dataset=yield_descriptor,
        auxiliary_dataset=aux_descriptor,
        crop_split=crop_split,
        yield_split=yield_split,
        leakage_report={
            "crop": crop_leakage,
            "yield": yield_leakage,
        },
        crop_split_feasibility=crop_feasibility,
        temporal_crop_generalization=temporal_note,
        overall_validity=overall_valid,
        errors=errors,
        warnings=warnings,
    )
