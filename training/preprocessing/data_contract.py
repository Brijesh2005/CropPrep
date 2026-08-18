"""Training-data contract: reject invalid mixed-target corpora (R5.2.1 Task D).

A training corpus is contract-valid only when it satisfies two hard rules:

1. **Single yield unit** — the regression target must not mix physical yields
   (kg/ha village records, e.g. ``data_season.csv``) with a normalized per-
   district proxy (``Yield_Proxy_NPP``, e.g. ``DK_Features_*.csv``). Mixing
   them trains one scaler on two incommensurable quantities and produces a
   meaningless ``R2``.

2. **Crop classifier needs real labels** — the final crop classifier must not
   be trained on unlabeled ``-1`` sentinel observations. A run that enables the
   crop head while the corpus has no (or too few) crop labels is rejected.

The module is pure: :func:`assess_training_data_contract` computes a
:class:`TrainingDataContract` report (never raises) and
:func:`validate_training_data_contract` enforces it (raises
:class:`~training.preprocessing.exceptions.DataContractViolationError`).

The report carries the fields the final training run must state:
``crop_training_samples``, ``yield_training_samples``, ``yield_unit``,
``yield_source``, ``image_samples``, ``tabular_samples``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .exceptions import DataContractViolationError

#: Yield-unit tokens recognised in a tabular source path.
_KG_HA_SOURCES = ("data_season", "icrisat", "kg_ha", "yield_kg", "yeilds")
_NPP_SOURCES = ("dk_features", "yield_proxy_npp", "npp", "proxy")


@dataclass(frozen=True)
class TrainingDataContract:
    """One training corpus's data-contract report (R5.2.1 Task D).

    ``errors`` are hard violations (the corpus must be rejected);
    ``warnings`` are degradations that do not block training but are reported
    so the final run states its limitations.
    """

    crop_training_samples: int = 0
    yield_training_samples: int = 0
    yield_unit: str | None = None
    yield_source: str | None = None
    image_samples: int = 0
    tabular_samples: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop_training_samples": self.crop_training_samples,
            "yield_training_samples": self.yield_training_samples,
            "yield_unit": self.yield_unit,
            "yield_source": self.yield_source,
            "image_samples": self.image_samples,
            "tabular_samples": self.tabular_samples,
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def infer_yield_unit(source_path: str | None, matched_level: str | None,
                     yield_value: float | None) -> str | None:
    """Infer a single yield unit for one observation (or ``None`` = unknown).

    Resolution order: source-path token > match level > magnitude heuristic.
    """
    if source_path:
        name = Path(source_path).name.lower()
        if any(tok in name for tok in _KG_HA_SOURCES):
            return "kg/ha"
        if any(tok in name for tok in _NPP_SOURCES):
            return "npp_proxy"

    if matched_level == "village":
        return "kg/ha"
    if matched_level == "district":
        return "npp_proxy"

    # Magnitude heuristic: physical kg/ha yields are typically > 100 while the
    # normalized NPP proxy lives in [0, ~1.5]. Only used as a last resort.
    if yield_value is not None and yield_value > 0:
        if yield_value > 100:
            return "kg/ha"
        if yield_value < 10:
            return "npp_proxy"
    return None


def assess_training_data_contract(
    observations: Iterable[Any],
    *,
    crop_head_enabled: bool = True,
) -> TrainingDataContract:
    """Compute the data-contract report for a training corpus (never raises).

    Args:
        observations: Accepted training observations.
        crop_head_enabled: Whether the run intends to train a crop classifier.
    """
    obs = list(observations)
    n = len(obs)
    if n == 0:
        return TrainingDataContract(
            crop_training_samples=0,
            yield_training_samples=0,
            yield_unit=None,
            yield_source=None,
            image_samples=0,
            tabular_samples=0,
            errors=["training corpus is EMPTY"],
        )

    crop_labeled = [o for o in obs if getattr(o, "crop", None) is not None]
    yield_bearing = [o for o in obs if getattr(o, "yield_value", None) is not None]
    image_samples = sum(1 for o in obs if getattr(o, "has_paired_images", False))
    tabular_samples = sum(
        1 for o in obs
        if getattr(getattr(o, "tabular", None), "matched_level", "none") != "none"
    )

    units: set[str] = set()
    sources: set[str] = set()
    for o in yield_bearing:
        unit = infer_yield_unit(
            getattr(getattr(o, "tabular", None), "source_path", None),
            getattr(getattr(o, "tabular", None), "matched_level", None),
            float(o.yield_value),
        )
        if unit is not None:
            units.add(unit)
        src = getattr(getattr(o, "tabular", None), "source_path", None)
        if src:
            sources.add(str(Path(src).name))

    errors: list[str] = []
    warnings: list[str] = []

    # -- Rule 1: single yield unit ---------------------------------------- #
    known = {u for u in units if u is not None}
    if len(known) > 1:
        errors.append(
            "yield target mixes incompatible units: "
            + ", ".join(sorted(known))
            + f" across {len(yield_bearing)} yield-labelled samples — "
            "train the regression head on ONE unit only"
        )
    elif len(known) == 1 and yield_bearing and len(yield_bearing) != len(obs):
        unit = next(iter(known))
        warnings.append(
            f"{len(yield_bearing)}/{n} samples carry a yield label (unit={unit}); "
            "the rest train no yield head"
        )

    # -- Rule 2: crop classifier needs real labels ------------------------ #
    if crop_head_enabled and crop_labeled:
        if len(crop_labeled) < max(2, len(yield_bearing) // 10):
            warnings.append(
                f"only {len(crop_labeled)}/{n} training samples carry a crop label; "
                "the crop classifier would train on a tiny labelled subset"
            )
        if len(crop_labeled) < n:
            unlabeled = n - len(crop_labeled)
            warnings.append(
                f"{unlabeled}/{n} training samples are UNLABELLED for crop "
                "(-1 sentinel); the crop head is trained only on the "
                f"{len(crop_labeled)} labelled ones"
            )
    elif crop_head_enabled and not crop_labeled and yield_bearing:
        errors.append(
            "crop classifier enabled but NO training sample carries a crop "
            "label — do not train the final crop classifier on unlabeled -1 "
            "observations"
        )

    unit_out = next(iter(known), None)
    source_out = ", ".join(sorted(sources)) or None
    return TrainingDataContract(
        crop_training_samples=len(crop_labeled),
        yield_training_samples=len(yield_bearing),
        yield_unit=unit_out,
        yield_source=source_out,
        image_samples=image_samples,
        tabular_samples=tabular_samples,
        errors=errors,
        warnings=warnings,
    )


def validate_training_data_contract(
    observations: Iterable[Any],
    *,
    crop_head_enabled: bool = True,
    strict: bool = True,
) -> TrainingDataContract:
    """Validate a training corpus against the data contract.

    Raises:
        DataContractViolationError: When ``strict`` and the corpus has hard
            violations (mixed yield units / crop head without labels).
    """
    report = assess_training_data_contract(
        observations, crop_head_enabled=crop_head_enabled
    )
    if strict and report.errors:
        raise DataContractViolationError(
            "training-data contract violated: " + "; ".join(report.errors),
            detail=report.to_dict(),
        )
    return report
