"""Canonical crop taxonomy and OGD label normalization.

Government crop survey data (data.gov.in) uses free-text crop names that must
be mapped through a deterministic pipeline before they can serve as training
labels:

    source crop name
    -> normalised lowercase
    -> canonical CropType member
    -> stable integer class ID  (assigned by LabelEncoder at fit time)

This module provides the authoritative mapping table.  Every source name that
enters the CropFusion training pipeline must pass through
:mfunc:`resolve_crop_label`.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from . import CropType


class LabelMatchStatus(str, Enum):
    """Classification of how a source crop name maps to the taxonomy."""

    EXACT = "exact"
    ALIAS = "alias"
    NORMALIZED = "normalized"
    UNMAPPED = "unmapped"
    AMBIGUOUS = "ambiguous"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True, slots=True)
class CropLabelResolution:
    """Result of resolving a single source crop name."""

    source_crop: str
    normalized_crop: str
    crop_type: CropType
    class_id: int | None
    status: LabelMatchStatus
    reason: str


# ---------------------------------------------------------------------------
# Canonical alias table
# ---------------------------------------------------------------------------
# Keys are normalised (lowercase, stripped) source names.
# Values are CropType members that the name unambiguously maps to.
#
# Only add entries here when the mapping is unambiguous and domain-certain.
# When in doubt, leave the crop as UNMAPPED or AMBIGUOUS.
# ---------------------------------------------------------------------------

_CROP_ALIASES: dict[str, CropType] = {
    # --- COCONUT ---
    "coconut": CropType.COCONUT,
    # --- PEPPER ---
    "pepper (black)": CropType.PEPPER,
    "black pepper": CropType.PEPPER,
    "pepper": CropType.PEPPER,
    # --- COFFEE ---
    "coffee robusta": CropType.COFFEE,
    "coffee arabica": CropType.COFFEE,
    "coffee": CropType.COFFEE,
    # --- CARDAMOM ---
    "cardamom": CropType.CARDAMOM,
    "small cardamom": CropType.CARDAMOM,
    "large cardamom": CropType.CARDAMOM,
    # --- BLACKGRAM ---
    "blackgram": CropType.BLACKGRAM,
    "black gram": CropType.BLACKGRAM,
    "urad": CropType.BLACKGRAM,
    "urad dal": CropType.BLACKGRAM,
    # --- ARECANUT ---
    "betel nuts (areca nuts)": CropType.ARECANUT,
    "betel nut": CropType.ARECANUT,
    "areca nut": CropType.ARECANUT,
    "arecanut": CropType.ARECANUT,
    # --- PADDY ---
    "paddy": CropType.PADDY,
    "paddy-h": CropType.PADDY,
    "paddy-l": CropType.PADDY,
    "rice": CropType.RICE,
    # --- RAGI ---
    "ragi": CropType.RAGI,
    "ragi-h": CropType.RAGI,
    "ragi-l": CropType.RAGI,
    "ragi -h": CropType.RAGI,
    "finger millet": CropType.RAGI,
    # --- MAIZE ---
    "maize": CropType.MAIZE,
    "corn": CropType.MAIZE,
    "jowar-h": CropType.MAIZE,
    "jowar-l": CropType.MAIZE,
    "sorghum": CropType.MAIZE,
}

# ---------------------------------------------------------------------------
# Out-of-scope patterns (non-crop records that appear in survey data)
# ---------------------------------------------------------------------------

_OUT_OF_SCOPE_PATTERNS: frozenset[str] = frozenset({
    "fallow",
    "na land",
    "harvest over crop",
    "trees and grooves",
    "grass",
    "green fodder",
    "nursery",
})

# ---------------------------------------------------------------------------
# Ambiguous crops (could map to multiple CropType members)
# ---------------------------------------------------------------------------

_AMBIGUOUS_CROPS: frozenset[str] = frozenset({
    "tapioca",
    "sugarcane-p",
})


def _normalise_crop_name(name: str) -> str:
    """Lowercase, strip whitespace, collapse internal spaces."""
    return " ".join(name.lower().split())


def _is_out_of_scope(normalized: str) -> bool:
    """Check if a crop name represents a non-crop record."""
    return any(pat in normalized for pat in _OUT_OF_SCOPE_PATTERNS)


def resolve_crop_label(
    source_name: str,
    *,
    class_id: int | None = None,
) -> CropLabelResolution:
    """Resolve a source crop name to a canonical CropType.

    Parameters
    ----------
    source_name:
        Raw crop name from government survey data.
    class_id:
        Optional pre-assigned class ID (from LabelEncoder).  When None, the
        caller is responsible for assigning the ID after LabelEncoder fitting.

    Returns
    -------
    CropLabelResolution
        Frozen dataclass with the resolution result and status classification.
    """
    normalized = _normalise_crop_name(source_name)

    # Out-of-scope check
    if _is_out_of_scope(normalized):
        return CropLabelResolution(
            source_crop=source_name,
            normalized_crop=normalized,
            crop_type=CropType.UNKNOWN,
            class_id=class_id,
            status=LabelMatchStatus.OUT_OF_SCOPE,
            reason=f"Non-crop record (matches out-of-scope pattern)",
        )

    # Ambiguous check
    if normalized in _AMBIGUOUS_CROPS:
        return CropLabelResolution(
            source_crop=source_name,
            normalized_crop=normalized,
            crop_type=CropType.UNKNOWN,
            class_id=class_id,
            status=LabelMatchStatus.AMBIGUOUS,
            reason=f"Ambiguous: maps to multiple potential CropType members",
        )

    # Direct alias lookup
    if normalized in _CROP_ALIASES:
        crop_type = _CROP_ALIASES[normalized]
        # Determine match quality
        if normalized == crop_type.value:
            status = LabelMatchStatus.EXACT
            reason = f"Exact match to CropType.{crop_type.name}"
        else:
            status = LabelMatchStatus.ALIAS
            reason = f"Alias '{source_name}' -> CropType.{crop_type.name}"
        return CropLabelResolution(
            source_crop=source_name,
            normalized_crop=normalized,
            crop_type=crop_type,
            class_id=class_id,
            status=status,
            reason=reason,
        )

    # No match found
    return CropLabelResolution(
        source_crop=source_name,
        normalized_crop=normalized,
        crop_type=CropType.UNKNOWN,
        class_id=class_id,
        status=LabelMatchStatus.UNMAPPED,
        reason=f"No alias entry for '{normalized}' in crop taxonomy",
    )


def resolve_all_ogd_labels(
    ogd_crops: dict[str, int],
) -> list[CropLabelResolution]:
    """Resolve all unique OGD crop names at once.

    Parameters
    ----------
    ogd_crops:
        Mapping of raw crop name -> record count (for context).

    Returns
    -------
    list[CropLabelResolution]
        One resolution per unique source name, sorted by status then count.
    """
    results = []
    for source_name, count in sorted(ogd_crops.items(), key=lambda x: -x[1]):
        results.append(resolve_crop_label(source_name))
    return results


def write_label_mapping_csv(
    resolutions: list[CropLabelResolution],
    output_path: str,
    ogd_counts: dict[str, int] | None = None,
) -> str:
    """Write the government_crop_label_mapping.csv file.

    Parameters
    ----------
    resolutions:
        Output of :func:`resolve_all_ogd_labels`.
    output_path:
        Full path for the CSV file.
    ogd_counts:
        Optional record counts per source crop name.

    Returns
    -------
    str
        Path to the written CSV.
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_crop",
            "normalized_crop",
            "crop_type",
            "class_id",
            "status",
            "reason",
            "record_count",
        ])
        for r in resolutions:
            count = ogd_counts.get(r.source_crop, 0) if ogd_counts else 0
            writer.writerow([
                r.source_crop,
                r.normalized_crop,
                r.crop_type.value,
                r.class_id if r.class_id is not None else "",
                r.status.value,
                r.reason,
                count,
            ])
    return output_path
