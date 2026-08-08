"""Name normalisation + special-case handling for the STAM tabular join.

The tabular record table (``data_season.csv``) stores colloquial district /
taluk names (``Mangalore``, ``Bangalore``, ...) while the KGIS boundary
shapefiles carry the official spellings (``Dakshina Kannada``,
``Bengaluru``, ...). STAM normalises **both sides** of the join to one
canonical name so a boundary name can be matched against the CSV
``Location`` column — see :func:`normalize_name`.

``SPECIAL_CASES`` covers locations that cannot be resolved from the
Karnataka boundary files at all (e.g. Kasaragodu in neighbouring Kerala).
These are injected into the location catalog as manual points and are never
sent through shapefile lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Colloquial / historical name -> official KGIS boundary spelling.
#: Keys are matched case-insensitively; values are returned verbatim (the
#: canonical names actually present in ``application/gis``, authoritative
#: for the join). Both spellings of Gulbarga map to the shapefile's
#: ``Kalaburgi``.
ALIASES: dict[str, str] = {
    "Mangalore": "Dakshina Kannada",
    "Bangalore": "Bengaluru",
    "Chikmangaluru": "Chikkamagaluru",
    "Davangere": "Davanagere",
    "Gulbarga": "Kalaburgi",
    "Kalaburagi": "Kalaburgi",
    "Madikeri": "Madikeri",  # taluk-level match required
}

#: CSV location names that already equal the official boundary spelling.
NO_ALIAS_NEEDED: set[str] = {
    "Hassan",
    "Kodagu",
    "Mysuru",
    "Raichur",
}

#: Locations outside the Karnataka boundary files, injected manually.
#: ``lat``/``lon`` are approximate (WGS-84).
SPECIAL_CASES: dict[str, dict[str, Any]] = {
    "Kasaragodu": {
        "type": "manual_point",
        "status": "manual",
        "lat": 12.49,
        "lon": 74.99,
    },
}


@dataclass(frozen=True)
class LocationResolution:
    """Outcome of resolving a raw location name.

    Attributes:
        name: The original (stripped) input name.
        normalized: The canonical name used for the tabular join.
        status: ``"matched"`` | ``"manual"`` | ``"unmatched"``. A special
            case resolves to ``"manual"``; an unresolvable/empty name to
            ``"unmatched"``.
        lon / lat: Coordinates for manual points (else None).
    """

    name: str
    normalized: str
    status: str
    lon: float | None = None
    lat: float | None = None


def _strip_parenthetical(value: str) -> str:
    """Drop ``(…)`` suffixes (``"Bengaluru (Urban)"`` -> ``"Bengaluru"``)."""
    if "(" not in value:
        return value
    return value.split("(", 1)[0].strip()


def _clean(value: str) -> str:
    """Strip + collapse whitespace (case-preserving)."""
    return " ".join(str(value).strip().split())


def _key(value: str) -> str:
    """Case-insensitive lookup key (cleaned + lowercased)."""
    return _strip_parenthetical(_clean(value)).lower()


#: Case-insensitive alias lookup: canonical key -> official spelling.
_ALIAS_MAP: dict[str, str] = {
    _key(name): _strip_parenthetical(value) for name, value in ALIASES.items()
}


def normalize_name(name: str | None) -> str:
    """Map ``name`` to its canonical spelling for the tabular join.

    Applies the alias table, strips surrounding whitespace, collapses inner
    whitespace and removes parenthetical qualifiers so both sides of the join
    compare equal. Non-aliased names are returned cleaned but otherwise
    unchanged (e.g. ``"Dakshina Kannada"`` stays ``"Dakshina Kannada"``).
    """
    if name is None:
        return ""
    cleaned = _strip_parenthetical(_clean(str(name)))
    if not cleaned:
        return ""
    return _ALIAS_MAP.get(_key(cleaned), cleaned)


def resolve_location(raw_name: str | None) -> LocationResolution:
    """Resolve a raw location name, routing special cases.

    Returns:
        A :class:`LocationResolution`. Names in :data:`SPECIAL_CASES`
        resolve to ``status="manual"`` with coordinates; other names resolve
        to ``status="matched"`` and an empty name to ``status="unmatched"``.
    """
    if raw_name is None:
        return LocationResolution(name="", normalized="", status="unmatched")
    cleaned = _clean(str(raw_name))
    if not cleaned:
        return LocationResolution(name="", normalized="", status="unmatched")
    special = SPECIAL_CASES.get(cleaned.lower()) or SPECIAL_CASES.get(cleaned)
    if special is not None:
        return LocationResolution(
            name=cleaned,
            normalized=normalize_name(cleaned),
            status=special.get("status", "manual"),
            lon=special.get("lon"),
            lat=special.get("lat"),
        )
    return LocationResolution(
        name=cleaned,
        normalized=normalize_name(cleaned),
        status="matched",
    )
