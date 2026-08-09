"""Tabular-source profiler for the STAM auto-wiring pipeline.

Given a CSV path, infer which columns hold the location / year / season /
crop / yield / feature fields with a per-field confidence score. The gated
pipeline (:mod:`training.stam.tabular_gates`) consumes a
:class:`SourceProfile` to decide whether a source may be auto-committed into
the live ``training/config/stam.yaml`` ``tabular.tables`` list.

The profiler is *inference only*: it never writes anything. Every score is
computed against the same vocabularies the matching code uses — the
``training/stam/name_aliases.py`` alias table, the KGIS district/taluk
boundary names (``application/gis/District/District.shp`` /
``application/gis/Taluk/Taluk.shp``) and the ``training/config/seasons.yaml``
Kharif/Rabi/Zaid calendar — so a high-confidence inference is one that will
actually resolve at match time.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .name_aliases import ALIASES, NO_ALIAS_NEEDED, SPECIAL_CASES

#: Season vocabulary from ``training/config/seasons.yaml`` (Kharif / Rabi /
#: Zaid). Sources whose season column uses another vocabulary (``Summer``,
#: numeric codes, ...) still count the overlapping values and lose confidence
#: rather than being forced onto the calendar.
SEASON_VOCAB: tuple[str, ...] = ("kharif", "rabi", "zaid")

#: Plausible year range for an Indian agricultural record table.
YEAR_MIN, YEAR_MAX = 1900, 2100

#: Gate A location threshold: a location column is high-confidence when at
#: least this fraction of its unique place names hit the boundary/alias
#: vocabulary with an exact match or a >= :data:`LOCATION_FUZZY_MIN` fuzzy
#: score.
LOCATION_CONFIDENCE_MIN = 0.9
LOCATION_FUZZY_MIN = 0.9

#: Gate B alias thresholds. A place name is alias-eligible only when its best
#: fuzzy boundary match is >= :data:`ALIAS_MIN_SIMILARITY` and the second-best
#: candidate is at least :data:`ALIAS_AMBIGUITY_DELTA` further away (so the
#: Bengaluru Urban/Rural/South style ambiguity fails the gate instead of
#: guessing).
ALIAS_MIN_SIMILARITY = 0.92
ALIAS_AMBIGUITY_DELTA = 0.05

#: Name-based yield column patterns (Gate A requires a name-based match, not
#: just "the most plausible remaining numeric column").
YIELD_NAME_RE = re.compile(
    r"(yield|yeild|productiv(?:ity)?|tonne?s?\s*/\s*ha|kg\s*per\s*ha)",
    re.IGNORECASE,
)

#: Wide-format ICRISAT-style "<CROP> AREA/YIELD/PRODUCTION (...)" triples.
_AREA_RE = re.compile(r"^(.*?)\s+AREA\s*\(", re.IGNORECASE)
_YIELD_TRIPLE_RE = re.compile(r"^(.*?)\s+YIELD\s*\(", re.IGNORECASE)

#: A modest crop-name vocabulary used to score the crop column inference.
CROP_VOCAB: tuple[str, ...] = (
    "rice", "wheat", "maize", "jowar", "sorghum", "bajra", "pearl millet",
    "ragi", "finger millet", "barley", "millets", "coconut", "arecanut",
    "paddy", "tur", "pigeonpea", "gram", "chickpea", "urad", "moong",
    "blackgram", "greengram", "lentil", "pulses", "groundnut", "peanut",
    "sesamum", "ginger", "pepper", "cardamum", "cardamom", "coffee", "tea",
    "cashew", "cocoa", "cotton", "sugarcane", "sunflower", "soybean",
    "soyabean", "potato", "onion", "tomato", "vegetables", "fruits",
    "rapeseed", "mustard", "castor", "safflower", "linseed", "oilseeds",
    "fodder", "small millets",
)

#: Indian state / union-territory vocabulary used to spot a state filter
#: column (e.g. ICRISAT's ``State Name``) so location inference can restrict
#: itself to the rows of the most relevant state.
STATE_VOCAB: tuple[str, ...] = (
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand",
    "karnataka", "kerala", "madhya pradesh", "maharashtra", "manipur",
    "meghalaya", "mizoram", "nagaland", "odisha", "orissa", "punjab",
    "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura",
    "uttar pradesh", "uttarakhand", "west bengal",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DISTRICT_SHP = _REPO_ROOT / "application" / "gis" / "District" / "District.shp"
_DEFAULT_TALUK_SHP = _REPO_ROOT / "application" / "gis" / "Taluk" / "Taluk.shp"


def clean_name(value: object) -> str:
    """Strip and collapse whitespace (case-preserving)."""
    return " ".join(str(value).strip().split())


def key_name(value: object) -> str:
    """Case-insensitive lookup key: cleaned + parenthetical stripped + lower."""
    cleaned = clean_name(value)
    if "(" in cleaned:
        cleaned = cleaned.split("(", 1)[0].strip()
    return cleaned.lower()


def _alternates(value: object) -> list[str]:
    """Split slash/comma alternates (``"Gulbarga / Kalaburagi"``)."""
    return [
        alt.strip()
        for alt in re.split(r"[/,]", clean_name(value))
        if alt.strip()
    ]


@dataclass(frozen=True)
class NameMatch:
    """The best vocabulary hit for one place name."""

    name: str
    score: float
    kind: str  # "district" | "taluk" | "alias_key" | "alias_value" | "none"


@dataclass(frozen=True)
class PlaceNameVerdict:
    """Gate-B classification for a single unique place name."""

    name: str
    #: ``exact`` | ``alias_existing`` | ``alias`` | ``ambiguous`` | ``unmapped``.
    status: str
    score: float
    matches: tuple[NameMatch, ...] = ()
    note: str = ""


class BoundaryVocabulary:
    """The name vocabulary the profiler scores against.

    Combines the KGIS district/taluk boundary spellings with the
    ``name_aliases`` table (both directions) and the special-case manual
    points, mirroring exactly what :mod:`training.stam.matcher` can resolve at
    match time.
    """

    def __init__(
        self,
        districts: Iterable[str],
        taluks: Iterable[str],
        alias_keys: Iterable[str] = (),
        alias_values: Iterable[str] = (),
        special_keys: Iterable[str] = (),
    ) -> None:
        self.districts = tuple(clean_name(d) for d in districts if clean_name(d))
        self.taluks = tuple(clean_name(t) for t in taluks if clean_name(t))
        self.alias_keys = {key_name(k) for k in alias_keys}
        self.alias_values = {key_name(v) for v in alias_values}
        self.special_keys = {key_name(k) for k in special_keys}
        self.boundaries = tuple(
            sorted({key_name(b) for b in self.districts} | {key_name(b) for b in self.taluks})
        )
        raw_by_key: dict[str, list[str]] = {}
        for b in self.districts + self.taluks:
            raw_by_key.setdefault(key_name(b), []).append(b)
        self._raw_by_key = raw_by_key

    # -- Public API ---------------------------------------------------------- #

    def exact_boundary_hits(self, value: object) -> list[str]:
        """All boundary names whose key equals ``value``'s key.

        More than one hit (e.g. ``Bengaluru`` matching ``Bengaluru (Urban)`` /
        ``Bengaluru (Rural)`` / ``Bengaluru South``) is the ambiguity the gate
        must reject, so this returns the full list rather than collapsing it.
        """
        key = key_name(value)
        return sorted(self._raw_by_key.get(key, []))

    def best_match(self, value: object) -> NameMatch:
        """Highest-confidence vocabulary hit for ``value`` (score 0..1)."""
        cleaned = clean_name(value)
        if not cleaned:
            return NameMatch("", 0.0, "none")
        key = key_name(value)
        if key in self.alias_keys:
            return NameMatch(cleaned, 1.0, "alias_key")
        if key in self.special_keys:
            return NameMatch(cleaned, 1.0, "alias_key")
        if key in self.alias_values:
            return NameMatch(cleaned, 1.0, "alias_value")
        # Slash / comma alternates (ICRISAT renamed-district spellings).
        for alt in _alternates(value):
            alt_key = key_name(alt)
            if alt_key in self.alias_keys or alt_key in self.special_keys:
                return NameMatch(alt, 1.0, "alias_key")
            if alt_key in self.alias_values:
                return NameMatch(alt, 1.0, "alias_value")
        hits = self.exact_boundary_hits(value)
        if hits:
            return NameMatch(hits[0], 1.0, self._kind(hits[0]))
        best: tuple[float, str] | None = None
        for boundary_key in self.boundaries:
            raw = self._raw_by_key.get(boundary_key, [boundary_key])[0]
            ratio = difflib.SequenceMatcher(None, key, boundary_key).ratio()
            if best is None or ratio > best[0]:
                best = (ratio, raw)
        if best is None:
            return NameMatch(cleaned, 0.0, "none")
        return NameMatch(best[1], round(best[0], 4), self._kind(best[1]))

    def classify(self, value: object) -> PlaceNameVerdict:
        """Gate-B classification of a single place name."""
        cleaned = clean_name(value)
        key = key_name(value)
        if not cleaned:
            return PlaceNameVerdict(cleaned, "unmapped", 0.0, note="empty name")
        if key in self.alias_keys or key in self.special_keys:
            return PlaceNameVerdict(cleaned, "alias_existing", 1.0,
                                    note="already in the alias table")
        hits = self.exact_boundary_hits(value)
        if len(hits) == 1:
            return PlaceNameVerdict(cleaned, "exact", 1.0,
                                    matches=(NameMatch(hits[0], 1.0, self._kind(hits[0])),))
        if len(hits) > 1:
            return PlaceNameVerdict(
                cleaned, "ambiguous", 1.0,
                matches=tuple(NameMatch(h, 1.0, self._kind(h)) for h in hits),
                note="exact key matches multiple boundaries "
                     f"({', '.join(sorted(hits))})",
            )
        # Fuzzy against boundary keys only (the alias keys are already exact).
        scored = sorted(
            ((difflib.SequenceMatcher(None, key, bk).ratio(), bk) for bk in self.boundaries),
            reverse=True,
        )
        if not scored:
            return PlaceNameVerdict(cleaned, "unmapped", 0.0)
        top = scored[0]
        if top[0] < ALIAS_MIN_SIMILARITY:
            return PlaceNameVerdict(cleaned, "unmapped", round(top[0], 4),
                                    matches=(NameMatch(self._raw_by_key.get(top[1], [top[1]])[0],
                                                       round(top[0], 4), self._kind(top[1])),))
        second = scored[1] if len(scored) > 1 else (0.0, "")
        if top[0] - second[0] < ALIAS_AMBIGUITY_DELTA:
            return PlaceNameVerdict(
                cleaned, "ambiguous", round(top[0], 4),
                matches=tuple(
                    NameMatch(self._raw_by_key.get(bk, [bk])[0], round(s, 4), self._kind(bk))
                    for s, bk in scored[:2]
                ),
                note="fuzzy match is not unambiguous "
                     f"(top-2 within {ALIAS_AMBIGUITY_DELTA})",
            )
        return PlaceNameVerdict(
            cleaned, "alias", round(top[0], 4),
            matches=(NameMatch(self._raw_by_key.get(top[1], [top[1]])[0],
                               round(top[0], 4), self._kind(top[1])),),
        )

    def _kind(self, boundary_name: str) -> str:
        key = key_name(boundary_name)
        if key in {key_name(t) for t in self.taluks}:
            return "taluk"
        return "district"


@dataclass(frozen=True)
class FieldInference:
    """One inferred column mapping with its confidence score."""

    column: str | None
    confidence: float
    method: str
    note: str = ""
    candidates: tuple[tuple[str, float], ...] = ()


@dataclass
class SourceProfile:
    """The full inferred field mapping for one CSV source."""

    source: Path
    columns: list[str]
    location: FieldInference
    taluk: FieldInference
    district: FieldInference
    year: FieldInference
    season: FieldInference
    crop: FieldInference
    yield_: FieldInference
    state: FieldInference
    wide_format: bool
    feature_columns: list[str]
    place_verdicts: list[PlaceNameVerdict]
    year_min: int | None = None
    year_max: int | None = None
    state_value: str | None = None
    table_config: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #


def load_boundary_names(
    district_shp: str | Path = _DEFAULT_DISTRICT_SHP,
    taluk_shp: str | Path = _DEFAULT_TALUK_SHP,
) -> tuple[list[str], list[str]]:
    """Read KGIS district + taluk names from the configured shapefiles."""
    import geopandas as gpd

    districts: list[str] = []
    taluks: list[str] = []
    for path, column in ((district_shp, "KGISDist_1"), (taluk_shp, "KGISTalukN")):
        if not Path(path).exists():
            raise FileNotFoundError(f"Boundary shapefile not found: {path}")
        frame = gpd.read_file(str(path))
        if column not in frame.columns:
            raise KeyError(f"Boundary file {path} has no column {column!r}")
        names = frame[column].dropna().astype(str).str.strip()
        names = names[names != ""]
        if "KGISTalukN" in frame.columns:
            taluks = sorted(set(names.tolist()))
        else:
            districts = sorted(set(names.tolist()))
    return districts, taluks


def build_default_vocabulary() -> BoundaryVocabulary:
    """The boundary vocabulary for the real repository (KGIS + aliases)."""
    districts, taluks = load_boundary_names()
    return BoundaryVocabulary(
        districts=districts,
        taluks=taluks,
        alias_keys=list(ALIASES) + list(SPECIAL_CASES) + list(NO_ALIAS_NEEDED),
        alias_values=list(ALIASES.values()),
    )


def read_source(path: str | Path, *, max_rows: int = 200_000) -> pd.DataFrame:
    """Read a CSV into a DataFrame (capped row count for giant tables)."""
    frame = pd.read_csv(path)
    if max_rows and len(frame) > max_rows:
        frame = frame.head(max_rows)
    return frame


def _textish(series: pd.Series) -> bool:
    return series.dtype == object or str(series.dtype).startswith(("str", "category"))


def _unique_values(series: pd.Series, *, cap: int = 2000) -> list[str]:
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if len(values) == 0:
        return []
    seen: list[str] = []
    seen_set: set[str] = set()
    for value in values:
        key = value.lower()
        if key not in seen_set:
            seen_set.add(key)
            seen.append(value)
            if len(seen) >= cap:
                break
    return seen


def _boundary_coverage(series: pd.Series, vocab: BoundaryVocabulary) -> tuple[float, dict[str, float]]:
    """Fraction of unique place names scoring >=0.9 against the vocabulary."""
    per_value: dict[str, float] = {}
    values = _unique_values(series)
    if not values:
        return 0.0, per_value
    for value in values:
        per_value[value] = vocab.best_match(value).score
    covered = sum(1 for s in per_value.values() if s >= LOCATION_CONFIDENCE_MIN)
    return round(covered / len(per_value), 4), per_value


def infer_state(frame: pd.DataFrame, vocab: BoundaryVocabulary, exclude: set[str]) -> FieldInference:
    """Spot a state-filter column (e.g. ICRISAT's ``State Name``)."""
    best: tuple[float, str] | None = None
    for column in frame.columns:
        if column in exclude:
            continue
        series = frame[column]
        if not _textish(series):
            continue
        values = _unique_values(series, cap=300)
        if not values:
            continue
        covered = sum(
            1 for v in values if key_name(v) in STATE_VOCAB
            or any(key_name(a) in STATE_VOCAB for a in _alternates(v))
        )
        if covered == 0:
            continue
        fraction = covered / len(values)
        if best is None or fraction > best[0]:
            best = (fraction, column)
    if best is None:
        return FieldInference(None, 0.0, "no-state-column")
    return FieldInference(
        best[1], round(best[0], 4), "state-name-overlap",
        note=f"{best[0]:.0%} of unique values are Indian state names",
    )


def infer_year(frame: pd.DataFrame) -> FieldInference:
    """The unique strong numeric-range candidate (or an ambiguity flag)."""
    candidates: list[tuple[str, float, tuple[int, int]]] = []
    for column in frame.columns:
        non_null = frame[column].dropna()
        if len(non_null) == 0:
            continue
        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        fraction = len(numeric) / len(non_null)
        if fraction < 0.9:
            continue
        ints = numeric.round()
        in_range = ints[(ints >= YEAR_MIN) & (ints <= YEAR_MAX)]
        if len(in_range) < max(1, int(0.9 * len(numeric))):
            continue
        distinct = sorted({int(v) for v in in_range.unique()})
        if not distinct:
            continue
        candidates.append((str(column), round(fraction, 4), (min(distinct), max(distinct))))

    if len(candidates) == 1:
        column, fraction, (lo, hi) = candidates[0]
        return FieldInference(
            column, fraction, "numeric-range",
            note=f"years {lo}-{hi}",
            candidates=((column, fraction),),
        )
    if not candidates:
        return FieldInference(None, 0.0, "no-year-column",
                              note="no column whose values are all plausible years (1900-2100)")
    ambiguous = tuple((c, s) for c, s, _ in candidates)
    return FieldInference(
        None, 0.0, "ambiguous-year",
        note="multiple columns look like years: "
             + ", ".join(f"{c} ({lo}-{hi})" for c, _, (lo, hi) in candidates),
        candidates=ambiguous,
    )


def infer_season(frame: pd.DataFrame, exclude: set[str]) -> FieldInference:
    """Column whose values overlap the Kharif/Rabi/Zaid vocabulary."""
    best: tuple[float, str] | None = None
    for column in frame.columns:
        if column in exclude:
            continue
        series = frame[column]
        if not _textish(series):
            continue
        values = _unique_values(series, cap=50)
        if not values:
            continue
        matched = [v for v in values if key_name(v) in SEASON_VOCAB
                   or any(key_name(a) in SEASON_VOCAB for a in _alternates(v))]
        if not matched:
            continue
        fraction = len(matched) / len(values)
        if best is None or fraction > best[0]:
            best = (fraction, column)
    if best is None or best[0] < 0.25:
        return FieldInference(None, 0.0, "no-season-overlap",
                              note="no column overlaps the Kharif/Rabi/Zaid vocabulary")
    return FieldInference(best[1], round(best[0], 4), "season-vocabulary-overlap",
                          note=f"{best[0]:.0%} of unique values are calendar seasons")


def infer_crop(frame: pd.DataFrame, exclude: set[str]) -> FieldInference:
    """Text column with high overlap against the crop vocabulary."""
    best: tuple[float, str] | None = None
    for column in frame.columns:
        if column in exclude:
            continue
        series = frame[column]
        if not _textish(series):
            continue
        values = _unique_values(series, cap=400)
        if len(values) < 2 or len(values) > 400:
            continue
        covered = sum(
            1 for v in values
            if key_name(v) in CROP_VOCAB
            or any(key_name(a) in CROP_VOCAB for a in _alternates(v))
        )
        fraction = covered / len(values)
        name_bonus = 1.0 if "crop" in key_name(column) else 0.0
        score = max(fraction, name_bonus * 0.6)
        if best is None or score > best[0]:
            best = (score, column)
    if best is None:
        return FieldInference(None, 0.0, "no-crop-column")
    return FieldInference(best[1], round(best[0], 4), "crop-name-overlap",
                          note=f"{best[0]:.0%} vocabulary overlap",
                          candidates=((best[1], best[0]),))


def infer_yield_and_wide(frame: pd.DataFrame) -> tuple[FieldInference, bool]:
    """Name-based yield column detection + wide-format flag."""
    area_cols: list[str] = []
    yield_cols: list[str] = []
    for column in frame.columns:
        name = str(column)
        if _AREA_RE.match(name):
            area_cols.append(column)
        if _YIELD_TRIPLE_RE.match(name):
            yield_cols.append(column)
    if len(area_cols) >= 2 or len(yield_cols) >= 2:
        return (
            FieldInference(None, 0.0, "wide-format",
                           note=f"<CROP> AREA/YIELD triples "
                                f"({len(area_cols)} crops); crop/yield are derived "
                                f"from the dominant crop at match time"),
            True,
        )
    name_matches = [
        column for column in frame.columns
        if YIELD_NAME_RE.search(str(column))
    ]
    if len(name_matches) == 1:
        column = name_matches[0]
        return FieldInference(column, 1.0, "name-based-match",
                              note="column name matches the yield vocabulary"), False
    if not name_matches:
        return (
            FieldInference(None, 0.0, "no-name-based-yield",
                           note="no column name matches yield/yeild/productivity"),
            False,
        )
    ambiguous = tuple((str(c), 1.0) for c in name_matches)
    return (
        FieldInference(None, 0.0, "ambiguous-yield",
                       note="multiple name-based yield columns: "
                            + ", ".join(str(c) for c in name_matches),
                       candidates=ambiguous),
        False,
    )


def infer_location(
    frame: pd.DataFrame,
    vocab: BoundaryVocabulary,
    exclude: set[str],
    state: FieldInference | None = None,
) -> tuple[FieldInference, FieldInference, dict[str, float] | None, str | None]:
    """Best boundary-coverage location column (+ restricted state, if any).

    When a state column exists the coverage is evaluated against the rows of
    the state with the best boundary overlap (e.g. ICRISAT's Karnataka), which
    is how a 20-state table resolves to its local district column.
    """
    sub = frame
    state_value: str | None = None
    if state is not None and state.column:
        column = state.column
        values = _unique_values(frame[column], cap=200)
        if values:
            best_state: tuple[float, str] | None = None
            for value in values:
                restricted = frame[frame[column].astype(str).str.strip().str.lower() == value.lower()]
                if len(restricted) == 0:
                    continue
                coverage, _ = _best_column_coverage(restricted, vocab, exclude | {column})
                score = coverage[1] if coverage[0] is not None else 0.0
                if best_state is None or score > best_state[0]:
                    best_state = (score, value)
            if best_state is not None and best_state[0] >= LOCATION_FUZZY_MIN:
                state_value = best_state[1]
                sub = frame[frame[column].astype(str).str.strip().str.lower() == state_value.lower()]

    coverage, per_value = _best_column_coverage(sub, vocab, exclude)
    best_col, best_score = coverage
    if best_col is None:
        return (
            FieldInference(None, 0.0, "no-location-column"),
            FieldInference(None, 0.0, "no-state-column"),
            None,
            None,
        )
    location = FieldInference(
        best_col, best_score, "boundary-name-overlap",
        note=f"{best_score:.0%} of unique place names hit the boundary/alias "
             f"vocabulary{(' (state=' + state_value + ')') if state_value else ''}",
    )
    state_inf = FieldInference(
        state.column if state and state.column else None,
        state.confidence if state else 0.0,
        state.method if state else "no-state-column",
        note=state.note if state else "",
    )
    return location, state_inf, per_value, state_value


def _best_column_coverage(
    frame: pd.DataFrame,
    vocab: BoundaryVocabulary,
    exclude: set[str],
) -> tuple[tuple[str | None, float], dict[str, float] | None]:
    best: tuple[float, str] | None = None
    best_per_value: dict[str, float] | None = None
    for column in frame.columns:
        if column in exclude:
            continue
        if not _textish(frame[column]):
            continue
        coverage, per_value = _boundary_coverage(frame[column], vocab)
        if coverage == 0.0:
            continue
        if best is None or coverage > best[0]:
            best = (coverage, column)
            best_per_value = per_value
    if best is None:
        return (None, 0.0), None
    return (best[1], best[0]), best_per_value


def _infer_features(frame: pd.DataFrame, key_columns: set[str]) -> list[str]:
    features: list[str] = []
    for column in frame.columns:
        if column in key_columns:
            continue
        name = str(column)
        if name.strip() == "" or name.startswith("Unnamed"):
            continue
        if frame[column].dropna().empty:
            continue
        features.append(column)
    return features


def _granularity(per_value: dict[str, float] | None, vocab: BoundaryVocabulary) -> tuple[int, int]:
    if not per_value:
        return 0, 0
    district_hits = 0
    taluk_hits = 0
    for value in per_value:
        kind = vocab.best_match(value).kind
        if kind == "district":
            district_hits += 1
        elif kind == "taluk":
            taluk_hits += 1
    return district_hits, taluk_hits


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def profile_tabular_source(
    source: str | Path,
    *,
    district_shp: str | Path = _DEFAULT_DISTRICT_SHP,
    taluk_shp: str | Path = _DEFAULT_TALUK_SHP,
    vocab: BoundaryVocabulary | None = None,
) -> SourceProfile:
    """Infer the STAM field mapping for ``source`` with confidence scores.

    Args:
        source: Path to the CSV to profile.
        district_shp / taluk_shp: KGIS boundary shapefiles (injected for tests).
        vocab: Pre-built boundary vocabulary (built from the repo defaults when
            None).

    Returns:
        A :class:`SourceProfile` with every field inference, the per-place-name
        Gate-B verdicts and the candidate ``table_config``.
    """
    source = Path(source)
    frame = read_source(source)
    vocab = vocab or build_default_vocabulary()

    state = infer_state(frame, vocab, exclude=set())
    location, state_inf, per_value, state_value = infer_location(frame, vocab, set(), state)

    key_columns: set[str] = {
        c for c in (location.column, state_inf.column) if c is not None
    }
    season = infer_season(frame, key_columns)
    key_columns |= {season.column} if season.column else set()
    year = infer_year(frame)
    key_columns |= {year.column} if year.column else set()
    yield_, wide_format = infer_yield_and_wide(frame)
    key_columns |= {yield_.column} if yield_.column else set()

    crop_exclude = key_columns
    crop = infer_crop(frame, crop_exclude)
    key_columns |= {crop.column} if crop.column else set()

    features = _infer_features(frame, key_columns)
    district_hits, taluk_hits = _granularity(per_value, vocab)

    # Place-name verdicts come from the selected location column.
    place_verdicts: list[PlaceNameVerdict] = []
    if location.column:
        for value in _unique_values(frame[location.column], cap=2000):
            verdict = vocab.classify(value)
            if verdict.status == "alias_existing":
                # Only surface genuinely new names; existing aliases are OK.
                verdict = PlaceNameVerdict(
                    verdict.name, "exact", 1.0,
                    matches=verdict.matches,
                    note="already in the alias table",
                )
            place_verdicts.append(verdict)

    year_lo = year_min = None
    year_hi = year_max = None
    if year.column and year.note.startswith("years "):
        bounds = year.note[len("years "):].split("-")
        if len(bounds) == 2:
            try:
                year_min, year_max = int(bounds[0]), int(bounds[1])
            except ValueError:
                pass

    profile = SourceProfile(
        source=source,
        columns=[str(c) for c in frame.columns],
        location=location,
        taluk=FieldInference(
            location.column if not wide_format and location.column else None,
            location.confidence if not wide_format else 0.0,
            "same-as-location" if not wide_format else "none",
            note=("narrow table: matched at village/taluk level"
                  if not wide_format else "wide-format tables use district column only"),
        ),
        district=FieldInference(
            location.column if wide_format and location.column else None,
            location.confidence if wide_format else 0.0,
            "district-boundary-overlap" if wide_format else "none",
        ),
        year=year,
        season=season,
        crop=FieldInference(
            None if wide_format else crop.column,
            crop.confidence if not wide_format else 0.0,
            "derived-dominant-crop" if wide_format else crop.method,
            note="derived from the dominant crop at match time" if wide_format else crop.note,
        ),
        yield_=yield_,
        state=state_inf,
        wide_format=wide_format,
        feature_columns=features,
        place_verdicts=place_verdicts,
        year_min=year_min,
        year_max=year_max,
        state_value=state_value,
    )

    notes = [f"granularity: {district_hits} district-like, {taluk_hits} taluk-like names"]
    if wide_format:
        notes.append("wide-format source: crop/yield derived from the dominant crop; "
                     "Gate A (name-based yield) cannot auto-approve this source")
    if year.column is None:
        notes.append("no unambiguous year column: the source cannot be auto-wired "
                     "(Gate A: ambiguous/missing year)")
    elif year_max is not None and year_max < 2018:
        notes.append(f"year range {year_min}-{year_max} does not overlap the imagery "
                     "window (2018-2025)")
    profile.notes.extend(notes)
    profile.table_config = candidate_table_entry(profile)
    return profile


def candidate_table_entry(profile: SourceProfile) -> dict:
    """The ``tabular.tables`` entry the auto-wire would commit for ``profile``."""
    entry: dict = {"name": profile.source.name}
    if profile.wide_format:
        entry.update(
            village_column=None,
            taluk_column=None,
            district_column=profile.location.column,
            season_column=profile.season.column if profile.season.column else None,
            year_column=profile.year.column,
            crop_column=None,
            yield_column=None,
            feature_columns=[],
            fallback_to_district=True,
        )
    else:
        entry.update(
            village_column=profile.location.column,
            taluk_column=profile.location.column,
            district_column=None,
            season_column=profile.season.column if profile.season.column else None,
            year_column=profile.year.column,
            crop_column=profile.crop.column,
            yield_column=profile.yield_.column,
            feature_columns=list(profile.feature_columns),
            fallback_to_district=False,
        )
    if profile.state.column:
        entry["state_column"] = profile.state.column
        entry["state_value"] = profile.state_value or profile.state.note
    return entry
