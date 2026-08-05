"""Temporal indexing for STAM: seasons, dates, gaps and matching.

The :class:`SeasonCalendar` encodes configurable cropping-season definitions
(e.g. Kharif Jun-Oct, Rabi Nov-Mar, Summer Apr-May). Seasons that cross the
calendar-year boundary (Rabi) are handled explicitly: a date in Jan-Mar is
attributed to the Rabi season that *started* in the previous year.

:class:`TemporalIndex` provides date-oriented helpers used by the sequence
builder — nearest-date matching, range filtering, sorting/deduplication and
gap detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Sequence

from .config import SeasonDef
from .logger import get_logger

logger = get_logger("temporal_index")


@dataclass(frozen=True, slots=True)
class Season:
    """A resolved season for a specific planting year."""

    name: str
    year: int  # planting year (the year the season started)
    start: date
    end: date

    @property
    def crosses_year_boundary(self) -> bool:
        return self.end.year > self.start.year


def _last_day(year: int, month: int) -> int:
    """Last day of a month in a given year."""
    next_month = date(year, month, 1) + timedelta(days=32)
    return (next_month - timedelta(days=next_month.day)).day


class SeasonCalendar:
    """Resolve dates/years against configurable season definitions."""

    def __init__(self, definitions: Sequence[SeasonDef]) -> None:
        self.definitions = list(definitions)
        self._by_name = {d.name: d for d in definitions}

    # -- Lookup --------------------------------------------------------------- #

    def names(self) -> list[str]:
        return [d.name for d in self.definitions]

    def has_season(self, name: str | None) -> bool:
        return name in self._by_name

    def months(self, name: str) -> list[int]:
        """Calendar months occupied by a season definition (may cross years).

        Example: Rabi (Nov-Mar) -> ``[11, 12, 1, 2, 3]``.
        """
        definition = self._by_name[name]
        start, end = definition.start_month, definition.end_month
        if start <= end:
            return list(range(start, end + 1))
        return list(range(start, 13)) + list(range(1, end + 1))

    def season_for_date(self, day: date) -> tuple[Season, int] | None:
        """Resolve ``(season, planting_year)`` for a calendar date.

        For a crossing season (e.g. Rabi Nov-Mar), dates in Jan-Mar resolve to
        the planting year ``day.year - 1``; dates in Nov-Dec to ``day.year``.

        Returns:
            A ``(Season, planting_year)`` tuple, or None when the date does
            not fall in any configured season.
        """
        for definition in self.definitions:
            if definition.crosses_year_boundary:
                # Window: Nov(prev)->Mar(current) OR Nov(current)->Mar(next).
                if day.month == 12 or day.month >= definition.start_month:
                    year = day.year
                    start = date(year, definition.start_month, definition.start_day)
                    end = date(
                        year + 1,
                        definition.end_month,
                        min(_last_day(year + 1, definition.end_month), definition.end_day),
                    )
                elif day.month <= definition.end_month:
                    year = day.year - 1
                    start = date(year, definition.start_month, definition.start_day)
                    end = date(
                        day.year,
                        definition.end_month,
                        min(_last_day(day.year, definition.end_month), definition.end_day),
                    )
                else:
                    continue
                if start <= day <= end:
                    return Season(definition.name, year, start, end), year
            else:
                if definition.start_month <= day.month <= definition.end_month:
                    start = date(day.year, definition.start_month, definition.start_day)
                    end = date(
                        day.year,
                        definition.end_month,
                        min(_last_day(day.year, definition.end_month), definition.end_day),
                    )
                    if start <= day <= end:
                        return Season(definition.name, day.year, start, end), day.year
        return None

    def season_window(self, name: str, year: int) -> Season:
        """The (start, end) date window of a season for a planting year.

        Raises:
            KeyError: When the season name is unknown.
        """
        definition = self._by_name[name]
        start = date(year, definition.start_month, definition.start_day)
        if definition.crosses_year_boundary:
            end = date(
                year + 1,
                definition.end_month,
                min(_last_day(year + 1, definition.end_month), definition.end_day),
            )
        else:
            end = date(
                year,
                definition.end_month,
                min(_last_day(year, definition.end_month), definition.end_day),
            )
        return Season(name, year, start, end)

    def contains(self, season: Season, day: date) -> bool:
        return season.start <= day <= season.end


class TemporalIndex:
    """Date helpers used throughout STAM."""

    @staticmethod
    def sort_unique(dates: Iterable[date]) -> list[date]:
        """Sort and deduplicate a collection of dates."""
        return sorted(set(dates))

    @staticmethod
    def nearest(target: date, dates: Sequence[date]) -> tuple[date | None, int | None]:
        """Nearest date and its index in ``dates`` (unsorted input ok)."""
        if not dates:
            return None, None
        best_date, best_idx = None, None
        best_delta: int | None = None
        for i, candidate in enumerate(dates):
            delta = abs((candidate - target).days)
            if best_delta is None or delta < best_delta:
                best_delta, best_date, best_idx = delta, candidate, i
        return best_date, best_idx

    @staticmethod
    def in_range(start: date, end: date, dates: Sequence[date]) -> list[date]:
        """Dates within the inclusive ``[start, end]`` window."""
        return [d for d in dates if start <= d <= end]

    @staticmethod
    def gaps(dates: Sequence[date]) -> list[float]:
        """Day-gaps between consecutive sorted dates."""
        ordered = TemporalIndex.sort_unique(dates)
        return [
            float((ordered[i + 1] - ordered[i]).days)
            for i in range(len(ordered) - 1)
        ]

    @staticmethod
    def max_gap_days(dates: Sequence[date]) -> float:
        gaps = TemporalIndex.gaps(dates)
        return max(gaps) if gaps else 0.0

    @staticmethod
    def within_tolerance(actual: date, expected: date, tolerance_days: int) -> bool:
        return abs((actual - expected).days) <= tolerance_days

    @staticmethod
    def dedupe_by_date(records: list[dict], date_key: str = "observation_date"):
        """Drop records sharing an observation date (keep the first)."""
        seen: set[date] = set()
        result: list[dict] = []
        for record in records:
            value = record.get(date_key)
            if value is None or value in seen:
                continue
            seen.add(value)
            result.append(record)
        return result
