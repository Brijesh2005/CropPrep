"""SeasonResolver — resolves the current season from the calendar date.

The farmer workflow requires only a location: the system infers the season
from today's date instead of asking the user to pick a year and season. This
module provides the date -> (season, planting-year) mapping backed by a
YAML-configurable calendar.

Configuration precedence:

* explicit ``seasons_file`` argument / ``ST_SEASONS_FILE`` environment
  variable / ``ST_TEMPORAL__SEASON_FILE`` config value
* the bundled ``seasons.yaml`` shipped with the package
* the :data:`~training.stam.config.DEFAULT_SEASONS` defaults

Each loaded calendar is hashed into a stable ``version`` string so callers
can record which calendar produced a given prediction.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import yaml

from .config import SeasonDef, StamConfig
from .exceptions import StamConfigurationError
from .logger import get_logger
from .temporal_index import Season, SeasonCalendar

logger = get_logger("season_resolver")

#: Bundled default calendar (shipped with the package).
DEFAULT_SEASONS_FILE = Path(__file__).resolve().parent / "seasons.yaml"


class SeasonResolver:
    """Resolve a calendar date to a ``(Season, planting_year)`` pair."""

    def __init__(
        self,
        definitions: Sequence[SeasonDef | dict[str, Any]],
        *,
        source: str = "default",
        today: date | None = None,
    ) -> None:
        self.definitions = [
            d if isinstance(d, SeasonDef) else SeasonDef(**d) for d in definitions
        ]
        self.calendar = SeasonCalendar(self.definitions)
        self.source = source
        self.today = today or date.today()
        self.version = _calendar_version(self.definitions)

    # ------------------------------------------------------------------ #
    # Factories
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(
        cls,
        config: StamConfig | None = None,
        *,
        seasons_file: str | Path | None = None,
        today: date | None = None,
    ) -> "SeasonResolver":
        """Build a resolver from config / env / the bundled calendar."""
        cfg = config or StamConfig()
        path = _resolve_seasons_file(seasons_file, cfg)
        definitions = _load_season_definitions(path)
        if definitions is None:
            return cls(cfg.seasons, source="config", today=today)
        return cls(definitions, source=str(path), today=today)

    # ------------------------------------------------------------------ #
    # Resolution
    # ------------------------------------------------------------------ #

    def resolve(self, day: date | None = None) -> tuple[Season, int] | None:
        """The ``(Season, planting_year)`` for a date (default: today).

        Returns ``None`` when the date falls outside every configured season.
        """
        return self.calendar.season_for_date(day or self.today)

    def season_name(self, day: date | None = None) -> str | None:
        """The season name for a date (default: today)."""
        match = self.resolve(day)
        return match[0].name if match else None

    def names(self) -> list[str]:
        """All configured season names, in calendar order."""
        return self.calendar.names()


# --------------------------------------------------------------------------- #
# Calendar loading
# --------------------------------------------------------------------------- #


def _resolve_seasons_file(
    seasons_file: str | Path | None,
    config: StamConfig,
) -> Path:
    if seasons_file is not None:
        return Path(seasons_file)
    env = os.environ.get("ST_SEASONS_FILE")
    if env:
        return Path(env)
    if config.temporal.season_file is not None:
        return Path(config.temporal.season_file)
    return DEFAULT_SEASONS_FILE


def _load_season_definitions(path: Path) -> list[SeasonDef] | None:
    """Load season definitions from a YAML file (None when absent)."""
    if not path.exists():
        logger.info("season calendar file not found; using defaults", path=str(path))
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise StamConfigurationError(
            f"Malformed season calendar YAML: {exc}", detail=str(path)
        ) from exc
    if not isinstance(raw, dict):
        raise StamConfigurationError(
            "Season calendar root must be a mapping", detail=str(path)
        )
    seasons = raw.get("seasons")
    if not isinstance(seasons, list) or not seasons:
        raise StamConfigurationError(
            "Season calendar file has no 'seasons' list", detail=str(path)
        )
    try:
        return [SeasonDef(**s) for s in seasons]
    except Exception as exc:  # pydantic.ValidationError
        raise StamConfigurationError(
            f"Invalid season definition in {path}: {exc}", detail=str(path)
        ) from exc


def _calendar_version(definitions: Sequence[SeasonDef]) -> str:
    """A stable version string for a set of season definitions."""
    canonical = [
        (d.name, d.start_month, d.end_month, d.start_day, d.end_day)
        for d in definitions
    ]
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode("utf-8")
    )
    return f"1.{digest.hexdigest()[:8]}"
