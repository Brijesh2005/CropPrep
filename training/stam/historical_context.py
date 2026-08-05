"""Historical context builder for STAM.

Before an observation is assembled for inference, STAM builds a multi-year
historical context for the resolved location + season: which past years in
the catalog cover the *same* season. The builder reads exclusively through
the Dataset Manager (the sole data access path) and records the dataset +
season-calendar versions so every prediction is traceable.
"""

from __future__ import annotations

from typing import Any

from training.dataset_manager import DatasetManager

from .config import StamConfig
from .logger import get_logger
from .observation import HistoricalContext
from .season_resolver import SeasonResolver

logger = get_logger("historical_context")


class HistoricalContextBuilder:
    """Build the multi-year historical context for a location + season.

    Args:
        manager: The Dataset Manager (sole data access path).
        config: Validated :class:`StamConfig`.
        season_resolver: The :class:`SeasonResolver` used to resolve seasons.
    """

    def __init__(
        self,
        manager: DatasetManager,
        config: StamConfig,
        season_resolver: SeasonResolver,
    ) -> None:
        self.manager = manager
        self.config = config
        self.season_resolver = season_resolver

    def build(
        self,
        *,
        season_name: str | None,
        resolved_year: int | None,
        resolution: str | None = None,
    ) -> HistoricalContext:
        """Per-year satellite availability for a season, plus provenance.

        Args:
            season_name: The resolved season name (e.g. ``"Kharif"``).
            resolved_year: The observation's resolved planting year.
            resolution: Resolution band (defaults to the configured one).

        Returns:
            A :class:`HistoricalContext` with per-year counts + versions.
        """
        resolution = resolution or self.config.image.resolution
        window_months = self._season_months(season_name)
        try:
            availability = self.manager.get_historical_context(
                window_months=window_months,
                index_type=None,
                resolution=resolution,
            )
            source = "dataset_manager"
        except Exception:  # noqa: BLE001 - context is best-effort
            logger.warning(
                "Historical context unavailable",
                extra={"season": season_name},
            )
            availability = None
            source = "unavailable"

        return HistoricalContext(
            season=season_name,
            resolved_year=resolved_year,
            years=availability.years if availability else [],
            per_year=(
                {str(k): v for k, v in availability.per_year.items()}
                if availability
                else {}
            ),
            total_records=availability.total_records if availability else 0,
            dataset_version=self.manager.current_version(),
            season_calendar_version=self.season_resolver.version,
            source=source,
        )

    def _season_months(self, season_name: str | None) -> list[int] | None:
        if season_name is None:
            return None
        try:
            return self.season_resolver.calendar.months(season_name)
        except KeyError:
            return None
