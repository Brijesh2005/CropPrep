"""Observation Resolver — build the R2.3 training-sample corpus.

STAM resolves a single ``(location, year, season)`` into one
:class:`AgriculturalObservation`. This module adds the **bulk** layer: it
enumerates a *sampling grid* over dataset locations, years and seasons and
resolves every cell into a :class:`ResolvedSample` record. The collected
records form an :class:`ObservationCorpus` — the R2.3 multimodal training
dataset that downstream feature engineering, statistics, quality control and
export consume.

Pipeline::

    locations (spatial index) x years (tabular + image catalog) x seasons
        └─► plan() -> ObservationPlan (cells)
        └─► resolve() -> ObservationCorpus (accepted / rejected / error)

Nothing reads the raw datasets directly: every cell goes through
:meth:`STAM.build_observation`, the single sanctioned data-access path.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .exceptions import SampleResolutionError, StamError
from .logger import get_logger

logger = get_logger("observation_resolver")


class ObservationResolverConfig(BaseModel):
    """Settings for the observation resolver / sample generation."""

    model_config = ConfigDict(extra="forbid")

    #: Minimum :class:`QualityReport.overall_score` for a resolved sample to
    #: be classified ``accepted`` (mirrors STAM's ``quality.fail_below``).
    min_quality_score: float = Field(default=40.0, ge=0.0, le=100.0)
    #: Keep resolved-but-low-quality samples in the corpus as ``rejected``.
    include_rejected: bool = True
    #: Keep failed cells in the corpus as ``error`` rows (no observation).
    include_errors: bool = True
    #: Explicit years to sample (empty => infer from the dataset catalog).
    years: list[int] = Field(default_factory=list)
    #: Infer years from the tabular table + image catalog when ``years`` empty.
    infer_years: bool = True
    #: Season names to sample (empty => every calendar season).
    seasons: list[str] = Field(default_factory=list)
    #: Cap the number of distinct locations (None => all).
    max_locations: int | None = Field(default=None, ge=1)
    #: Restrict locations to a bounding box ``(lon_min, lat_min, lon_max,
    #: lat_max)``.
    bbox: tuple[float, float, float, float] | None = None
    #: Serve observations from / store in the STAM observation cache.
    use_cache: bool = True
    #: Seed for the deterministic location sub-sample when ``max_locations``
    #: is set (None => unordered).
    seed: int | None = 42


class SamplingCell(BaseModel):
    """One (location, year, season) grid cell to resolve."""

    location_id: str
    name: str
    lon: float
    lat: float
    year: int
    season: str

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.location_id, self.year, self.season)


class ObservationPlan(BaseModel):
    """The full sampling grid produced by :meth:`ObservationResolver.plan`."""

    model_config = ConfigDict(extra="forbid")

    cells: list[SamplingCell] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    years: list[int] = Field(default_factory=list)
    seasons: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def total(self) -> int:
        return len(self.cells)

    def counts(self) -> dict[str, int]:
        return {
            "locations": len(self.locations),
            "years": len(self.years),
            "seasons": len(self.seasons),
            "cells": self.total,
        }


class ResolvedSample(BaseModel):
    """Outcome of resolving one :class:`SamplingCell`."""

    location_id: str
    name: str
    lon: float
    lat: float
    year: int
    season: str
    #: ``accepted`` | ``rejected`` | ``error``.
    status: str = Field(pattern="^(accepted|rejected|error)$")
    #: Overall quality score (None when the cell failed to resolve).
    quality_score: float | None = None
    #: ``{"code": ..., "message": ...}`` when the cell failed.
    error: dict[str, Any] | None = None
    #: The resolved observation (None on error).
    observation: Any | None = None
    #: Wall-clock resolution time in milliseconds.
    duration_ms: int = 0

    @property
    def cell_key(self) -> tuple[str, int, str]:
        return (self.location_id, self.year, self.season)


class ObservationCorpus(BaseModel):
    """The collected set of resolved training samples.

    :class:`ResolvedSample` records are kept for every cell so acceptance,
    rejection and failure statistics can be computed later. Use
    :meth:`accepted_observations` to feed the Phase 4 preprocessing
    (:func:`~training.preprocessing.dataset.split_observations` +
    :class:`~training.preprocessing.dataset.CropFusionDataset`).
    """

    model_config = ConfigDict(extra="forbid")

    samples: list[ResolvedSample] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    config: dict[str, Any] = Field(default_factory=dict)

    # -- Selection ------------------------------------------------------------ #

    @property
    def total(self) -> int:
        return len(self.samples)

    def by_status(self, status: str) -> list[ResolvedSample]:
        return [s for s in self.samples if s.status == status]

    def accepted(self) -> list[ResolvedSample]:
        return self.by_status("accepted")

    def rejected(self) -> list[ResolvedSample]:
        return self.by_status("rejected")

    def errors(self) -> list[ResolvedSample]:
        return self.by_status("error")

    def accepted_observations(self) -> list[Any]:
        """The accepted :class:`AgriculturalObservation` objects in order."""
        return [s.observation for s in self.accepted() if s.observation is not None]

    # -- Summary -------------------------------------------------------------- #

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {"accepted": 0, "rejected": 0, "error": 0}
        for sample in self.samples:
            counts[sample.status] += 1
        return counts

    def summary(self) -> dict[str, Any]:
        counts = self.status_counts()
        total = self.total
        scored = [s.quality_score for s in self.accepted() if s.quality_score is not None]
        return {
            "total": total,
            "accepted": counts["accepted"],
            "rejected": counts["rejected"],
            "errors": counts["error"],
            "acceptance_rate": round(counts["accepted"] / total, 4) if total else 0.0,
            "quality": _score_summary(scored),
            "config": self.config,
            "created_at": self.created_at.isoformat(),
        }

    def to_dict(self, *, mode: str = "json") -> dict[str, Any]:
        return {
            "created_at": self.created_at.isoformat(),
            "config": self.config,
            "samples": [s.model_dump(mode=mode) for s in self.samples],
        }

    # -- Persistence ---------------------------------------------------------- #

    def save(self, path: str | Any) -> "ObservationCorpus":
        """Write the corpus to a JSON file (pydantic-safe serialisation)."""
        from pathlib import Path

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return self

    @classmethod
    def load(cls, path: str | Any) -> "ObservationCorpus":
        """Reconstruct a corpus from a JSON file written by :meth:`save`."""
        from pathlib import Path

        from .observation import AgriculturalObservation

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        samples: list[ResolvedSample] = []
        for item in raw.get("samples", []):
            payload = dict(item)
            obs = payload.get("observation")
            payload["observation"] = (
                AgriculturalObservation.model_validate(obs) if obs is not None else None
            )
            samples.append(ResolvedSample.model_validate(payload))
        return cls(samples=samples, config=raw.get("config", {}),
                   created_at=raw.get("created_at"))


class ObservationResolver:
    """Enumerate and resolve training samples over the dataset grid.

    Args:
        stam: An initialized :class:`~training.stam.stam.STAM` instance.
        config: Validated resolver settings (defaults when None).

    The resolver reads locations from STAM's built spatial index and years
    from the Dataset Manager (tabular table + image catalog) — both through
    STAM / its adapters, never directly from disk.
    """

    def __init__(
        self,
        stam: Any,
        config: ObservationResolverConfig | None = None,
    ) -> None:
        self.stam = stam
        self.config = config or ObservationResolverConfig()

    # -- Catalog helpers ------------------------------------------------------ #

    def _require_indexes(self) -> None:
        if not self.stam.matcher.initialized:
            self.stam.initialize()

    def available_years(self) -> list[int]:
        """Distinct data years across the tabular table and image catalog."""
        years: set[int] = set()
        try:
            for value in self.stam.matcher.tabular_source.available_years():
                years.add(int(value))
        except Exception:  # noqa: BLE001 - best effort
            pass
        try:
            catalog = self.stam.manager.image_catalog()
            years.update(int(y) for y in catalog.years)
        except Exception:  # noqa: BLE001 - best effort
            pass
        return sorted(years)

    def available_seasons(self) -> list[str]:
        return list(self.stam.season_resolver.names())

    def locations(self) -> list[Any]:
        """Dataset locations from STAM's spatial index (LocationPoints)."""
        self._require_indexes()
        return list(self.stam.matcher.spatial_index.points)

    # -- Plan ----------------------------------------------------------------- #

    def plan(
        self,
        *,
        years: Sequence[int] | None = None,
        seasons: Sequence[str] | None = None,
        locations: Iterable[Any] | None = None,
        max_locations: int | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> ObservationPlan:
        """Build the sampling grid ``locations x years x seasons``.

        Args:
            years: Explicit years (default: config / inferred catalog years).
            seasons: Explicit season names (default: all calendar seasons).
            locations: Explicit location points (default: spatial index).
            max_locations: Cap on distinct locations (config fallback).
            bbox: ``(lon_min, lat_min, lon_max, lat_max)`` location filter.

        Raises:
            SampleResolutionError: When no years or seasons resolve.
        """
        self._require_indexes()

        year_list = _resolve_years(self.config, years)
        season_list = list(seasons) if seasons is not None else list(self.config.seasons)
        if not season_list:
            season_list = self.available_seasons()
        if not year_list and self.config.infer_years:
            year_list = self.available_years()
        if not year_list:
            raise SampleResolutionError(
                "No sample years available (tabular table + image catalog empty)",
                suggested_resolution="Provide explicit 'years' or ingest tabular/image data first",
            )
        if not season_list:
            raise SampleResolutionError(
                "No seasons configured on the season calendar"
            )

        locs = list(locations) if locations is not None else self.locations()
        if bbox is None and self.config.bbox is not None:
            bbox = self.config.bbox
        if bbox is not None:
            lon_min, lat_min, lon_max, lat_max = bbox
            locs = [
                p for p in locs
                if lon_min <= p.lon <= lon_max and lat_min <= p.lat <= lat_max
            ]
        cap = max_locations if max_locations is not None else self.config.max_locations
        if cap is not None and len(locs) > cap:
            rng = random.Random(self.config.seed)
            locs = rng.sample(locs, cap)
            logger.info("Location sub-sample applied", extra={"cap": cap})

        cells = [
            SamplingCell(
                location_id=p.id, name=p.name, lon=float(p.lon), lat=float(p.lat),
                year=year, season=season,
            )
            for p in locs
            for year in year_list
            for season in season_list
        ]
        plan = ObservationPlan(
            cells=cells,
            locations=[p.id for p in locs],
            years=year_list,
            seasons=season_list,
        )
        logger.info("Observation plan built", extra=plan.counts())
        return plan

    # -- Resolution ----------------------------------------------------------- #

    def resolve_cell(self, cell: SamplingCell, *, resolution: str | None = None) -> ResolvedSample:
        """Resolve one grid cell into a :class:`ResolvedSample`.

        Failures are captured on the sample record (``status="error"``) rather
        than raised, mirroring how bulk resolution behaves.
        """
        import time

        started = time.perf_counter()
        try:
            observation = self.stam.build_observation(
                cell.lon, cell.lat,
                year=cell.year, season=cell.season,
                resolution=resolution,
                use_cache=self.config.use_cache,
            )
        except StamError as exc:
            return _error_sample(cell, started, exc.code, str(exc))
        except Exception as exc:  # noqa: BLE001 - unexpected failures are recorded
            logger.warning(
                "Cell resolution failed unexpectedly",
                extra={"cell": cell.key, "reason": str(exc)},
            )
            return _error_sample(cell, started, "ST-RESOLVE-999", str(exc))

        accepted = (
            observation.quality.passed
            and observation.quality.overall_score >= self.config.min_quality_score
        )
        duration = int((time.perf_counter() - started) * 1000)
        return ResolvedSample(
            location_id=cell.location_id,
            name=cell.name,
            lon=cell.lon,
            lat=cell.lat,
            year=cell.year,
            season=cell.season,
            status="accepted" if accepted else "rejected",
            quality_score=observation.quality.overall_score,
            observation=observation,
            duration_ms=duration,
        )

    def resolve(
        self,
        plan: ObservationPlan | None = None,
        *,
        resolution: str | None = None,
        progress_every: int = 50,
    ) -> ObservationCorpus:
        """Resolve a plan (or build one) into an :class:`ObservationCorpus`.

        Args:
            plan: Pre-built plan; a default plan is built when None.
            resolution: Optional image resolution band override.
            progress_every: Log progress every N cells.
        """
        if plan is None:
            plan = self.plan()
        samples: list[ResolvedSample] = []
        for index, cell in enumerate(plan.cells, start=1):
            sample = self.resolve_cell(cell, resolution=resolution)
            if sample.status == "error" and not self.config.include_errors:
                continue
            if sample.status == "rejected" and not self.config.include_rejected:
                continue
            samples.append(sample)
            if progress_every and index % progress_every == 0:
                logger.info(
                    "Corpus progress", extra={"resolved": index, "total": plan.total}
                )
        corpus = ObservationCorpus(
            samples=samples, config=self.config.model_dump(mode="json")
        )
        logger.info(
            "Corpus resolved",
            extra={"total": corpus.total, **corpus.status_counts()},
        )
        return corpus


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _resolve_years(config: ObservationResolverConfig, explicit: Sequence[int] | None) -> list[int]:
    if explicit is not None:
        return sorted(set(int(y) for y in explicit))
    if config.years:
        return sorted(set(int(y) for y in config.years))
    if config.infer_years:
        return []
    return []


def _error_sample(cell: SamplingCell, started: float, code: str, message: str) -> ResolvedSample:
    import time

    return ResolvedSample(
        location_id=cell.location_id,
        name=cell.name,
        lon=cell.lon,
        lat=cell.lat,
        year=cell.year,
        season=cell.season,
        status="error",
        error={"code": code, "message": message},
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


def _score_summary(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"min": None, "max": None, "mean": None, "median": None, "count": 0}
    ordered = sorted(scores)
    count = len(ordered)
    median = ordered[count // 2] if count % 2 else (
        (ordered[count // 2 - 1] + ordered[count // 2]) / 2.0
    )
    return {
        "count": count,
        "min": round(ordered[0], 2),
        "max": round(ordered[-1], 2),
        "mean": round(sum(ordered) / count, 2),
        "median": round(median, 2),
    }
