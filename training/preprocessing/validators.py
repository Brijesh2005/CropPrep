"""Quality filtering of observations before they enter the pipeline.

Observations that fail any configured check are rejected with a machine
readable reason, so downstream stages only ever see clean samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import QualityConfig
from .logger import get_logger

logger = get_logger("validators")


@dataclass(slots=True)
class FilterDecision:
    """Outcome of the quality filter for one observation."""

    observation_id: str
    accepted: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def code(self) -> str:
        return "ACCEPTED" if self.accepted else "REJECTED"


def filter_observation(observation: object, config: QualityConfig) -> FilterDecision:
    """Apply all configured checks to an :class:`AgriculturalObservation`."""
    reasons: list[str] = []
    obs_id = str(getattr(observation, "observation_id", "unknown"))

    # -- Coordinates ---------------------------------------------------------- #
    if config.require_valid_coordinates:
        location = observation.location
        if not (-180.0 <= location.lon <= 180.0 and -90.0 <= location.lat <= 90.0):
            reasons.append("invalid_coordinates")

    # -- Quality score --------------------------------------------------------- #
    score = float(observation.quality.overall_score)
    if score < config.min_quality_score:
        reasons.append(f"quality_score_below_{config.min_quality_score}")

    # -- Labels ---------------------------------------------------------------- #
    if config.require_crop_label and observation.crop is None:
        reasons.append("missing_crop_label")
    if config.require_yield_label and observation.yield_value is None:
        reasons.append("missing_yield_label")

    # -- Sequence --------------------------------------------------------------- #
    if config.min_observations > 0 and observation.num_observations() < config.min_observations:
        reasons.append(f"too_few_observations({observation.num_observations()})")

    if config.reject_unpaired and not observation.has_paired_images:
        reasons.append("unpaired_images")

    decision = FilterDecision(observation_id=obs_id, accepted=not reasons, reasons=reasons)
    if not decision.accepted:
        logger.debug(
            "Observation rejected",
            extra={"observation_id": obs_id, "reasons": reasons},
        )
    return decision


def filter_observations(observations: list, config: QualityConfig) -> tuple[list, list[FilterDecision]]:
    """Filter a batch; returns ``(accepted, decisions)``."""
    decisions = [filter_observation(obs, config) for obs in observations]
    accepted = [
        obs for obs, decision in zip(observations, decisions) if decision.accepted
    ]
    logger.info(
        "Quality filtering complete",
        extra={"accepted": len(accepted), "total": len(observations)},
    )
    return accepted, decisions
