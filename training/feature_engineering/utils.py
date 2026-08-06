"""Small helpers for the feature-engineering package."""

from __future__ import annotations

from typing import Any, Sequence


def observations_from_corpus(observations: Sequence[Any]) -> list[Any]:
    """Normalise an input into a list of :class:`AgriculturalObservation`.

    Accepts a sequence of observations directly, or an
    :class:`~training.stam.observation_resolver.ObservationCorpus` (its
    accepted observations are used). A single observation is wrapped.
    """
    if observations is None:
        return []
    if hasattr(observations, "accepted_observations"):
        return list(observations.accepted_observations())
    if hasattr(observations, "crop") or hasattr(observations, "location"):
        return [observations]
    return list(observations)


def group_counts(rows: Sequence[Any], key: str) -> dict[str, int]:
    """Count occurrences of a non-None attribute across rows."""
    counts: dict[str, int] = {}
    for row in rows:
        value = getattr(row, key, None)
        if value is None:
            continue
        name = str(value)
        counts[name] = counts.get(name, 0) + 1
    return counts
