"""Drift package exceptions."""

from __future__ import annotations


class DriftError(Exception):
    """Base error for the drift-monitoring framework."""


class InsufficientDataError(DriftError):
    """Raised when a dataset does not meet the minimum sample requirement."""
