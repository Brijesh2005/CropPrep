"""Inference module models (no ORM — predictions live in the predictions module)."""

from __future__ import annotations

# The inference engine is stateless apart from the loaded model; prediction
# persistence belongs to the predictions / history modules.
