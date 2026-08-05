"""Validator registry: pick a validator by target name."""

from __future__ import annotations

from ..exceptions import ValidationNotSupportedError
from .base import Validator


class ValidatorRegistry:
    """Named registry of :class:`Validator` implementations."""

    def __init__(self) -> None:
        self._validators: dict[str, Validator] = {}

    def register(self, validator: Validator) -> None:
        """Register a validator under its ``name``."""
        self._validators[validator.name] = validator

    def get(self, name: str) -> Validator:
        """Return the validator registered as ``name``."""
        try:
            return self._validators[name]
        except KeyError:
            raise ValidationNotSupportedError(
                f"No validator registered for target: {name}",
                detail=name,
                suggested_resolution=f"Register one via ValidatorRegistry.register, or use one of {sorted(self._validators)}",
            ) from None

    def names(self) -> list[str]:
        """Names of all registered validators."""
        return sorted(self._validators)


#: Process-wide registry pre-populated with the built-in validators.
default_registry = ValidatorRegistry()
