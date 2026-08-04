"""Password policy enforcement.

A configurable policy object validates new passwords and exposes human-readable
violation messages. Combined with :class:`~app.core.config.PasswordPolicySettings`.
"""

from __future__ import annotations

import re

from app.core.config import PasswordPolicySettings


class PolicyViolationError(ValueError):
    """Raised when a password does not satisfy the configured policy."""


class PasswordPolicy:
    """Validates passwords against the configured complexity policy."""

    def __init__(self, settings: PasswordPolicySettings) -> None:
        self.settings = settings

    def validate(self, password: str, *, email: str | None = None) -> None:
        settings = self.settings
        if len(password) < settings.min_length:
            raise PolicyViolationError(
                f"password must be at least {settings.min_length} characters"
            )
        if len(password) > settings.max_length:
            raise PolicyViolationError(
                f"password must be at most {settings.max_length} characters"
            )
        if settings.require_uppercase and not re.search(r"[A-Z]", password):
            raise PolicyViolationError("password must contain an uppercase letter")
        if settings.require_lowercase and not re.search(r"[a-z]", password):
            raise PolicyViolationError("password must contain a lowercase letter")
        if settings.require_digit and not re.search(r"\d", password):
            raise PolicyViolationError("password must contain a digit")
        if settings.require_special and not re.search(r"[^A-Za-z0-9]", password):
            raise PolicyViolationError("password must contain a special character")
        if settings.prevent_email_substring and email:
            if email.lower().split("@")[0].lower() in password.lower():
                raise PolicyViolationError("password must not contain your email/username")
