"""Account lockout policy.

Tracks failed login attempts per account (persisted on the user row) and
exposes lock/unlock decisions. Time-based lockout is deterministic from
``locked_until``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


class AccountLockout:
    """Applies the configured failed-attempts/lockout policy."""

    def __init__(self, max_failed_attempts: int, lockout_minutes: int) -> None:
        self.max_failed_attempts = max_failed_attempts
        self.lockout_minutes = lockout_minutes

    def is_locked(self, failed_attempts: int, locked_until: datetime | None) -> bool:
        if locked_until is not None and locked_until > datetime.now(timezone.utc):
            return True
        return failed_attempts >= self.max_failed_attempts

    def next_lockout(self, current_attempts: int) -> datetime | None:
        """Return the new ``locked_until`` (if the threshold is reached)."""
        if current_attempts + 1 >= self.max_failed_attempts:
            return datetime.now(timezone.utc) + timedelta(minutes=self.lockout_minutes)
        return None

    def lockout_expires_in_seconds(self, locked_until: datetime) -> int:
        delta = locked_until - datetime.now(timezone.utc)
        return max(int(delta.total_seconds()), 0)
