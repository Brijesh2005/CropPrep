"""Audit trail helpers."""

from __future__ import annotations

import logging
from typing import Any

_AUDIT_LOGGER = "cropfusion.audit"


def audit(
    action: str,
    *,
    actor: str | None = None,
    target: str | None = None,
    outcome: str = "success",
    detail: Any = None,
    logger: logging.Logger | None = None,
) -> None:
    """Emit an audit log record (``INFO`` level, structured fields).

    Args:
        action: Stable action name, e.g. ``"dataset.validated"``.
        actor: Principal performing the action.
        target: Artefact the action affected.
        outcome: ``"success"`` or ``"failure"``.
        detail: Optional structured payload.
    """
    use = logger or logging.getLogger(_AUDIT_LOGGER)
    use.info(
        action,
        extra={
            "audit": True,
            "actor": actor,
            "target": target,
            "outcome": outcome,
            "detail": detail,
        },
    )
