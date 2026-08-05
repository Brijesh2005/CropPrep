"""Canonical exception base for the CropFusion shared framework.

Every platform (Training and Application) and every domain module raises
exceptions that derive from :class:`CropFusionError`.  The base carries a
stable machine-readable ``code``, a human readable ``message``, optional
structured ``detail`` and an optional ``suggested_resolution`` so that CLI
tooling, REST adapters and dashboards can map failures deterministically and
guide users towards a fix without string matching.

Error codes follow the convention ``<PREFIX>-<AREA>-<NNN>`` where ``PREFIX``
identifies the raising component (``DM-``, ``TD-``, ``ST-``, ``PPT-``,
``MOD-``, ``EXP-``, ``ML-``, ``API-``) and ``AREA`` a stable area tag.
"""

from __future__ import annotations

from typing import Any


class CropFusionError(Exception):
    """Base class for every CropFusion error.

    Attributes:
        code: Stable machine-readable error code, e.g. ``"DM-DL-001"``.
        message: Human readable description of the failure.
        detail: Optional structured detail (offending path, expected value,
            actual value, ...) attached to the error.
        suggested_resolution: Optional human readable guidance for recovering.
    """

    code: str = "CF-ERROR"

    def __init__(
        self,
        message: str,
        *,
        detail: Any = None,
        suggested_resolution: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.suggested_resolution = suggested_resolution

    def __str__(self) -> str:
        text = f"{self.code}: {self.message}"
        if self.detail is not None:
            text += f" (detail={self.detail!r})"
        return text
