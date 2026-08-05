"""Semantic versioning implementation.

Versioning follows MAJOR.MINOR.PATCH:

* **MAJOR** — incompatible structural change (schema / layout breaking).
* **MINOR** — backward compatible additions (new years, new index types).
* **PATCH** — corrections (fixes, metadata refresh).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..exceptions import InvalidVersionError

_VERSION_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


@dataclass(frozen=True, slots=True)
class SemanticVersion:
    """A validated ``MAJOR.MINOR.PATCH`` version.

    The string form is ``"{major}.{minor}.{patch}"``.
    """

    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        for part in (self.major, self.minor, self.patch):
            if not isinstance(part, int) or part < 0:
                raise InvalidVersionError(
                    f"Version parts must be non-negative integers: {self}"
                )

    @classmethod
    def from_string(cls, value: str) -> "SemanticVersion":
        """Parse a ``MAJOR.MINOR.PATCH`` string.

        Raises:
            InvalidVersionError: When ``value`` is not valid semver.
        """
        if not isinstance(value, str):
            raise InvalidVersionError(
                "Version must be a string", detail=repr(value)
            )
        match = _VERSION_RE.match(value.strip())
        if not match:
            raise InvalidVersionError(
                "Invalid semantic version (expected MAJOR.MINOR.PATCH)",
                detail=value,
                suggested_resolution="Use a three-part numeric version, e.g. '1.0.0'",
            )
        major, minor, patch = (int(g) for g in match.groups())
        return cls(major=major, minor=minor, patch=patch)

    def bump(self, part: str = "patch") -> "SemanticVersion":
        """Return the version after bumping ``part`` (major/minor/patch)."""
        normalised = part.lower()
        if normalised == "major":
            return SemanticVersion(self.major + 1, 0, 0)
        if normalised == "minor":
            return SemanticVersion(self.major, self.minor + 1, 0)
        if normalised == "patch":
            return SemanticVersion(self.major, self.minor, self.patch + 1)
        raise InvalidVersionError(
            f"Unknown version part: {part}",
            detail=part,
            suggested_resolution="Use one of 'major', 'minor' or 'patch'",
        )

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) < (
            other.major,
            other.minor,
            other.patch,
        )

    def __le__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) <= (
            other.major,
            other.minor,
            other.patch,
        )

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) > (
            other.major,
            other.minor,
            other.patch,
        )

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) >= (
            other.major,
            other.minor,
            other.patch,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) == (
            other.major,
            other.minor,
            other.patch,
        )
