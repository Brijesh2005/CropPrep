"""Password hashing with Argon2id (primary) and passlib (legacy hashes).

New passwords are hashed with Argon2id via ``argon2-cffi``. Verification is
backward compatible: hashes that do not start with ``$argon2`` are verified
against the configured legacy passlib scheme (e.g. ``pbkdf2_sha256``) so users
created by the Phase 8 auth flow can still log in.

Note: argon2-cffi 25.1.0's convenience ``PasswordHasher.verify`` has an
upstream bug that raises ``InvalidHashError`` for argon2id hashes (it slices 9
bytes while its header map stores 8-byte keys). We therefore verify through
``argon2.low_level.verify_secret`` with an explicit hash type, which is
unaffected.
"""

from __future__ import annotations

from argon2 import PasswordHasher, extract_parameters
from argon2.exceptions import VerifyMismatchError
from argon2.low_level import Type, verify_secret
from passlib.context import CryptContext

_TYPE_BY_HEADER = {
    "$argon2i$": Type.I,
    "$argon2d$": Type.D,
    "$argon2id$": Type.ID,
    "$argon2id": Type.ID,
}


class PasswordService:
    """Argon2id hashing + legacy passlib verification."""

    def __init__(
        self,
        *,
        time_cost: int = 3,
        memory_kib: int = 65536,
        parallelism: int = 4,
        legacy_scheme: str = "pbkdf2_sha256",
    ) -> None:
        self._argon = PasswordHasher(
            time_cost=time_cost, memory_cost=memory_kib, parallelism=parallelism
        )
        self._legacy = CryptContext(schemes=[legacy_scheme], deprecated="auto")
        self._legacy_scheme = legacy_scheme

    def hash(self, password: str) -> str:
        return self._argon.hash(password)

    def verify(self, password: str, hashed: str) -> bool:
        if not hashed.startswith("$argon2"):
            try:
                return self._legacy.verify(password, hashed)
            except (ValueError, TypeError):
                return False
        hash_type = _TYPE_BY_HEADER.get(hashed[:9], _TYPE_BY_HEADER.get(hashed[:8]))
        if hash_type is None:
            return False
        try:
            result = verify_secret(hashed.encode("ascii"), password.encode("utf-8"), hash_type)
            return bool(result)
        except (VerifyMismatchError, ValueError, TypeError):
            return False

    @staticmethod
    def is_argon2(hashed: str) -> bool:
        return hashed.startswith("$argon2")

    def needs_rehash(self, hashed: str) -> bool:
        """True when a legacy hash or non-conforming params should be re-hashed."""
        if not self.is_argon2(hashed):
            return True
        try:
            params = extract_parameters(hashed)
        except ValueError:
            return True
        return (
            params.type != Type.ID
            or params.time_cost != self._argon.time_cost
            or params.memory_cost != self._argon.memory_cost
            or params.parallelism != self._argon.parallelism
        )
