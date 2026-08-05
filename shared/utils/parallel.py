"""Parallel helpers."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_parallel(
    items: Sequence[T],
    worker: Callable[[T], R],
    *,
    workers: int | None = None,
    raise_on_error: bool = False,
) -> list[R]:
    """Apply ``worker`` to every item in ``items`` using a thread pool.

    The order of results matches the order of ``items``. When
    ``raise_on_error`` is False (default) a worker exception is captured as
    the result for that slot; when True the exception is re-raised.

    Args:
        items: Iterable of inputs (materialised once).
        worker: Callable applied to each item.
        workers: Thread count. Defaults to ``min(32, cpu_count + 4)``.
        raise_on_error: Re-raise worker exceptions instead of capturing them.
    """
    pool_size = workers or min(32, (os.cpu_count() or 1) + 4)
    if not items:
        return []
    results: list[R] = [None] * len(items)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=max(1, pool_size)) as pool:
        future_map = {pool.submit(worker, item): i for i, item in enumerate(items)}
        for future in as_completed(future_map):
            index = future_map[future]
            try:
                results[index] = future.result()
            except Exception as exc:  # noqa: BLE001 - captured per design
                if raise_on_error:
                    raise
                results[index] = exc  # type: ignore[assignment]
    return results
