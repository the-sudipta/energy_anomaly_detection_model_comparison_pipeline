"""Timing helpers."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from src.utils.logger import get_logger

F = TypeVar("F", bound=Callable[..., Any])
_log = get_logger(__name__)


def format_seconds(seconds: float) -> str:
    """Format a duration for humans.

    Args:
        seconds: Duration in seconds.

    Returns:
        A string such as ``"4.2s"`` or ``"3m 12s"``.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes}m {rest:02d}s"


@contextmanager
def timed(label: str) -> Iterator[dict[str, float]]:
    """Log the start and end of a block with its elapsed time.

    Args:
        label: Name shown in the log lines.

    Yields:
        A dict whose ``"seconds"`` key is filled when the block exits.
    """
    result: dict[str, float] = {"seconds": 0.0}
    _log.info("Started: %s", label)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
        _log.info("Finished: %s in %s", label, format_seconds(result["seconds"]))


def timed_stage(label: str) -> Callable[[F], F]:
    """Decorator form of :func:`timed`.

    Args:
        label: Name shown in the log lines.

    Returns:
        A decorator that wraps the function in a timed block.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with timed(label):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
