from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from functools import wraps
from typing import ParamSpec, TypeVar

from pypair.profiling import record_timing

P = ParamSpec("P")
R = TypeVar("R")


def timeit(f: Callable[P, R]) -> Callable[P, R]:
    """
    Records execution time when profiling is enabled.
    """

    @wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = perf_counter()
        try:
            return f(*args, **kwargs)
        finally:
            record_timing(f"{f.__module__}.{f.__qualname__}", perf_counter() - start)

    return wrapper


def similarity(f: Callable[P, R]) -> Callable[P, R]:
    """
    Marker for similarity functions.
    """

    @wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return f(*args, **kwargs)

    return wrapper


def distance(f: Callable[P, R]) -> Callable[P, R]:
    """
    Marker for distance functions.
    """

    @wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return f(*args, **kwargs)

    return wrapper
