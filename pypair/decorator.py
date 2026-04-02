from time import perf_counter
from functools import wraps

from pypair.profiling import record_timing


def timeit(f):
    """
    Records execution time when profiling is enabled.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        try:
            return f(*args, **kwargs)
        finally:
            record_timing(f"{f.__module__}.{f.__qualname__}", perf_counter() - start)

    return wrapper


def similarity(f):
    """
    Marker for similarity functions.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


def distance(f):
    """
    Marker for distance functions.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper
