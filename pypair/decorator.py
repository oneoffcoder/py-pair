import time
from functools import wraps


def timeit(f):
    """
    Benchmarks the time it takes (seconds) to execute.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        output = f(*args, **kwargs)
        # diff = time.time() - start
        time.time() - start
        # print(f'{f.__name__}: {diff}')
        return output

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
