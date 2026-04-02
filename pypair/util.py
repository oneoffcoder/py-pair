from abc import ABC
from functools import lru_cache
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


class MeasureMixin(ABC):
    """
    Measure mixin. Able to get list the functions decorated with `@property` and also
    access such property based on name.
    """

    @classmethod
    def measures(cls):
        """Gets a list of all the measures."""
        return get_measures(cls)

    @lru_cache(maxsize=None)
    def get(self, measure):
        """Gets the specified measure."""
        return getattr(self, measure)

    @lru_cache(maxsize=None)
    def get_measures(self):
        """Gets a list of all the measures."""
        return get_measures(self.__class__)


def get_measures(clazz):
    """Gets all the measures of a clazz."""
    from itertools import chain

    def is_property(value):
        return isinstance(value, property)

    def is_public(name):
        return not name.startswith("_")

    def is_valid(name, value):
        return is_public(name) and is_property(value)

    measures = sorted(list(chain(*[[n for n, v in vars(c).items() if is_valid(n, v)] for c in clazz.__mro__])))
    return measures


def to_numpy(values: Any, dtype=None) -> np.ndarray:
    """Converts common sequence / series inputs to a numpy array."""
    if isinstance(values, np.ndarray):
        return values.astype(dtype) if dtype is not None else values

    if hasattr(values, "to_numpy"):
        arr = values.to_numpy()
    else:
        arr = np.asarray(list(values))

    return arr.astype(dtype) if dtype is not None else arr


def corr(df, f):
    """
    Computes the pairwise association matrix for a pandas dataframe.

    :param df: Pandas data frame.
    :param f: Callable function; e.g. lambda a, b: categorical_categorical(a, b, measure='phi')
    """
    fields = list(df.columns)
    idx_map = {field: i for i, field in enumerate(fields)}
    n = len(fields)
    mat = np.zeros((n, n), dtype=float)

    for a, b in combinations(fields, 2):
        i, j = idx_map[a], idx_map[b]
        assoc = f(df[a], df[b])
        mat[i, j] = assoc
        mat[j, i] = assoc

    return pd.DataFrame(mat, columns=fields, index=fields)
