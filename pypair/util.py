from abc import ABC
from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd


class MeasureMixin(ABC):
    """
    Measure mixin. Able to get list the functions decorated with `@property` and also
    access such property based on name.
    """

    @classmethod
    def measures(cls):
        """
        Gets a list of all the measures.

        :return: List of all the measures.
        """
        return get_measures(cls)

    @lru_cache(maxsize=None)
    def get(self, measure):
        """
        Gets the specified measure.

        :param measure: Name of measure.
        :return: Measure.
        """
        return getattr(self, measure)

    @lru_cache(maxsize=None)
    def get_measures(self):
        """
        Gets a list of all the measures.

        :return: List of all the measures.
        """
        return get_measures(self.__class__)


def get_measures(clazz):
    """
    Gets all the measures of a clazz.

    :param clazz: Clazz.
    :return: List of measures.
    """
    from itertools import chain

    is_property = lambda v: isinstance(v, property)
    is_public = lambda n: not n.startswith('_')
    is_valid = lambda n, v: is_public(n) and is_property(v)

    measures = sorted(list(chain(*[[n for n, v in vars(c).items() if is_valid(n, v)] for c in clazz.__mro__])))

    return measures


def corr(df, f):
    """
    Computes the pairwise association matrix. ALL fields/columns must be the same type and so that the specified field
    ``f`` will be able to compute the pairwise associations.

    :param df: Pandas data frame.
    :param f: Callable function; e.g. lambda a, b: categorical_categorical(a, b, measure='phi')
    """
    fields = list(df.columns)
    idx_map = {f: i for i, f in enumerate(fields)}
    associations = ((idx_map[a], idx_map[b], f(df[a], df[b])) for a, b in combinations(fields, 2))

    n = df.shape[1]
    mat = np.empty((n, n))
    for i, j, a in associations:
        mat[i, j] = mat[j, i] = a

    df = pd.DataFrame(mat, columns=fields, index=fields)
    return df
