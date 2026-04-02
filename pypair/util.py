from abc import ABC
from functools import lru_cache
from itertools import combinations
from typing import Any
import warnings

import numpy as np
import pandas as pd


class UndefinedMeasureError(ValueError):
    """Raised when a measure is undefined for the provided data."""


def _has_non_finite_numeric(value) -> bool:
    if isinstance(value, (str, bytes)):
        return False

    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False

    return arr.size > 0 and not np.all(np.isfinite(arr))


def _measure_error_message(measure: str, owner: object | str, detail: str, context: str | None = None) -> str:
    owner_name = owner if isinstance(owner, str) else owner.__class__.__name__
    message = f"Measure '{measure}' is undefined for {owner_name}: {detail}"
    if context is not None:
        message = f"{message} ({context})"
    return message


def raise_undefined_measure(measure: str, owner: object | str, detail: str, context: str | None = None) -> None:
    raise UndefinedMeasureError(_measure_error_message(measure, owner, detail, context=context))


def compute_all_measures(computer, context: str | None = None):
    measures = {}
    for measure in computer.measures():
        try:
            measures[measure] = computer.get(measure)
        except UndefinedMeasureError as exc:
            if context is None:
                raise
            raise UndefinedMeasureError(f"{exc} ({context})") from exc
    return measures


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
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                value = getattr(self, measure)
        except UndefinedMeasureError:
            raise
        except ZeroDivisionError as exc:
            raise UndefinedMeasureError(_measure_error_message(measure, self, "division by zero")) from exc
        except FloatingPointError as exc:
            detail = str(exc) or exc.__class__.__name__
            raise UndefinedMeasureError(_measure_error_message(measure, self, detail)) from exc
        except RuntimeWarning as exc:
            detail = str(exc) or exc.__class__.__name__
            raise UndefinedMeasureError(_measure_error_message(measure, self, detail)) from exc
        except ValueError as exc:
            if "math domain error" in str(exc):
                raise UndefinedMeasureError(_measure_error_message(measure, self, "math domain error")) from exc
            raise

        if _has_non_finite_numeric(value):
            raise UndefinedMeasureError(_measure_error_message(measure, self, "non-finite result"))

        return value

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
