from __future__ import annotations

from functools import lru_cache
from math import sqrt

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from pypair.typing import ArrayLike1D, NumericArrayLike1D
from pypair.util import MeasureMixin, to_numpy


class BiserialMixin(object):
    r"""Biserial computations based off of :math:`n, p, q, y_0, y_1, \sigma`."""

    @property
    @lru_cache(maxsize=None)
    def __params(self) -> tuple[int, float, float, float, float, float]:
        return self._n, self._p, self._q, self._y_0, self._y_1, self._std

    @property
    @lru_cache(maxsize=None)
    def biserial(self) -> float:
        n, p, q, y_0, y_1, std = self.__params
        r_pb = (y_1 - y_0) * sqrt(p * q) / std
        y = norm.pdf(norm.ppf(q))
        return r_pb * sqrt(p * q) / y

    @property
    @lru_cache(maxsize=None)
    def point_biserial(self) -> float:
        n, p, q, y_0, y_1, std = self.__params
        return (y_1 - y_0) * sqrt(p * q) / std

    @property
    @lru_cache(maxsize=None)
    def rank_biserial(self) -> float:
        n, p, q, y_0, y_1, std = self.__params
        return 2 * (y_1 - y_0) / n


class Biserial(MeasureMixin, BiserialMixin, object):
    """Biserial association between a binary and continuous variable."""

    def __init__(self, b: ArrayLike1D, c: NumericArrayLike1D, b_0: object = 0, b_1: object = 1) -> None:
        b_arr = to_numpy(b)
        c_arr = to_numpy(c, dtype=float)

        mask = (~pd_isna(b_arr)) & (~np.isnan(c_arr))
        b_clean = b_arr[mask]
        c_clean = c_arr[mask]

        n = b_clean.size
        if n == 0:
            raise ValueError("No valid samples available to compute biserial statistics.")

        b1_mask = b_clean == b_1
        b0_mask = b_clean == b_0

        p = float(np.sum(b1_mask)) / n
        y_0 = float(np.mean(c_clean[b0_mask]))
        y_1 = float(np.mean(c_clean[b1_mask]))
        std = float(np.std(c_clean, ddof=1))

        self._n = n
        self._p = p
        self._q = 1.0 - p
        self._y_0 = y_0
        self._y_1 = y_1
        self._std = std


class BiserialStats(MeasureMixin, BiserialMixin, object):
    """Computes biserial stats."""

    def __init__(self, n: int, p: float, y_0: float, y_1: float, std: float) -> None:
        self._n = n
        self._p = p
        self._q = 1.0 - p
        self._y_0 = y_0
        self._y_1 = y_1
        self._std = std


def pd_isna(values: ArrayLike1D) -> npt.NDArray[np.bool_]:
    # lightweight NA handling without requiring pandas internally
    try:
        import pandas as pd

        return pd.isna(values)
    except Exception:
        return np.equal(values, None)
