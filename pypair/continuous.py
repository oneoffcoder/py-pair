from __future__ import annotations

from functools import lru_cache
from math import sqrt

import numpy as np
import numpy.typing as npt
from scipy.stats import pearsonr, spearmanr, kendalltau, f_oneway, kruskal, linregress
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from pypair.typing import ArrayLike1D, ConcordanceCounts, NumericArrayLike1D
from pypair.util import MeasureMixin, to_numpy


class Continuous(MeasureMixin, object):
    def __init__(self, a: NumericArrayLike1D, b: NumericArrayLike1D) -> None:
        """
        ctor.

        :param a: Continuous variable (iterable).
        :param b: Continuous variable (iterable).
        """
        self.__a = a
        self.__b = b

    @property
    @lru_cache(maxsize=None)
    def pearson(self) -> tuple[float, float]:
        """
        `Pearson's r <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html>`_.

        :return: Pearson's r, p-value.
        """
        return pearsonr(self.__a, self.__b)

    @property
    @lru_cache(maxsize=None)
    def spearman(self) -> tuple[float, float]:
        """
        `Spearman's r <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.spearmanr.html>`_.

        :return: Spearman's r, p-value.
        """
        r = spearmanr(self.__a, self.__b)
        return r.correlation, r.pvalue

    @property
    @lru_cache(maxsize=None)
    def kendall(self) -> tuple[float, float]:
        """
        `Kendall's tau <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.kendalltau.html>`_.

        :return: Kendall's tau, p-value.
        """
        r = kendalltau(self.__a, self.__b)
        return r.correlation, r.pvalue

    @property
    @lru_cache(maxsize=None)
    def regression(self) -> tuple[float, float]:
        """
        `Line regression <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.linregress.html>`_.

        :return: Coefficient, p-value
        """
        slope, intercept, r_value, p_value, std_err = linregress(self.__a, self.__b)
        return r_value, p_value


class CorrelationRatio(MeasureMixin, object):
    """
    `Correlation ratio <https://en.wikipedia.org/wiki/Correlation_ratio>`_.

    """

    def __init__(self, x: ArrayLike1D, y: NumericArrayLike1D) -> None:
        """
        ctor.

        :param x: Categorical variable (iterable).
        :param y: Continuous variable (iterable).
        """
        x_arr = to_numpy(x)
        y_arr = to_numpy(y, dtype=float)

        mask = (~pd_isna(x_arr)) & (~np.isnan(y_arr))
        self.__x = x_arr[mask]
        self.__y = y_arr[mask]

        if self.__x.size == 0:
            raise ValueError("No valid samples available to compute categorical-continuous statistics.")

    @property
    @lru_cache(maxsize=None)
    def __mean(self) -> float:
        """
        Gets the mean of :math:`\\bar{y}`.

        :return: :math:`\\bar{y}`.
        """
        return float(np.mean(self.__y))

    @property
    @lru_cache(maxsize=None)
    def __sigma_cat(self) -> float:
        """
        Gets :math:`\\sigma_{\\bar{y}}^2`

        :return: :math:`\\sigma_{\\bar{y}}^2`.
        """
        y = self.__mean
        sigma = 0.0
        for value in np.unique(self.__x):
            grp = self.__y[self.__x == value]
            sigma += float(grp.size) * (float(np.mean(grp)) - y) ** 2
        return sigma

    @property
    def __sigma_sam(self) -> float:
        """
        Gets :math:`\\sigma_{y}^2`

        :return: :math:`\\sigma_{y}^2`.
        """
        y = self.__mean
        return float(np.sum((self.__y - y) ** 2))

    @property
    @lru_cache(maxsize=None)
    def eta_squared(self) -> float:
        """
        Gets :math:`\\eta^2 = \\frac{\\sigma_{\\bar{y}}^2}{\\sigma_{y}^2}`

        :return: :math:`\\eta^2`.
        """
        sigma_cat = self.__sigma_cat
        sigma_sam = self.__sigma_sam
        eta = sigma_cat / sigma_sam
        return eta

    @property
    @lru_cache(maxsize=None)
    def eta(self) -> float:
        """
        Gets :math:`\\eta`.

        :returns: :math:`\\eta`.
        """
        return sqrt(self.eta_squared)

    @property
    @lru_cache(maxsize=None)
    def anova(self) -> tuple[float, float]:
        """
        Computes an `ANOVA test <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.f_oneway.html>`_.

        :return: F-statistic, p-value.
        """
        samples = [self.__y[self.__x == value] for value in np.unique(self.__x)]
        r = f_oneway(*samples)
        return r.statistic, r.pvalue

    @property
    @lru_cache(maxsize=None)
    def kruskal(self) -> tuple[float, float]:
        """
        Computes the `Kruskal-Wallis H-test <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.kruskal.html>`_.

        :return: H-statistic, p-value.
        """
        samples = [self.__y[self.__x == value] for value in np.unique(self.__x)]
        r = kruskal(*samples)
        return r.statistic, r.pvalue

    @property
    @lru_cache(maxsize=None)
    def silhouette(self) -> float:
        """
        `Silhouette coefficient <https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient>`_.

        :return: Silhouette coefficient.
        """
        return silhouette_score(self.__y.reshape(-1, 1), self.__x)

    @property
    @lru_cache(maxsize=None)
    def davies_bouldin(self) -> float:
        """
        `Davies-Bouldin Index <https://scikit-learn.org/stable/modules/clustering.html#davies-bouldin-index>`_.

        :return: Davies-Bouldin Index.
        """
        return davies_bouldin_score(self.__y.reshape(-1, 1), self.__x)

    @property
    @lru_cache(maxsize=None)
    def calinski_harabasz(self) -> float:
        """
        `Calinski-Harabasz Index <https://scikit-learn.org/stable/modules/clustering.html#calinski-harabasz-index>`_.

        :return: Calinski-Harabasz Index.
        """
        return calinski_harabasz_score(self.__y.reshape(-1, 1), self.__x)


class ConcordanceMixin(object):
    @property
    @lru_cache(maxsize=None)
    def __counts(self) -> ConcordanceCounts:
        return self._d, self._t_xy, self._t_x, self._t_y, self._c, self._n

    @property
    @lru_cache(maxsize=None)
    def __probs(self) -> tuple[float, float, float, float, float, int]:
        n = self._n
        return self._d / n, self._t_xy / n, self._t_x / n, self._t_y / n, self._c / n, n

    @property
    @lru_cache(maxsize=None)
    def kendall_tau(self) -> float:
        """
        Kendall's :math:`\\tau` is defined as follows.

        :math:`\\tau = \\frac{C - D}{{{n}\\choose{2}}}`

        Where

        - :math:`C` is the number of concordant pairs
        - :math:`D` is the number of discordant pairs
        - :math:`n` is the sample size

        :return: :math:`\\tau`.
        """
        d, t_xy, t_x, t_y, c, n = self.__counts
        t = (c - d) / (n * (n - 1) / 2)
        return t

    @property
    @lru_cache(maxsize=None)
    def somers_d(self) -> tuple[float, float]:
        """
        Computes `Somers' d <https://en.wikipedia.org/wiki/Somers%27_D>`_ for two continuous
        variables. Note that Somers' d is defined for :math:`d_{X \\cdot Y}` and :math:`d_{Y \\cdot X}`
        and in general :math:`d_{X \\cdot Y} \\neq d_{Y \\cdot X}`.

        - :math:`d_{Y \\cdot X} = \\frac{\\pi_c - \\pi_d}{\\pi_c + \\pi_d + \\pi_t^Y}`
        - :math:`d_{X \\cdot Y} = \\frac{\\pi_c - \\pi_d}{\\pi_c + \\pi_d + \\pi_t^X}`

        Where

        - :math:`\\pi_c = \\frac{C}{n}`
        - :math:`\\pi_d = \\frac{D}{n}`
        - :math:`\\pi_t^X = \\frac{T^X}{n}`
        - :math:`\\pi_t^Y = \\frac{T^Y}{n}`
        - :math:`C` is the number of concordant pairs
        - :math:`D` is the number of discordant pairs
        - :math:`T^X` is the number of ties on :math:`X`
        - :math:`T^Y` is the number of ties on :math:`Y`
        - :math:`n` is the sample size

        :return: :math:`d_{X \\cdot Y}`, :math:`d_{Y \\cdot X}`.
        """
        p_d, p_txy, p_tx, p_ty, p_c, n = self.__probs

        d_yx = (p_c - p_d) / (p_c + p_d + p_ty)
        d_xy = (p_c - p_d) / (p_c + p_d + p_tx)

        return d_yx, d_xy

    @property
    @lru_cache(maxsize=None)
    def goodman_kruskal_gamma(self) -> float:
        """
        Goodman-Kruskal :math:`\\gamma` is like Somer's D. It is defined as follows.

        :math:`\\gamma = \\frac{\\pi_c - \\pi_d}{1 - \\pi_t}`

        Where

        - :math:`\\pi_c = \\frac{C}{n}`
        - :math:`\\pi_d = \\frac{D}{n}`
        - :math:`\\pi_t = \\frac{T}{n}`
        - :math:`C` is the number of concordant pairs
        - :math:`D` is the number of discordant pairs
        - :math:`T` is the number of ties
        - :math:`n` is the sample size

        :return: :math:`\\gamma`.
        """
        p_d, p_txy, p_tx, p_ty, p_c, n = self.__probs
        p_t = p_txy + p_tx + p_ty

        gamma = (p_c - p_d) / (1 - p_t)

        return gamma


class Concordance(MeasureMixin, ConcordanceMixin, object):
    """
    Concordance for continuous and ordinal data.
    """

    def __init__(self, x: NumericArrayLike1D, y: NumericArrayLike1D) -> None:
        """
        ctor.

        :param x: Continuous or ordinal data (iterable).
        :param y: Continuous or ordinal data (iterable).
        """
        d, t_xy, t_x, t_y, c, n = Concordance.__to_counts(x, y)
        self._d = d
        self._t_xy = t_xy
        self._t_x = t_x
        self._t_y = t_y
        self._c = c
        self._n = n

    @staticmethod
    def __to_counts(x: NumericArrayLike1D, y: NumericArrayLike1D) -> ConcordanceCounts:
        """
        Gets the count of concordance, discordance or tie. Two pairs of variables :math:`(X_i, Y_i)`
        and :math:`(X_j, Y_j)` are

        - concordant if :math:`X_i < X_j` and :math:`Y_i < Y_j` **or** :math:`X_i > X_j` and :math:`Y_i > Y_j`,
        - discordant if :math:`X_i < X_j` and :math:`Y_i > Y_j` **or** :math:`X_i > X_j` and :math:`Y_i < Y_j`, and
        - tied if :math:`X_i = X_j` and :math:`Y_i = Y_j`.

        Equivalently.

        - concordant if :math:`(X_j - X_i)(Y_j - Y_i) > 0`
        - discordant if :math:`(X_j - X_i)(Y_j - Y_i) < 0`
        - tied if :math:`(X_j - X_i)(Y_j - Y_i) = 0`

        Any two pairs of observations are necessarily concordant, discordant or tied.

        :return: Counts(D, T_XY, T_X, T_Y, C), n.
        """

        def is_valid(a: object, b: object) -> bool:
            return a is not None and b is not None

        x_vals = []
        y_vals = []
        for a, b in zip(x, y):
            if is_valid(a, b):
                x_vals.append(a)
                y_vals.append(b)

        n = len(x_vals)
        if n < 2:
            raise ValueError("At least two valid paired samples are required to compute concordance statistics.")

        d = 0
        t_xy = 0
        t_x = 0
        t_y = 0
        c = 0

        for i in range(n - 1):
            x_i = x_vals[i]
            y_i = y_vals[i]
            for j in range(i + 1, n):
                x_j = x_vals[j]
                y_j = y_vals[j]
                r = (x_j - x_i) * (y_j - y_i)

                if r > 0:
                    c += 1
                elif r < 0:
                    d += 1
                elif x_i == x_j and y_i == y_j:
                    t_xy += 1
                elif x_i == x_j:
                    t_x += 1
                elif y_i == y_j:
                    t_y += 1

        return d, t_xy, t_x, t_y, c, n


class ConcordanceStats(MeasureMixin, ConcordanceMixin):
    """
    Computes concordance stats.
    """

    def __init__(self, d: int, t_xy: int, t_x: int, t_y: int, c: int, n: int) -> None:
        """
        ctor.

        :param d: Number of discordant pairs.
        :param t_xy: Number of ties on XY pairs.
        :param t_x: Number of ties on X pairs.
        :param t_y: Number of ties on Y pairs.
        :param c: Number of concordant pairs.
        :param n: Total number of pairs.
        """
        self._d = d
        self._t_xy = t_xy
        self._t_x = t_x
        self._t_y = t_y
        self._t_c = c
        self._c = c
        self._n = n


def pd_isna(values: ArrayLike1D) -> npt.NDArray[np.bool_]:
    try:
        import pandas as pd

        return pd.isna(values)
    except Exception:
        return np.equal(values, None)
