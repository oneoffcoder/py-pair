from functools import reduce, lru_cache
from itertools import combinations
from math import sqrt

import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau, f_oneway, kruskal, linregress
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from pypair.util import MeasureMixin


class ConcordantCounts(object):
    """
    Stores the concordance, discordant and tie counts.
    """

    def __init__(self, d, t_xy, t_x, t_y, c):
        """
        ctor.

        :param d: Discordant.
        :param t_xy: Tie.
        :param t_x: Tie on X.
        :param t_y: Tie on Y.
        :param c: Concordant.
        """
        self.d = d
        self.t_xy = t_xy
        self.t_x = t_x
        self.t_y = t_y
        self.c = c

    def __add__(self, other):
        d = self.d + other.d
        t_xy = self.t_xy + other.t_xy
        t_x = self.t_x + other.t_x
        t_y = self.t_y + other.t_y
        c = self.c + other.c
        return ConcordantCounts(d, t_xy, t_x, t_y, c)


class Continuous(MeasureMixin, object):
    def __init__(self, a, b):
        """
        ctor.

        :param a: Continuous variable (iterable).
        :param b: Continuous variable (iterable).
        """
        self.__a = a
        self.__b = b

    @property
    @lru_cache(maxsize=None)
    def pearson(self):
        """
        `Pearson's r <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html>`_.

        :return: Pearson's r, p-value.
        """
        return pearsonr(self.__a, self.__b)

    @property
    @lru_cache(maxsize=None)
    def spearman(self):
        """
        `Spearman's r <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.spearmanr.html>`_.

        :return: Spearman's r, p-value.
        """
        r = spearmanr(self.__a, self.__b)
        return r.correlation, r.pvalue

    @property
    @lru_cache(maxsize=None)
    def kendall(self):
        """
        `Kendall's tau <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.kendalltau.html>`_.

        :return: Kendall's tau, p-value.
        """
        r = kendalltau(self.__a, self.__b)
        return r.correlation, r.pvalue

    @property
    @lru_cache(maxsize=None)
    def regression(self):
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

    def __init__(self, x, y):
        """
        ctor.

        :param x: Categorical variable (iterable).
        :param y: Continuous variable (iterable).
        """
        is_valid = lambda a, b: a is not None and b is not None
        self.__df = pd.DataFrame([(a, b) for a, b, in zip(x, y) if is_valid(a, b)], columns=['x', 'y'])

    @property
    @lru_cache(maxsize=None)
    def __mean(self):
        """
        Gets the mean of :math:`\\bar{y}`.

        :return: :math:`\\bar{y}`.
        """
        return self.__df.y.mean()

    @property
    @lru_cache(maxsize=None)
    def __sigma_cat(self):
        """
        Gets :math:`\\sigma_{\\bar{y}}^2`

        :return: :math:`\\sigma_{\\bar{y}}^2`.
        """
        stats = self.__df.groupby(['x']).agg(['count', 'mean']).reset_index()
        stats.columns = stats.columns.droplevel(0)
        stats = stats.rename(columns={'': 'x', 'count': 'n_x', 'mean': 'y_x'})
        y = self.__mean

        sigma = sum([r.n_x * (r.y_x - y) ** 2 for _, r in stats.iterrows()])

        return sigma

    @property
    def __sigma_sam(self):
        """
        Gets :math:`\\sigma_{y}^2`

        :return: :math:`\\sigma_{y}^2`.
        """
        y = self.__mean
        sigma = sum((self.__df.y - y) ** 2)

        return sigma

    @property
    @lru_cache(maxsize=None)
    def eta_squared(self):
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
    def eta(self):
        """
        Gets :math:`\\eta`.

        :returns: :math:`\\eta`.
        """
        return sqrt(self.eta_squared)

    @property
    @lru_cache(maxsize=None)
    def anova(self):
        """
        Computes an `ANOVA test <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.f_oneway.html>`_.

        :return: F-statistic, p-value.
        """
        df = self.__df
        samples = [df[df.x == x].y for x in df.x.unique()]
        r = f_oneway(*samples)
        return r.statistic, r.pvalue

    @property
    @lru_cache(maxsize=None)
    def kruskal(self):
        """
        Computes the `Kruskal-Wallis H-test <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.kruskal.html>`_.

        :return: H-statistic, p-value.
        """
        df = self.__df
        samples = [df[df.x == x].y for x in df.x.unique()]
        r = kruskal(*samples)
        return r.statistic, r.pvalue

    @property
    @lru_cache(maxsize=None)
    def silhouette(self):
        """
        `Silhouette coefficient <https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient>`_.

        :return: Silhouette coefficient.
        """
        labels = self.__df.x
        X = self.__df[['y']]
        return silhouette_score(X, labels)

    @property
    @lru_cache(maxsize=None)
    def davies_bouldin(self):
        """
        `Davies-Bouldin Index <https://scikit-learn.org/stable/modules/clustering.html#davies-bouldin-index>`_.

        :return: Davies-Bouldin Index.
        """
        labels = self.__df.x
        X = self.__df[['y']]
        return davies_bouldin_score(X, labels)

    @property
    @lru_cache(maxsize=None)
    def calinski_harabasz(self):
        """
        `Calinski-Harabasz Index <https://scikit-learn.org/stable/modules/clustering.html#calinski-harabasz-index>`_.

        :return: Calinski-Harabasz Index.
        """
        labels = self.__df.x
        X = self.__df[['y']]
        return calinski_harabasz_score(X, labels)


class ConcordanceMixin(object):

    @property
    @lru_cache(maxsize=None)
    def __counts(self):
        return self._d, self._t_xy, self._t_x, self._t_y, self._c, self._n

    @property
    @lru_cache(maxsize=None)
    def __probs(self):
        n = self._n
        return self._d / n, self._t_xy / n, self._t_x / n, self._t_y / n, self._c / n, n

    @property
    @lru_cache(maxsize=None)
    def kendall_tau(self):
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
    def somers_d(self):
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
    def goodman_kruskal_gamma(self):
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

    def __init__(self, x, y):
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
    def __to_counts(x, y):
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

        def get_concordance(p1, p2):
            x_i, y_i = p1
            x_j, y_j = p2

            d = 0
            t_xy = 0
            t_x = 0
            t_y = 0
            c = 0

            r = (x_j - x_i) * (y_j - y_i)

            if r > 0:
                c = 1
            elif r < 0:
                d = 1
            else:
                if x_i == x_j and y_i == y_j:
                    t_xy = 1
                elif x_i == x_j:
                    t_x = 1
                elif y_i == y_j:
                    t_y = 1

            return ConcordantCounts(d, t_xy, t_x, t_y, c)

        is_valid = lambda a, b: a is not None and b is not None
        data = [(a, b) for a, b in zip(x, y) if is_valid(a, b)]
        results = combinations(data, 2)
        results = map(lambda tup: get_concordance(tup[0], tup[1]), results)
        c = reduce(lambda c1, c2: c1 + c2, results)
        n = len(data)
        return c.d, c.t_xy, c.t_x, c.t_y, c.c, n


class ConcordanceStats(MeasureMixin, ConcordanceMixin):
    """
    Computes concordance stats.
    """

    def __init__(self, d, t_xy, t_x, t_y, c, n):
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
