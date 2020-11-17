from functools import lru_cache
from math import sqrt

import pandas as pd
from scipy.stats import norm

from pypair.util import MeasureMixin


class Biserial(MeasureMixin, object):
    """
    Biserial association between a binary and continuous variable.
    """

    def __init__(self, b, c, b_0=0, b_1=1):
        """
        ctor.

        :param b: Binary variable (iterable).
        :param c: Continuous variable (iterable).
        :param b_0: Value for b is zero. Default 0.
        :param b_1: Value for b is one. Default 1.
        """
        self.__df = pd.DataFrame([(x, y) for x, y in zip(b, c) if pd.notna(x)], columns=['b', 'c'])
        self.__b_0 = b_0
        self.__b_1 = b_1

    @property
    @lru_cache(maxsize=None)
    def __params(self):
        """
        Gets the parameters associated with the data.

        - n: total
        - p: P(b=0)
        - q: 1 - p
        - y_0: average of c when b=0
        - y_1: average of c when b=1
        - std: standard deviation of c

        :return: n, p, q, y_0, y_1, std
        """
        n = self.__df.shape[0]
        p = self.__df[self.__df.b == self.__b_1].shape[0] / n
        q = 1.0 - p

        y_0 = self.__df[self.__df.b == self.__b_0].c.mean()
        y_1 = self.__df[self.__df.b == self.__b_1].c.mean()
        std = self.__df.c.std()
        return n, p, q, y_0, y_1, std

    @property
    @lru_cache(maxsize=None)
    def biserial(self):
        """
        Computes the biserial correlation between a binary and continuous variable. The biserial correlation
        :math:`r_b` can be computed from the point-biserial correlation :math:`r_{\\mathrm{pb}}` as follows.

        :math:`r_b = \\frac{r_{\\mathrm{pb}}}{h} \\sqrt{pq}`

        The tricky thing to explain is the :math:`h` parameter. :math:`h` is defined as the
        height of the standard normal distribution at z, where :math:`P(z'<z) = q` and :math:`P(z’>z) = p`.
        The way to get :math:`h` in practice is take the inverse standard normal of :math:`q`, and
        then take the standard normal probability of that result. Using Scipy `norm.pdf(norm.ppf(q))`.

        References

        - `Point-Biserial Correlation & Biserial Correlation: Definition, Examples <https://www.statisticshowto.com/point-biserial-correlation/>`_
        - `Point-Biserial and Biserial Correlations <https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Point-Biserial_and_Biserial_Correlations.pdf>`_
        - `Real Statistics Using Excel <https://www.real-statistics.com/correlation/biserial-correlation/>`_
        - `NORM.S.DIST function <https://support.microsoft.com/en-us/office/norm-s-dist-function-1e787282-3832-4520-a9ae-bd2a8d99ba88>`_
        - `NORM.S.INV function <https://support.microsoft.com/en-us/office/norm-s-inv-function-d6d556b4-ab7f-49cd-b526-5a20918452b1>`_
        - `scipy.stats.norm <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html>`_
        - `How to calculate the inverse of the normal cumulative distribution function in python? <https://stackoverflow.com/questions/20626994/how-to-calculate-the-inverse-of-the-normal-cumulative-distribution-function-in-p>`_

        :return: Biserial correlation coefficient.
        """
        n, p, q, y_0, y_1, std = self.__params

        r_pb = (y_1 - y_0) * sqrt(p * q) / std

        y = norm.pdf(norm.ppf(q))
        r_b = r_pb * sqrt(p * q) / y

        return r_b

    @property
    @lru_cache(maxsize=None)
    def point_biserial(self):
        """
        Computes the `point-biserial correlation coefficient <https://www.andrews.edu/~calkins/math/edrm611/edrm13.htm>`_
        between a binary variable :math:`X` and a continuous variable :math:`Y`.

        :math:`r_{\\mathrm{pb}} = \\frac{(Y_1 - Y_0) \\sqrt{pq}}{\\sigma_Y}`

        Where

        - :math:`Y_0` is the average of :math:`Y` when :math:`X=0`
        - :math:`Y_1` is the average of :math:`Y` when :math:`X=1`
        - :math:`\\sigma_Y`` is the standard deviation of :math:`Y`
        - :math:`p` is :math:`P(X=1)`
        - :math:`q` is :math:`1 - p`

        :return: Point-biserial correlation coefficient.
        """
        n, p, q, y_0, y_1, std = self.__params

        r = (y_1 - y_0) * sqrt(p * q) / std
        return r

    @property
    @lru_cache(maxsize=None)
    def rank_biserial(self):
        """
        Computes the rank-biserial correlation between a binary variable :math:`X` and a continuous variable :math:`Y`.

        :math:`r_r = \\frac{2 (Y_1 - Y_0)}{n}`

        Where

        - :math:`Y_0` is the average of :math:`Y` when :math:`X=0`
        - :math:`Y_1` is the average of :math:`Y` when :math:`X=1`
        - :math:`n` is the total number of data

        :return: Rank-biserial correlation.
        """
        n, p, q, y_0, y_1, std = self.__params

        r = 2 * (y_1 - y_0) / n
        return r
