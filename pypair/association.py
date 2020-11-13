from math import sqrt

import pandas as pd
from scipy.stats import norm

from pypair.table import BinaryTable, CategoricalTable
from itertools import chain
from functools import reduce


def rank_biserial(b, c, b_0=0, b_1=1):
    """
    Computes the rank-biserial correlation between a binary variable :math:`X` and a continuous variable :math:`Y`.

    :math:`r_r = \\frac{2 \\times (Y_1 - Y_0)}{n}`

    Where

    - :math:`Y_0` is the average of :math:`Y` when :math:`X=0`
    - :math:`Y_1` is the average of :math:`Y` when :math:`X=1`
    - :math:`n` is the total number of data

    :param b: Binary data (iterable).
    :param c: Continuous data (iterable).
    :param b_0: The zero value for the binary data. Default is 0.
    :param b_1: The one value for the binary data. Default is 1.
    :return: Rank-biserial correlation.
    """
    df = pd.DataFrame([(x, y) for x, y in zip(b, c) if pd.notna(x)], columns=['b', 'c'])

    n = df.shape[0]

    y_0 = df[df.b == b_0].c.mean()
    y_1 = df[df.b == b_1].c.mean()

    r = 2 * (y_1 - y_0) / n
    return r


def point_biserial(b, c, b_0=0, b_1=1):
    """
    Computes the `point-biserial correlation coefficient <https://www.andrews.edu/~calkins/math/edrm611/edrm13.htm>`_
    between a binary variable :math:`X` and a continuous variable :math:`Y`.

    :math:`r_{\\mathrm{pb}} = \\frac{(Y_1 - Y_0) \\sqrt pq}{\\sigma_Y}`

    Where

    - :math:`Y_0` is the average of :math:`Y` when :math:`X=0`
    - :math:`Y_1` is the average of :math:`Y` when :math:`X=1`
    - :math:`\\sigma_Y`` is the standard deviation of :math:`Y`
    - :math:`p` is :math:`P(X=1)`
    - :math:`q` is :math:`1 - p`

    :param b: Binary data (iterable).
    :param c: Continuous data (iterable).
    :param b_0: The zero value for the binary data. Default is 0.
    :param b_1: The one value for the binary data. Default is 1.
    :return: Point-biserial correlation coefficient.
    """
    df = pd.DataFrame([(x, y) for x, y in zip(b, c) if pd.notna(x)], columns=['b', 'c'])

    n = df.shape[0]
    p = df[df.b == b_1].shape[0] / n
    q = 1.0 - p

    y_0 = df[df.b == b_0].c.mean()
    y_1 = df[df.b == b_1].c.mean()
    std = df.c.std()

    r = (y_1 - y_0) * sqrt(p * q) / std
    return r


def biserial(b, c, b_0=0, b_1=1):
    """
    Computes the biserial correlation between a binary and continuous variable. The biserial correlation
    :math:`r_b` can be computed from the point-biserial correlation :math:`r_{\\mathrm{pb}}` as follows.

    :math:`r_b = \\frac{r_{\\mathrm{pb}}}{h} \\sqrt pq`

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

    :param b: Binary data (iterable).
    :param c: Continuous data (iterable).
    :param b_0: The zero value for the binary data. Default is 0.
    :param b_1: The one value for the binary data. Default is 1.
    :return: Biserial correlation coefficient.
    """
    df = pd.DataFrame([(x, y) for x, y in zip(b, c) if pd.notna(x)], columns=['b', 'c'])

    n = df.shape[0]
    p = df[df.b == b_1].shape[0] / n
    q = 1.0 - p

    y_0 = df[df.b == b_0].c.mean()
    y_1 = df[df.b == b_1].c.mean()
    std = df.c.std()

    r_pb = (y_1 - y_0) * sqrt(p * q) / std

    y = norm.pdf(norm.ppf(q))
    r_b = r_pb * sqrt(p * q) / y

    return r_b


def phi_coefficient(a, b):
    """
    Computes the :math:`\\phi` coefficient.

    :param a: Categorical data (iterable).
    :param b: Categorical data (iterable).
    :return: :math:`\\phi`.
    """
    return CategoricalTable(a, b).phi


def cramer_v(a, b, a_0=0, a_1=1, b_0=0, b_1=1):
    """
    Computes Cramer's V. Only for two binary variables.

    :param a: Binary data (iterable).
    :param b: Binary data (iterable).
    :param a_0: Zero value for a. Default is 0.
    :param a_1: One value for a. Default is 1.
    :param b_0: Zero value for b. Default is 0.
    :param b_1: One value for b. Default is 1.
    :return: Cramer's V.
    """
    return BinaryTable(a, b, a_0=a_0, a_1=a_1, b_0=b_0, b_1=b_1).cramer_v


def goodman_kruskal_lambda(a, b):
    """
    Computes Goodman-Kruskal's :math:`\\lambda`. Requires two categorical variables.

    :param a: Categorical data (iterable).
    :param b: Categorical data (iterable).
    :return: :math:`\\lambda`.
    """
    return CategoricalTable(a, b).goodman_kruskal_lambda


def tetrachoric(a, b, a_0=0, a_1=1, b_0=0, b_1=1):
    """
    Computes the tetrachoric correlation. Only for two binary variables.

    :param a: Binary data (iterable).
    :param b: Binary data (iterable).
    :param a_0: Zero value for a. Default is 0.
    :param a_1: One value for a. Default is 1.
    :param b_0: Zero value for b. Default is 0.
    :param b_1: One value for b. Default is 1.
    :return: Tetrachoric correlation.
    """
    return BinaryTable(a, b, a_0=a_0, a_1=a_1, b_0=b_0, b_1=b_1).tetrachoric_correlation


def somers_d(x, y):
    """
    Compute `Somer's D <https://en.wikipedia.org/wiki/Somers%27_D>`_ for two continuous
    variables.

    :param x: Continuous data (iterable).
    :param y: Continuous data (iterable).
    :return: Somer's D.
    """
    def get_concordance(p1, p2):
        x_i, y_i = p1
        x_j, y_j = p2
        if x_i > x_j and y_i > y_j:
            return 0, 0, 1

        if x_i > x_j and y_i < y_j:
            return 1, 0, 0

        if x_i < x_j and y_i > y_j:
            return 1, 0, 0

        return 0, 1, 0

    is_valid = lambda a, b: a is not None and b is not None
    data = [(a, b) for a, b in zip(x, y) if is_valid(a, b)]
    results = ([get_concordance(p1, p2) for j, p2 in enumerate(data) if j > i] for i, p1 in enumerate(data))
    results = chain(*results)
    add_tup = lambda tup1, tup2: (tup1[0] + tup2[0], tup1[1] + tup2[1], tup1[2] + tup2[2])
    results = reduce(lambda tup1, tup2: add_tup(tup1, tup2), results)
    n_d, n_n, n_c = results
    n = len(data)

    t = (n_c - n_d) / (n * (n - 1) / 2)
    return t

