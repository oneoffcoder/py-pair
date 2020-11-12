from math import sqrt

import pandas as pd
from scipy.stats import norm

from pypair.table import BinaryTable, CategoricalTable


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
    Computes the biserial correlation between a binary and continuous variable.

    References

    - https://www.statisticshowto.com/point-biserial-correlation/
    - https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Point-Biserial_and_Biserial_Correlations.pdf
    - https://www.real-statistics.com/correlation/biserial-correlation/
    - https://support.microsoft.com/en-us/office/norm-s-dist-function-1e787282-3832-4520-a9ae-bd2a8d99ba88
    - https://support.microsoft.com/en-us/office/norm-s-inv-function-d6d556b4-ab7f-49cd-b526-5a20918452b1
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    - https://stackoverflow.com/questions/20626994/how-to-calculate-the-inverse-of-the-normal-cumulative-distribution-function-in-p

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
    return BinaryTable(a, b).cramer_v


def goodman_kruskal_lambda(a, b):
    """
    Computes Goodman-Kruskal's :math:`\\lambda`. Requires two categorical variables.

    :param a: Categorical data (iterable).
    :param b: Categorical data (iterable).
    :return: :math:`\\lambda`.
    """
    return CategoricalTable(a, b).goodman_kruskal_lambda
