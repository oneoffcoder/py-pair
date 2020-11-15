from functools import reduce
from itertools import chain
from math import sqrt

import pandas as pd
from scipy.stats import norm

from pypair.table import BinaryTable, CategoricalTable


class Concordance(object):
    """
    Stores the concordance, discordant and tie counts.
    """
    def __init__(self, d, t, t_x, t_y, c):
        """
        ctor.

        :param d: Discordant.
        :param t: Tie.
        :param t_x: Tie on X.
        :param t_y: Tie on Y.
        :param c: Concordant.
        """
        self.d = d
        self.t = t
        self.t_x = t_x
        self.t_y = t_y
        self.c = c

    def __add__(self, other):
        d = self.d + other.d
        t = self.t + other.t
        t_x = self.t_x + other.t_x
        t_y = self.t_y + other.t_y
        c = self.c + other.c
        return Concordance(d, t, t_x, t_y, c)


def rank_biserial(b, c, b_0=0, b_1=1):
    """
    Computes the rank-biserial correlation between a binary variable :math:`X` and a continuous variable :math:`Y`.

    :math:`r_r = \\frac{2 (Y_1 - Y_0)}{n}`

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

    :math:`r_{\\mathrm{pb}} = \\frac{(Y_1 - Y_0) \\sqrt{pq}}{\\sigma_Y}`

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
    Computes Goodman-Kruskal's :math:`\\lambda_{A|B}`. Requires two categorical variables `A` and `B`.
    :math:`\\lambda_{A|B}` is the proportional reduction in error (PRE) of `A` knowing (given) `B`.
    In general, :math:`\\lambda_{A|B} \\neq \\lambda_{B|A}`, and so this association measure
    is `asymmetric`.

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
    return BinaryTable(a, b, a_0=a_0, a_1=a_1, b_0=b_0, b_1=b_1).tetrachoric


def __get_concordance(x, y):
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

    :param x: Continuous variable (iterable).
    :param y: Continuous variable (iterable).
    :return: Concordance(D, T, T_X, T_Y, C), n.
    """

    def get_concordance(p1, p2):
        x_i, y_i = p1
        x_j, y_j = p2

        d = 0
        t = 0
        t_x = 0
        t_y = 0
        c = 0

        r = (x_j - x_i) * (y_j - y_i)

        if r > 0:
            c = 1
        elif r < 0:
            d = 1
        else:
            t = 1

        if x_i == x_j:
            t_x = 1

        if y_i == y_j:
            t_y = 1

        return Concordance(d, t, t_x, t_y, c)

    is_valid = lambda a, b: a is not None and b is not None
    data = [(a, b) for a, b in zip(x, y) if is_valid(a, b)]
    results = ((get_concordance(p1, p2) for j, p2 in enumerate(data) if j > i) for i, p1 in enumerate(data))
    results = chain(*results)
    concordance = reduce(lambda c1, c2: c1 + c2, results)
    n = len(data)
    return concordance, n


def kendall_tau(x, y):
    """
    Kendall's :math:`\\tau` is defined as follows.

    :math:`\\tau = \\frac{C - D}{{{n}\\choose{2}}}`

    Where

    - :math:`C` is the number of concordant pairs
    - :math:`D` is the number of discordant pairs
    - :math:`n` is the sample size

    :param x: Continuous data (iterable).
    :param y: Continuous data (iterable).
    :return: :math:`\\tau`.
    """
    c, n = __get_concordance(x, y)
    t = (c.c - c.d) / (n * (n - 1) / 2)
    return t


def somers_d(x, y):
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

    :param x: Continuous data (iterable).
    :param y: Continuous data (iterable).
    :return: :math:`d_{X \\cdot Y}`, :math:`d_{Y \\cdot X}`.
    """
    c, n = __get_concordance(x, y)

    p_d = c.d / n
    p_tx = c.t_x / n
    p_ty = c.t_y / n
    p_c = c.c / n

    d_yx = (p_c - p_d) / (p_c + p_d + p_ty)
    d_xy = (p_c - p_d) / (p_c + p_d + p_tx)

    return d_yx, d_xy


def goodman_kruskal_gamma(x, y):
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

    :param x: Continuous data (iterable).
    :param y: Continuous data (iterable).
    :return: :math:`\\gamma`.
    """
    c, n = __get_concordance(x, y)
    p_d, p_t, p_c = c.d / n, c.t / n, c.c / n
    gamma = (p_c - p_d) / (1 - p_t)
    return gamma
