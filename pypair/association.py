from functools import reduce
from itertools import combinations

from pypair.biserial import Biserial
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


def binary_binary(a, b, measure='chisq', a_0=0, a_1=1, b_0=0, b_1=1):
    """
    Gets the binary-binary association.

    :param a: Binary variable (iterable).
    :param b: Binary variable (iterable).
    :param measure: Measure. Default is `chisq`.
    :param a_0: The a zero value. Default 0.
    :param a_1: The a one value. Default 1.
    :param b_0: The b zero value. Default 0.
    :param b_1: The b one value. Default 1.
    :return: Measure.
    """
    if measure not in BinaryTable.get_measures():
        raise ValueError(f'{measure} is not a valid association measure.')
    return BinaryTable(a, b, a_0=a_0, a_1=a_1, b_0=b_0, b_1=b_1).get(measure)


def categorical_categorical(a, b, measure='chisq', a_vals=None, b_vals=None):
    """
    Gets the categorical-categorical association.

    :param a: Categorical variable (iterable).
    :param b: Categorical variable (iterable).
    :param measure: Measure. Default is `chisq`.
    :param a_vals: The unique values in `a`.
    :param b_vals: The unique values in `b`.
    :return: Measure.
    """
    if measure not in CategoricalTable.get_measures():
        raise ValueError(f'{measure} is not a valid association measure.')
    return CategoricalTable(a, b, a_vals=a_vals, b_vals=b_vals).get(measure)


def binary_continuous(b, c, measure='biserial', b_0=0, b_1=1):
    """
    Gets the binary-continuous association.

    :param b: Binary variable (iterable).
    :param c: Continuous variable (iterable).
    :param measure: Measure. Default is `biserial`.
    :param b_0: Value when `b` is zero. Default 0.
    :param b_1: Value when `b` is one. Default is 1.
    :return: Measure.
    """
    if measure not in Biserial.get_measures():
        raise ValueError(f'{measure} is not a valid association measure.')
    return Biserial(b, c, b_0=b_0, b_1=b_1).get(measure)


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
    results = combinations(data, 2)
    results = map(lambda tup: get_concordance(tup[0], tup[1]), results)
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
