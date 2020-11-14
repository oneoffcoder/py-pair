from abc import ABC
from functools import lru_cache, reduce
from itertools import chain, product
from math import sqrt, log2, pi, log

import pandas as pd
from scipy import stats
from scipy.special import binom


class ContingencyTable(ABC):
    """
    Abstract contingency table. All other tables inherit from this one.
    """

    def __init__(self):
        pass


class CategoricalTable(ContingencyTable):
    """
    Represents a contingency table for categorical variables.

    References

    - `Contingency table <https://en.wikipedia.org/wiki/Contingency_table>`_
    - `More Correlation Coefficients <https://www.andrews.edu/~calkins/math/edrm611/edrm13.htm#TETRA>`_
    """

    def __init__(self, a, b, a_vals=None, b_vals=None):
        """
        ctor. If `a_vals` or `b_vals` are `None`, then the possible
        values will be determined empirically from the data.

        :param a: Iterable list.
        :param b: Iterable list.
        :param a_vals: All possible values in a. Defaults to `None`.
        :param b_vals: All possible values in b. Defaults to `None`.
        """
        df = pd.DataFrame([(x, y) for x, y in zip(a, b)], columns=['a', 'b'])

        if a_vals is None:
            a_vals = sorted(list(set(a)))
        else:
            a_vals = sorted(list(df.a.unique()))

        if b_vals is None:
            b_vals = sorted(list(set(b)))
        else:
            b_vals = sorted(list(df.b.unique()))

        observed = [[df.query(f'a=="{x}" and b=="{y}"').shape[0] for y in b_vals] for x in a_vals]

        n_rows = len(a_vals)
        n_cols = len(b_vals)

        row_marginals = [sum(r) for r in observed]
        col_marginals = [sum([observed[r][c] for r in range(n_rows)]) for c in range(n_cols)]

        n = sum([sum(o) for o in observed])
        get_expected = lambda r, c: r * c / n
        expected = [[get_expected(row_marginals[i], col_marginals[j]) for j, _ in enumerate(b_vals)] for i, _ in
                    enumerate(a_vals)]

        chisq = sum([(o - e) ** 2 / e for o, e in zip(chain(*observed), chain(*expected))])

        p_observed = [[c / n for c in r] for r in observed]
        p_r_marginals = [sum(r) for r in p_observed]
        p_c_marginals = [sum([p_observed[r][c] for r in range(n_rows)]) for c in range(n_cols)]

        self.observed = observed
        self.expected = expected
        self._df = df
        self._chisq = chisq
        self._n = n
        self._a_map = {v: i for i, v in enumerate(a_vals)}
        self._b_map = {v: i for i, v in enumerate(b_vals)}
        self._n_cols = n_cols
        self._n_rows = n_rows
        self._row_marginals = row_marginals
        self._col_marginals = col_marginals
        self._p_observed = p_observed
        self._p_r_marginals = p_r_marginals
        self._p_c_marginals = p_c_marginals

    @lru_cache(maxsize=None)
    def _count(self, a=None, b=None):
        if a is not None and b is not None:
            q = f'a=="{a}" and b=="{b}"'
        elif a is not None and b is None:
            q = f'a=="{a}"'
        elif a is None and b is not None:
            q = f'b=="{b}"'
        else:
            return self._df.shape[0]

        return self._df.query(q).shape[0]

    @property
    def chisq(self):
        """
        The `chi-square statistic <https://en.wikipedia.org/wiki/Chi-square_distribution>`_ :math:`\\chi^2`,
        is defined as follows.

        :math:`\\sum_i \\sum_j \\frac{(O_{ij} - E_{ij})^2}{E_{ij}}`

        In a contingency table, :math:`O_ij` is the observed cell count corresponding to the :math:`i` row
        and :math:`j` column. :math:`E_ij` is the expected cell count corresponding to the :math:`i` row and
        :math:`j` column.

        :math:`E_i = \\frac{N_{i*} N_{*j}}{N}`

        Where :math:`N_{i*}` is the i-th row marginal, :math:`N_{*j}` is the j-th column marginal and
        :math:`N` is the sum of all the values in the contingency cells (or the total size of the data).

        References

        - `Chi-Square (χ2) Statistic Definition <https://www.investopedia.com/terms/c/chi-square-statistic.asp>`_

        :return: Chi-square statistic.
        """
        return self._chisq

    @property
    def chisq_dof(self):
        """
        Returns the degrees of freedom form :math:`\\chi^2`, which is defined as :math:`(R - 1)(C - 1)`,
        where :math:`R` is the number of rows and :math:`C` is the number of columns in a contingency
        table induced by two categorical variables.

        :return: Degrees of freedom.
        """
        return (self._n_rows - 1) * (self._n_cols - 1)

    @property
    def phi(self):
        """
        The `phi coefficient <https://en.wikipedia.org/wiki/Phi_coefficient>`_ :math:`\\phi` is defined
        as :math:`\\sqrt \\frac{\\chi^2}{N}`. For two binary variables, the range will be like canonical
        Pearson :math:`[-1, 1]`, where -1 indicates perfect disagreement, 1 indicates perfect agreement and
        0 indicates no relationship. When at least one variable is not binary (has more than 2 values possible),
        then the upper-bound of :math:`\\phi`` is determined by the distribution of the two variables.

        References

        - `Matthews correlation coefficient <https://en.wikipedia.org/wiki/Matthews_correlation_coefficient>`_

        :return: Phi.
        """
        return sqrt(self.chisq / self._n)

    def __get_cond_entropy(self, a, b, n, reversed=False):
        p_ab = self._df.query(f'a=="{a}" and b=="{b}"').shape[0] / n
        if not reversed:
            p_a = self._df.query(f'a=="{a}"').shape[0] / n
        else:
            p_a = self._df.query(f'b=="{b}"').shape[0] / n
        c_ba = p_ab / p_a
        return p_ab * log(c_ba)

    @property
    def uncertainty_coefficient(self):
        """
        The `uncertainty coefficient <https://en.wikipedia.org/wiki/Uncertainty_coefficient>`_ :math:`U(X|Y)`
        for two variables :math:`X` and :math:`Y` is defined as follows.

        :math:`U(X|Y) = \\frac{I(X;Y)}{H(X)}`

        Where,

        - :math:`H(X) = -\\sum_x P(x) \\log P(x)`
        - :math:`I(X;Y) = \\sum_y \\sum_x P(x, y) \\log \\frac{P(x, y)}{P(x) P(y)}`

        :math:`H(X)` is called the entropy of :math:`X` and :math:`I(X;Y)` is the mutual information
        between :math:`X` and :math:`Y`. Note that :math:`I(X;Y) < H(X)` and both values are positive.
        As such, the uncertainty coefficient may be viewed as the normalized mutual information
        between :math:`X` and :math:`Y` and in the range :math:`[0, 1]`.

        :return: Uncertainty coefficient.
        """
        b_keys = list(self._b_map.keys())
        df = self._df[self._df.b.isin(b_keys)]
        n = df.shape[0]

        h_b = map(lambda b: df.query(f'b=="{b}"').shape[0] / n, b_keys)
        h_b = map(lambda p: p * log(p), h_b)
        h_b = -reduce(lambda x, y: x + y, h_b)

        i_ab = self.mutual_information

        e = i_ab / h_b

        return e

    @property
    def uncertainty_coefficient_reversed(self):
        """
        `Uncertainty coefficient <https://en.wikipedia.org/wiki/Uncertainty_coefficient>`_.

        :return: Uncertainty coefficient.
        """
        a_keys = list(self._a_map.keys())
        df = self._df[self._df.a.isin(a_keys)]
        n = df.shape[0]

        h_a = map(lambda a: df.query(f'a=="{a}"').shape[0] / n, a_keys)
        h_a = map(lambda p: p * log(p), h_a)
        h_a = -reduce(lambda x, y: x + y, h_a)

        i_ab = self.mutual_information

        e = i_ab / h_a

        return e

    @property
    def mutual_information(self):
        """
        The `mutual information <https://en.wikipedia.org/wiki/Mutual_information>`_ between
        two variables :math:`X` and :math:`Y` is denoted as :math:`I(X;Y)`.  :math:`I(X;Y)` is
        unbounded and in the range :math:`[0, \\infty]`. A higher mutual information
        value implies strong association. The formula for :math:`I(X;Y)` is defined as follows.

        :math:`I(X;Y) = \\sum_y \\sum_x P(x, y) \\log \\frac{P(x, y)}{P(x) P(y)}`

        :return: Mutual information.
        """
        a_keys = list(self._a_map.keys())
        b_keys = list(self._b_map.keys())
        df = self._df[self._df.a.isin(a_keys) & self._df.b.isin(b_keys)]
        n = df.shape[0]

        get_p_a = lambda a: df.query(f'a=="{a}"').shape[0] / n
        get_p_b = lambda b: df.query(f'b=="{b}"').shape[0] / n
        get_p_ab = lambda a, b: df.query(f'a=="{a}" and b=="{b}"').shape[0] / n
        get_sub_i = lambda a, b: get_p_ab(a, b) * log(get_p_ab(a, b) / get_p_a(a) / get_p_b(b))
        mi = sum((get_sub_i(a, b) for a, b in product(*[a_keys, b_keys])))
        return mi

    @property
    def goodman_kruskal_lambda(self):
        """
        Goodman-Kruskal's lambda is the `proportional reduction in error`
        of predicting one variable `b` given another `a`: :math:`\\lambda_{B|A}`.

        - The probability of an error in predicting the column category: :math:`P_e = 1 - \\frac{\\max_{c} N_{* c}}{N}`
        - The probability of an error in predicting the column category given the row category: :math:`P_{e|r} = 1 - \\frac{\\sum_r \\max_{c} N_{r c}}{N}`

        Where,

        - :math:`\\max_{c} N_{* c}` is the maximum of the column marginals
        - :math:`\\sum_r \\max_{c} N_{r c}` is the sum over the maximum value per row
        - :math:`N` is the total

        Thus, :math:`\\lambda_{B|A} = \\frac{P_e - P_{e|r}}{P_e}`.

        The way the contingency table is setup by default is that `a` is on
        the rows and `b` is on the columns. Note that Goodman-Kruskal's lambda
        is not symmetric: :math:`\\lambda_{B|A}` does not necessarily equal
        :math:`\\lambda_{A|B}`. By default, :math:`\\lambda_{B|A}` is computed, but
        if you desire the reverse, use `goodman_kruskal_lambda_reversed()`.

        References

        - `Goodman-Kruskal's lambda <https://en.wikipedia.org/wiki/Goodman_and_Kruskal%27s_lambda>`_.
        - `Correlation <http://cda.psych.uiuc.edu/web_407_spring_2014/correlation_week4.pdf>`_.

        :return: Goodman-Kruskal's lambda.
        """
        n = self._n
        x = sum([max(self.observed[r]) for r in range(self._n_rows)])
        y = max(self._col_marginals)
        gkl = (x - y) / (n - y)
        return gkl

    @property
    def goodman_kruskal_lambda_reversed(self):
        """
        Computes :math:`\\lambda_{A|B}`.

        :return: Goodman-Kruskal's lambda.
        """
        n = self._n
        x = sum([max([self.observed[r][c] for r in range(self._n_rows)]) for c in range(self._n_cols)])
        y = max(self._row_marginals)
        gkl = (x - y) / (n - y)
        return gkl

    @property
    def adjusted_rand_index(self):
        """
        The Adjusted Rand Index (ARI) should yield a value between
        [0, 1], however, negative values can also arise when the index
        is less than the expected value. This function uses `binom()`
        from `scipy.special`, and when n >= 300, the results are too
        large and may cause overflow.

        TODO: use a different way to compute binomial coefficient

        References

        - `Adjusted Rand Index <https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index>`_.
        - `Python binomial coefficient <https://stackoverflow.com/questions/26560726/python-binomial-coefficient>`_.

        :return: Adjusted Rand Index.
        """
        a_i = sum([int(binom(a, 2)) for a in self._row_marginals])
        b_j = sum([int(binom(b, 2)) for b in self._col_marginals])
        n_ij = sum([int(binom(n, 2)) for n in chain(*self.observed)])
        n = binom(self._n, 2)

        top = (n_ij - (a_i * b_j) / n)
        bot = 0.5 * (a_i + b_j) - (a_i * b_j) / n
        s = top / bot
        return s


class AgreementTable(CategoricalTable):
    """
    Represents a contingency table for agreement data against one variable. The variable is typically
    a rating variable (e.g. dislike, neutral, like), and the data is a pairing of ratings over
    the same set of items.
    """

    def __init__(self, a, b, a_vals=None, b_vals=None):
        super().__init__(a, b, a_vals=a_vals, b_vals=b_vals)
        if self._n_cols != self._n_rows:
            raise ValueError(f'Table not symmetric: rows={self._n_rows}, cols={self._n_cols}')

        a_set = set(self._a_map.keys())
        b_set = set(self._b_map.keys())
        if len(a_set) != len(b_set):
            raise ValueError(f'Table not symmetric: rows={len(a_set)}, cols={len(b_set)}')
        if len(a_set & b_set) != len(a_set):
            raise ValueError(f'Rating value mismatches: {a_set ^ b_set}')

    @property
    def chohen_k(self):
        """
        Computes Cohen's :math:`\\kappa`.

        - :math:`\\kappa = \\frac{\\theta_1 - \\theta_2}{1 - \\theta_2}`
        - :math:`\\theta_1 = \\sum_i p_{ii}`
        - :math:`\\theta_2 = \\sum_i p_{i+}p_{+i}`

        :return: :math:`\\kappa`.
        """
        I = len(self._a_map.keys())
        theta_1 = sum([self._p_observed[i][i] for i in range(I)])
        theta_2 = sum([self._p_r_marginals[i] * self._p_c_marginals[i] for i in range(I)])
        k = (theta_1 - theta_2) / (1 - theta_2)
        return k

    @property
    def cohen_light_k(self):
        """
        Computes Cohen-Light :math:`\\kappa` is a measure of conditional agreement. Several :math:`\\kappa`,
        one for each unique value, will be computed and returned.

        - :math:`\\kappa = \\frac{\\theta_1 - \\theta_2}{1 - \\theta_2}`
        - :math:`\\theta_1 = \\frac{p_{ii}}{p_{i+}}`
        - :math:`\\theta_2 = p_{+i}`

        :return: A list of :math:`\\kappa`.
        """
        theta_1 = lambda i: self._p_observed[i][i] / self._p_r_marginals[i]
        theta_2 = lambda i: self._p_c_marginals[i]
        kappa = lambda t_1, t_2: (t_1 - t_2) / (1 - t_2)

        I = len(self._a_map.keys())
        kappas = [kappa(theta_1(i), theta_2(i)) for i in range(I)]
        return kappas


class BinaryTable(CategoricalTable):
    """
    Represents a contingency table for binary variables.
    """

    def __init__(self, a, b, a_0=0, a_1=1, b_0=0, b_1=1):
        """
        ctor.

        :param a: Iterable list.
        :param b: Iterable list.
        :param a_0: The zero value for a. Defaults to 0.
        :param a_1: The one value for a. Defaults to 1.
        :param b_0: The zero value for b. Defaults to 0.
        :param b_1: The zero value for b. Defaults to 1.
        """
        super().__init__(a, b, a_vals=[a_0, a_1], b_vals=[b_0, b_1])
        self._a_0 = a_0
        self._a_1 = a_1
        self._b_0 = b_0
        self._b_1 = b_1

    @property
    def jaccard_similarity(self):
        """
        `Jaccard Index <https://en.wikipedia.org/wiki/Jaccard_index>`_.

        :return: Jaccard similarity.
        """
        a_0 = self._a_map[self._a_0]
        a_1 = self._a_map[self._a_1]
        b_0 = self._b_map[self._b_0]
        b_1 = self._b_map[self._b_1]

        m_11 = self.observed[a_1][b_1]
        m_01 = self.observed[a_0][b_1]
        m_10 = self.observed[a_1][b_0]

        s = m_11 / (m_01 + m_10 + m_11)
        return s

    @property
    def jaccard_distance(self):
        """
        `Jaccard Index <https://en.wikipedia.org/wiki/Jaccard_index>`_.

        :return: Jaccard distance.
        """
        d = 1.0 - self.jaccard_similarity
        return d

    @property
    def tanimoto_similarity(self):
        """
        `Tanimoto similarity and distance <https://en.wikipedia.org/wiki/Jaccard_index#Tanimoto_similarity_and_distance>`_.

        :return: Tanimoto similarity.
        """
        count_11 = self._count(self._a_1, self._b_1)
        count_01 = self._count(self._a_0, self._b_1)
        count_10 = self._count(self._a_1, self._b_0)
        s = count_11 / (count_01 + count_10)
        return s

    @property
    def tanimoto_distance(self):
        """
        `Tanimoto similarity and distance <https://en.wikipedia.org/wiki/Jaccard_index#Tanimoto_similarity_and_distance>`_.

        :return: Tanimoto distance.
        """
        d = -log2(self.tanimoto_similarity)
        return d

    @property
    def cramer_v(self):
        """
        `Cramer's V <https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V>`_.

        :return: Cramer's V.
        """
        s = sqrt(self.chisq / self._n / min(self._n_cols - 1, self._n_rows - 1))
        return s

    @property
    def contingency_coefficient(self):
        """
        `Contingency coefficient <https://en.wikipedia.org/wiki/Contingency_table#Cram%C3%A9r's_V_and_the_contingency_coefficient_C>`_.

        :return: Contingency coefficient.
        """
        s = sqrt(self.chisq / (self._n + self.chisq))
        return s

    @property
    def tschuprow_t(self):
        """
        `Tschuprow's T <https://en.wikipedia.org/wiki/Tschuprow%27s_T>`_.

        :return: Tschuprow's T.
        """
        s = sqrt(self.chisq / sqrt((self._n_cols - 1) * (self._n_rows - 1)))
        return s

    @property
    def rand_index(self):
        """
        `Rand Index <https://en.wikipedia.org/wiki/Rand_index>`_.

        :return: Rand Index.
        """
        tp = self._count(self._a_1, self._b_1)
        fp = self._count(self._a_0, self._b_1)
        fn = self._count(self._a_1, self._b_0)
        tn = self._count(self._a_0, self._b_0)
        s = (tp + tn) / (tp + fp + fn + tn)
        return s

    @property
    def fowlkes_mallows_index(self):
        """
        `Fowlkes-Mallows Index <https://en.wikipedia.org/wiki/Fowlkes%E2%80%93Mallows_index>`_ is
        typically used to judge the similarity between two clusters. A larger value indicates
        that the clusters are more similar.

        :return: Fowlkes-Mallows Index.
        """
        tp = self._count(self._a_1, self._b_1)
        fp = self._count(self._a_0, self._b_1)
        fn = self._count(self._a_1, self._b_0)
        s = sqrt((tp / (tp + fp)) * (tp / (tp + fn)))
        return s

    @property
    def mcnemar_test(self):
        """
        `McNemar's test <https://en.wikipedia.org/wiki/McNemar%27s_test>`_.

        :return: A tuple. First element is chi-square test statistics. Second element is p-value.
        """
        c = self._count(self._a_0, self._b_1)
        b = self._count(self._a_1, self._b_0)
        chisq = (b - c) ** 2 / (b + c)
        p = 1 - stats.chi2.cdf(chisq, 1)
        return chisq, p

    @property
    def odds_ratio(self):
        """
        `Odds ratio <https://en.wikipedia.org/wiki/Contingency_table#Odds_ratio>`_. The odds
        ratio is also referred to as the `cross-product ratio`.

        :return: Odds ratio.
        """
        p_00 = self._count(self._a_0, self._b_0) / self._n
        p_01 = self._count(self._a_0, self._b_1) / self._n
        p_10 = self._count(self._a_1, self._b_0) / self._n
        p_11 = self._count(self._a_1, self._b_1) / self._n

        ratio = (p_11 * p_00) / (p_10 * p_01)
        return ratio

    @property
    def yule_q(self):
        """
        Yule's Q is based off of the odds ratio or cross-product ratio, :math:`\\alpha`.

        :math:`Q = \\frac{\\alpha - 1}{\\alpha + 1}`

        Yule's Q is the same as Goodman-Kruskal's :math:`\\lambda` for 2 x 2 contingency tables and is also
        a measure of proportional reduction in error (PRE).

        :return: Yule's Q.
        """
        alpha = self.odds_ratio
        q = (alpha - 1) / (alpha + 1)
        return q

    @property
    def yule_y(self):
        """
        Yule's Y is based off of the odds ratio or cross-product ratio, :math:`\\alpha`.

        :math:`Y = \\frac{\\sqrt\\alpha - 1}{\\sqrt\\alpha + 1}`

        :return: Yule's Y.
        """
        alpha = sqrt(self.odds_ratio)
        q = (alpha - 1) / (alpha + 1)
        return q

    @property
    def tetrachoric_correlation(self):
        """
        Tetrachoric correlation ranges from :math:`[0, 1]`, where 0 indicates no agreement and
        1 indicates perfect agreement.

        References

        - `Tetrachoric correlation <https://www.andrews.edu/~calkins/math/edrm611/edrm13.htm#TETRA>`_.
        - `Tetrachoric Correlation: Definition, Examples, Formula <https://www.statisticshowto.com/tetrachoric-correlation/>`_.
        - `Tetrachoric Correlation Estimation <https://www.real-statistics.com/correlation/polychoric-correlation/tetrachoric-correlation-estimation/>`_.

        :return: Tetrachoric correlation.
        """
        n_00 = self._count(self._a_0, self._b_0)
        n_01 = self._count(self._a_0, self._b_1)
        n_10 = self._count(self._a_1, self._b_0)
        n_11 = self._count(self._a_1, self._b_1)

        if n_10 == 0 or n_01 == 0:
            return 1.0
        if n_00 == 0 or n_11 == 0:
            return -1.0

        y = pow((n_00 * n_11) / (n_10 * n_01), pi / 4.0)
        p = (y - 1) / (y + 1)
        return p


class ConfusionMatrix(BinaryTable):
    """
    Represents a `confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`_. The confusion
    matrix looks like what is shown below for two binary variables `a` and `b`;
    `a` is in the rows and `b` in the columns. Most of the statistics around performance comes
    from the counts of `TN`, `FN`, `FP` and `TP`.

    .. list-table:: Confusion Matrix
       :widths: 25 25 25
       :header-rows: 1

       * -
         - b=0
         - b=1
       * - a=0
         - TN
         - FP
       * - a=1
         - FN
         - TP
    """

    def __init__(self, a, b, a_0=0, a_1=1, b_0=0, b_1=1):
        """
        ctor. Note that `a` is the ground truth and `b` is the prediction.

        :param a: Binary variable (iterable). Ground truth.
        :param b: Binary variable (iterable). Prediction.
        :param a_0: The zero value for a. Defaults to 0.
        :param a_1: The one value for a. Defaults to 1.
        :param b_0: The zero value for b. Defaults to 0.
        :param b_1: The zero value for b. Defaults to 1.
        """
        super().__init__(a, b, a_0=a_0, a_1=a_1, b_0=b_0, b_1=b_1)

    @property
    def tp(self):
        """
        True positive. Count of when `a=1` and `b=1`.

        :return: TP.
        """
        return self._count(self._a_1, self._b_1)

    @property
    def fp(self):
        """
        False positive. Count of when `a=0` and `b=1`.

        :return: FP.
        """
        return self._count(self._a_0, self._b_1)

    @property
    def fn(self):
        """
        False negative. Count of when `a=1` and `b=0`.

        :return: FN.
        """
        return self._count(self._a_1, self._b_0)

    @property
    def tn(self):
        """
        True negative. Count of when `a=0` and `b=0`.

        :return: TN.
        """
        return self._count(self._a_0, self._b_0)

    @property
    @lru_cache(maxsize=None)
    def n(self):
        """
        Total number of data. Count of `TP + FP + FN + TN`.

        :return: N.
        """
        return self.tp + self.fp + self.fn + self.tn

    @property
    def tpr(self):
        """
        True positive rate.

        :math:`TPR = \\frac{TP}{TP + FN}`

        Aliases

        - sensitivity
        - recall
        - hit rate
        - power
        - probability of detection

        :return: TPR.
        """
        return self.tp / (self.tp + self.fn)

    @property
    def tnr(self):
        """
        True negative rate.

        :math:`TNR = \\frac{TN}{TN + FP}`

        Aliases

        - specificity
        - selectivity

        :return: TNR.
        """
        return self.tn / (self.tn + self.fp)

    @property
    def ppv(self):
        """
        Positive predictive value.

        :math:`PPV = \\frac{TP}{TP + FP}`

        Aliases

        - precision

        :return: PPV.
        """
        return self.tp / (self.tp + self.fp)

    @property
    def npv(self):
        """
        Negative predictive value.

        :math:`NPV = \\frac{TN}{TN + FN}`

        :return: NPV.
        """
        return self.tn / (self.tn + self.fn)

    @property
    def fnr(self):
        """
        False negative rate.

        :math:`FNR = \\frac{FN}{FN + TP}`

        Aliases

        - miss rate

        :return: FNR.
        """
        return self.fn / (self.fn + self.tp)

    @property
    def fpr(self):
        """
        False positive rate.

        :math:`FPR = \\frac{FP}{FP + TN}`

        Aliases

        - fall-out
        - probability of false alarm

        :return: FPR.
        """
        return self.fp / (self.fp + self.tn)

    @property
    def fdr(self):
        """
        False discovery rate.

        :math:`FDR = \\frac{FP}{FP + TP}`

        :return: FDR.
        """
        return self.fp / (self.fp + self.tp)

    @property
    def fomr(self):
        """
        False omission rate.

        :math:`FOR = \\frac{FN}{FN + TN}`

        :return: FOR.
        """
        return self.fn / (self.fn + self.tn)

    @property
    def pt(self):
        """
        Prevalence threshold.

        :math:`PT = \\frac{\\sqrt{TPR(-TNR + 1)} + TNR - 1}{TPR + TNR - 1}`

        :return: Prevalence threshold.
        """
        tpr = self.tpr
        tnr = self.tnr

        return (sqrt(tpr * (-tnr + 1)) + tnr - 1) / (tpr + tnr - 1)

    @property
    def ts(self):
        """
        Threat score.

        :math:`TS = \\frac{TP}{TP + FN + FP}`

        Aliases

        - critical success index (CSI).

        :return: TS.
        """
        return self.tp / (self.tp + self.fn + self.fp)

    @property
    def acc(self):
        """
        Accuracy.

        :math:`ACC = \\frac{TP + TN}{TP + TN + FP + FN}`

        :return: Accuracy.
        """
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    @property
    def ba(self):
        """
        Balanced accuracy.

        :math:`BA = \\frac{TPR + TNR}{2}`

        :return: Balanced accuracy.
        """
        return (self.tpr + self.tnr) / 2

    @property
    def f1(self):
        """
        F1 score: harmonic mean of precision and sensitivity.

        :math:`F1 = \\frac{PPV \\times TPR}{PPV + TPR}`

        :return: F1.
        """
        return 2 * (self.ppv * self.tpr) / (self.ppv + self.tpr)

    @property
    def mcc(self):
        """
        Matthew's correlation coefficient.

        :math:`MCC = \\frac{TP + TN - FP \\times FN}{\\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}`

        :return: MCC.
        """
        tp = self.tp
        tn = self.tn
        fp = self.fp
        fn = self.fn

        return (tp + tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    @property
    def bm(self):
        """
        Bookmaker informedness.

        :math:`BI = TPR + TNR - 1`

        :return: BM.
        """
        return self.tpr + self.tnr - 1

    @property
    def mk(self):
        """
        Markedness.

        :math:`MK = PPV + NPV - 1`

        Aliases

        - deltaP

        :return: Markedness.
        """
        return self.ppv + self.npv - 1

    @property
    def sensitivity(self):
        """
        Alias to TPR.

        :return: Sensitivity.
        """
        return self.tpr

    @property
    def specificity(self):
        """
        Alias to TNR.

        :return: Specificity.
        """
        return self.tnr

    @property
    def precision(self):
        """
        Alias to PPV.

        :return: PPV.
        """
        return self.ppv

    @property
    def recall(self):
        """
        Alias to TPR.

        :return: TPR.
        """
        return self.tpr

    @property
    def prevalence(self):
        """
        Prevalence.

        :math:`\\frac{TP + FN}{N}`

        :return: Prevalence.
        """
        return (self.tp + self.fn) / self.n

    @property
    def plr(self):
        """
        Positive likelihood ratio.

        :math:`PLR = \\frac{TPR}{FPR}`

        Aliases

        - LR+

        :return: PLR.
        """
        return self.tpr / self.fpr

    @property
    def nlr(self):
        """
        Negative likelihood ratio.

        :math:`NLR = \\frac{FNR}{TNR}`

        Aliases

        - LR-

        :return: NLR.
        """
        return self.fnr / self.tnr

    @property
    def dor(self):
        """
        Diagnostic odds ratio.

        :math:`\\frac{PLR}{NLR}`

        :return: DOR.
        """
        return self.plr / self.nlr
