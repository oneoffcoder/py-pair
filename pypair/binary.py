from itertools import chain
from math import sqrt, log2
from scipy.special import binom


class CategoricalTable(object):
    """
    https://en.wikipedia.org/wiki/Fowlkes%E2%80%93Mallows_index
    https://en.wikipedia.org/wiki/Polychoric_correlation
    https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    https://en.wikipedia.org/wiki/Contingency_table
    https://en.wikipedia.org/wiki/Chi-squared_test#:~:text=Pearson's%20chi%2Dsquared%20test%20is,categories%20of%20a%20contingency%20table.
    https://en.wikipedia.org/wiki/McNemar%27s_test
    """

    def __init__(self, a, b, a_vals=None, b_vals=None):
        if a_vals is None:
            a_vals = sorted(list({v for v in a}))
        else:
            a_vals = sorted(list(set(a_vals)))

        if b_vals is None:
            b_vals = sorted(list({v for v in b}))
        else:
            b_vals = sorted(list(set(b_vals)))

        data = [(x, y) for x, y in zip(a, b)]
        observed = [[data.count((x, y)) for y in b_vals] for x in a_vals]

        n_rows = len(a_vals)
        n_cols = len(b_vals)

        rows = [sum(r) for r in observed]
        cols = [sum([observed[r][c] for r in range(n_rows)]) for c in range(n_cols)]

        n = sum([sum(o) for o in observed])
        get_expected = lambda r, c: r * c / n
        expected = [[get_expected(rows[i], cols[j]) for j, _ in enumerate(b_vals)] for i, _ in enumerate(a_vals)]

        chisq = sum([(o - e) ** 2 / e for o, e in zip(chain(*observed), chain(*expected))])

        self.observed = observed
        self.expected = expected
        self._data = data
        self._chisq = chisq
        self._n = n
        self._a_map = {v: i for i, v in enumerate(a_vals)}
        self._b_map = {v: i for i, v in enumerate(b_vals)}
        self._k = n_cols
        self._r = n_rows
        self._row_marginals = rows
        self._col_marginals = cols

    @property
    def chisq(self):
        """
        https://en.wikipedia.org/wiki/Chi-square_distribution
        :return: Chi-squared.
        """
        return self._chisq

    @property
    def phi(self):
        """
        https://en.wikipedia.org/wiki/Phi_coefficient
        :return: Phi.
        """
        return sqrt(self.chisq / self._n)

    @property
    def uncertainty_coefficient(self):
        """
        https://en.wikipedia.org/wiki/Uncertainty_coefficient
        :return:
        """
        pass

    @property
    def mutual_information(self):
        """
        https://en.wikipedia.org/wiki/Mutual_information
        :return:
        """
        pass

    @property
    def goodman_kruskal_lambda(self):
        """
        https://en.wikipedia.org/wiki/Goodman_and_Kruskal%27s_lambda
        :return:
        """
        pass

    @property
    def adjusted_rand_index(self):
        """
        The Adjusted Rand Index (ARI) should yield a value between
        [0, 1], however, negative values can also arise when the index
        is less than the expected value. This function uses `binom()`
        from `scipy.special`, and when n >= 300, the results are too
        large and may cause overflow.

        TODO: use a different way to compute binomial coefficient
        https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index
        https://stackoverflow.com/questions/26560726/python-binomial-coefficient
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


class BinaryTable(CategoricalTable):
    def __init__(self, a, b, a_0=0, a_1=1, b_0=0, b_1=1):
        super().__init__(a, b, a_vals=[a_0, a_1], b_vals=[b_0, b_1])
        self._a_0 = a_0
        self._a_1 = a_1
        self._b_0 = b_0
        self._b_1 = b_1

    @property
    def jaccard_similarity(self):
        """
        https://en.wikipedia.org/wiki/Jaccard_index
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
        https://en.wikipedia.org/wiki/Jaccard_index
        :return: Jaccard distance.
        """
        d = 1.0 - self.jaccard_similarity
        return d

    @property
    def tanimoto_similarity(self):
        """
        https://en.wikipedia.org/wiki/Jaccard_index#Tanimoto_similarity_and_distance
        :return: Tanimoto similarity.
        """
        count_11 = self._data.count((self._a_1, self._b_1))
        count_01 = self._data.count((self._a_0, self._b_1))
        count_10 = self._data.count((self._a_1, self._b_0))
        s = count_11 / (count_01 + count_10)
        return s

    @property
    def tanimoto_distance(self):
        """
        https://en.wikipedia.org/wiki/Jaccard_index#Tanimoto_similarity_and_distance
        :return: Tanimoto distance.
        """
        d = -log2(self.tanimoto_similarity)
        return d

    @property
    def cramer_v(self):
        """
        https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
        :return: Cramer's V.
        """
        s = sqrt(self.chisq / self._n / min(self._k - 1, self._r - 1))
        return s

    @property
    def tschuprow_t(self):
        """
        https://en.wikipedia.org/wiki/Tschuprow%27s_T
        :return: Tschuprow's T.
        """
        s = sqrt(self.chisq / sqrt((self._k - 1) * (self._r - 1)))
        return s

    @property
    def rand_index(self):
        """
        https://en.wikipedia.org/wiki/Rand_index
        :return: Rand index.
        """
        tp = self._data.count((self._a_1, self._b_1))
        fp = self._data.count((self._a_0, self._b_1))
        fn = self._data.count((self._a_1, self._b_0))
        tn = self._data.count((self._a_0, self._b_0))
        s = (tp + tn) / (tp + fp + fn + tn)
        return s
