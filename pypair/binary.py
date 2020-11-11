from itertools import chain
from math import sqrt, log2


class CategoricalTable(object):
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
        s = sqrt(self.chisq / sqrt((self._k -1) * (self._r - 1)))
        return s
