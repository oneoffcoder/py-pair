from itertools import chain
from math import sqrt


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
        phi = sqrt(chisq / n)

        self.observed = observed
        self.expected = expected
        self.chisq = chisq
        self.phi = phi
        self._a_map = {v: i for i, v in enumerate(a_vals)}
        self._b_map = {v: i for i, v in enumerate(b_vals)}


class BinaryTable(CategoricalTable):
    def __init__(self, a, b, a_0=0, a_1=1, b_0=0, b_1=1):
        super().__init__(a, b, a_vals=[a_0, a_1], b_vals=[b_0, b_1])
        self.a_0 = a_0
        self.a_1 = a_1
        self.b_0 = b_0
        self.b_1 = b_1

    @property
    def jaccard_similarity(self):
        a_0 = self._a_map[self.a_0]
        a_1 = self._a_map[self.a_1]
        b_0 = self._b_map[self.b_0]
        b_1 = self._b_map[self.b_1]

        m_11 = self.observed[a_1][b_1]
        m_01 = self.observed[a_0][b_1]
        m_10 = self.observed[a_1][b_0]

        return m_11 / (m_01 + m_10 + m_11)

    @property
    def jaccard_distance(self):
        return 1.0 - self.jaccard_similarity



