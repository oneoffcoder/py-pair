from collections import namedtuple
from itertools import combinations, product, chain
from math import sqrt

from pypair.biserial import BiserialStats
from pypair.contingency import ConfusionStats, CategoricalStats, \
    BinaryStats, AgreementStats
from pypair.continuous import ConcordanceStats


def __as_key(k1, k2):
    """
    Creates a key (tuple) out of the two specified. The key is always ordered.
    If k2 < k1, then (k2, k1), else, (k1, k2).

    :param k1: Key (string).
    :param k2: Key (string).
    :return: (k1, k2) or (k2, k1).
    """
    keys = sorted([k1, k2])
    return keys[0], keys[1]


def __to_abcd_counts(d):
    """
    Maps the paired keys in the dictionary and their associated values to a form: ``(k1, k2), (a, b, c, d)``.

    :param d: A dictionary. Names are variable names. Values are 0 or 1.
    :return: A list of tuples of the form: (k1, k2), (a, b, c, d).
    """

    def as_count(v1, v2):
        """
        Maps the specified values to a (TP or 11), b (FN or 10), c (FP or 01) and d (TN or 00).
        Only one of these will be 1, and the others will be 0. Look below for example.

        - 1, 1 = (1, 0, 0, 0)
        - 1, 0 = (0, 1, 0, 0)
        - 0, 1 = (0, 0, 1, 0)
        - 0, 0 = (0, 0, 0, 1)

        :param v1: Value (0 or 1).
        :param v2: Value (0 or 1).
        :return: a, b, c, d
        """
        a, b, c, d = 0, 0, 0, 0
        if v1 is not None and v2 is not None:
            if v1 == 1 and v2 == 1:
                a = 1
            elif v1 == 1 and v2 == 0:
                b = 1
            elif v1 == 0 and v2 == 1:
                c = 1
            else:
                d = 1
        return a, b, c, d

    def transform(k1, k2):
        """
        Transforms the keys and associated value to the form (a tuple of tuples): (k1, k2), (a, b, c, d).

        :param k1: Key (string).
        :param k2: Key (string).
        :return: (k1, k2), (a, b, c, d)
        """
        v1, v2 = d[k1], d[k2]
        return __as_key(k1, k2), as_count(v1, v2)

    return [transform(k1, k2) for k1, k2 in combinations(d.keys(), 2)]


def __add_abcd_counts(x, y):
    """
    Adds two tuples. For example.

    :math:`x + y = (x_a + y_a, x_b + y_b, x_c + y_c, x_d + y_d)`

    :param x: Tuple (a, b, c, d).
    :param y: Tuple (a, b, c, d).
    :return: Tuple (a, b, c, d).
    """
    return x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]


def __add_concordance_counts(x, y):
    """
    Adds two tuples. For example.

    :math:`x + y = (x_d + y_d, x_t_{xy} + y_t_{xy}, x_t_x + y_t_x, x_t_y + y_t_y, x_c + y_c, x_n + y_n)`

    :param x: Tuple (d, t_xy, t_x, t_y, c, n).
    :param y: Tuple (d, t_xy, t_x, t_y, c, n).
    :return: Tuple (d, t_xy, t_x, t_y, c, n).
    """
    return x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3], x[4] + y[4], x[5] + y[5]


def __get_contingency_table(sdf):
    """
    Gets the pairwise contingency tables. Each record in the pair-RDD returns has the following form.

    `(k1, k2), (table, row_marginals, col_marginals, domain1, domain2)`

    - k1 is the name of a variable
    - k2 is the name of a variable
    - table is a list of list (a table, matrix) of counts
    - row_marginals contain the row marginals
    - col_marginals contain the column marginals
    - domain1 is a list of all the values of variable 1
    - domain2 is a list of all the values of variable 2

    :param sdf: Spark dataframe.
    :return: Spark pair-RDD.
    """

    def to_count(d):
        def count(k1, k2):
            tups = [(k1, d[k1]), (k2, d[k2])]
            tups = sorted(tups, key=lambda t: t[0])

            return (tups[0][0], tups[1][0], tups[0][1], tups[1][1]), 1

        return [count(k1, k2) for k1, k2 in combinations(d.keys(), 2)]

    def attach_domains(tup):
        key, d = tup
        v1 = sorted(list({k[0] for k, _ in d.items()}))
        v2 = sorted(list({k[1] for k, _ in d.items()}))

        return key, (d, v1, v2)

    def to_contingency_table(tup):
        key, (d, v1, v2) = tup
        table = [[d[(a, b)] if (a, b) in d else 0 for b in v2] for a in v1]

        return key, table

    return sdf.rdd \
        .flatMap(lambda r: to_count(r.asDict())) \
        .reduceByKey(lambda a, b: a + b) \
        .map(lambda tup: ((tup[0][0], tup[0][1]), (tup[0][2], tup[0][3], tup[1]))) \
        .map(lambda tup: (tup[0], {(tup[1][0], tup[1][1]): tup[1][2]})) \
        .reduceByKey(lambda a, b: {**a, **b}) \
        .map(lambda tup: attach_domains(tup)) \
        .map(lambda tup: to_contingency_table(tup)) \
        .sortByKey()


def binary_binary(sdf):
    """
    Gets all the pairwise binary-binary association measures. The result is a Spark pair-RDD,
    where the keys are tuples of variable names e.g. (k1, k2), and values are dictionaries
    of association names and measures e.g. {'phi': 1, 'lambda': 0.8}. Each record in the pair-RDD is of the form.

    - (k1, k2), {'phi': 1, 'lambda': 0.8, ...}

    :param sdf: Spark dataframe. Should be all 1's and 0's.
    :return: Spark pair-RDD.
    """

    def to_results(counts):
        """
        Converts the result of the contingency table counts to a dictionary of association measures.

        :param counts: Tuple of tuples: (k1, k2), (a, b, c, d).
        :return: (x1, x2), {'measure1': val1, 'measure2': val2, ...}.
        """
        (x1, x2), (a, b, c, d) = counts

        computer = BinaryStats([[a + 1, b + 1], [c + 1, d + 1]])
        measures = {m: computer.get(m) for m in computer.measures()}
        return (x1, x2), measures

    return sdf.rdd \
        .flatMap(lambda r: __to_abcd_counts(r.asDict())) \
        .reduceByKey(lambda a, b: __add_abcd_counts(a, b)) \
        .sortByKey() \
        .map(lambda counts: to_results(counts))


def confusion(sdf):
    """
    Gets all the pairwise confusion matrix metrics. The result is a Spark pair-RDD,
    where the keys are tuples of variable names e.g. (k1, k2), and values are dictionaries
    of association names and metrics e.g. {'acc': 0.9, 'fpr': 0.2}.
    Each record in the pair-RDD is of the form.

    - (k1, k2), {'acc': 0.9, 'fpr': 0.2, ...}

    :param sdf: Spark dataframe. Should be all 1's and 0's.
    :return: Spark pair-RDD.
    """

    def to_results(counts):
        """
        Converts the result of the contingency table counts to a dictionary of association measures.

        :param counts: Tuple of tuples: (x1, x2), (tp, fn, fp, tn).
        :return: (x1, x2), {'metric1': val1, 'metric2': val2, ...}.
        """
        (x1, x2), (tp, fn, fp, tn) = counts

        tp = max(1, tp)
        fn = max(1, fn)
        fp = max(1, fp)
        tn = max(1, tn)

        computer = ConfusionStats([[tp, fn], [fp, tn]])
        measures = {m: computer.get(m) for m in computer.measures()}
        return (x1, x2), measures

    return sdf.rdd \
        .flatMap(lambda r: __to_abcd_counts(r.asDict())) \
        .reduceByKey(lambda a, b: __add_abcd_counts(a, b)) \
        .map(lambda counts: to_results(counts)) \
        .sortByKey()


def categorical_categorical(sdf):
    """
    Gets all pairwise categorical-categorical association measures. The result is a Spark pair-RDD,
    where the keys are tuples of variable names e.g. (k1, k2), and values are dictionaries of
    association names and metrics e.g. {‘phi’: 0.9, ‘chisq’: 0.2}. Each record in the pair-RDD is of the form.

    - (k1, k2), {‘phi’: 0.9, ‘chisq’: 0.2, ...}

    :param sdf: Spark dataframe. Should be strings or whole numbers to represent the values.
    :return: Spark pair-RDD.
    """

    def to_results(tup):
        key, table = tup
        computer = CategoricalStats(table)
        measures = {m: computer.get(m) for m in computer.measures()}
        return key, measures

    return __get_contingency_table(sdf) \
        .map(lambda tup: to_results(tup)) \
        .sortByKey()


def agreement(sdf):
    """
    Gets all pairwise categorical-categorical `agreement` association measures. The result is a Spark pair-RDD,
    where the keys are tuples of variable names e.g. (k1, k2), and values are dictionaries of
    association names and metrics e.g. {‘kappa’: 0.9, ‘delta’: 0.2}. Each record in the pair-RDD is of the form.

    - (k1, k2), {‘kappa’: 0.9, ‘delta’: 0.2, ...}

    :param sdf: Spark dataframe. Should be strings or whole numbers to represent the values.
    :return: Spark pair-RDD.
    """

    def to_results(tup):
        key, table = tup
        computer = AgreementStats(table)
        measures = {m: computer.get(m) for m in computer.measures()}
        return key, measures

    return __get_contingency_table(sdf) \
        .map(lambda tup: to_results(tup)) \
        .sortByKey()


def binary_continuous(sdf, binary, continuous, b_0=0, b_1=1):
    """
    Gets all pairwise binary-continuous association measures. The result is a Spark pair-RDD,
    where the keys are tuples of variable names e.g. (k1, k2), and values are dictionaries of
    association names and metrics e.g. {‘biserial’: 0.9, ‘point_biserial’: 0.2}. Each record
    in the pair-RDD is of the form.

    - (k1, k2), {‘biserial’: 0.9, ‘point_biserial’: 0.2, ...}

    All the binary fields/columns should be encoded in the same way. For example, if you
    are using 1 and 0, then all binary fields should only have those values, not a mixture
    of 1 and 0, True and False, -1 and 1, etc.

    :param sdf: Spark dataframe.
    :param binary: List of fields that are binary.
    :param continuous: List of fields that are continuous.
    :param b_0: Zero value for binary field.
    :param b_1: One value for binary field.
    :return: Spark pair-RDD.
    """

    def to_pair1(d):
        """
        Creates a list of tuples.

        :param d: Dictionary of data.
        :return: List of (b, c, b_val), (sum_c, sum_c_sq, sum_b).
        """
        return [((b, c, d[b]), (d[c], d[c] ** 2, 1)) for b, c in product(*[binary, continuous])]

    def to_pair2(tup):
        """
        Makes a new pair.

        :param tup: (b, c, b_val), (sum_c, sum_c_sq, sum_b)
        :return: (b, c), (b_val, sum_c, sum_c_sq, sum_b)
        """
        (b, c, b_val), (sum_c, sum_c_sq, sum_b) = tup
        return (b, c), (b_val, sum_c, sum_c_sq, sum_b)

    def compute_stats(tup):
        """
        `Computational formula for variance and standard deviation <http://www.ablongman.com/graziano6e/text_site/MATERIAL/Stats/manvar.htm>`_.

        - :math:`SS = \\sum (X - \\bar{X})^2 = \\sum X^2 - \\frac{\\left(\\sum X\\right)^2}{N}`
        - :math:`\\sigma^2 = \\frac{SS}{N - 1}`
        - :math:`\\sigma = \\sqrt{\\sigma^2}`

        :param tup: (b, c), [(b_val, sum_c, sum_c_sq, sum_b), (b_val, sum_c, sum_c_sq, sum_b)]
        :return: (b, c), (n, p, y_0, y_1, std)
        """
        (b, c), data = tup

        data = list(data)
        data_0 = data[0] if data[0][0] == b_0 else data[1]
        data_1 = data[0] if data[0][0] == b_1 else data[0]

        _, sum_c_0, sum_c_sq_0, sum_b_0 = data_0
        _, sum_c_1, sum_c_sq_1, sum_b_1 = data_1

        n = sum_b_0 + sum_b_1
        p = sum_b_1 / n
        y_0 = sum_c_0 / sum_b_0
        y_1 = sum_c_1 / sum_b_1
        ss = (sum_c_sq_0 + sum_c_sq_1) - ((sum_c_0 + sum_c_1) ** 2 / n)
        v = ss / (n - 1)
        std = sqrt(v)

        return (b, c), (n, p, y_0, y_1, std)

    def to_results(tup):
        """
        Computes the results.

        :param tup: (b, c), (n, p, y_0, y_1, std)
        :return: (b, c), {'measure1': val1, 'measure2': val2, ...}
        """
        key, (n, p, y_0, y_1, std) = tup
        computer = BiserialStats(n, p, y_0, y_1, std)
        measures = {m: computer.get(m) for m in computer.measures()}
        return key, measures

    return sdf.rdd \
        .flatMap(lambda r: to_pair1(r.asDict())) \
        .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2])) \
        .map(lambda tup: to_pair2(tup)) \
        .groupByKey() \
        .map(lambda tup: compute_stats(tup)) \
        .map(lambda tup: to_results(tup)) \
        .sortByKey()


def categorical_continuous(sdf, categorical, continuous):
    """
    Gets all pairwise categorical-continuous association measures. The result is a Spark pair-RDD,
    where the keys are tuples of variable names e.g. (k1, k2), and values are dictionaries of
    association names and metrics e.g. {‘eta_sq’: 0.9, 'eta': 0.95}. Each record
    in the pair-RDD is of the form.

    - (k1, k2), {‘eta_sq’: 0.9, 'eta': 0.95}

    For now, only ``eta`` :math:`\\eta^2` is supported.

    :param sdf: Spark dataframe.
    :param categorical: List of categorical variables.
    :param continuous: List of continuous variables.
    :return: Spark pair-RDD.
    """

    def to_pair1(d):
        """
        Creates a list of tuples.

        :param d: Dictionary of data.
        :return: List of (b, c, b_val), (sum_c, sum_c_sq, sum_b).
        """
        kv_0 = lambda cat, con: ((cat, con, d[cat]), (d[con], 0, 1))
        kv_1 = lambda cat, con: ((cat, con, '__*_avg_*__'), (d[con], 0, 1))
        kv_2 = lambda cat, con: ((cat, con, '__*_den_*__'), (d[con], d[con] ** 2, 1))
        explode = lambda cat, con: [kv_0(cat, con), kv_1(cat, con), kv_2(cat, con)]
        return chain(*(explode(cat, con) for cat, con in product(*[categorical, continuous])))

    def to_pair2(tup):
        """
        Makes a new pair.

        :param tup: (b, c, b_val), (sum_c, sum_c_sq, sum_b)
        :return: (b, c), (b_val, stats)
        """
        ss = lambda x, x_sq, n: (x_sq - (x ** 2 / n))
        (cat, con, flag), (sum_c, sum_c_sq, sum_b) = tup
        key = cat, con

        if flag == '__*_den_*__':
            val = ss(sum_c, sum_c_sq, sum_b)
        elif flag == '__*_avg_*__':
            val = sum_c / sum_b
        else:
            val = sum_c / sum_b, sum_b

        return key, (flag, val)

    def to_results(tup):
        """
        Computes the results.

        :param tup: (b, c), (flag, val)
        :return: (b, c), {'measure1': val1, 'measure2': val2, ...}
        """
        (b, c), data = tup
        data = {k: v for k, v in data}

        y_avg = data['__*_avg_*__']
        num = sum([v[1] * ((v[0] - y_avg) ** 2) for k, v in data.items() if isinstance(v, tuple)])
        den = data['__*_den_*__']

        eta = num / den
        return (b, c), {'eta_sq': eta, 'eta': sqrt(eta)}

    return sdf.rdd \
        .flatMap(lambda r: to_pair1(r.asDict())) \
        .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2])) \
        .map(lambda tup: to_pair2(tup)) \
        .groupByKey() \
        .map(lambda tup: to_results(tup)) \
        .sortByKey()


def concordance(sdf):
    """
    Gets all the pairwise ordinal-ordinal concordance measures. The result is a Spark pair-RDD,
    where the keys are tuples of variable names e.g. (k1, k2), and values are dictionaries
    of association names and measures e.g. {'kendall': 1, 'gamma': 0.8}. Each record in the pair-RDD is of the form.

    - (k1, k2), {'kendall': 1, 'gamma': 0.8, ...}

    :param sdf: Spark dataframe. Should be all ordinal data (numeric).
    :return: Spark pair-RDD.
    """

    def as_pair1(n1, n2, v1, v2):
        """
        Creates a pair of the form (n1, n2), (v1, v2) where the first tuple are sorted and the second
        tuple have corresponding values to the elements of the first tuple.

        :param n1: String (variable name).
        :param n2: String (Variable name).
        :param v1: Value.
        :param v2: Value.
        :return: (n1, n2), (v1, v2).
        """
        tups = sorted([(n1, v1), (n2, v2)], key=lambda t: t[0])

        k1, j1 = tups[0]
        k2, j2 = tups[1]

        return (k1, k2), (j1, j2)

    def to_pair1(d):
        """
        Creates a list of pairs of variables and values. Keys are names of variables and values are values of
        those variables.

        :param d: Dictionary.
        :return: List of (k1, k2), (v1, v2).
        """
        return [as_pair1(n1, n2, d[n1], d[n2]) for n1, n2 in combinations(d.keys(), 2)]

    def as_count(v1, v2):
        """
        Maps the specified pairs of values to concordance status. Concordance status can be the follow.

        - discordant: :math:`(y_j - y_i)(x_j - x_i) < 0`
        - tied: :math:`(y_j - y_i)(x_j - x_i) = 0`
        - concordant: :math:`(y_j - y_i)(x_j - x_i) > 0`

        Ties are differentiated as follows.

        - tied on ``x``: `x_i = x_j`
        - tied on ``y``: `y_i = y_j`
        - tied on ``xy``: `x_i = x_j \\land y_i = y_j`

        A tuple that looks like the following will be mapped from the concordance status.

        - discordant: (1, 0, 0, 0, 0, 1)
        - tie on ``x`` and ``y``: (0, 1, 0, 0, 0, 1)
        - tie on ``x``: (0, 0, 1, 0, 0, 1)
        - tie on ``y``: (0, 0, 0, 1, 0, 1)
        - concordant: (0, 0, 0, 0, 1, 1)

        :param v1: Pair (x_i, y_i).
        :param v2: Pair (x_j, y_j).
        :return: d, t_xy, t_x, t_y, c, n
        """
        d, t_xy, t_x, t_y, c, n = 0, 0, 0, 0, 0, 1

        if v1 is not None and v2 is not None:
            x_i, y_i = v1
            x_j, y_j = v2
            r = (y_j - y_i) * (x_j - x_i)

            if r > 0:
                c = 1
            elif r < 0:
                d = 1
            else:
                if x_i == x_j and y_i == y_j:
                    t_xy = 1
                elif x_i == x_j:
                    t_x = 1
                else:
                    t_y = 1

        return d, t_xy, t_x, t_y, c, n

    def to_pair2(tup):
        """
        Creates concordant status counts for each pair of observations.

        :param tup: (key, iterable).
        :return: Generator of (k1, k2), (d, t_xy, t_x, t_y, c, n).
        """
        key, data = tup

        return ((key, as_count(v1, v2)) for v1, v2 in combinations(data, 2))

    def to_results(counts):
        """
        Converts the results of concordance to a dictionary of measures.

        :param counts: Tuple of tuples: (x1, x2), (a, b, c, d).
        :return: (x1, x2), {'measure1': val1, 'measure2': val2, ...}.
        """
        (x1, x2), (d, t_xy, t_x, t_y, c, n) = counts

        d += 1
        t_xy += 1
        t_x += 1
        t_y += 1
        c += 1
        n += 5

        computer = ConcordanceStats(d, t_xy, t_x, t_y, c, n)
        measures = {m: computer.get(m) for m in computer.measures()}
        return (x1, x2), measures

    return sdf.rdd \
        .flatMap(lambda r: to_pair1(r.asDict())) \
        .groupByKey() \
        .flatMap(lambda tup: to_pair2(tup)) \
        .reduceByKey(lambda x, y: __add_concordance_counts(x, y)) \
        .map(lambda tup: to_results(tup)) \
        .sortByKey()


def continuous_continuous(sdf):
    """
    Gets all the pairwise continuous-continuous association measures. The result is a Spark pair-RDD,
    where the keys are tuples of variable names e.g. (k1, k2), and values are dictionaries
    of association names and measures e.g. {'pearson': 1}. Each record in the pair-RDD is of the form.

    - (k1, k2), {'pearson': 1}

    Only pearson is supported at the moment.

    :param sdf: Spark dataframe. Should be all ordinal data (numeric).
    :return: Spark pair-RDD.
    """

    CorrItem = namedtuple('CorrItem', 'x y xy x_sq y_sq n')

    def to_items(d):
        """
        Converts the dictionary to (n1, n2), CorrItem.

        :param d: Dictionary.
        :return: (n1, n2), CorrItem.
        """
        as_item = lambda n1, n2: CorrItem(d[n1], d[n2], d[n1] * d[n2], d[n1] ** 2, d[n2] ** 2, 1)
        return (((n1, n2), as_item(n1, n2)) for n1, n2 in combinations(d.keys(), 2))

    def add_items(a, b):
        """
        Adds two CorrItems.

        :param a: CorrItem.
        :param b: CorrItem.
        :return: CorrItem.
        """
        return CorrItem(a.x + b.x, a.y + b.y, a.xy + b.xy, a.x_sq + b.x_sq, a.y_sq + b.y_sq, a.n + b.n)

    def to_results(tup):
        """
        Converts the tup to a result.

        :param tup: (n1, n2), CorrItem.
        :return: (n1, n2), {'measure': value}.
        """
        (n1, n2), item = tup
        n = item.xy - (item.x * item.y) / item.n
        d = sqrt(item.x_sq - (item.x ** 2 / item.n)) * sqrt(item.y_sq - (item.y ** 2 / item.n))
        r = n / d
        return (n1, n2), {'pearson': r}

    return sdf.rdd \
        .flatMap(lambda r: to_items(r.asDict())) \
        .reduceByKey(lambda a, b: add_items(a, b)) \
        .map(lambda tup: to_results(tup)) \
        .sortByKey()
