from itertools import combinations

from pypair.contigency import BinaryMeasures, CmMeasures


def __to_abcd_counts(d):
    """
    Maps the paired keys in the dictionary and their associated values to a form: ``(k1, k2), (a, b, c, d)``.

    :param d: A dictionary. Names are variable names. Values are 0 or 1.
    :return: A list of tuples of the form: (k1, k2), (a, b, c, d).
    """

    def as_key(k1, k2):
        """
        Creates a key (tuple) out of the two specified. The key is always ordered.
        If k2 < k1, then (k2, k1), else, (k1, k2).

        :param k1: Key (string).
        :param k2: Key (string).
        :return: (k1, k2) or (k2, k1).
        """
        keys = sorted([k1, k2])
        return keys[0], keys[1]

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
        return as_key(k1, k2), as_count(v1, v2)

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


def binary_binary(sdf):
    """
    Gets all the pairwise binary-binary association measures. The result is a Spark pair-RDD,
    where the keys are tuples of variable names e.g. (k1, k2), and values are dictionaries
    of association names and measures e.g. {'phi': 1, 'lambda': 0.8}.

    :param sdf: Spark dataframe.
    :return: Spark pair-RDD.
    """

    def to_results(counts):
        (x1, x2), (a, b, c, d) = counts

        a = max(1, a)
        b = max(1, b)
        c = max(1, c)
        d = max(1, d)

        computer = BinaryMeasures(a, b, c, d)
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

    :param sdf: Spark dataframe.
    :return: Spark pair-RDD.
    """

    def to_results(counts):
        (x1, x2), (tp, fn, fp, tn) = counts

        tp = max(1, tp)
        fn = max(1, fn)
        fp = max(1, fp)
        tn = max(1, tn)

        computer = CmMeasures(tp, fn, fp, tn)
        measures = {m: computer.get(m) for m in computer.measures()}
        return (x1, x2), measures

    return sdf.rdd \
        .flatMap(lambda r: __to_abcd_counts(r.asDict())) \
        .reduceByKey(lambda a, b: __add_abcd_counts(a, b)) \
        .sortByKey() \
        .map(lambda counts: to_results(counts))
