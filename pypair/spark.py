from itertools import combinations

from pypair.contigency import BinaryMeasures


def binary_binary(sdf):
    def to_counts(d):
        def as_key(k1, k2):
            keys = sorted([k1, k2])
            return keys[0], keys[1]

        def as_count(v1, v2):
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
            v1, v2 = d[k1], d[k2]
            return as_key(k1, k2), as_count(v1, v2)

        return [transform(k1, k2) for k1, k2 in combinations(d.keys(), 2)]

    def add_counts(a, b):
        return a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]

    def to_results(counts):
        (x1, x2), (a, b, c, d) = counts
        a = max(1, a)
        b = max(1, b)
        c = max(1, c)
        d = max(1, d)
        computer = BinaryMeasures(a, b, c, d)
        measures = {m: computer.get(m) for m in computer.measures()}
        return (x1, x2), measures

    results = sdf.rdd \
        .flatMap(lambda r: to_counts(r.asDict())) \
        .reduceByKey(lambda a, b: add_counts(a, b)) \
        .sortByKey() \
        .map(lambda counts: to_results(counts)) \
        .collect()
    return {tup[0]: tup[1] for tup in results}
