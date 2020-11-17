from pypair.association import confusion
from pypair.contigency import ConfusionMatrix


def get_data():
    """
    Data taken from `here <https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/>`_.
    A pair of binary variables, `a` and `p`, are returned.

    :return: a, p
    """
    tn = [(0, 0) for _ in range(50)]
    fp = [(0, 1) for _ in range(10)]
    fn = [(1, 0) for _ in range(5)]
    tp = [(1, 1) for _ in range(100)]
    data = tn + fp + fn + tp
    a = [a for a, _ in data]
    p = [b for _, b in data]
    return a, p


a, p = get_data()

# if you need to quickly get just one association measure
r = confusion(a, p, measure='acc')
print(r)

print('-' * 15)

# you can also get a list of available association measures
# and loop over to call confusion(...)
# this is more convenient, but less fast
for m in ConfusionMatrix.measures():
    r = confusion(a, p, m)
    print(f'{r}: {m}')

print('-' * 15)

# if you need multiple association measures, then
# build the confusion matrix table
# this is less convenient, but much faster
matrix = ConfusionMatrix(a, p)
for m in matrix.measures():
    r = matrix.get(m)
    print(f'{r}: {m}')
