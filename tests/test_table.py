import numpy as np

from pypair.contingency import BinaryTable, CategoricalTable, ConfusionMatrix


def test_confusion_matrix_creation():
    a = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    b = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1]

    table = ConfusionMatrix(a, b)
    for measure in ConfusionMatrix.measures():
        value = table.get(measure)
        if isinstance(value, tuple):
            assert np.all(np.isfinite(value))
        else:
            assert np.isfinite(value)


def test_binary_and_categorical_table_creation():
    data = [(1, 1)] * 50 + [(1, 0)] * 25 + [(0, 1)] * 30 + [(0, 0)] * 45
    a = [x for x, _ in data]
    b = [y for _, y in data]

    binary = BinaryTable(a, b)
    categorical = CategoricalTable(a, b)

    assert binary.get('chisq') > 0
    assert categorical.get('phi') > 0
