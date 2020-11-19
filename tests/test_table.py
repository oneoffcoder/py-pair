import random

import numpy as np
from nose import with_setup

from pypair.contingency import BinaryTable, CategoricalTable, ConfusionMatrix, CategoricalMeasures


def setup():
    """
    Setup.
    :return: None.
    """
    np.random.seed(37)
    random.seed(37)


def teardown():
    """
    Teardown.
    :return: None.
    """
    pass


@with_setup(setup, teardown)
def test_confusion_matrix_creation():
    """
    Tests creating ConfusionMatrix.

    :return: None.
    """
    a = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    b = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1]

    table = ConfusionMatrix(a, b)
    for measure in ConfusionMatrix.measures():
        stats = table.get(measure)
        if isinstance(stats, tuple):
            print(f'{stats[0]:.8f}, {stats[1]:.8f}: {measure}')
        else:
            print(f'{stats:.8f}: {measure}')


@with_setup(setup, teardown)
def test_binary_table_creation():
    """
    Tests creating BinaryTable. The data is simulated from this `site <https://www.mathsisfun.com/data/chi-square-test.html>`_.

    :return: None.
    """
    get_data = lambda x, y, n: [(x, y) for _ in range(n)]
    data = get_data(1, 1, 207) + get_data(1, 0, 282) + get_data(0, 1, 231) + get_data(0, 0, 242)
    a = [a for a, _ in data]
    b = [b for _, b in data]

    table = BinaryTable(a, b)
    for measure in BinaryTable.measures():
        stats = table.get(measure)
        if isinstance(stats, tuple):
            print(f'{stats[0]:.8f}, {stats[1]:.8f}: {measure}')
        else:
            print(f'{stats:.8f}: {measure}')


@with_setup(setup, teardown)
def test_categorical_table_creation():
    """
    Tests creating CategoricalTable.

    :return: None.
    """
    a = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    b = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1]

    table = CategoricalTable(a, b)
    assert 0.4 == table.get('chisq')
    assert 0.2 == table.get('phi')


@with_setup(setup, teardown)
def test_categorical_measures():
    """
    Tests categorical measures.

    :return: None.
    """
    data = [(('x0', 'x1'), [[21, 9], [17, 23], [16, 14]]), (('x0', 'x2'), [[8, 12, 10], [16, 7, 17], [6, 11, 13]]),
            (('x0', 'x3'), [[21, 9], [25, 15], [13, 17]]), (('x1', 'x2'), [[16, 19, 19], [14, 11, 21]]),
            (('x1', 'x3'), [[35, 19], [24, 22]]), (('x2', 'x3'), [[19, 11], [16, 14], [24, 16]])]

    for (x1, x2), table in data:
        print(x1, x2, table)
        computer = CategoricalMeasures(table)
        for m in computer.measures():
            print(m, computer.get(m))
        print('-' * 15)
