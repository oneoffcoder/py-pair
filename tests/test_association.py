import random

import numpy as np
from nose import with_setup

from pypair.association import binary_binary, categorical_categorical, \
    binary_continuous, concordance, categorical_continuous, continuous_continuous, confusion
from pypair.biserial import Biserial
from pypair.contigency import BinaryTable, CategoricalTable, ConfusionMatrix
from pypair.continuous import Concordance, CorrelationRatio, Continuous


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
def test_binary_binary():
    """
    Tests binary-binary.

    :return: None.
    """
    get_data = lambda x, y, n: [(x, y) for _ in range(n)]
    data = get_data(1, 1, 207) + get_data(1, 0, 282) + get_data(0, 1, 231) + get_data(0, 0, 242)
    a = [a for a, _ in data]
    b = [b for _, b in data]

    for m in BinaryTable.measures():
        r = binary_binary(a, b, m)
        print(f'{r}: {m}')


@with_setup(setup, teardown)
def test_categorical_categorical():
    """
    Tests categorical-categorical.

    :return: None.
    """
    get_data = lambda x, y, n: [(x, y) for _ in range(n)]
    data = get_data(1, 1, 207) + get_data(1, 0, 282) + get_data(0, 1, 231) + get_data(0, 0, 242)
    a = [a for a, _ in data]
    b = [b for _, b in data]

    for m in CategoricalTable.measures():
        r = categorical_categorical(a, b, m)
        print(f'{r}: {m}')


@with_setup(setup, teardown)
def test_binary_continuous():
    """
    Tests binary-continuous.

    :return: None.
    """
    get_data = lambda x, y, n: [(x, y) for _ in range(n)]
    data = get_data(1, 1, 207) + get_data(1, 0, 282) + get_data(0, 1, 231) + get_data(0, 0, 242)
    a = [a for a, _ in data]
    b = [b for _, b in data]

    for m in Biserial.measures():
        r = binary_continuous(a, b, m)
        print(f'{r}: {m}')


@with_setup(setup, teardown)
def test_concordance():
    """
    Tests concordance.

    :return: None.
    """
    a = [1, 2, 3]
    b = [3, 2, 1]

    for m in Concordance.measures():
        r = concordance(a, b, m)
        print(f'{r}: {m}')


@with_setup(setup, teardown)
def test_categorical_continuous():
    """
    Tests categorical-continuous. Data taken from `Wikipedia <https://en.wikipedia.org/wiki/Correlation_ratio>`_.

    :return: None.
    """
    data = [
        ('a', 45), ('a', 70), ('a', 29), ('a', 15), ('a', 21),
        ('g', 40), ('g', 20), ('g', 30), ('g', 42),
        ('s', 65), ('s', 95), ('s', 80), ('s', 70), ('s', 85), ('s', 73)
    ]
    x = [x for x, _ in data]
    y = [y for _, y in data]
    for m in CorrelationRatio.measures():
        r = categorical_continuous(x, y, m)
        print(f'{r}: {m}')


@with_setup(setup, teardown)
def test_continuous_continuous():
    """
    Tests continuous-continuous.

    :return: None.
    """
    x = [x for x in range(10)]
    y = [y for y in range(10)]
    for m in Continuous.measures():
        r = continuous_continuous(x, y, m)
        print(f'{r}: {m}')


@with_setup(setup, teardown)
def test_confusion():
    """
    Tests confusion matrix. Data taken from `here <https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/>`_.

    :return: None
    """
    tn = [(0, 0) for _ in range(50)]
    fp = [(0, 1) for _ in range(10)]
    fn = [(1, 0) for _ in range(5)]
    tp = [(1, 1) for _ in range(100)]
    data = tn + fp + fn + tp
    a = [a for a, _ in data]
    b = [b for _, b in data]

    for m in ConfusionMatrix.measures():
        r = confusion(a, b, m)
        print(f'{r}: {m}')
