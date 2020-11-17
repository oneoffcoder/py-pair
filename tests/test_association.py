import random

import numpy as np
from nose import with_setup

from pypair.association import binary_binary, categorical_categorical, \
    binary_continuous, concordance
from pypair.biserial import Biserial
from pypair.continuous import Concordance
from pypair.table import BinaryTable, CategoricalTable


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

    for m in BinaryTable.get_measures():
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

    for m in CategoricalTable.get_measures():
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

    for m in Biserial.get_measures():
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

    for m in Concordance.get_measures():
        r = concordance(a, b, m)
        print(f'{r}: {m}')
