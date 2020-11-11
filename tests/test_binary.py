import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from nose import with_setup
import random

from pypair.binary import BinaryTable, CategoricalTable


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
def test_binary_table_creation():
    """
    Tests creating BinaryTable.
    :return: None.
    """
    a = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    b = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1]

    table = BinaryTable(a, b)
    assert_array_equal(table.observed, [[3, 2], [2, 3]])
    assert_array_equal(table.expected, [[2.5, 2.5], [2.5, 2.5]])
    assert 0.4 == table.chisq
    assert 0.2 == table.phi

    assert_almost_equal(0.5714285714285714, table.jaccard_similarity, decimal=0.5)
    assert_almost_equal(0.42857142857142855, table.jaccard_distance, decimal=0.5)


@with_setup(setup, teardown)
def test_categorical_table_creation():
    """
    Tests creating CategoricalTable.
    :return: None.
    """
    a = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    b = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1]

    table = CategoricalTable(a, b)
    assert_array_equal(table.observed, [[3, 2], [2, 3]])
    assert_array_equal(table.expected, [[2.5, 2.5], [2.5, 2.5]])
    assert 0.4 == table.chisq
    assert 0.2 == table.phi
