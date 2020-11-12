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

    assert_almost_equal(table.jaccard_similarity, 0.42857142857142855, decimal=5)
    assert_almost_equal(table.jaccard_distance, 0.5714285714285714, decimal=5)

    assert_almost_equal(table.tanimoto_similarity, 0.75, decimal=5)
    assert_almost_equal(table.tanimoto_distance, 0.4150374992788438, decimal=5)

    assert_almost_equal(table.cramer_v, 0.2, decimal=5)
    assert_almost_equal(table.rand_index, 0.6, decimal=5)
    print(table.adjusted_rand_index)
    print(table.mcnemar_test)
    print(table.odds_ratio)
    print(table.contigency_coefficient)
    print(table.tetrachoric_correlation)
    print(table.goodman_kruskal_lambda)
    print(table.goodman_kruskal_lambda_reversed)
    assert 1 == 1


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
