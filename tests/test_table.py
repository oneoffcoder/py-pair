import random

import numpy as np
from nose import with_setup
from numpy.testing import assert_array_equal, assert_almost_equal

from pypair.table import BinaryTable, CategoricalTable, ConfusionMatrix


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
    Tests creating BinaryTable.
    :return: None.
    """
    a = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    b = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1]

    table = ConfusionMatrix(a, b)
    print(table.tp)
    print(table.fp)
    print(table.fn)
    print(table.tn)
    print(table.n)
    print(table.tpr)
    print(table.tnr)
    print(table.ppv)
    print(table.npv)
    print(table.fnr)
    print(table.fpr)
    print(table.fdr)
    print(table.fomr)
    print(table.pt)
    print(table.ts)
    print(table.acc)
    print(table.ba)
    print(table.f1)
    print(table.mcc)
    print(table.bm)
    print(table.mk)
    print(table.sensitivity)
    print(table.specificity)
    print(table.precision)
    print(table.recall)


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

    assert_almost_equal(table.chisq, 0.4, decimal=5)
    assert_almost_equal(table.phi, 0.2, decimal=5)

    assert_almost_equal(table.jaccard_similarity, 0.42857142857142855, decimal=5)
    assert_almost_equal(table.jaccard_distance, 0.5714285714285714, decimal=5)

    assert_almost_equal(table.tanimoto_similarity, 0.75, decimal=5)
    assert_almost_equal(table.tanimoto_distance, 0.4150374992788438, decimal=5)

    assert_almost_equal(table.cramer_v, 0.2, decimal=5)
    assert_almost_equal(table.rand_index, 0.6, decimal=5)
    print(table.adjusted_rand_index)
    print(table.fowlkes_mallows_index)
    print(table.mcnemar_test)
    print(table.odds_ratio)
    print(table.contingency_coefficient)
    print(table.tetrachoric_correlation)
    print(table.goodman_kruskal_lambda)
    print(table.goodman_kruskal_lambda_reversed)
    print(table.tschuprow_t)
    print(table.uncertainty_coefficient)
    print(table.uncertainty_coefficient_reversed)
    print(table.mutual_information)
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
