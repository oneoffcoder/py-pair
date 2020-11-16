import random

import numpy as np
from nose import with_setup
from numpy.testing import assert_array_equal

from pypair.table import BinaryTable, CategoricalTable, ConfusionMatrix
from pypair.association import kendall_tau


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


def test_kendall_tau():
    """
    Tests Kendall's :math:`\\tau`.

    :return: None.
    """
    a = [1, 2, 3]
    b = [3, 2, 1]

    t = kendall_tau(a, b)
    print(t)
