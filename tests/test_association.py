import random

import numpy as np
from nose import with_setup

from pypair.association import kendall_tau, somers_d, goodman_kruskal_gamma


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
def test_kendall_tau():
    """
    Tests Kendall's :math:`\\tau`.

    :return: None.
    """
    a = [1, 2, 3]
    b = [3, 2, 1]

    t = kendall_tau(a, b)
    assert t == -1.0


@with_setup(setup, teardown)
def test_somers_d():
    """
    Tests Somers' d.

    :return: None.
    """
    a = [1, 2, 3]
    b = [3, 2, 1]

    t = somers_d(a, b)
    assert t == (-1.0, -1.0)


@with_setup(setup, teardown)
def test_goodman_kruskal_gamma():
    """
    Tests Goodman-Kruskal :math:`\\gamma`

    :return: None.
    """
    a = [1, 2, 3]
    b = [3, 2, 1]

    t = goodman_kruskal_gamma(a, b)
    assert t == -1.0
