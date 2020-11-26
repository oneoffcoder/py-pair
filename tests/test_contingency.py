import random

import numpy as np
import pandas as pd
from nose import with_setup

from pypair.association import categorical_categorical


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
def test_contingency_with_nulls():
    """
    Tests creating contingency table with nulls.

    :return: None.
    """
    df = pd.DataFrame([
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, None),
        (None, 0),
        (None, None)
    ], columns=['a', 'b'])
    v = categorical_categorical(df.a, df.b, measure='phi')
    print(v)
