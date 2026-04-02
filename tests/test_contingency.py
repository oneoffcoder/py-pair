import numpy as np

from pypair.association import categorical_categorical


def test_contingency_with_nulls_is_finite():
    a = [0, 0, 1, 1, 0, None, None]
    b = [0, 1, 0, 1, None, 0, None]

    phi = categorical_categorical(a, b, measure='phi')
    assert np.isfinite(phi)
