import numpy as np
import pandas as pd

from pypair.association import (
    binary_binary,
    binary_continuous,
    categorical_categorical,
    categorical_continuous,
    concordance,
    confusion,
    continuous_continuous,
)
from pypair.util import to_numpy


def test_to_numpy_accepts_series_and_generators():
    series = pd.Series([1, 2, 3])
    generator = (value for value in [4, 5, 6])

    assert np.array_equal(to_numpy(series), np.array([1, 2, 3]))
    assert np.array_equal(to_numpy(generator), np.array([4, 5, 6]))


def test_local_association_functions_accept_common_1d_containers():
    binary_a = [1, 1, 0, 0, 1, 0]
    binary_b = [1, 0, 1, 0, 1, 0]
    categorical_a = ["a", "a", "b", "b", "c", "c"]
    categorical_b = ["x", "y", "x", "y", "x", "y"]
    continuous_a = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    continuous_b = [0.1, 0.3, 0.5, 0.7, 1.8, 3.1]

    def variants(values):
        return [
            list(values),
            np.asarray(values),
            pd.Series(values),
        ]

    expected_jaccard = binary_binary(np.asarray(binary_a), np.asarray(binary_b), measure="jaccard")
    expected_acc = confusion(np.asarray(binary_a), np.asarray(binary_b), measure="acc")
    expected_phi = categorical_categorical(np.asarray(categorical_a), np.asarray(categorical_b), measure="phi")
    expected_biserial = binary_continuous(np.asarray(binary_a), np.asarray(continuous_a), measure="point_biserial")
    expected_eta = categorical_continuous(np.asarray(categorical_a), np.asarray(continuous_a), measure="eta")
    expected_tau = concordance(np.asarray(continuous_a), np.asarray(continuous_b), measure="kendall_tau")
    expected_pearson = continuous_continuous(np.asarray(continuous_a), np.asarray(continuous_b), measure="pearson")

    for left, right in zip(variants(binary_a), variants(binary_b)):
        assert binary_binary(left, right, measure="jaccard") == expected_jaccard
        assert confusion(left, right, measure="acc") == expected_acc

    for left, right in zip(variants(categorical_a), variants(categorical_b)):
        assert categorical_categorical(left, right, measure="phi") == expected_phi

    for left, right in zip(variants(binary_a), variants(continuous_a)):
        assert binary_continuous(left, right, measure="point_biserial") == expected_biserial

    for left, right in zip(variants(categorical_a), variants(continuous_a)):
        assert categorical_continuous(left, right, measure="eta") == expected_eta

    for left, right in zip(variants(continuous_a), variants(continuous_b)):
        assert concordance(left, right, measure="kendall_tau") == expected_tau
        assert continuous_continuous(left, right, measure="pearson") == expected_pearson
