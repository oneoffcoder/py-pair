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
from pypair.contingency import BinaryTable, CategoricalTable, ConfusionMatrix
from pypair.continuous import Concordance, Continuous, CorrelationRatio
from pypair.util import corr


def test_binary_binary_and_confusion_measures_available():
    data = [(1, 1)] * 40 + [(1, 0)] * 20 + [(0, 1)] * 10 + [(0, 0)] * 30
    a = [x for x, _ in data]
    b = [y for _, y in data]

    for measure in BinaryTable.measures():
        value = binary_binary(a, b, measure)
        if isinstance(value, tuple):
            assert np.all(np.isfinite(value))
        else:
            assert np.all(np.isfinite(value))

    for measure in ConfusionMatrix.measures():
        value = confusion(a, b, measure)
        if isinstance(value, tuple):
            assert np.all(np.isfinite(value))
        else:
            assert np.isfinite(value)


def test_categorical_and_continuous_interfaces():
    x = np.array(["a", "a", "a", "g", "g", "s", "s", "s", "s"])
    y = np.array([45, 70, 29, 40, 20, 65, 95, 80, 70], dtype=float)

    for measure in CategoricalTable.measures():
        assert np.isfinite(categorical_categorical(x, x, measure))

    for measure in CorrelationRatio.measures():
        value = categorical_continuous(x, y, measure)
        if isinstance(value, tuple):
            assert np.all(np.isfinite(value))
        else:
            assert np.isfinite(value)


def test_biserial_concordance_continuous_interfaces():
    b = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0])
    c = np.array([10, 11, 6, 4, 3, 12, 2, 2, 1], dtype=float)

    for measure in ["biserial", "point_biserial", "rank_biserial"]:
        assert np.isfinite(binary_continuous(b, c, measure))

    x = np.arange(12, dtype=float)
    y = np.arange(12, dtype=float)

    for measure in Continuous.measures():
        corr_value, pvalue = continuous_continuous(x, y, measure)
        assert np.isfinite(corr_value)
        assert np.isfinite(pvalue)

    x_ord = [1, 2, 3, 4]
    y_ord = [4, 3, 2, 1]
    for measure in Concordance.measures():
        value = concordance(x_ord, y_ord, measure)
        if isinstance(value, tuple):
            assert np.all(np.isfinite(value))
        else:
            assert np.all(np.isfinite(value))


def test_corr_matrix_for_dataframe():
    df = pd.DataFrame(
        {
            "x1": ["on", "on", "off", "off"],
            "x2": ["on", "off", "on", "off"],
            "x3": ["off", "off", "on", "on"],
        }
    )

    matrix = corr(df, lambda a, b: categorical_categorical(a, b, measure="mutual_information"))
    assert matrix.shape == (3, 3)
    assert np.allclose(np.diag(matrix.values), 0.0)
    assert np.allclose(matrix.values, matrix.values.T)
