import builtins

import numpy as np
import pandas as pd
import pytest

from pypair.association import (
    agreement,
    binary_binary,
    binary_continuous,
    categorical_categorical,
    categorical_continuous,
    concordance,
    confusion,
    continuous_continuous,
)
from pypair.biserial import Biserial, BiserialStats, pd_isna as biserial_pd_isna
from pypair.contingency import (
    AgreementStats,
    AgreementTable,
    BinaryMixin,
    BinaryStats,
    BinaryTable,
    CategoricalStats,
    ConfusionStats,
    ContingencyTable,
)
from pypair.continuous import ConcordanceStats, pd_isna as continuous_pd_isna
from pypair.util import MeasureMixin, to_numpy


class ExampleMeasure(MeasureMixin):
    @property
    def alpha(self):
        return 1


@pytest.mark.parametrize(
    ("func", "args"),
    [
        (confusion, ([0, 1], [0, 1])),
        (binary_binary, ([0, 1], [0, 1])),
        (categorical_categorical, (["a", "b"], ["a", "b"])),
        (agreement, (["a", "b"], ["a", "b"])),
        (binary_continuous, ([0, 1], [1.0, 2.0])),
        (categorical_continuous, (["a", "b"], [1.0, 2.0])),
        (concordance, ([1, 2], [2, 1])),
        (continuous_continuous, ([1.0, 2.0], [2.0, 3.0])),
    ],
)
def test_association_functions_reject_invalid_measures(func, args):
    with pytest.raises(ValueError, match="not a valid association measure"):
        func(*args, measure="not-a-measure")


def test_agreement_and_stat_builders_cover_remaining_paths():
    ratings_a = ["yes", "no", "yes", "no"]
    ratings_b = ["yes", "no", "no", "yes"]

    agreement_table = AgreementTable(ratings_a, ratings_b, a_vals=["no", "yes"], b_vals=["no", "yes"])
    assert isinstance(agreement(ratings_a, ratings_b, measure="chohen_k"), float)
    assert len(agreement_table.cohen_light_k) == 2

    with pytest.raises(ValueError, match="Table not symmetric"):
        AgreementTable(["x"], ["y"], a_vals=["x"], b_vals=["y", "z"])

    categorical_stats = CategoricalStats([[2, 1], [1, 2]])
    binary_stats = BinaryStats([[4, 2], [1, 5]])
    confusion_stats = ConfusionStats([[5, 1], [2, 4]])
    agreement_stats = AgreementStats([[3, 1], [1, 3]])

    assert categorical_stats.phi > 0
    assert BinaryMixin.chisq.fget(binary_stats) == binary_stats.pearson_i
    assert binary_stats.tversky_index() > 0
    assert confusion_stats.acc > 0
    assert len(agreement_stats.cohen_light_k) == 2

    with pytest.raises(ValueError, match="Table not symmetric"):
        AgreementStats([[1, 2, 3], [4, 5, 6]])


def test_contingency_helpers_cover_edge_cases():
    assert ContingencyTable._to_binary_counts([1, 1, 0, None], [1, 0, 1, 1]) == (2, 2, 2, 1)

    table = ContingencyTable._to_categorical_counts(
        ["x", None, "z", "ignore"],
        ["left", "left", "right", "outside"],
        a_vals=["x", "z"],
        b_vals=["left", "right"],
    )
    assert table == [[2, 1], [1, 2]]


def test_binary_table_special_cases_and_biserial_stats():
    positive_only = BinaryStats([[3, 0], [1, 2]])
    negative_only = BinaryStats([[0, 2], [1, 3]])

    assert positive_only.tetrachoric == 1.0
    assert negative_only.tetrachoric == -1.0
    assert BinaryTable([1, 1, 0], [1, 0, 1]).tversky_index() > 0

    stats = BiserialStats(10, 0.6, 1.0, 4.0, 2.0)
    assert np.isfinite(stats.biserial)
    assert np.isfinite(stats.point_biserial)
    assert np.isfinite(stats.rank_biserial)

    with pytest.raises(ValueError, match="No valid samples"):
        Biserial([None, None], [np.nan, np.nan])


def test_concordance_stats_and_pd_isna_fallbacks(monkeypatch):
    stats = ConcordanceStats(1, 1, 1, 1, 2, 6)
    assert np.isfinite(stats.kendall_tau)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("forced failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert np.array_equal(biserial_pd_isna(np.array([None, 1])), np.array([True, False]))
    assert np.array_equal(continuous_pd_isna(np.array([None, 1])), np.array([True, False]))


def test_measure_mixin_and_to_numpy_cover_all_paths():
    example = ExampleMeasure()
    assert ExampleMeasure.measures() == ["alpha"]
    assert example.get("alpha") == 1
    assert example.get_measures() == ["alpha"]

    arr = np.array([1, 2, 3])
    assert to_numpy(arr).tolist() == [1, 2, 3]
    assert to_numpy(arr, dtype=float).dtype == float
    assert to_numpy(pd.Series([1, 2, 3])).tolist() == [1, 2, 3]
    assert to_numpy((value for value in [1, 2, 3]), dtype=float).tolist() == [1.0, 2.0, 3.0]
