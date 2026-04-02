from dataclasses import dataclass

import pytest

import pypair.spark as spark
from pypair.util import UndefinedMeasureError


@dataclass
class FakeRow:
    values: dict

    def asDict(self):
        return dict(self.values)


class FakeRDD:
    def __init__(self, items):
        self.items = list(items)

    def flatMap(self, func):
        return FakeRDD(item for value in self.items for item in func(value))

    def reduceByKey(self, func):
        reduced = {}
        for key, value in self.items:
            reduced[key] = func(reduced[key], value) if key in reduced else value
        return FakeRDD(reduced.items())

    def map(self, func):
        return FakeRDD(func(value) for value in self.items)

    def sortByKey(self):
        return FakeRDD(sorted(self.items, key=lambda item: item[0]))

    def groupByKey(self):
        grouped = {}
        for key, value in self.items:
            grouped.setdefault(key, []).append(value)
        return FakeRDD((key, tuple(values)) for key, values in grouped.items())

    def collect(self):
        return list(self.items)

    def count(self):
        return len(self.items)


class FakeDataFrame:
    def __init__(self, rows):
        self.rdd = FakeRDD(FakeRow(row) for row in rows)


def test_spark_private_helpers():
    assert spark.__as_key("z", "a") == ("a", "z")
    assert spark.__add_abcd_counts((1, 2, 3, 4), (4, 3, 2, 1)) == (5, 5, 5, 5)
    assert spark.__add_concordance_counts((1, 2, 3, 4, 5, 6), (6, 5, 4, 3, 2, 1)) == (7, 7, 7, 7, 7, 7)

    counts = spark.__to_abcd_counts({"b": 1, "a": 0, "c": None})
    assert (("a", "b"), (0, 1, 0, 0)) in counts
    assert any(key == ("a", "c") and value == (0, 0, 0, 0) for key, value in counts)


def test_spark_contingency_helper_and_public_functions():
    binary_rows = [
        {"x1": 1, "x2": 1, "x3": 1, "x4": 1},
        {"x1": 1, "x2": 0, "x3": 1, "x4": 0},
        {"x1": 0, "x2": 1, "x3": 0, "x4": 1},
        {"x1": 0, "x2": 0, "x3": 0, "x4": 0},
        {"x1": 1, "x2": 0, "x3": 0, "x4": 1},
        {"x1": 0, "x2": 1, "x3": 1, "x4": 0},
    ]
    categorical_rows = [
        {"a": "up", "b": "up", "c": "up"},
        {"a": "up", "b": "up", "c": "down"},
        {"a": "up", "b": "down", "c": "up"},
        {"a": "up", "b": "down", "c": "down"},
        {"a": "down", "b": "up", "c": "up"},
        {"a": "down", "b": "up", "c": "down"},
        {"a": "down", "b": "down", "c": "up"},
        {"a": "down", "b": "down", "c": "down"},
    ]
    binary_continuous_rows = [
        {"flag1": 1, "flag2": 0, "score1": 1.0, "score2": 4.0},
        {"flag1": 0, "flag2": 1, "score1": 2.0, "score2": 3.0},
        {"flag1": 1, "flag2": 0, "score1": 3.0, "score2": 2.0},
        {"flag1": 0, "flag2": 1, "score1": 4.0, "score2": 1.0},
    ]
    concordance_rows = [
        {"x": 0, "y": 0},
        {"x": 1, "y": 1},
        {"x": 0, "y": 1},
        {"x": 1, "y": 0},
        {"x": 0, "y": 0},
    ]
    continuous_rows = [
        {"u": 1.0, "v": 2.0, "w": 3.0},
        {"u": 2.0, "v": 3.0, "w": 5.0},
        {"u": 3.0, "v": 5.0, "w": 8.0},
    ]

    contingency = spark.__get_contingency_table(FakeDataFrame(categorical_rows)).collect()
    assert contingency[0][1]

    assert spark.binary_binary(FakeDataFrame(binary_rows)).count() == 6
    assert spark.confusion(FakeDataFrame(binary_rows)).count() == 6
    assert spark.categorical_categorical(FakeDataFrame(categorical_rows)).count() == 3
    assert spark.agreement(FakeDataFrame(categorical_rows)).count() == 3

    binary_continuous_result = spark.binary_continuous(
        FakeDataFrame(binary_continuous_rows),
        ["flag1", "flag2"],
        ["score1", "score2"],
    ).collect()
    assert binary_continuous_result
    assert set(binary_continuous_result[0][1]) == {"biserial", "point_biserial", "rank_biserial"}

    categorical_continuous_result = spark.categorical_continuous(
        FakeDataFrame(
            [
                {"group": "a", "segment": "x", "score": 1.0, "other": 2.0},
                {"group": "a", "segment": "y", "score": 2.0, "other": 1.5},
                {"group": "b", "segment": "x", "score": 3.0, "other": 3.0},
                {"group": "b", "segment": "y", "score": 4.0, "other": 4.5},
            ]
        ),
        ["group", "segment"],
        ["score", "other"],
    ).collect()
    assert categorical_continuous_result
    assert categorical_continuous_result[0][1]["eta"] >= 0

    concordance_result = spark.concordance(FakeDataFrame(concordance_rows)).collect()
    assert concordance_result
    assert "kendall_tau" in concordance_result[0][1]

    continuous_result = spark.continuous_continuous(FakeDataFrame(continuous_rows)).collect()
    assert continuous_result
    assert "pearson" in continuous_result[0][1]


def test_spark_pseudocount_can_be_disabled():
    degenerate_binary = FakeDataFrame([{"x1": 1, "x2": 1}, {"x1": 1, "x2": 1}, {"x1": 1, "x2": 1}])
    assert spark.binary_binary(degenerate_binary).count() == 1
    with pytest.raises(UndefinedMeasureError, match=r"pair \(x1, x2\) in binary_binary"):
        spark.binary_binary(degenerate_binary, pseudocount=False).collect()

    degenerate_concordance = FakeDataFrame([{"x": 1, "y": 1}, {"x": 1, "y": 1}, {"x": 1, "y": 1}])
    assert spark.concordance(degenerate_concordance).count() == 1
    with pytest.raises(UndefinedMeasureError, match=r"pair \(x, y\) in concordance"):
        spark.concordance(degenerate_concordance, pseudocount=False).collect()
