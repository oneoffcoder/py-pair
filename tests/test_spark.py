import pytest

pyspark = pytest.importorskip('pyspark')
from pyspark.sql import SparkSession

from pypair.spark import (
    agreement,
    binary_binary,
    binary_continuous,
    categorical_continuous,
    concordance,
    confusion,
    continuous_continuous,
)


@pytest.fixture(scope='module')
def spark():
    session = SparkSession.builder.master('local[2]').appName('pypair-tests').getOrCreate()
    yield session
    session.stop()


def test_spark_pairwise_operations(spark):
    sdf_binary = spark.createDataFrame(
        [(1, 1, 1, 1), (1, 0, 1, 0), (0, 1, 0, 1), (0, 0, 0, 0), (1, 0, 0, 1), (0, 1, 1, 0)] * 10,
        ['x1', 'x2', 'x3', 'x4'],
    )
    assert binary_binary(sdf_binary).count() > 0
    assert confusion(sdf_binary).count() > 0
    assert agreement(sdf_binary).count() > 0

    sdf_bc = spark.createDataFrame(
        [(1, 2.0), (1, 3.5), (0, 1.1), (0, 0.7), (1, 4.2), (0, 2.4), (1, 5.1), (0, 1.8)] * 6,
        ['gender', 'years'],
    )
    assert binary_continuous(sdf_bc, ['gender'], ['years']).count() > 0

    sdf_cat = spark.createDataFrame(
        [('a', 1.0, 'a', 2.0), ('b', 2.0, 'b', 2.5), ('a', 1.5, 'b', 3.0), ('c', 2.5, 'a', 1.0)] * 8,
        ['x1', 'x2', 'x3', 'x4'],
    )
    assert categorical_continuous(sdf_cat, ['x1', 'x3'], ['x2', 'x4']).count() > 0
    assert continuous_continuous(sdf_cat.select('x2', 'x4')).count() > 0

    sdf_ord = spark.createDataFrame([(1, 4), (2, 3), (3, 2), (4, 1)] * 4, ['a', 'b'])
    assert concordance(sdf_ord).count() > 0
