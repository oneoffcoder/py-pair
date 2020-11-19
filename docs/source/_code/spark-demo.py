import json
from random import choice

import pandas as pd
from pyspark.sql import SparkSession

from pypair.spark import binary_binary, confusion, categorical_categorical, agreement


def _get_binary_binary_data(spark):
    """
    Gets dummy binary-binary data in a Spark dataframe.

    :return: Spark dataframe.
    """
    get_data = lambda x, y, n: [(x, y) * 2 for _ in range(n)]
    data = get_data(1, 1, 207) + get_data(1, 0, 282) + get_data(0, 1, 231) + get_data(0, 0, 242)
    pdf = pd.DataFrame(data, columns=['x1', 'x2', 'x3', 'x4'])
    sdf = spark.createDataFrame(pdf)
    return sdf


def _get_confusion_data(spark):
    """
    Gets dummy binary-binary data in Spark dataframe. For use with confusion matrix analysis.

    :return: Spark dataframe.
    """
    tn = [(0, 0) * 2 for _ in range(50)]
    fp = [(0, 1) * 2 for _ in range(10)]
    fn = [(1, 0) * 2 for _ in range(5)]
    tp = [(1, 1) * 2 for _ in range(100)]
    data = tn + fp + fn + tp
    pdf = pd.DataFrame(data, columns=['x1', 'x2', 'x3', 'x4'])
    sdf = spark.createDataFrame(pdf)
    return sdf


def _get_categorical_categorical_data(spark):
    """
    Gets dummy categorical-categorical data in Spark dataframe.

    :return: Spark dataframe.
    """
    x_domain = ['a', 'b', 'c']
    y_domain = ['a', 'b']

    get_x = lambda: choice(x_domain)
    get_y = lambda: choice(y_domain)
    get_data = lambda: {f'x{i}': v for i, v in enumerate((get_x(), get_y(), get_x(), get_y()))}

    pdf = pd.DataFrame([get_data() for _ in range(100)])
    sdf = spark.createDataFrame(pdf)
    return sdf


spark = None

try:
    # create a spark session
    spark = (SparkSession.builder
             .master('local[4]')
             .appName('local-testing-pyspark')
             .getOrCreate())

    # create some spark dataframes
    bin_sdf = _get_binary_binary_data(spark)
    con_sdf = _get_confusion_data(spark)
    cat_sdf = _get_categorical_categorical_data(spark)

    # call these methods to get the association measures
    bin_results = binary_binary(bin_sdf).collect()
    con_results = confusion(con_sdf).collect()
    cat_results = categorical_categorical(cat_sdf).collect()
    agr_results = agreement(bin_sdf).collect()

    # convert the lists to dictionaries
    bin_results = {tup[0]: tup[1] for tup in bin_results}
    con_results = {tup[0]: tup[1] for tup in con_results}
    cat_results = {tup[0]: tup[1] for tup in cat_results}
    agr_results = {tup[0]: tup[1] for tup in agr_results}

    # pretty print
    to_json = lambda r: json.dumps({f'{k[0]}_{k[1]}': v for k, v in r.items()}, indent=1)
    print(to_json(bin_results))
    print('-' * 10)
    print(to_json(con_results))
    print('*' * 10)
    print(to_json(cat_results))
    print('~' * 10)
    print(to_json(agr_results))
except Exception as e:
    print(e)
finally:
    try:
        spark.stop()
        print('closed spark')
    except Exception as e:
        print(e)
