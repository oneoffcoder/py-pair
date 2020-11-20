import json
from random import choice

import pandas as pd
from pyspark.sql import SparkSession

from pypair.spark import binary_binary, confusion, categorical_categorical, agreement, binary_continuous, concordance, \
    categorical_continuous, continuous_continuous


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


def _get_binary_continuous_data(spark):
    """
    Gets dummy `binary-continuous data <https://www.slideshare.net/MuhammadKhalil66/point-biserial-correlation-example>`_.

    :return: Spark dataframe.
    """
    data = [
        (1, 10), (1, 11), (1, 6), (1, 11), (0, 4),
        (0, 3), (1, 12), (0, 2), (0, 2), (0, 1)
    ]
    pdf = pd.DataFrame(data, columns=['gender', 'years'])
    sdf = spark.createDataFrame(pdf)
    return sdf


def _get_concordance_data(spark):
    """
    Gets dummy concordance data.

    :return: Spark dataframe.
    """
    a = [1, 2, 3]
    b = [3, 2, 1]
    pdf = pd.DataFrame({'a': a, 'b': b, 'c': a, 'd': b})
    sdf = spark.createDataFrame(pdf)
    return sdf


def _get_categorical_continuous_data(spark):
    data = [
        ('a', 45), ('a', 70), ('a', 29), ('a', 15), ('a', 21),
        ('g', 40), ('g', 20), ('g', 30), ('g', 42),
        ('s', 65), ('s', 95), ('s', 80), ('s', 70), ('s', 85), ('s', 73)
    ]
    data = [tup * 2 for tup in data]
    pdf = pd.DataFrame(data, columns=['x1', 'x2', 'x3', 'x4'])
    sdf = spark.createDataFrame(pdf)
    return sdf


def _get_continuous_continuous_data(spark):
    """
    Gets dummy continuous-continuous data.
    See `site <http://onlinestatbook.com/2/describing_bivariate_data/calculation.html>`_.

    :return: Spark dataframe.
    """
    data = [
        (12, 9),
        (10, 12),
        (9, 12),
        (14, 11),
        (10, 8),
        (11, 9),
        (10, 9),
        (10, 6),
        (14, 12),
        (9, 11),
        (11, 12),
        (10, 7),
        (11, 13),
        (15, 14),
        (8, 11),
        (11, 11),
        (9, 8),
        (9, 9),
        (10, 11),
        (12, 9),
        (11, 12),
        (10, 12),
        (9, 7),
        (7, 9),
        (12, 14)
    ]
    pdf = pd.DataFrame([item * 2 for item in data], columns=['x1', 'x2', 'x3', 'x4'])
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
    bcn_sdf = _get_binary_continuous_data(spark)
    crd_sdf = _get_concordance_data(spark)
    ccn_sdf = _get_categorical_continuous_data(spark)
    cnt_sdf = _get_continuous_continuous_data(spark)

    # call these methods to get the association measures
    bin_results = binary_binary(bin_sdf).collect()
    con_results = confusion(con_sdf).collect()
    cat_results = categorical_categorical(cat_sdf).collect()
    agr_results = agreement(bin_sdf).collect()
    bcn_results = binary_continuous(bcn_sdf, binary=['gender'], continuous=['years']).collect()
    crd_results = concordance(crd_sdf).collect()
    ccn_results = categorical_continuous(ccn_sdf, ['x1', 'x3'], ['x2', 'x4']).collect()
    cnt_results = continuous_continuous(cnt_sdf).collect()

    # convert the lists to dictionaries
    bin_results = {tup[0]: tup[1] for tup in bin_results}
    con_results = {tup[0]: tup[1] for tup in con_results}
    cat_results = {tup[0]: tup[1] for tup in cat_results}
    agr_results = {tup[0]: tup[1] for tup in agr_results}
    bcn_results = {tup[0]: tup[1] for tup in bcn_results}
    crd_results = {tup[0]: tup[1] for tup in crd_results}
    ccn_results = {tup[0]: tup[1] for tup in ccn_results}
    cnt_results = {tup[0]: tup[1] for tup in cnt_results}

    # pretty print
    to_json = lambda r: json.dumps({f'{k[0]}_{k[1]}': v for k, v in r.items()}, indent=1)
    print(to_json(bin_results))
    print('-' * 10)
    print(to_json(con_results))
    print('*' * 10)
    print(to_json(cat_results))
    print('~' * 10)
    print(to_json(agr_results))
    print('-' * 10)
    print(to_json(bcn_results))
    print('=' * 10)
    print(to_json(crd_results))
    print('`' * 10)
    print(to_json(ccn_results))
    print('/' * 10)
    print(to_json(cnt_results))
except Exception as e:
    print(e)
finally:
    try:
        spark.stop()
        print('closed spark')
    except Exception as e:
        print(e)
