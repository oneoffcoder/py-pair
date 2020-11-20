import json
import logging
import random
import unittest
from random import choice

import pandas as pd
from pyspark.sql import SparkSession

from pypair.spark import binary_binary, confusion, categorical_categorical, agreement, binary_continuous, concordance, \
    categorical_continuous, continuous_continuous


class PySparkTest(unittest.TestCase):
    """
    PySpark test class.
    """

    @classmethod
    def supress_py4j_logging(cls):
        """
        Supresses p4j logging.

        :return: None.
        """
        logger = logging.getLogger('py4j')
        logger.setLevel(logging.WARN)

    @classmethod
    def create_pyspark_session(cls):
        """
        Creates a PySpark session.

        :return: PySpark session.
        """
        return (SparkSession.builder
                .master('local[4]')
                .appName('local-testing-pyspark')
                .getOrCreate())

    @classmethod
    def setUpClass(cls):
        """
        Sets up the class.

        :return: None.
        """
        cls.supress_py4j_logging()
        cls.spark = cls.create_pyspark_session()
        random.seed(37)

    @classmethod
    def tearDownClass(cls):
        """
        Tears down the class.

        :return: None.
        """
        cls.spark.stop()

    def _get_binary_binary_data(self):
        """
        Gets dummy binary-binary data in a Spark dataframe.

        :return: Spark dataframe.
        """
        get_data = lambda x, y, n: [(x, y) * 2 for _ in range(n)]
        data = get_data(1, 1, 207) + get_data(1, 0, 282) + get_data(0, 1, 231) + get_data(0, 0, 242)
        pdf = pd.DataFrame(data, columns=['x1', 'x2', 'x3', 'x4'])
        sdf = self.spark.createDataFrame(pdf)
        return sdf

    def _get_confusion_data(self):
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
        sdf = self.spark.createDataFrame(pdf)
        return sdf

    def _get_categorical_categorical_data(self):
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
        sdf = self.spark.createDataFrame(pdf)
        return sdf

    def _get_binary_continuous_data(self):
        """
        Gets dummy `binary-continuous data <https://www.slideshare.net/MuhammadKhalil66/point-biserial-correlation-example>`_.

        :return: Spark dataframe.
        """
        data = [
            (1, 10), (1, 11), (1, 6), (1, 11), (0, 4),
            (0, 3), (1, 12), (0, 2), (0, 2), (0, 1)
        ]
        pdf = pd.DataFrame(data, columns=['gender', 'years'])
        sdf = self.spark.createDataFrame(pdf)
        return sdf

    def _get_concordance_data(self):
        """
        Gets dummy concordance data.

        :return: Spark dataframe.
        """
        a = [1, 2, 3]
        b = [3, 2, 1]
        pdf = pd.DataFrame({'a': a, 'b': b, 'c': a, 'd': b})
        sdf = self.spark.createDataFrame(pdf)
        return sdf

    def _get_categorical_continuous_data(self):
        """
        Gets dummy categorical-continuous data.
        See `site <https://en.wikipedia.org/wiki/Correlation_ratio>`_.

        :return: Spark dataframe.
        """
        data = [
            ('a', 45), ('a', 70), ('a', 29), ('a', 15), ('a', 21),
            ('g', 40), ('g', 20), ('g', 30), ('g', 42),
            ('s', 65), ('s', 95), ('s', 80), ('s', 70), ('s', 85), ('s', 73)
        ]
        data = [tup * 2 for tup in data]
        pdf = pd.DataFrame(data, columns=['x1', 'x2', 'x3', 'x4'])
        sdf = self.spark.createDataFrame(pdf)
        return sdf

    def _get_continuous_continuous_data(self):
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
        sdf = self.spark.createDataFrame(pdf)
        return sdf


class SparkTest(PySparkTest):
    """
    Tests Spark operations.
    """

    def test_binary_binary(self):
        """
        Tests binary-binary Spark operation.

        :return: None.
        """
        sdf = self._get_binary_binary_data()
        results = {tup[0]: tup[1] for tup in binary_binary(sdf).collect()}

        print(json.dumps({f'{k[0]}_{k[1]}': v for k, v in results.items()}, indent=1))

    def test_confusion(self):
        """
        Tests confusion Spark operation.

        :return: None.
        """
        sdf = self._get_confusion_data()
        results = {tup[0]: tup[1] for tup in confusion(sdf).collect()}

        print(json.dumps({f'{k[0]}_{k[1]}': v for k, v in results.items()}, indent=1))

    def test_categorical_categorical(self):
        """
        Tests categorical-categorical Spark operation.

        :return: None.
        """
        sdf = self._get_categorical_categorical_data()
        results = {tup[0]: tup[1] for tup in categorical_categorical(sdf).collect()}

        print(json.dumps({f'{k[0]}_{k[1]}': v for k, v in results.items()}, indent=1))

    def test_agreement(self):
        """
        Tests agreement Spark operation.

        :return: None.
        """
        sdf = self._get_binary_binary_data()
        results = {tup[0]: tup[1] for tup in agreement(sdf).collect()}

        print(json.dumps({f'{k[0]}_{k[1]}': v for k, v in results.items()}, indent=1))

    def test_biserial(self):
        """
        Tests binary-continuous Spark operation.

        :return: None.
        """
        sdf = self._get_binary_continuous_data()
        results = {tup[0]: tup[1] for tup in binary_continuous(sdf, binary=['gender'], continuous=['years']).collect()}

        print(json.dumps({f'{k[0]}_{k[1]}': v for k, v in results.items()}, indent=1))

    def test_concordance(self):
        """
        Tests concordance Spark operation.

        :return: None.
        """
        sdf = self._get_concordance_data()
        results = {tup[0]: tup[1] for tup in concordance(sdf).collect()}

        print(json.dumps({f'{k[0]}_{k[1]}': v for k, v in results.items()}, indent=1))

    def test_categorical_continuous(self):
        """
        Tests categorical-continuous Spark operation.

        :return: None.
        """
        sdf = self._get_categorical_continuous_data()
        results = {tup[0]: tup[1] for tup in categorical_continuous(sdf, ['x1', 'x3'], ['x2', 'x4']).collect()}

        print(json.dumps({f'{k[0]}_{k[1]}': v for k, v in results.items()}, indent=1))

    def test_continuous_continuous(self):
        """
        Tests continuous-continuous Spark operation.

        :return: None.
        """
        sdf = self._get_continuous_continuous_data()
        results = {tup[0]: tup[1] for tup in continuous_continuous(sdf).collect()}

        print(json.dumps({f'{k[0]}_{k[1]}': v for k, v in results.items()}, indent=1))
