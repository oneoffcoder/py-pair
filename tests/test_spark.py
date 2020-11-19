import logging
import random
import unittest
from random import choice

import pandas as pd
from pyspark.sql import SparkSession

from pypair.spark import binary_binary, confusion, categorical_categorical, agreement


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
        result = {tup[0]: tup[1] for tup in binary_binary(sdf).collect()}

        import json
        print(json.dumps({f'{k[0]}_{k[1]}': v for k, v in result.items()}, indent=1))

    def test_confusion(self):
        """
        Tests confusion Spark operation.

        :return: None.
        """
        sdf = self._get_confusion_data()
        result = {tup[0]: tup[1] for tup in confusion(sdf).collect()}

        import json
        print(json.dumps({f'{k[0]}_{k[1]}': v for k, v in result.items()}, indent=1))

    def test_categorical_categorical(self):
        """
        Tests categorical-categorical Spark operation.

        :return: None.
        """
        sdf = self._get_categorical_categorical_data()
        results = categorical_categorical(sdf).collect()
        print(results)

    def test_agreement(self):
        """
        Tests agreement Spark operation.

        :return: None.
        """
        sdf = self._get_binary_binary_data()
        results = agreement(sdf).collect()
        print(results)
