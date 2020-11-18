import logging
import unittest

import pandas as pd
from pyspark.sql import SparkSession

from pypair.spark import binary_binary


class PySparkTest(unittest.TestCase):
    @classmethod
    def supress_py4j_logging(cls):
        logger = logging.getLogger('py4j')
        logger.setLevel(logging.WARN)

    @classmethod
    def create_pyspark_session(cls):
        return (SparkSession.builder
                .master('local[4]')
                .appName('local-testing-pyspark')
                .getOrCreate())

    @classmethod
    def setUpClass(cls):
        cls.supress_py4j_logging()
        cls.spark = cls.create_pyspark_session()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def get_binary_binary_data(self):
        get_data = lambda x, y, n: [(x, y) * 2 for _ in range(n)]
        data = get_data(1, 1, 207) + get_data(1, 0, 282) + get_data(0, 1, 231) + get_data(0, 0, 242)
        pdf = pd.DataFrame(data, columns=['x1', 'x2', 'x3', 'x4'])
        sdf = self.spark.createDataFrame(pdf)
        return sdf


class BinaryBinaryTest(PySparkTest):
    def test(self):
        sdf = self.get_binary_binary_data()
        result = binary_binary(sdf)
        import json
        print(json.dumps({f'{k[0]}_{k[1]}': v for k, v in result.items()}, indent=1))
