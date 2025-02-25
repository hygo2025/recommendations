import logging

from pyspark.sql import SparkSession

from src.dataset.movielens.loader import Loader
from src.runners.abstract_runnner import AbstractRunner
from src.utils.enums import MovieLensDataset, MovieLensType


class SarRunner(AbstractRunner):
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = logging.getLogger(name="SarRunner")
        self.loader = Loader()

    def run(self) -> None:
        df = self.loader.load_spark(MovieLensDataset.ML_100K, MovieLensType.MOVIES)
        df.show(10, truncate=False)
        self.logger.info("Running example runner")