from pyspark.sql import SparkSession

from src.dataset.movielens.loader import Loader
from src.runners.abstract_runnner import AbstractRunner
from src.utils.enums import MovieLensDataset, MovieLensType
from src.utils.logger import Logger
from src.utils.spark_splitter import random_split


class SarRunner(AbstractRunner):
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = Logger.get_logger(name="SarRunner")
        self.loader = Loader()

    def run(self) -> None:
        movies_df = self.loader.load_spark(dataset=MovieLensDataset.ML_100K, ml_type=MovieLensType.MOVIES)
        ratings_df = self.loader.load_spark(dataset=MovieLensDataset.ML_100K, ml_type=MovieLensType.RATINGS)

        df = ratings_df.join(movies_df, on="movieId", how="inner")

        train, test = random_split(df = df, train_ratio=0.75)

        df.show(10, truncate=False)

        self.logger.info("Running example runner")