import logging

from pyspark.sql import SparkSession

from src.runners.abstract_runnner import AbstractRunner


class ExampleRunner(AbstractRunner):
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = logging.getLogger(name="ExampleRunner")

    def run(self) -> None:
        self.logger.info("Running example runner")