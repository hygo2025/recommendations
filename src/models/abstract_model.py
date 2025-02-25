from abc import ABC, abstractmethod

from pyspark.sql import DataFrame


class AbstractModel(ABC):
    @abstractmethod
    def fit(self, df: DataFrame) -> DataFrame:
        raise NotImplementedError("This method should be overridden in derived classes")

    @abstractmethod
    def predict(self, df: DataFrame) -> DataFrame:
        raise NotImplementedError("This method should be overridden in derived classes")