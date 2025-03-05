import os
import pandas as pd

from pyspark.sql import SparkSession, DataFrame

from src.dataset.movielens.downloader import Downloader
from src.utils.enums import MovieLensDataset, MovieLensType
from src.utils.logger import Logger


class Loader:
    def __init__(self, download_folder="/tmp/dataset", extract_folder="/tmp/dataset"):
        self.extract_folder = extract_folder
        self.download_folder = download_folder
        self.logger = Logger.get_logger(name="Loader")

    def _get_file_path(self, dataset: MovieLensDataset, ml_type: MovieLensType) -> str:
        dataset_folder = os.path.join(self.extract_folder, dataset.name)
        return os.path.join(dataset_folder, ml_type.value)

    def _ensure_dataset(self, dataset: MovieLensDataset):
        dataset_folder = os.path.join(self.extract_folder, dataset.name)
        if not os.path.exists(dataset_folder):
            self.logger.info(f"Dataset {dataset.name} não encontrado em {dataset_folder}. Iniciando download e extração...")
            downloader = Downloader(download_folder=self.download_folder, extract_folder=self.extract_folder)
            downloader.download_and_extract_dataset(dataset)
        else:
            self.logger.info(f"Dataset {dataset.name} já existe em {dataset_folder}.")

    def load_pandas(self, dataset: MovieLensDataset, ml_type: MovieLensType) -> pd.DataFrame:
        file_path = self._get_file_path(dataset, ml_type)
        if dataset == MovieLensDataset.ML_1M:
            file_path = file_path.replace(".csv", ".dat")
        if not os.path.exists(file_path):
            self._ensure_dataset(dataset)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado mesmo após o download.")


        if dataset == MovieLensDataset.ML_1M and ml_type == MovieLensType.RATINGS: #TODO: Implementar o resto dos tipos
            df = pd.read_csv(file_path, sep="::", engine="python", names=["userId", "movieId", "rating", "timestamp"])
        else:
            df = pd.read_csv(file_path)
        return df

    def load_spark(self, dataset: MovieLensDataset, ml_type: MovieLensType) -> DataFrame:
        file_path = self._get_file_path(dataset, ml_type)
        if not os.path.exists(file_path):
            self._ensure_dataset(dataset)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado mesmo após o download.")
        spark = SparkSession.builder.appName("MovieLensDataLoader").getOrCreate()
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        return df
