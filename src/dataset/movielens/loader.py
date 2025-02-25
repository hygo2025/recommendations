import os
import pandas as pd

from pyspark.sql import SparkSession

from src.dataset.movielens.downloader import Downloader
from src.utils.enums import MovieLensDataset, MovieLensType


class Loader:
    def __init__(self, download_folder="/tmp/dataset", extract_folder="/tmp/dataset"):
        self.extract_folder = extract_folder
        self.download_folder = download_folder

    def _get_file_path(self, dataset: MovieLensDataset, ml_type: MovieLensType) -> str:
        dataset_folder = os.path.join(self.extract_folder, dataset.name)
        return os.path.join(dataset_folder, ml_type.value)

    def _ensure_dataset(self, dataset: MovieLensDataset):
        dataset_folder = os.path.join(self.extract_folder, dataset.name)
        if not os.path.exists(dataset_folder):
            print(f"Dataset {dataset.name} não encontrado em {dataset_folder}. Iniciando download e extração...")
            downloader = Downloader(download_folder=self.download_folder, extract_folder=self.extract_folder)
            downloader.download_and_extract_dataset(dataset)
        else:
            print(f"Dataset {dataset.name} já existe em {dataset_folder}.")

    def load_pandas(self, dataset: MovieLensDataset, ml_type: MovieLensType) -> pd.DataFrame:
        file_path = self._get_file_path(dataset, ml_type)
        if not os.path.exists(file_path):
            self._ensure_dataset(dataset)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado mesmo após o download.")
        df = pd.read_csv(file_path)
        return df

    def load_spark(self, dataset: MovieLensDataset, ml_type: MovieLensType):
        file_path = self._get_file_path(dataset, ml_type)
        if not os.path.exists(file_path):
            self._ensure_dataset(dataset)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado mesmo após o download.")
        spark = SparkSession.builder.appName("MovieLensDataLoader").getOrCreate()
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        return df


# # Exemplo de uso:
# if __name__ == "__main__":
#     loader = MovieLensDataLoader(
#         extract_folder="/home/hygo/Development/trabalho_final/data/extract",
#         download_folder="/home/hygo/Development/trabalho_final/data/zip"
#     )
#
#     # Exemplo: carregar o arquivo tags.csv do dataset ML_100K em um DataFrame do Pandas
#     try:
#         df_pd = loader.load_pandas(MovieLensDataset.ML_100K, MovieLensType.MOVIES)
#         print("DataFrame do Pandas:")
#         print(df_pd.head())
#     except Exception as e:
#         print(e)
#
#     # Exemplo: carregar o arquivo movies.csv do dataset ML_100K em um DataFrame do Spark
#     try:
#         df_spark = loader.load_spark(MovieLensDataset.ML_100K, MovieLensType.MOVIES)
#         print("DataFrame do Spark:")
#         df_spark.show(5)
#     except Exception as e:
#         print(e)
