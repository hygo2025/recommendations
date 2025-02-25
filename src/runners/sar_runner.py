from pyspark.sql import SparkSession

from src.dataset.movielens.loader import Loader
from src.models.sar_spark import SarSpark
from src.runners.abstract_runnner import AbstractRunner
from src.utils.enums import MovieLensDataset, MovieLensType, ItemSimMeasure
from src.utils.logger import Logger
from src.utils.spark_splitter import random_split


class SarRunner(AbstractRunner):
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = Logger.get_logger(name="SarRunner")
        self.loader = Loader()

    def run(self) -> None:
        # Carrega os dados do MovieLens
        movies_df = self.loader.load_spark(dataset=MovieLensDataset.ML_100K, ml_type=MovieLensType.MOVIES)
        ratings_df = self.loader.load_spark(dataset=MovieLensDataset.ML_100K, ml_type=MovieLensType.RATINGS)

        # Faz o join entre ratings e movies (supondo que o join seja feito pela coluna "movieId")
        df = ratings_df.join(movies_df, on="movieId", how="inner")

        # Renomeia as colunas para que o modelo SAR utilize os nomes esperados: "user_id", "item_id" e "rating"
        df = df.withColumnRenamed("userId", "user_id") \
            .withColumnRenamed("movieId", "item_id") \
            .select("user_id", "item_id", "rating")

        # Divide os dados em treino e teste (75% treino, 25% teste)
        train, test = random_split(df=df, train_ratio=0.75)

        df.show(10, truncate=False)
        self.logger.info("Running SAR Runner")

        # Instancia o modelo SAR (neste exemplo, usando cosine similarity)
        sar_model = SarSpark(
            spark=self.spark,
            rating_threshold=0,
            topK=10,
            threshold=1,
            similarity_type=ItemSimMeasure.SIM_COSINE
        )

        # Treina o modelo com os dados de treino
        sar_model.fit(train)

        recs = sar_model.recommend(user_id=1)
        recs.show()