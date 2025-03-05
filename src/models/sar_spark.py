# import logging
#
# from pyspark.sql import DataFrame
# from pyspark.sql import functions as F
# from pyspark.sql.connect.session import SparkSession
#
# from src.models.abstract_model import AbstractModel
# from src.similarity.cosine_similarity import CosineSimilarity
# from src.utils.enums import ItemSimMeasure
#
# logger = logging.getLogger(__name__)
#
#
# class SarSpark(AbstractModel):
#     def __init__(self, spark: SparkSession, rating_threshold=0, topK=10, threshold=1, similarity_type=ItemSimMeasure.SIM_COSINE):
#         """
#         rating_threshold: valor mínimo de rating para considerar uma interação.
#         topK: número de recomendações a retornar.
#         threshold: valor mínimo para a co-ocorrência (mantém apenas pares com co_count >= threshold).
#         similarity_type: enum que define o método de similaridade.
#         """
#         self.spark = spark
#         self.rating_threshold = rating_threshold
#         self.topK = topK
#         self.threshold = threshold
#         self.similarity_type = similarity_type
#
#         # Atributos gerados durante o fit
#         self.user_affinity = None  # Matriz de afinidade usuário-item (interações filtradas)
#         self.item_frequencies = None  # Frequência (popularidade) de cada item
#         self.item_similarity = None  # Matriz de similaridade entre itens
#
#     def filter_interactions(self, interactions: DataFrame) -> DataFrame:
#         """
#         Filtra as interações com rating >= rating_threshold, mantendo as colunas "user_id", "item_id" e "rating".
#         """
#         filtered = interactions.filter(F.col("rating") >= self.rating_threshold)
#         return filtered.select("user_id", "item_id", "rating").distinct()
#
#     def compute_affinity_matrix(self, df: DataFrame, rating_col: str) -> DataFrame:
#         """
#         Constrói a matriz de afinidade do usuário.
#         Neste exemplo, usamos o próprio DataFrame de interações filtradas.
#         """
#         return df.select("user_id", "item_id", rating_col)
#
#     def compute_item_popularity(self, user_affinity: DataFrame) -> DataFrame:
#         """
#         Calcula a popularidade de cada item (número de interações).
#         """
#         return user_affinity.groupBy("item_id").agg(F.count("user_id").alias("item_count"))
#
#     def compute_cooccurrence_matrix(self, df: DataFrame) -> DataFrame:
#         """
#         Calcula a matriz de co-ocorrência a partir do DataFrame de interações.
#
#         Args:
#             df (DataFrame): DataFrame contendo, ao menos, as colunas "user_id" e "item_id".
#
#         Returns:
#             DataFrame: com as colunas "i1", "i2" e "co_count", mantendo apenas os pares com co_count >= threshold.
#         """
#         # Seleciona interações únicas
#         interactions = df.select("user_id", "item_id").distinct()
#
#         # Realiza self-join para obter pares de itens por usuário
#         joined = interactions.alias("a").join(
#             interactions.alias("b"), on="user_id"
#         ).filter(F.col("a.item_id") != F.col("b.item_id")) \
#          .select(F.col("a.item_id").alias("i1"), F.col("b.item_id").alias("i2"))
#
#         # Agrupa os pares e conta as co-ocorrências
#         co_occurrence_df = joined.groupBy("i1", "i2").agg(F.count("*").alias("co_count"))
#
#         # Filtra os pares com contagem abaixo do threshold
#         co_occurrence_df = co_occurrence_df.filter(F.col("co_count") >= self.threshold)
#
#         return co_occurrence_df
#
#     def fit(self, interactions: DataFrame):
#         """
#         Treina o modelo SAR a partir de um DataFrame de interações (sem duplicatas), realizando:
#           - Construção da matriz de afinidade (user_affinity)
#           - Cálculo das frequências dos itens (item_frequencies)
#           - Cálculo da matriz de similaridade conforme o similarity_type:
#                 * Se SIM_COCCURRENCE: usa a matriz de co-ocorrência.
#                 * Se SIM_COSINE: utiliza a classe CosineSimilarity.
#         """
#         logger.info("Building user affinity sparse matrix")
#         temp_df = self.filter_interactions(interactions)
#         self.user_affinity = self.compute_affinity_matrix(temp_df, rating_col="rating")
#
#         logger.info("Calculating item frequencies")
#         self.item_frequencies = self.compute_item_popularity(self.user_affinity)
#
#         logger.info("Calculating item similarity")
#         if self.similarity_type == ItemSimMeasure.SIM_COCCURRENCE:
#             logger.info("Using co-occurrence based similarity")
#             co_occurrence = self.compute_cooccurrence_matrix(temp_df)
#             self.item_similarity = co_occurrence
#         elif self.similarity_type == ItemSimMeasure.SIM_COSINE:
#             logger.info("Using cosine similarity")
#             cosine_sim = CosineSimilarity(sim_col="cosine_similarity")
#             self.item_similarity = cosine_sim.get_similarity(
#                 df=self.user_affinity,
#                 col_user="user_id",
#                 col_item="item_id",
#                 n_partitions=200
#             )
#         else:
#             raise ValueError("Unknown similarity type: {}".format(self.similarity_type))
#
#         del temp_df
#         logger.info("Done training")
#
#     def recommend(self, user_id) -> DataFrame:
#         """
#         Para um dado user_id, recomenda os topK itens que o usuário ainda não consumiu,
#         agregando as similaridades dos itens consumidos.
#         Retorna um DataFrame com as colunas:
#           - i2: item recomendado
#           - score: pontuação (soma das similaridades)
#         """
#         user_items_df = self.user_affinity.filter(F.col("user_id") == user_id).select("item_id").distinct()
#         user_items = [row.item_id for row in user_items_df.collect()]
#
#         # Filtra a matriz de similaridade para os pares onde o item consumido aparece como "i1"
#         sim_df = self.item_similarity.filter(F.col("i1").isin(user_items))
#         recs = sim_df.groupBy("i2").agg(F.sum("cosine_similarity").alias("score"))
#         recs = recs.filter(~F.col("i2").isin(user_items))
#         recommendations = recs.orderBy(F.col("score").desc()).limit(self.topK)
#         return recommendations
#
#     def predict(self, user_id) -> DataFrame:
#         """
#         Implementação do método abstrato predict.
#         Neste exemplo, delega a recomendação para o método recommend.
#         """
#         return self.recommend(user_id)