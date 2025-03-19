from enum import Enum

class MovieLensDataset(Enum):
    ML_100K = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    ML_1M = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    ML_20M = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    ML_32M = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"

class MovieLensType(Enum):
    LINKS = "links.csv"
    MOVIES = "movies.csv"
    RATINGS = "ratings.csv"
    TAGS = "tags.csv"


class TargetMetric(Enum):
    HR = "HR"
    MRR = "MRR"
    NDCG = "NDCG"