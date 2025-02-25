from enum import Enum

class MovieLensDataset(Enum):
    ML_100K = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    ML_20M = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    ML_32M = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"

class MovieLensType(Enum):
    LINKS = "links.csv"
    MOVIES = "movies.csv"
    RATINGS = "ratings.csv"
    TAGS = "tags.csv"


class RunnerEventType(Enum):
    SAR = "sar"

    @classmethod
    def value_of(cls, value):
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"{value} is not a valid {cls.__name__}")


class ItemSimMeasure(Enum):
    SIM_COCCURRENCE = "cooccurrence"
    SIM_COSINE = "cosise"
