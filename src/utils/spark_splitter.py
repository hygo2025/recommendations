from typing import Optional

import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.dataframe import DataFrame


def random_split(df: DataFrame, seed: Optional[int] = None, train_ratio: float = 0.75):
    return df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)


def chrono_split(data, train_ratio=0.75, timestamp_col="timestamp"):
    data_ordered = data.orderBy(timestamp_col)
    split_index = int(data_ordered.count() * train_ratio)

    data_with_rownum = data_ordered.withColumn("row_num", F.row_number().over(
        Window.orderBy(timestamp_col)
    ))

    train = data_with_rownum.filter(F.col("row_num") <= split_index).drop("row_num")
    test = data_with_rownum.filter(F.col("row_num") > split_index).drop("row_num")
    return train, test
