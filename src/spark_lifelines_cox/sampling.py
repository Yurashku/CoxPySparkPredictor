from __future__ import annotations

from typing import Optional

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import col, count, lit, rand, row_number

from .utils import deterministic_seed


def cap_sample_by_key(
    sdf: DataFrame,
    key_col: str,
    max_rows_per_key: int,
    seed: Optional[int] = None,
) -> DataFrame:
    """Limit rows per key using deterministic random sampling.

    If count(key) <= limit, return all rows. Otherwise sample up to max_rows_per_key rows
    using a deterministic random order produced by ``rand(seed)``.
    """

    if max_rows_per_key <= 0:
        return sdf

    counts = sdf.groupBy(key_col).agg(count(lit(1)).alias("cnt"))
    sdf_with_count = sdf.join(counts, on=key_col, how="left")
    window = Window.partitionBy(key_col).orderBy(rand(deterministic_seed(seed)))
    sampled = sdf_with_count.withColumn("rn", row_number().over(window)).where(
        (col("cnt") <= lit(max_rows_per_key)) | (col("rn") <= lit(max_rows_per_key))
    )
    return sampled.drop("rn", "cnt")
