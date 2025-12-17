from __future__ import annotations

from typing import Dict, List

from pyspark.sql import DataFrame
from pyspark.sql.functions import col


def ensure_columns_exist(df: DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def cast_columns(df: DataFrame, casts: Dict[str, str]) -> DataFrame:
    for name, dtype in casts.items():
        if name in df.columns:
            df = df.withColumn(name, col(name).cast(dtype))
    return df
