from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import col, count, lit, rand, row_number

logger = logging.getLogger(__name__)


def deterministic_seed(seed: Optional[int] = None, extra: Optional[int] = None) -> int:
    """Return a deterministic non-negative seed combining base seed and optional extra."""
    base = seed if seed is not None else 0
    if extra is not None:
        base = (base * 31 + extra) % (2**31 - 1)
    return base


def safe_json_dump(obj: Any) -> str:
    """Serialize object to JSON string."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def safe_json_load(data: str) -> Any:
    return json.loads(data)


def now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class CoxModelError(RuntimeError):
    pass


def log_skip(type_value: Any, reason: str) -> None:
    logger.warning("Skipping type %s: %s", type_value, reason)


def ensure_columns_exist(df: DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def cast_columns(df: DataFrame, casts: Dict[str, str]) -> DataFrame:
    for name, dtype in casts.items():
        if name in df.columns:
            df = df.withColumn(name, col(name).cast(dtype))
    return df


def cap_sample_by_key(
    sdf: DataFrame,
    key_col: str,
    max_rows_per_key: int,
    seed: Optional[int] = None,
) -> DataFrame:
    """Limit rows per key using deterministic random sampling.

    If count(key) <= limit, return all rows. Otherwise sample up to ``max_rows_per_key``
    rows using a deterministic random order produced by ``rand(seed)``.
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
