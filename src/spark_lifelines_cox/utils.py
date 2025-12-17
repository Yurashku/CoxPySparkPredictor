from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import col, count, lit, rand, row_number

logger = logging.getLogger(__name__)


def deterministic_seed(seed: Optional[int] = None, extra: Optional[int] = None) -> int:
    """Возвращает детерминированное неотрицательное зерно с учётом базового seed и дополнительного смещения."""
    base = seed if seed is not None else 0
    if extra is not None:
        base = (base * 31 + extra) % (2**31 - 1)
    return base


def safe_json_dump(obj: Any) -> str:
    """Сериализует объект в JSON-строку без лишних пробелов, чтобы сохранять компактные артефакты."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def safe_json_load(data: str) -> Any:
    """Декодирует JSON-строку, возвращая оригинальные артефакты или конфиг."""
    return json.loads(data)


def now_utc_iso() -> str:
    """Фиксирует текущий UTC без микросекунд в ISO-формате для единообразных меток времени."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class CoxModelError(RuntimeError):
    """Исключение верхнего уровня для ошибок, связанных с обучением или инференсом модели Cox."""
    pass


def log_skip(type_value: Any, reason: str) -> None:
    """Записывает причину пропуска типа, чтобы легче отлаживать пайплайн."""
    logger.warning("Skipping type %s: %s", type_value, reason)


def ensure_columns_exist(df: DataFrame, cols: List[str]) -> None:
    """Проверяет наличие обязательных колонок в DataFrame, чтобы упасть раньше с понятной ошибкой."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def cast_columns(df: DataFrame, casts: Dict[str, str]) -> DataFrame:
    """Приводит указанные колонки к нужным типам, чтобы избежать сюрпризов при работе Spark с double/int."""
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
    """Ограничивает число строк на ключ с детерминированной случайной выборкой, чтобы защищаться от перекоса крупных групп."""

    if max_rows_per_key <= 0:
        return sdf

    counts = sdf.groupBy(key_col).agg(count(lit(1)).alias("cnt"))
    sdf_with_count = sdf.join(counts, on=key_col, how="left")
    # rn служит порядковым номером внутри группы: он позволяет отсечь строки сверх лимита при избыточных группах
    window = Window.partitionBy(key_col).orderBy(rand(deterministic_seed(seed)))
    sampled = sdf_with_count.withColumn("rn", row_number().over(window)).where(
        (col("cnt") <= lit(max_rows_per_key)) | (col("rn") <= lit(max_rows_per_key))
    )
    return sampled.drop("rn", "cnt")
