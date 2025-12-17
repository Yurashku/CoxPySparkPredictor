from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
from lifelines import CoxPHFitter
from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


@dataclass
class BaselineModel:
    model_key: str
    weights: List[float]
    baseline_survival: List[float]
    sample_size: int


CONFIG_ROW_TYPE = "__config__"


def _vector_size(sample: Iterable[DenseVector | SparseVector]) -> int:
    for v in sample:
        return int(v.size)
    raise ValueError("Vector column is empty; cannot infer dimensionality")


def _vector_to_columns(pdf, vector_col: str, prefix: str = "f"):
    vectors = pdf[vector_col]
    size = _vector_size(vectors)
    feature_matrix = np.vstack([np.array(v.toArray()) for v in vectors])
    for idx in range(size):
        pdf[f"{prefix}{idx}"] = feature_matrix[:, idx]
    return pdf.drop(columns=[vector_col]), [f"{prefix}{idx}" for idx in range(size)]


def _hazard_sequence(cph: CoxPHFitter) -> List[float]:
    hazard_series = cph.baseline_hazard_.iloc[:, 0]
    durations = [int(round(v)) for v in hazard_series.index.to_list()]
    values = hazard_series.to_list()
    if not durations:
        return []

    max_t = max(durations)
    hazard: List[float] = []
    last_value = values[0]
    duration_iter = iter(zip(durations, values))
    current_duration, current_value = next(duration_iter)

    for t in range(1, max_t + 1):
        if t == current_duration:
            last_value = current_value
            try:
                current_duration, current_value = next(duration_iter)
            except StopIteration:
                current_duration = max_t + 1
        hazard.append(float(last_value))
    return hazard


def _extend_hazard(hazard: List[float], target_length: int, tail_cycle: int = 12) -> List[float]:
    if target_length <= len(hazard):
        return hazard[:target_length]
    if not hazard:
        return [1.0] * target_length
    cycle = hazard[-tail_cycle:] if len(hazard) >= tail_cycle else hazard
    cycle = (cycle * (target_length // len(cycle) + 1))[: target_length - len(hazard)]
    return hazard + cycle


def _hazard_to_survival(hazard: List[float]) -> List[float]:
    survival = [1.0]
    for h in hazard:
        survival.append(float(survival[-1] * np.exp(-h)))
    return survival


def build_baseline_model(
    pdf,
    model_key: str,
    duration_col: str,
    event_col: str,
    vector_col: str,
    max_length: int,
    tail_cycle: int,
) -> Optional[BaselineModel]:
    if len(pdf) == 0:
        return None
    if pdf[event_col].sum() == 0:
        return None
    pdf = pdf.copy()
    pdf[duration_col] = pdf[duration_col].astype(int)
    pdf[event_col] = pdf[event_col].astype(int)
    pdf, feature_cols = _vector_to_columns(pdf, vector_col)
    pdf = pdf[[duration_col, event_col, *feature_cols]]

    cph = CoxPHFitter()
    cph.fit(pdf, duration_col=duration_col, event_col=event_col)
    hazard = _extend_hazard(_hazard_sequence(cph), target_length=max_length, tail_cycle=tail_cycle)
    survival = _hazard_to_survival(hazard)
    weights = cph.params_.reindex(feature_cols).to_numpy(dtype=float).tolist()
    return BaselineModel(
        model_key=model_key,
        weights=weights,
        baseline_survival=survival,
        sample_size=len(pdf),
    )


def save_models(path: str, config: Dict[str, object], models: Dict[str, BaselineModel]) -> None:
    rows = [
        {"type": CONFIG_ROW_TYPE, "payload": json.dumps(config, ensure_ascii=False, separators=(",", ":"))}
    ]
    for key, model in models.items():
        rows.append({
            "type": key,
            "payload": json.dumps(model.__dict__, ensure_ascii=False, separators=(",", ":")),
        })
    schema = T.StructType(
        [
            T.StructField("type", T.StringType(), False),
            T.StructField("payload", T.StringType(), False),
        ]
    )
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("Spark session is not active")
    spark.createDataFrame(rows, schema=schema).coalesce(1).write.mode("overwrite").csv(path, header=True)


def load_models(path: str) -> tuple[Dict[str, BaselineModel], Dict[str, object]]:
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("Spark session is not active")
    df = spark.read.option("header", True).csv(path)
    rows = df.collect()
    models: Dict[str, BaselineModel] = {}
    config: Dict[str, object] = {}
    for row in rows:
        payload = json.loads(row["payload"])
        if row["type"] == CONFIG_ROW_TYPE:
            config = payload
            continue
        models[str(row["type"])] = BaselineModel(**payload)
    return models, config


def build_baseline_column(models: Dict[str, BaselineModel]) -> F.Column:
    items = []
    for key, model in models.items():
        items.append(F.lit(key))
        baseline_array = F.array([F.lit(float(v)) for v in model.baseline_survival])
        items.append(baseline_array)
    return F.create_map(*items)


def attach_baseline(sdf: DataFrame, key_col: str, models: Dict[str, BaselineModel], output_col: str) -> DataFrame:
    mapping_col = build_baseline_column(models)
    return sdf.withColumn(output_col, mapping_col.getItem(F.col(key_col)))


def adjust_survival_tail(sdf: DataFrame, duration_col: str, baseline_col: str, output_col: str) -> DataFrame:
    def _trim_and_scale(baseline: Optional[List[float]], lived: Optional[int]):
        if baseline is None or lived is None:
            return None
        lived_int = int(lived)
        if lived_int < 0:
            return baseline
        if lived_int >= len(baseline):
            return [1.0]
        tail = baseline[lived_int:]
        start = tail[0]
        if start == 0:
            return [float("nan") for _ in tail]
        return [float(v / start) for v in tail]

    udf = F.udf(_trim_and_scale, T.ArrayType(T.DoubleType()))
    return sdf.withColumn(output_col, udf(F.col(baseline_col), F.col(duration_col)))

