from __future__ import annotations

import logging
import json
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, StructField, StructType

from .artifacts import TrainingResult, TypeArtifacts
from .io import load_artifacts, save_artifacts
from .sampling import cap_sample_by_key
from .schemas import cast_columns, ensure_columns_exist
from .udfs import build_period_prob_udf, build_survival_udf
from .utils import CoxModelError, now_utc_iso

logger = logging.getLogger(__name__)


class SparkCoxPHByType:
    def __init__(
        self,
        type_col: str,
        duration_col: str,
        event_col: str,
        feature_cols: Iterable[str],
        max_rows_per_type: int = 500_000,
        min_events_per_type: int = 5,
        penalizer: Optional[float] = 0.0,
        l1_ratio: Optional[float] = 0.0,
        baseline_estimation_method: str = "breslow",
        seed: Optional[int] = None,
        unknown_type_policy: str = "null",
        null_policy: str = "nan",
    ) -> None:
        self.type_col = type_col
        self.duration_col = duration_col
        self.event_col = event_col
        self.feature_cols = list(feature_cols)
        self.max_rows_per_type = max_rows_per_type
        self.min_events_per_type = min_events_per_type
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.baseline_estimation_method = baseline_estimation_method
        self.seed = seed
        self.unknown_type_policy = unknown_type_policy
        self.null_policy = null_policy

        self.artifacts: Dict[str, TypeArtifacts] = {}
        self.skipped: Dict[str, str] = {}

    @property
    def spark(self) -> SparkSession:
        return SparkSession.getActiveSession()  # type: ignore[return-value]

    def fit(self, sdf: DataFrame) -> "SparkCoxPHByType":
        ensure_columns_exist(sdf, [self.type_col, self.duration_col, self.event_col, *self.feature_cols])
        sdf = cast_columns(
            sdf,
            {
                self.duration_col: "double",
                self.event_col: "int",
                **{c: "double" for c in self.feature_cols},
            },
        )
        sdf = sdf.filter(col(self.duration_col) >= 0)
        sdf = cap_sample_by_key(sdf, self.type_col, self.max_rows_per_type, seed=self.seed)

        schema = StructType(
            [
                StructField(self.type_col, StringType(), False),
                StructField("payload", StringType(), True),
                StructField("status", StringType(), True),
            ]
        )

        def train(pdf: pd.DataFrame) -> pd.DataFrame:
            type_value = str(pdf[self.type_col].iloc[0])
            pdf = pdf.dropna()
            if pdf.empty:
                return pd.DataFrame({self.type_col: [type_value], "payload": [None], "status": ["empty"]})
            if pdf[self.event_col].sum() < self.min_events_per_type:
                return pd.DataFrame({
                    self.type_col: [type_value],
                    "payload": [None],
                    "status": ["not_enough_events"],
                })
            pdf_local = pdf[[self.duration_col, self.event_col] + self.feature_cols]
            penalizer = 0.0 if self.penalizer is None else float(self.penalizer)
            l1_ratio = 0.0 if self.l1_ratio is None else float(self.l1_ratio)
            cph = CoxPHFitter(
                penalizer=penalizer,
                l1_ratio=l1_ratio,
                baseline_estimation_method=self.baseline_estimation_method,
            )
            cph.fit(
                pdf_local,
                duration_col=self.duration_col,
                event_col=self.event_col,
            )
            mean_values = pdf_local[self.feature_cols].mean().to_dict()
            timeline = np.arange(0, int(pdf_local[self.duration_col].max()) + 1, dtype=int)
            survival_series = cph.baseline_survival_.iloc[:, 0]
            survival_series.index = survival_series.index.astype(int)
            survival_series = survival_series[~survival_series.index.duplicated(keep="last")]
            survival_series = survival_series.reindex(timeline, method="ffill")
            survival_series.iloc[0] = 1.0
            s0 = survival_series.ffill().fillna(1.0).to_numpy()
            ratios = np.ones_like(s0)
            ratios[1:] = s0[1:] / s0[:-1]
            artifacts = TypeArtifacts(
                type_value=type_value,
                beta={k: float(v) for k, v in cph.params_.to_dict().items()},
                mean_={k: float(v) for k, v in mean_values.items()},
                baseline_survival=s0.tolist(),
                baseline_ratio=ratios.tolist(),
                sample_size=int(len(pdf_local)),
                event_count=int(pdf_local[self.event_col].sum()),
                fitted_at=now_utc_iso(),
                penalizer=penalizer,
                l1_ratio=l1_ratio,
                feature_cols=self.feature_cols,
                baseline_method=self.baseline_estimation_method,
            )
            payload = json.dumps(artifacts.__dict__)
            return pd.DataFrame({self.type_col: [type_value], "payload": [payload], "status": ["ok"]})

        results = (
            sdf.groupBy(self.type_col)
            .applyInPandas(train, schema=schema)
            .toPandas()
        )

        training = TrainingResult()
        for _, row in results.iterrows():
            type_value = str(row[self.type_col])
            if row["status"] == "ok" and isinstance(row["payload"], str):
                training.add(type_value, TypeArtifacts(**json.loads(row["payload"])), None)
            else:
                training.add(type_value, None, row["status"])
        self.artifacts = training.artifacts_by_type
        self.skipped = training.skipped
        return self

    def extend_baselines(
        self, max_time: int, extend_fn: Optional[Callable] = None, tail_k: int = 6
    ) -> None:
        for art in self.artifacts.values():
            art.extend_baseline(max_time=max_time, extend_fn=extend_fn, tail_k=tail_k)

    def predict_survival_at_t(
        self,
        sdf: DataFrame,
        t: Optional[int] = None,
        t_col: Optional[str] = None,
        output_col: str = "survival",
    ) -> DataFrame:
        if t is None and t_col is None:
            raise ValueError("Provide either t or t_col")
        if t is not None and t_col is not None:
            raise ValueError("Only one of t or t_col should be set")
        ensure_columns_exist(sdf, [self.type_col, *self.feature_cols])
        if t_col:
            ensure_columns_exist(sdf, [t_col])
        sdf = cast_columns(sdf, {c: "double" for c in self.feature_cols})
        udf = build_survival_udf(
            spark=self.spark,
            artifacts=self.artifacts,
            feature_cols=self.feature_cols,
            unknown_policy=self.unknown_type_policy,
            t=t,
        )
        cols = [col(self.type_col)] + [col(c) for c in self.feature_cols]
        if t_col:
            cols.append(col(t_col))
        result = sdf.withColumn(output_col, udf(*cols))
        return result

    def predict_period_event_prob(
        self, sdf: DataFrame, period_col: str, output_col: str = "p_event"
    ) -> DataFrame:
        ensure_columns_exist(sdf, [self.type_col, period_col, *self.feature_cols])
        sdf = cast_columns(sdf, {period_col: "int", **{c: "double" for c in self.feature_cols}})
        udf = build_period_prob_udf(
            spark=self.spark,
            artifacts=self.artifacts,
            feature_cols=self.feature_cols,
            unknown_policy=self.unknown_type_policy,
        )
        cols = [col(self.type_col)] + [col(c) for c in self.feature_cols] + [col(period_col)]
        return sdf.withColumn(output_col, udf(*cols))

    def save(self, path: str) -> None:
        save_artifacts(path, self.artifacts, self.skipped, self._config_dict())

    def _config_dict(self) -> Dict[str, object]:
        return {
            "type_col": self.type_col,
            "duration_col": self.duration_col,
            "event_col": self.event_col,
            "feature_cols": self.feature_cols,
            "max_rows_per_type": self.max_rows_per_type,
            "min_events_per_type": self.min_events_per_type,
            "penalizer": self.penalizer,
            "l1_ratio": self.l1_ratio,
            "baseline_estimation_method": self.baseline_estimation_method,
            "seed": self.seed,
            "unknown_type_policy": self.unknown_type_policy,
            "null_policy": self.null_policy,
        }

    @classmethod
    def load(cls, path: str) -> "SparkCoxPHByType":
        artifacts, skipped, config = load_artifacts(path)
        if not artifacts:
            raise CoxModelError("No artifacts found in path")
        model = cls(
            type_col=config.get("type_col", "type"),
            duration_col=config.get("duration_col", "duration"),
            event_col=config.get("event_col", "event"),
            feature_cols=config.get("feature_cols", []),
            max_rows_per_type=int(config.get("max_rows_per_type", 0)),
            min_events_per_type=int(config.get("min_events_per_type", 0)),
            penalizer=config.get("penalizer"),
            l1_ratio=config.get("l1_ratio"),
            baseline_estimation_method=config.get("baseline_estimation_method", "breslow"),
            seed=config.get("seed"),
            unknown_type_policy=config.get("unknown_type_policy", "null"),
            null_policy=config.get("null_policy", "nan"),
        )
        model.artifacts = artifacts
        model.skipped = skipped
        return model
