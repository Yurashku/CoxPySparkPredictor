from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .utils import (
    BaselineModel,
    adjust_survival_tail,
    attach_baseline,
    build_baseline_model,
    load_models,
    save_models,
)


@dataclass
class BaselinePipelineConfig:
    model_key_col: str = "model_key"
    duration_col: str = "duration"
    event_col: str = "event"
    vector_col: str = "x"
    max_baseline_length: int = 120
    tail_cycle: int = 12
    sample_fraction: float = 1.0
    seed: int = 0


class BaselinePipeline:
    """Полный цикл: обучение бейзлайнов, продление, сохранение, загрузка и инференс."""

    def __init__(self, config: Optional[BaselinePipelineConfig] = None):
        self.config = config or BaselinePipelineConfig()
        self.models: Dict[str, BaselineModel] = {}

    @property
    def spark(self) -> SparkSession:
        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("Spark session is not active")
        return spark

    def fit(self, sdf: DataFrame) -> "BaselinePipeline":
        c = self.config
        required = [c.model_key_col, c.duration_col, c.event_col, c.vector_col]
        missing = [col for col in required if col not in sdf.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        sdf = sdf.select(*required)
        sdf = sdf.withColumn(c.duration_col, F.col(c.duration_col).cast("int")).withColumn(
            c.event_col, F.col(c.event_col).cast("int")
        )

        keys = [r[0] for r in sdf.select(c.model_key_col).distinct().collect()]
        models: Dict[str, BaselineModel] = {}
        for key in keys:
            keyed = sdf.filter(F.col(c.model_key_col) == F.lit(key))
            if c.sample_fraction < 1.0:
                keyed = keyed.sample(withReplacement=False, fraction=c.sample_fraction, seed=c.seed)
            pdf = keyed.toPandas()
            model = build_baseline_model(
                pdf=pdf,
                model_key=key,
                duration_col=c.duration_col,
                event_col=c.event_col,
                vector_col=c.vector_col,
                max_length=c.max_baseline_length,
                tail_cycle=c.tail_cycle,
            )
            if model is not None:
                models[key] = model
        self.models = models
        return self

    def save(self, path: str) -> None:
        if not self.models:
            raise RuntimeError("Nothing to save: fit the pipeline first")
        config_dict = self.config.__dict__
        save_models(path, config=config_dict, models=self.models)

    @classmethod
    def load(cls, path: str) -> "BaselinePipeline":
        pipeline = cls()
        models, config = load_models(path)
        pipeline.models = models
        pipeline.config = BaselinePipelineConfig(**config) if config else BaselinePipelineConfig()
        return pipeline

    def infer_baseline(self, sdf: DataFrame, output_col: str = "baseline") -> DataFrame:
        if not self.models:
            raise RuntimeError("No baselines loaded. Call fit or load first.")
        return attach_baseline(sdf, key_col=self.config.model_key_col, models=self.models, output_col=output_col)

    def adjust_for_lived(
        self,
        sdf: DataFrame,
        duration_col: Optional[str] = None,
        baseline_col: str = "baseline",
        output_col: str = "adjusted_baseline",
    ) -> DataFrame:
        duration_col = duration_col or self.config.duration_col
        return adjust_survival_tail(sdf, duration_col=duration_col, baseline_col=baseline_col, output_col=output_col)
