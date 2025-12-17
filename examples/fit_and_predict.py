"""Полный цикл работы с BaselinePipeline на маленьком датасете."""
import numpy as np
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

from spark_lifelines_cox.model import BaselinePipeline, BaselinePipelineConfig


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("baseline-example").getOrCreate()

    rng = np.random.default_rng(42)
    rows = []
    for key in ["alpha", "beta"]:
        for _ in range(30):
            duration = int(rng.integers(1, 10))
            event = int(rng.binomial(1, 0.6))
            features = Vectors.dense([float(rng.normal()), float(rng.normal())])
            rows.append((key, duration, event, features))

    sdf = spark.createDataFrame(rows, ["model_key", "duration", "event", "x"])

    config = BaselinePipelineConfig(max_baseline_length=48, tail_cycle=12, sample_fraction=0.8, seed=7)
    pipeline = BaselinePipeline(config)
    pipeline.fit(sdf)

    pipeline.save("/tmp/baseline_csv")
    restored = BaselinePipeline.load("/tmp/baseline_csv")

    with_baseline = restored.infer_baseline(sdf, output_col="baseline")
    adjusted = restored.adjust_for_lived(with_baseline, duration_col="duration", baseline_col="baseline", output_col="tail")

    print("=== Baseline ===")
    with_baseline.show(5, truncate=False)

    print("=== Tail adjusted for lived duration ===")
    adjusted.select("model_key", "duration", "tail").show(5, truncate=False)

    spark.stop()
