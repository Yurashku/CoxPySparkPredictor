import numpy as np
from pyspark.ml.linalg import Vectors

from spark_lifelines_cox.model import BaselinePipeline, BaselinePipelineConfig


def build_dataset(spark, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for key in ["A", "B"]:
        for _ in range(50):
            duration = int(rng.integers(1, 8))
            event = int(rng.binomial(1, 0.7))
            features = Vectors.dense([float(rng.normal()), float(rng.normal())])
            rows.append((key, duration, event, features))
    return spark.createDataFrame(rows, ["model_key", "duration", "event", "x"])


def test_full_cycle_save_load_and_adjust(spark, tmp_path):
    sdf = build_dataset(spark)
    config = BaselinePipelineConfig(max_baseline_length=24, tail_cycle=12, sample_fraction=1.0)
    pipeline = BaselinePipeline(config=config)
    pipeline.fit(sdf)

    assert set(pipeline.models.keys()) == {"A", "B"}
    for model in pipeline.models.values():
        assert len(model.baseline_survival) == config.max_baseline_length + 1
        assert model.baseline_survival[0] == 1.0

    save_path = tmp_path / "baseline_csv"
    pipeline.save(str(save_path))

    loaded = BaselinePipeline.load(str(save_path))
    assert loaded.config.max_baseline_length == config.max_baseline_length

    with_baseline = loaded.infer_baseline(sdf)
    assert "baseline" in with_baseline.columns
    sample_row = with_baseline.limit(1).collect()[0]
    assert len(sample_row.baseline) == config.max_baseline_length + 1

    adjusted = loaded.adjust_for_lived(with_baseline, duration_col="duration", baseline_col="baseline")
    adjusted_row = adjusted.limit(1).collect()[0]
    assert adjusted_row.adjusted_baseline[0] == 1.0
    assert len(adjusted_row.adjusted_baseline) == len(sample_row.baseline) - sample_row.duration
