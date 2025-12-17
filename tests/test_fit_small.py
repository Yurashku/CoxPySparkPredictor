import numpy as np
import pandas as pd

from spark_lifelines_cox.model import SparkCoxPHByType


def make_data():
    rng = np.random.default_rng(0)
    n = 200
    types = np.where(rng.random(n) > 0.5, "A", "B")
    x = rng.normal(size=n)
    base_duration = rng.exponential(scale=5, size=n)
    # make higher x increase hazard
    duration = np.clip(base_duration - x, 0.1, None)
    event = rng.binomial(1, 0.7, size=n)
    return pd.DataFrame({"type": types, "duration": duration, "event": event, "x": x})


def test_fit_creates_artifacts(spark):
    pdf = make_data()
    sdf = spark.createDataFrame(pdf)
    model = SparkCoxPHByType(
        type_col="type",
        duration_col="duration",
        event_col="event",
        feature_cols=["x"],
        max_rows_per_type=1000,
        min_events_per_type=3,
        seed=123,
    )
    model.fit(sdf)
    assert set(model.artifacts.keys()) <= {"A", "B"}
    for art in model.artifacts.values():
        assert art.sample_size > 0
        assert len(art.baseline_survival) == len(art.baseline_ratio)
        assert art.baseline_survival[0] == 1.0


def test_extend_baseline(spark):
    pdf = make_data()
    sdf = spark.createDataFrame(pdf)
    model = SparkCoxPHByType(
        type_col="type",
        duration_col="duration",
        event_col="event",
        feature_cols=["x"],
        max_rows_per_type=1000,
        min_events_per_type=3,
        seed=123,
    )
    model.fit(sdf)
    model.extend_baselines(max_time=50, tail_k=3)
    for art in model.artifacts.values():
        assert len(art.baseline_survival) >= 51
        assert not any(np.isnan(art.baseline_survival))
        assert not any(np.array(art.baseline_survival) < 0)
