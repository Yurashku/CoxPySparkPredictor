import numpy as np
import pandas as pd

from spark_lifelines_cox.model import SparkCoxPHByType


def build_model(spark):
    rng = np.random.default_rng(1)
    n = 150
    pdf = pd.DataFrame({
        "type": np.where(rng.random(n) > 0.4, "A", "B"),
        "duration": rng.exponential(scale=4, size=n),
        "event": rng.binomial(1, 0.7, size=n),
        "x": rng.normal(size=n),
    })
    sdf = spark.createDataFrame(pdf)
    model = SparkCoxPHByType(
        type_col="type",
        duration_col="duration",
        event_col="event",
        feature_cols=["x"],
        seed=10,
    )
    model.fit(sdf)
    model.extend_baselines(max_time=20)
    return model, sdf


def test_survival_monotonic(spark):
    model, sdf = build_model(spark)
    pred = model.predict_survival_at_t(sdf, t=5, output_col="s5")
    pred = model.predict_survival_at_t(pred, t=10, output_col="s10")
    rows = pred.select("s5", "s10").collect()
    for r in rows:
        assert 0 <= r.s5 <= 1
        assert 0 <= r.s10 <= 1
        assert r.s10 <= r.s5


def test_period_probability(spark):
    model, sdf = build_model(spark)
    long_df = sdf.withColumn("period", sdf.duration.cast("int")).select("type", "x", "period")
    pred = model.predict_period_event_prob(long_df, period_col="period", output_col="p_event")
    rows = pred.select("p_event").collect()
    for r in rows:
        assert r.p_event is None or (0 <= r.p_event <= 1)
