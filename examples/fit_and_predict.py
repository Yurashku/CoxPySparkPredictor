"""Minimal example for fitting and predicting survival with SparkCoxPHByType."""
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

from spark_lifelines_cox.model import SparkCoxPHByType


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("example").getOrCreate()

    rng = np.random.default_rng(0)
    pdf = pd.DataFrame({
        "type": np.where(rng.random(50) > 0.5, "A", "B"),
        "duration": rng.exponential(scale=5, size=50),
        "event": rng.binomial(1, 0.8, size=50),
        "x": rng.normal(size=50),
    })
    sdf = spark.createDataFrame(pdf)

    model = SparkCoxPHByType(
        type_col="type",
        duration_col="duration",
        event_col="event",
        feature_cols=["x"],
        max_rows_per_type=1000,
    )
    model.fit(sdf)
    model.extend_baselines(max_time=30)

    pred = model.predict_survival_at_t(sdf, t=10)
    pred.show(5, truncate=False)

    model.save("/tmp/cox_model")
    loaded = SparkCoxPHByType.load("/tmp/cox_model")
    pred2 = loaded.predict_period_event_prob(
        sdf.withColumn("period", sdf.duration.cast("int")),
        period_col="period",
    )
    pred2.show(5, truncate=False)

    spark.stop()
