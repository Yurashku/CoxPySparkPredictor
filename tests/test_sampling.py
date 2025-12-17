import pandas as pd
from pyspark.sql import functions as F

from spark_lifelines_cox.sampling import cap_sample_by_key


def test_cap_sampling_deterministic(spark):
    pdf = pd.DataFrame({
        "type": ["a"] * 100 + ["b"] * 10,
        "duration": list(range(110)),
        "event": [1] * 110,
        "x": [0.1] * 110,
    })
    sdf = spark.createDataFrame(pdf)
    sampled = cap_sample_by_key(sdf, key_col="type", max_rows_per_key=20, seed=42)
    result = sampled.groupBy("type").agg(F.count(F.lit(1)).alias("cnt")).orderBy("type").collect()
    counts = {r["type"]: r["cnt"] for r in result}
    assert counts == {"a": 20, "b": 10}

    sampled2 = cap_sample_by_key(sdf, key_col="type", max_rows_per_key=20, seed=42)
    ids1 = [r.duration for r in sampled.orderBy("duration").limit(5).collect()]
    ids2 = [r.duration for r in sampled2.orderBy("duration").limit(5).collect()]
    assert ids1 == ids2
