from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType

from .artifacts import TypeArtifacts
from .utils import CoxModelError


UNKNOWN_TYPE_ERROR = "error"
UNKNOWN_TYPE_NULL = "null"


class _BroadcastArtifacts:
    def __init__(self, feature_order: List[str], artifacts: Dict[str, TypeArtifacts]):
        self.feature_order = feature_order
        self.artifacts = artifacts

    def make_survival_udf(self, unknown_policy: str, t: Optional[int]):
        feature_order = self.feature_order
        artifacts = self.artifacts

        @pandas_udf(DoubleType())
        def predict(*cols: pd.Series) -> pd.Series:
            type_series = cols[0]
            feature_cols_series = cols[1 : 1 + len(feature_order)]
            features = np.vstack([c.to_numpy(dtype=float) for c in feature_cols_series]).T
            if t is not None:
                t_values = np.full(len(type_series), t, dtype=int)
            else:
                t_values = cols[1 + len(feature_order)].astype(int).to_numpy()

            out = np.empty(len(type_series), dtype=float)
            for i, (type_val, feats, t_val) in enumerate(zip(type_series, features, t_values)):
                art = artifacts.get(str(type_val))
                if art is None:
                    if unknown_policy == UNKNOWN_TYPE_ERROR:
                        raise CoxModelError(f"Unknown type {type_val}")
                    out[i] = np.nan
                    continue
                if np.any(np.isnan(feats)):
                    out[i] = np.nan
                    continue
                centered = feats - np.array([art.mean_[c] for c in feature_order], dtype=float)
                eta = float(np.dot(centered, np.array([art.beta[c] for c in feature_order], dtype=float)))
                base = art.survival_at(int(t_val))
                out[i] = float(base ** np.exp(eta))
            return pd.Series(out)

        return predict

    def make_period_prob_udf(self, unknown_policy: str):
        feature_order = self.feature_order
        artifacts = self.artifacts

        @pandas_udf(DoubleType())
        def predict(*cols: pd.Series) -> pd.Series:
            type_series = cols[0]
            feature_cols_series = cols[1 : 1 + len(feature_order)]
            features = np.vstack([c.to_numpy(dtype=float) for c in feature_cols_series]).T
            periods = cols[1 + len(feature_order)].astype(int).to_numpy()
            out = np.empty(len(type_series), dtype=float)
            for i, (type_val, feats, t_val) in enumerate(zip(type_series, features, periods)):
                art = artifacts.get(str(type_val))
                if art is None:
                    if unknown_policy == UNKNOWN_TYPE_ERROR:
                        raise CoxModelError(f"Unknown type {type_val}")
                    out[i] = np.nan
                    continue
                if np.any(np.isnan(feats)):
                    out[i] = np.nan
                    continue
                centered = feats - np.array([art.mean_[c] for c in feature_order], dtype=float)
                eta = float(np.dot(centered, np.array([art.beta[c] for c in feature_order], dtype=float)))
                ratio = art.ratio_at(int(t_val))
                out[i] = float(1.0 - (ratio ** np.exp(eta)))
            return pd.Series(out)

        return predict


def build_survival_udf(
    spark: SparkSession,
    artifacts: Dict[str, TypeArtifacts],
    feature_cols: List[str],
    unknown_policy: str,
    t: Optional[int],
):
    broadcasted = spark.sparkContext.broadcast(_BroadcastArtifacts(feature_cols, artifacts))
    return broadcasted.value.make_survival_udf(unknown_policy=unknown_policy, t=t)


def build_period_prob_udf(
    spark: SparkSession,
    artifacts: Dict[str, TypeArtifacts],
    feature_cols: List[str],
    unknown_policy: str,
):
    broadcasted = spark.sparkContext.broadcast(_BroadcastArtifacts(feature_cols, artifacts))
    return broadcasted.value.make_period_prob_udf(unknown_policy=unknown_policy)
