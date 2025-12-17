from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, StructField, StructType

from .udfs import build_survival_udf
from .utils import (
    CoxModelError,
    cap_sample_by_key,
    cast_columns,
    ensure_columns_exist,
    now_utc_iso,
    safe_json_dump,
    safe_json_load,
)

logger = logging.getLogger(__name__)

CSV_FILENAME = "artifacts.csv"
CONFIG_ROW_TYPE = "__config__"


@dataclass
class TypeArtifacts:
    """Хранит все артефакты, полученные из lifelines для конкретного значения type_col."""

    type_value: str
    beta: Dict[str, float]
    mean_: Dict[str, float]
    baseline_survival: List[float]
    baseline_ratio: List[float]
    sample_size: int
    event_count: int
    fitted_at: str
    penalizer: Optional[float]
    l1_ratio: Optional[float]
    feature_cols: List[str]
    baseline_method: str

    def survival_at(self, t: int) -> float:
        """Возвращает вероятность выживания на момент t, подхватывая крайние значения за пределами ряда."""
        if t < 0:
            return 1.0
        if t < len(self.baseline_survival):
            return float(self.baseline_survival[t])
        return float(self.baseline_survival[-1])

    def extend_baseline(self, max_time: int, extend_fn=None, tail_k: int = 12) -> None:
        """Продлевает базовую кривую выживаемости до указанного горизонта, чтобы прогнозировать дальше тренировочных данных."""
        current_T = len(self.baseline_survival) - 1
        if max_time <= current_T:
            return
        tail_start = max(1, len(self.baseline_ratio) - tail_k)
        tail_values = np.array(self.baseline_ratio[tail_start:])
        if extend_fn is None:
            if len(tail_values) == 0:
                new_ratios = np.ones(max_time - current_T, dtype=float)
            else:
                repeated = np.resize(tail_values[-12:], 12)
                cycle = np.resize(repeated, max_time - current_T)
                new_ratios = np.array(cycle, dtype=float)
        else:
            new_times = np.arange(current_T + 1, max_time + 1, dtype=int)
            new_ratios = np.array(extend_fn(tail_values, new_times), dtype=float)
        last_survival = self.baseline_survival[-1]
        cumulative = last_survival * np.cumprod(new_ratios)
        self.baseline_ratio.extend(new_ratios.tolist())
        self.baseline_survival.extend(cumulative.tolist())


@dataclass
class TrainingResult:
    """Агрегирует результаты обучения по типам: успешные артефакты и причины пропусков."""

    artifacts_by_type: Dict[str, TypeArtifacts] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)

    def add(self, type_value: str, artifacts: Optional[TypeArtifacts], reason: Optional[str]) -> None:
        """Сохраняет артефакты обученного типа или фиксирует причину его пропуска."""
        if artifacts is not None:
            self.artifacts_by_type[type_value] = artifacts
        elif reason:
            self.skipped[type_value] = reason


def _resolve_csv_path(path: str) -> str:
    """Нормализует путь до CSV, позволяя указывать как директорию, так и файл."""
    if path.endswith(".csv"):
        return path
    return os.path.join(path, CSV_FILENAME)


def save_artifacts(path: str, artifacts: Dict[str, TypeArtifacts], skipped: Dict[str, str], config: Dict[str, object]) -> None:
    """Сохраняет конфиг и артефакты в один CSV, чтобы упростить переносимость модели."""
    csv_path = _resolve_csv_path(path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    rows = []
    rows.append(
        {
            "type": CONFIG_ROW_TYPE,
            "payload": safe_json_dump(config),
            "status": "config",
        }
    )
    for t, art in artifacts.items():
        rows.append({"type": t, "payload": safe_json_dump(art.__dict__), "status": "ok"})
    for t, reason in skipped.items():
        rows.append({"type": t, "payload": None, "status": reason})
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def load_artifacts(path: str) -> Tuple[Dict[str, TypeArtifacts], Dict[str, str], Dict[str, object]]:
    """Загружает сохранённые артефакты и конфигурацию, проверяя наличие файла."""
    csv_path = _resolve_csv_path(path)
    if not os.path.exists(csv_path):
        raise CoxModelError(f"Artifacts file not found at {csv_path}")
    df = pd.read_csv(csv_path)
    config: Dict[str, object] = {}
    artifacts: Dict[str, TypeArtifacts] = {}
    skipped: Dict[str, str] = {}
    for _, row in df.iterrows():
        row_type = str(row["type"])
        status = None if pd.isna(row.get("status")) else str(row.get("status"))
        if row_type == CONFIG_ROW_TYPE:
            payload = row.get("payload")
            config = safe_json_load(payload) if isinstance(payload, str) else {}
            continue
        if status == "ok":
            payload = row.get("payload")
            if not isinstance(payload, str):
                continue
            artifacts[row_type] = TypeArtifacts(**safe_json_load(payload))
        else:
            skipped[row_type] = status or "skipped"
    return artifacts, skipped, config


class SparkCoxPHByType:
    """Оркестрирует обучение и инференс Cox PH по каждому значению категориального столбца."""

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
        trim_tail_on_wide_ci: bool = True,
        ci_width_quantile: float = 0.9,
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
        self.trim_tail_on_wide_ci = trim_tail_on_wide_ci
        self.ci_width_quantile = ci_width_quantile

        self.artifacts: Dict[str, TypeArtifacts] = {}
        self.skipped: Dict[str, str] = {}

    @property
    def spark(self) -> SparkSession:
        return SparkSession.getActiveSession()  # type: ignore[return-value]

    def fit(self, sdf: DataFrame) -> "SparkCoxPHByType":
        """Обучает отдельную Cox-модель на каждый тип, сохраняя артефакты и причины пропуска."""
        print("[SparkCoxPHByType] Starting fit pipeline")
        ensure_columns_exist(sdf, [self.type_col, self.duration_col, self.event_col, *self.feature_cols])
        print("[SparkCoxPHByType] Casting columns to numeric types")
        sdf = cast_columns(
            sdf,
            {
                self.duration_col: "double",
                self.event_col: "int",
                **{c: "double" for c in self.feature_cols},
            },
        )
        sdf = sdf.filter(col(self.duration_col) >= 0)
        print(
            "[SparkCoxPHByType] Applying cap-sampling by type with max_rows_per_type=",
            self.max_rows_per_type,
        )
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
            print(f"[SparkCoxPHByType][{type_value}] Start training on {len(pdf)} rows")
            pdf = pdf.dropna()
            if pdf.empty:
                print(f"[SparkCoxPHByType][{type_value}] Skipped because dataset is empty")
                return pd.DataFrame({self.type_col: [type_value], "payload": [None], "status": ["empty"]})
            if pdf[self.event_col].sum() < self.min_events_per_type:
                print(
                    f"[SparkCoxPHByType][{type_value}] Skipped because events < min_events_per_type"
                )
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
            # histograms фиксируют события и риск на каждой точке времени для оценки ширины доверительных интервалов хвоста
            durations_int = pdf_local[self.duration_col].astype(int).to_numpy()
            events_mask = pdf_local[self.event_col].to_numpy() == 1
            event_durations = durations_int[events_mask]
            max_time = int(timeline[-1])
            event_hist = np.bincount(event_durations, minlength=max_time + 1)
            at_risk_hist = np.bincount(durations_int, minlength=max_time + 1)
            at_risk = np.cumsum(at_risk_hist[::-1])[::-1]
            var_terms = np.zeros_like(s0)
            valid = (event_hist > 0) & (at_risk > event_hist)
            var_terms[valid] = event_hist[valid] / (at_risk[valid] * (at_risk[valid] - event_hist[valid]))
            cumulative_var = np.cumsum(var_terms)
            ci_widths = 1.96 * np.sqrt(cumulative_var)
            trim_threshold = float(np.quantile(ci_widths, self.ci_width_quantile))
            trim_idx = len(s0)
            if self.trim_tail_on_wide_ci:
                # отрезаем хвост baseline там, где доверительные интервалы становятся слишком широкими
                while trim_idx > 1 and ci_widths[trim_idx - 1] >= trim_threshold:
                    trim_idx -= 1
                if trim_idx < len(s0):
                    print(
                        f"[SparkCoxPHByType][{type_value}] Trimming trailing hazards: removed {len(s0) - trim_idx} points"
                    )
                    s0 = s0[:trim_idx]
                    ratios = ratios[:trim_idx]
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
            print(f"[SparkCoxPHByType][{type_value}] Training finished")
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
        print(
            f"[SparkCoxPHByType] Training completed. Fitted types: {len(self.artifacts)}, skipped: {len(self.skipped)}"
        )
        return self

    def extend_baselines(
        self, max_time: int, extend_fn: Optional[Callable] = None, tail_k: int = 12
    ) -> None:
        """Продлевает baseline для всех типов одной операцией, чтобы синхронизировать горизонты."""
        print(f"[SparkCoxPHByType] Extending baselines to max_time={max_time}")
        for art in self.artifacts.values():
            art.extend_baseline(max_time=max_time, extend_fn=extend_fn, tail_k=tail_k)

    def predict_survival_at_t(
        self,
        sdf: DataFrame,
        t: Optional[int] = None,
        t_col: Optional[str] = None,
        output_col: str = "survival",
    ) -> DataFrame:
        """Прогнозирует выживаемость на фиксированном горизонте или в столбце, используя предобученные артефакты."""
        if t is None and t_col is None:
            raise ValueError("Provide either t or t_col")
        if t is not None and t_col is not None:
            raise ValueError("Only one of t or t_col should be set")
        ensure_columns_exist(sdf, [self.type_col, *self.feature_cols])
        if t_col:
            ensure_columns_exist(sdf, [t_col])
        sdf = cast_columns(sdf, {c: "double" for c in self.feature_cols})
        print("[SparkCoxPHByType] Building survival UDF for prediction")
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
        print("[SparkCoxPHByType] Starting prediction DataFrame transformation")
        result = sdf.withColumn(output_col, udf(*cols))
        return result

    def save(self, path: str) -> None:
        """Сохраняет артефакты и конфиг в CSV, чтобы использовать модель в других средах."""
        print(f"[SparkCoxPHByType] Saving artifacts to {path}")
        save_artifacts(path, self.artifacts, self.skipped, self._config_dict())

    def _config_dict(self) -> Dict[str, object]:
        """Возвращает словарь конфигурации для сериализации вместе с артефактами."""
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
            "trim_tail_on_wide_ci": self.trim_tail_on_wide_ci,
            "ci_width_quantile": self.ci_width_quantile,
        }

    @classmethod
    def load(cls, path: str) -> "SparkCoxPHByType":
        """Загружает модель с артефактами и статистикой пропущенных типов."""
        print(f"[SparkCoxPHByType] Loading artifacts from {path}")
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
            trim_tail_on_wide_ci=bool(config.get("trim_tail_on_wide_ci", True)),
            ci_width_quantile=float(config.get("ci_width_quantile", 0.9)),
        )
        model.artifacts = artifacts
        model.skipped = skipped
        return model

    @classmethod
    def from_config(cls, path: str) -> "SparkCoxPHByType":
        """Создаёт новый экземпляр только по конфигурации, не подгружая артефакты."""
        print(f"[SparkCoxPHByType] Loading configuration only from {path}")
        _, _, config = load_artifacts(path)
        if not config:
            raise CoxModelError("No configuration found in artifacts file")
        return cls(
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
            trim_tail_on_wide_ci=bool(config.get("trim_tail_on_wide_ci", True)),
            ci_width_quantile=float(config.get("ci_width_quantile", 0.9)),
        )
