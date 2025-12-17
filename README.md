# spark-lifelines-cox

Репозиторий демонстрирует, как обучать отдельные модели Cox Proportional Hazards (библиотека `lifelines`) на данных PySpark по каждому значению категориального столбца и выполнять масштабный инференс через векторизованные pandas UDF. Код рассчитан на Python 3.9+ и Spark 3.5+.

## Формула и центрирование признаков

Модель Cox оценивает частичную правдоподобность и представляет индивидуальную выживаемость как

```
S(t | x) = S0(t) ** exp((x - \bar{x})' β)
```

`lifelines` центрирует признаки (`x - \bar{x}`), поэтому средние значения тренировочных признаков (`mean_train`) сохраняются в артефактах и используются при инференсе. Базовая функция выживаемости `S0(t)` извлекается из `CoxPHFitter.baseline_survival_` и переводится на целочисленную шкалу.

## Возможности

- Обучение отдельной Cox-модели на каждый тип (`type_col`) с ограничением по числу строк на тип — прокси для контроля памяти executors.
- Детерминированный cap-sampling: при превышении `max_rows_per_type` выбираются случайные, но повторяемые строки.
- Аккуратные артефакты по типам: коэффициенты β, средние признаков, `S0(t)` и периодные коэффициенты `p0(t)=S0(t)/S0(t-1)`.
- Продление baseline до нужного горизонта с настраиваемой функцией хвоста (по умолчанию среднее последних `tail_k` периодов).
- Масштабный инференс без `groupBy.applyInPandas` на полном датасете: векторизованные pandas UDF c broadcast артефактов.
- Обработка edge-cases: типы без событий пропускаются, неизвестные типы на инференсе возвращают `null` или ошибку.

## Установка

```bash
pip install -e .[dev]
```

## Быстрый старт

```python
from pyspark.sql import SparkSession
from spark_lifelines_cox.model import SparkCoxPHByType

spark = SparkSession.builder.master("local[*]").getOrCreate()

# Подготовьте Spark DataFrame
sdf = spark.createDataFrame([
    ("A", 5, 1, 0.4),
    ("A", 3, 0, -0.2),
    ("B", 4, 1, 1.1),
], ["type", "duration", "event", "x"])

model = SparkCoxPHByType(
    type_col="type",
    duration_col="duration",
    event_col="event",
    feature_cols=["x"],
    max_rows_per_type=100_000,
)
model.fit(sdf)
model.extend_baselines(max_time=120, tail_k=6)

pred = model.predict_survival_at_t(sdf, t=12, output_col="s12")
pred.show()
```

### Расширенный пример (long format по периодам)

Если у вас уже раскрытые по периодам строки, можно предсказывать вероятность события в периоде и затем получить кумулятивную выживаемость через оконную агрегацию:

```python
long_df = sdf.withColumn("period", sdf.duration.cast("int")).select("type", "x", "period")
period_pred = model.predict_period_event_prob(long_df, period_col="period", output_col="p_event")
# Затем на стороне пользователя:
# survival = F.exp(F.sum(F.log1p(-F.col("p_event"))).over(Window.partitionBy("id").orderBy("period")))
```

### Сохранение и загрузка

```python
model.save("/tmp/cox_model")
loaded = SparkCoxPHByType.load("/tmp/cox_model")
```

Артефакты сериализуются в JSON: manifest c конфигурацией и списком типов, плюс файл на каждый тип.

## Параметры класса

- `max_rows_per_type`: лимит строк на тип. Используется как грубая оценка памяти: реальный объём зависит от числа признаков и количества типов.
- `min_events_per_type`: минимальное число событий, иначе тип пропускается.
- `unknown_type_policy`: поведение на инференсе для неизвестных типов (`"null"` или `"error"`).
- `baseline_estimation_method`, `penalizer`, `l1_ratio`: проксируются в `lifelines.CoxPHFitter`.

## Edge cases и политики

- **Типы без событий или с малым числом событий**: пропускаются с причиной `not_enough_events`; инференс по таким типам вернёт `null`.
- **Неизвестные типы на инференсе**: по умолчанию `null`, можно выбросить ошибку через `unknown_type_policy="error"`.
- **Пропуски в признаках**: при обучении строки с `NaN` дропаются; на инференсе возвращается `NaN` в прогнозе.
- **Продление baseline**: по умолчанию хвост `p0(t)` берётся как среднее последних `tail_k` периодов, но можно передать функцию `extend_fn` для кастомной логики.

## CI

В репозитории есть GitHub Actions (`.github/workflows/ci.yml`), которые прогоняют `ruff` и `pytest`.
