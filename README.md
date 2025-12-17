# CoxPySparkPredictor

Репозиторий реализует единственный пайплайн, который обучает и применяет базовые кривые выживаемости для набора категориальных моделей (`model_key`) на данных PySpark. Все шаги строго следуют требованиям:

1. **Сэмплирование и обучение:** для каждого `model_key` берётся (опционально) подвыборка строк, из которой оцениваются веса признаков и базовый hazard/выживаемость с помощью `lifelines.CoxPHFitter`. Столбец `duration` всегда приводится к целому.
2. **Продление бейзлайна:** полученный hazard циклически продолжает последние 12 значений до заданной длины горизонта (`max_baseline_length`).
3. **Сохранение в CSV:** конфигурация пайплайна и все базовые кривые записываются в один CSV-каталог (Spark `DataFrameWriter.csv`).
4. **Загрузка и инференс:** артефакты читаются из CSV, после чего можно восстанавливать полный бейзлайн для новых объектов по их `model_key`.
5. **Учёт прожитого срока:** у «живых» объектов хвост бейзлайна обрезается на `duration` точек и нормируется так, чтобы новая кривальная часть начиналась с единицы (мы учитываем факт, что объект уже прожил `duration`).

Никакого дополнительного функционала нет: только перечисленные шаги и вспомогательные функции, необходимые для их выполнения.

## Требования к данным

Входной `DataFrame` Spark обязан содержать следующие столбцы:

- `model_key` — строковый идентификатор группы.
- `duration` — целочисленная продолжительность жизни объекта (начинается с 1).
- `event` — индикатор события (0/1).
- `x` — вектор числовых признаков (`pyspark.ml.linalg.Vector`, как правило полученный из VectorAssembler).

## Установка

```bash
pip install -e .[dev]
```

> Spark 3.2.1 требует JDK 8–11. Убедитесь, что `JAVA_HOME` указывает на совместимую версию (например, `/usr/lib/jvm/java-11-openjdk-amd64`).

## Быстрый пример

```python
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from spark_lifelines_cox.model import BaselinePipeline, BaselinePipelineConfig

spark = SparkSession.builder.master("local[*]").getOrCreate()

rows = [
    ("A", 3, 1, Vectors.dense([0.1, -0.2])),
    ("A", 5, 0, Vectors.dense([0.3, 0.4])),
    ("B", 2, 1, Vectors.dense([-0.5, 0.7])),
]
sdf = spark.createDataFrame(rows, ["model_key", "duration", "event", "x"])

config = BaselinePipelineConfig(max_baseline_length=36, tail_cycle=12, sample_fraction=1.0)
pipeline = BaselinePipeline(config)
pipeline.fit(sdf)
pipeline.save("/tmp/baseline_csv")

loaded = BaselinePipeline.load("/tmp/baseline_csv")
with_baseline = loaded.infer_baseline(sdf, output_col="baseline")
adjusted = loaded.adjust_for_lived(with_baseline, duration_col="duration", baseline_col="baseline", output_col="tail")

with_baseline.show(truncate=False)
adjusted.show(truncate=False)
```

## Что внутри

- `src/spark_lifelines_cox/model.py` — класс `BaselinePipeline`, который оркестрирует обучение, продление, сохранение/загрузку и инференс бейзлайнов.
- `src/spark_lifelines_cox/utils.py` — утилиты для работы с векторными признаками, преобразования hazard в выживаемость, сериализации моделей и построения UDF для инференса.
- `examples/fit_and_predict.py` — скрипт, воспроизводящий полный цикл на маленьком синтетическом датасете.
- `examples/tutorial.ipynb` — ноутбук с теми же шагами, но с подробными выводами и иллюстрациями каждого этапа.
- `tests/test_pipeline.py` — автотест, проверяющий полный цикл: обучение, сохранение/загрузка, построение и корректировку хвостов для «живых» объектов.

## Ключевые параметры

- `max_baseline_length` — желаемая длина hazard-массива; выживаемость содержит на одну точку больше (начинается с 1.0).
- `tail_cycle` — число последних hazard-значений, которые зацикливаются при продлении.
- `sample_fraction` — доля данных, используемая при обучении каждого `model_key`.
- `seed` — фиксирует сэмплирование для воспроизводимости.

## Примечания по использованию

- Если для какого-то `model_key` данных нет, бейзлайн для него не строится.
- При обрезке хвостов для «живых» объектов пустой или слишком короткий бейзлайн заменяется на `[1.0]`, чтобы не возвращать пустые массивы.
- Сохранение идёт в формате Spark CSV (каталог с одним файлом), поэтому при передаче пути нужно указывать директорию, а не конечный файл.

## Обновление артефактов

Каждое видоизменение кода сопровождается обновлением `README.md` и примеров. Все Jupyter-ноутбуки должны быть прогнаны до конца и сохранены с актуальными выводами перед коммитом.
