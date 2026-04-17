import os
from typing import List, Tuple

import numpy
import pandas
from crunch.api import Metric, Target
from crunch.unstructured import ScoredMetric
from crunch.unstructured.utils import delta_message
from crunch.utils import Tracer
from sklearn.metrics import roc_auc_score


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


tracer = Tracer()


def check(
    prediction_directory_path: str,
    data_directory_path: str,
):
    prediction = _load_prediction(prediction_directory_path)

    with tracer.log("Check for required index"):
        difference = delta_message(
            {"id", "time"},
            set(prediction.index.names),
        )

        if difference:
            raise ParticipantVisibleError(f"Index do not match: {difference}")

    with tracer.log("Check for required columns"):
        difference = delta_message(
            {"prediction"},
            set(prediction.columns),
        )

        if difference:
            raise ParticipantVisibleError(f"Columns do not match: {difference}")

    with tracer.log("Check for dtypes"):
        if not numpy.issubdtype(prediction.dtypes["prediction"], numpy.floating):
            raise ParticipantVisibleError("Column 'prediction' must be of type float")

        if not numpy.issubdtype(prediction.index.dtypes["id"], numpy.integer):
            raise ParticipantVisibleError("Index 'id' must be of type int")

        if not numpy.issubdtype(prediction.index.dtypes["time"], numpy.integer):
            raise ParticipantVisibleError("Index 'time' must be of type int")

    with tracer.log("Check for nan and inf"):
        if prediction["prediction"].isna().any():
            raise ParticipantVisibleError("Prediction must not contain NaN values")

        replaced = prediction["prediction"].replace([numpy.inf, -numpy.inf], numpy.nan)
        if replaced.isna().any():
            raise ParticipantVisibleError("Prediction must not contain infinite values")

    y_test = _load_y_test(data_directory_path)

    with tracer.log("Check for ids"):
        difference = delta_message(
            y_test.index,
            prediction.index,
        )

        if difference:
            raise ParticipantVisibleError(f"IDs do not match: {difference}")

        if len(prediction.index) != len(y_test.index):
            raise ParticipantVisibleError("Duplicate IDs found")


def score(
    prediction_directory_path: str,
    data_directory_path: str,
    target_and_metrics: List[Tuple[Target, List[Metric]]],
):
    metric = target_and_metrics[0][1][0]  # [first entry], [take list of metrics], [take first metric]
    assert metric.name == "roc-auc", "missing roc-auc metric"

    prediction = _load_prediction(prediction_directory_path)
    y_test = _load_y_test(data_directory_path)

    with tracer.log("Reindex prediction"):
        prediction = prediction.reindex(y_test.index, copy=False)

        if prediction["prediction"].isna().any():
            raise ParticipantVisibleError("Reindex resulted in NaN values")

    with tracer.log("Call roc_auc_score"):
        value = roc_auc_score(
            y_test,
            prediction,
        )

    return {
        metric.id: ScoredMetric(value, []),
    }


def _load_prediction(prediction_directory_path: str) -> pandas.DataFrame:
    path = os.path.join(prediction_directory_path, "prediction.parquet")

    with tracer.log("Loading prediction"):
        return pandas.read_parquet(path)


def _load_y_test(data_directory_path: str) -> pandas.Series:
    path = os.path.join(data_directory_path, "y_test.parquet")

    with tracer.log("Loading y_test"):
        return pandas.read_parquet(path)
