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

    with tracer.log("Merge prediction with y_test"):
        merged = prediction.merge(
            y_test,
            how="left",
            left_index=True,
            right_index=True
        )

        if merged["target"].isna().any():
            raise ParticipantVisibleError("Merge resulted in NaN values")

        # Add a column for the online time step (0, 1, 2, ...)
        merged["time_online"] = merged.groupby("id").cumcount()

    with tracer.log("Scoring predictions"):
        weighted_auc_sum = 0.0
        total_weight = 0.0

        # Step 1: select all observations at time t
        for time, group in merged.groupby("time_online"):
            labels = group["target"].values
            scores = group["prediction"].values

            # Step 2: count positives and negatives
            n_positive = int(labels.sum())
            n_negative = int((1 - labels).sum())

            # Step 3: skip if only one class is present
            if n_positive == 0 or n_negative == 0:
                continue

            # Step 4: AUC at this time step and its weight
            auc_at_time = float(roc_auc_score(labels, scores))
            weight = float(n_positive * n_negative)

            weighted_auc_sum += weight * auc_at_time
            total_weight += weight

        if total_weight == 0.0:
            score_value = 0.5
        else:
            score_value = weighted_auc_sum / total_weight

    return {
        metric.id: ScoredMetric(score_value, []),
    }


def _load_prediction(prediction_directory_path: str) -> pandas.DataFrame:
    path = os.path.join(prediction_directory_path, "prediction.parquet")

    with tracer.log("Loading prediction"):
        return pandas.read_parquet(path)


def _load_y_test(data_directory_path: str) -> pandas.Series:
    path = os.path.join(data_directory_path, "y_test.parquet")

    with tracer.log("Loading y_test"):
        return pandas.read_parquet(path)
