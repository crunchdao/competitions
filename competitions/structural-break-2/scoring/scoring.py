import os
from typing import List, Tuple

import numpy
import pandas
import sklearn.metrics
from crunch.api import Metric, Target
from crunch.scoring import ScoredMetric
from crunch.unstructured.utils import delta_message
from crunch.utils import Tracer


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


tracer = Tracer()


def check(
    prediction_directory_path: str,
    data_directory_path: str,
):
    y_test = _load_y_test(data_directory_path)

    with tracer.log(f"Checking: prediction.parquet"):
        prediction = _load_prediction(prediction_directory_path, "prediction.parquet")
        _check_prediction(prediction, y_test)

    with tracer.log(f"Checking: prediction.noisy.parquet"):
        noisy_prediction = _load_prediction(prediction_directory_path, "prediction.noisy.parquet")
        _check_prediction(prediction, y_test)

    with tracer.log("Ensure not too correlated with previous top 10"):
        with tracer.log("Load the data"):
            top_results = pandas.read_parquet(os.path.join(data_directory_path, "top_results.noisy.parquet"))

        acceptable = True
        for column_name in tracer.loop(top_results.columns, "Checking against: {value}"):
            correlation = noisy_prediction["prediction"].corr(
                top_results[column_name],
                method="spearman",
            )

            print(f"correlation against {column_name:40} is {correlation:0.8f}")

            if correlation > 0.9:
                acceptable = False

        if not acceptable:
            raise ParticipantVisibleError(f"Too correlated with the official competition's top 10 leaderboard")


def score(
    prediction_directory_path: str,
    data_directory_path: str,
    target_and_metrics: List[Tuple[Target, List[Metric]]],
):
    metric = target_and_metrics[0][1][0]  # [first entry], [take list of metrics], [take first metric]
    assert metric.name == "roc-auc", "missing roc-auc metric"

    prediction = _load_prediction(prediction_directory_path, "prediction.parquet")
    y_test = _load_y_test(data_directory_path)

    with tracer.log("Reindex prediction"):
        prediction = prediction.reindex(y_test.index, copy=False)

        if prediction["prediction"].isna().any():
            raise ParticipantVisibleError("Reindex resulted in NaN values")

    with tracer.log("Call roc_auc_score"):
        value = sklearn.metrics.roc_auc_score(
            y_test,
            prediction,
        )

    return {
        metric.id: ScoredMetric(value, [])
    }


def _check_prediction(
    prediction: pandas.DataFrame,
    y_test: pandas.Series,
):
    with tracer.log(f"Check for required columns"):
        difference = delta_message(
            {'prediction'},
            set(prediction.columns),
        )

        if difference:
            raise ParticipantVisibleError(f"Columns do not match: {difference}")

    with tracer.log("Check for dtypes"):
        if not numpy.issubdtype(prediction.dtypes["prediction"], numpy.floating):
            raise ParticipantVisibleError("Column 'prediction' must be of type float")

        if not numpy.issubdtype(prediction.index.dtype, numpy.integer):
            raise ParticipantVisibleError("Index must be of type int")

    with tracer.log("Check for nan and inf"):
        if prediction["prediction"].isna().any():
            raise ParticipantVisibleError("Prediction must not contain NaN values")

        replaced = prediction["prediction"].replace([numpy.inf, -numpy.inf], numpy.nan)
        if replaced.isna().any():
            raise ParticipantVisibleError("Prediction must not contain infinite values")

    with tracer.log("Check for ids"):
        difference = delta_message(
            y_test.index,
            prediction.index,
        )

        if difference:
            raise ParticipantVisibleError(f"IDs do not match: {difference}")

        if len(prediction.index) != len(y_test.index):
            raise ParticipantVisibleError("Duplicate IDs found.")


def _load_prediction(prediction_directory_path: str, name: str) -> pandas.DataFrame:
    path = os.path.join(prediction_directory_path, name)

    with tracer.log("Loading prediction"):
        return pandas.read_parquet(path)


def _load_y_test(data_directory_path: str) -> pandas.Series:
    path = os.path.join(data_directory_path, "y_test.parquet")

    with tracer.log("Loading y_test"):
        return pandas.read_parquet(path)["structural_breakpoint"]
