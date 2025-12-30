import json
import os
from typing import List, Tuple

import crunch
import crunch.unstructured
import numpy
import pandas
from crunch.api import Metric, PhaseType, Target
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
    phase_type: PhaseType,
):
    prediction = _load_prediction(prediction_directory_path)

    with tracer.log("Check for required columns"):
        difference = crunch.unstructured.utils.delta_message(
            {"id", "moon", "prediction"},
            set(prediction.columns),
        )

        if difference:
            raise ParticipantVisibleError(f"Columns do not match: {difference}")

    with tracer.log("Check for dtypes"):
        if not numpy.issubdtype(prediction.dtypes["id"], numpy.integer):
            raise ParticipantVisibleError("Column 'id' must be of type int")

        if not numpy.issubdtype(prediction.dtypes["moon"], numpy.integer):
            raise ParticipantVisibleError("Column 'id' must be of type int")

        if not numpy.issubdtype(prediction.dtypes["prediction"], numpy.floating):
            raise ParticipantVisibleError("Column 'prediction' must be of type float")

    with tracer.log("Check for nan and inf"):
        if prediction["prediction"].isna().any():
            raise ParticipantVisibleError("Prediction must not contain NaN values")

        replaced = prediction["prediction"].replace([numpy.inf, -numpy.inf], numpy.nan)
        if replaced.isna().any():
            raise ParticipantVisibleError("Prediction must not contain infinite values")

    with tracer.log("Check for [-1, 1]"):
        has_less = (prediction["prediction"] < -1).any()
        has_greater = (prediction["prediction"] > 1).any()

        if has_less or has_greater:
            raise ParticipantVisibleError("Prediction must be between -1.0 and 1.0")

    moons = _load_moons(
        data_directory_path,
        is_submission_phase=phase_type == PhaseType.SUBMISSION,
    )

    with tracer.log("Check for moons"):
        difference = delta_message(
            moons,
            prediction["moon"].unique(),
        )

        if difference:
            raise ParticipantVisibleError(f"Moons do not match: {difference}")

    y = _load_y(data_directory_path)
    with tracer.log("Check for ids"):
        for moon, group in tracer.loop(prediction.groupby("moon")["id"], lambda x: f"Checking moon {x[0]}"):
            y_ids = y.loc[y["moon"] == moon, "id"]

            difference = crunch.unstructured.utils.delta_message(
                y_ids,
                group,
            )

            if difference:
                raise ParticipantVisibleError(f"IDs do not match for moon {moon}: {difference}")

            if len(group) != len(y_ids):
                raise ParticipantVisibleError(f"Duplicate IDs found for moon {moon}.")


def score(
    prediction_directory_path: str,
    data_directory_path: str,
    target_and_metrics: List[Tuple[Target, List[Metric]]],
):
    _, metrics = target_and_metrics[0]
    score_metric = _find_metric_by_name(metrics, "score")

    prediction = _load_prediction(prediction_directory_path)
    y = _load_y(data_directory_path)

    with tracer.log("Merge prediction with y"):
        merged = y.merge(prediction, on=["moon", "id"], how="right")
        # merged["prediction"] = merged["target"] * 0.999

    with tracer.log("Compute pearson"):
        pearson = merged.groupby("moon")\
            .apply(
                lambda group: group["prediction"].corr(
                    group["target"],
                    method="pearson"
                ),
                include_groups=False
            )\
            .fillna(0)

        if len(pearson) > 0:
            score_value = pearson.iloc[-1]
        else:
            raise ValueError("Error in score computation: len(pearson) == 0")

    return {
        score_metric.id: ScoredMetric(score_value, []),
    }


def _find_metric_by_name(
    metrics: List[crunch.api.Metric],
    name: str
) -> Metric:
    for metric in metrics:
        if metric.name == name:
            return metric

    raise ValueError(f"metric {name} not found")


def _load_moons(
    data_directory_path: str,
    is_submission_phase: bool,
) -> List[int]:
    path = os.path.join(data_directory_path, "moons_split.json")

    with tracer.log("Loading moons"):
        with open(path, "r") as fd:
            splits = json.load(fd)

        moons = []

        if is_submission_phase:
            # moons.extend(splits["reduced_local"])
            moons.extend(splits["reduced_cloud"])
        else:
            moons.extend(splits["oos_private"])
            moons.extend(splits["live_to_predict"])

    return moons


def _load_prediction(
    prediction_directory_path: str,
) -> pandas.DataFrame:
    path = os.path.join(prediction_directory_path, "prediction.parquet")

    with tracer.log("Loading prediction"):
        return pandas.read_parquet(path)


def _load_y(
    data_directory_path: str,
) -> pandas.DataFrame:
    path = os.path.join(data_directory_path, "y.parquet")

    with tracer.log("Loading y"):
        return pandas.read_parquet(path)
