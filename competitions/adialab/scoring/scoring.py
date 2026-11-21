import json
import os
from typing import Any, Dict, List, Tuple

import numpy
import pandas
from crunch.api import Metric, Target
from crunch.unstructured import ScoredMetric
from crunch.unstructured.utils import delta_message
from crunch.utils import Tracer

ID_COLUMN_NAME = "id"
MOON_COLUMN_NAME = "date"
TARGET_COLUMN_NAME = "y"
PREDICTION_COLUMN_NAME = "value"

SPLITS_FILE_NAME = "splits.json"
PREDICTION_FILE_NAME = "prediction.parquet"
Y_FILE_NAME = "y.parquet"

# the target is uniform on [-1, 1], but only the ranks are scored: this only rejects broken values
MINIMUM_VALUE = -10.0
MAXIMUM_VALUE = 10.0


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


tracer = Tracer()


def check(
    prediction_directory_path: str,
    data_directory_path: str,
):
    prediction = _load_prediction(prediction_directory_path)

    with tracer.log("Check for column names"):
        difference = delta_message(
            {MOON_COLUMN_NAME, ID_COLUMN_NAME, PREDICTION_COLUMN_NAME},
            set(prediction.columns),
        )

        if difference:
            raise ParticipantVisibleError(f"Columns do not match: {difference}")

    with tracer.log("Check for dtypes"):
        if not numpy.issubdtype(prediction.dtypes[MOON_COLUMN_NAME], numpy.integer):
            raise ParticipantVisibleError(f"Column `{MOON_COLUMN_NAME}` must be of type int")

        if not numpy.issubdtype(prediction.dtypes[PREDICTION_COLUMN_NAME], numpy.number):
            raise ParticipantVisibleError(f"Column `{PREDICTION_COLUMN_NAME}` must be numeric")

    values = prediction[PREDICTION_COLUMN_NAME]

    with tracer.log("Check for NaN and infinite values"):
        if values.isna().any():
            raise ParticipantVisibleError("Prediction must not contain NaN values")

        if numpy.isinf(values).any():
            raise ParticipantVisibleError("Prediction must not contain infinite values")

    with tracer.log(f"Check for [{MINIMUM_VALUE}, {MAXIMUM_VALUE}]"):
        if (values < MINIMUM_VALUE).any() or (values > MAXIMUM_VALUE).any():
            raise ParticipantVisibleError(f"Prediction must be between {MINIMUM_VALUE} and {MAXIMUM_VALUE}")

    splits = _load_splits(data_directory_path)
    moons = _get_test_moons(splits)

    with tracer.log("Check for moons"):
        difference = delta_message(
            set(moons),
            set(prediction[MOON_COLUMN_NAME].unique()),
        )

        if difference:
            raise ParticipantVisibleError(f"Moons do not match: {difference}")

    y = _load_y(data_directory_path, moons)

    with tracer.log("Check for ids and constant values"):
        y_ids_by_moon = y.groupby(MOON_COLUMN_NAME)[ID_COLUMN_NAME]

        for moon, group in prediction.groupby(MOON_COLUMN_NAME):
            ids = group[ID_COLUMN_NAME]

            difference = delta_message(
                set(y_ids_by_moon.get_group(moon)),
                set(ids),
            )

            if difference:
                raise ParticipantVisibleError(f"IDs do not match for moon {moon}: {difference}")

            if ids.duplicated().any():
                raise ParticipantVisibleError(f"Duplicate IDs found for moon {moon}")

            # the spearman correlation of a constant is undefined
            if group[PREDICTION_COLUMN_NAME].nunique() == 1:
                raise ParticipantVisibleError(f"Constant values found for moon {moon}")


def score(
    prediction_directory_path: str,
    data_directory_path: str,
    target_and_metrics: List[Tuple[Target, List[Metric]]],
):
    _, metrics = target_and_metrics[0]
    spearman_metric = _find_metric_by_name(metrics, "spearman")

    prediction = _load_prediction(prediction_directory_path)

    splits = _load_splits(data_directory_path)
    y = _load_y(data_directory_path, _get_scored_moons(splits))

    with tracer.log("Merge prediction with y"):
        merged = y.merge(prediction, on=[MOON_COLUMN_NAME, ID_COLUMN_NAME])

    with tracer.log("Compute spearman"):
        spearman_by_moon = merged.groupby(MOON_COLUMN_NAME).apply(
            lambda group: group[PREDICTION_COLUMN_NAME].corr(group[TARGET_COLUMN_NAME], method="spearman"),
            include_groups=False,
        )

        if not len(spearman_by_moon):
            raise ValueError("no moon to score")

        score_value = float(spearman_by_moon.mean())

    return {
        spearman_metric.id: ScoredMetric(
            value=score_value,
        ),
    }


def _find_metric_by_name(
    metrics: List[Metric],
    name: str,
) -> Metric:
    for metric in metrics:
        if metric.name == name:
            return metric

    raise ValueError(f"metric {name} not found")


def _load_splits(
    data_directory_path: str,
) -> List[Dict[str, Any]]:
    """
    The splits of the data release, as `{"key": <moon>, "group": "TRAIN" | "TEST", "reduced": "XY" | "X" | None}`.
    """

    with tracer.log("Load splits"):
        with open(os.path.join(data_directory_path, SPLITS_FILE_NAME), "r") as fd:
            return json.load(fd)


def _get_test_moons(
    splits: List[Dict[str, Any]],
) -> List[int]:
    """
    The moons the cloud predicts, the reduced ones included.
    """

    return sorted(
        split["key"]
        for split in splits
        if split["group"] == "TEST"
    )


def _get_scored_moons(
    splits: List[Dict[str, Any]],
) -> List[int]:
    """
    The moons that count: the reduced moons are public (given locally).
    """

    return sorted(
        split["key"]
        for split in splits
        if split["group"] == "TEST" and split["reduced"] is None
    )


def _load_prediction(
    prediction_directory_path: str,
) -> pandas.DataFrame:
    path = os.path.join(prediction_directory_path, PREDICTION_FILE_NAME)
    if not os.path.exists(path):
        raise ParticipantVisibleError(f"Prediction file `{PREDICTION_FILE_NAME}` not found")

    with tracer.log("Load prediction"):
        return pandas.read_parquet(path)


def _load_y(
    data_directory_path: str,
    moons: List[int],
) -> pandas.DataFrame:
    with tracer.log("Load y"):
        return pandas.read_parquet(
            os.path.join(data_directory_path, Y_FILE_NAME),
            columns=[MOON_COLUMN_NAME, ID_COLUMN_NAME, TARGET_COLUMN_NAME],
            filters=[(MOON_COLUMN_NAME, "in", moons)],
            engine="pyarrow",
        )
