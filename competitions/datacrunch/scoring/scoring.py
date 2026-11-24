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
MOON_COLUMN_NAME = "moon"

# target name -> (target column name, prediction column name)
TARGET_COLUMN_NAMES = {
    "white": ("target_w", "prediction_w"),
    "red": ("target_r", "prediction_r"),
    "green": ("target_g", "prediction_g"),
    "blue": ("target_b", "prediction_b"),
}

PREDICTION_COLUMN_NAMES = [prediction_column_name for _, prediction_column_name in TARGET_COLUMN_NAMES.values()]

SPLITS_FILE_NAME = "splits.json"
PREDICTION_FILE_NAME = "prediction.parquet"

# every row the runner gives (the ids to predict), with the targets as the participants know them
Y_FILE_NAME = "y.parquet"

# the returns of the scored moons, as they resolved: what is scored
Y_RAW_FILE_NAME = "y_raw.parquet"

# the platform's check of 2024
MINIMUM_VALUE = 0.0
MAXIMUM_VALUE = 1.0


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
            {MOON_COLUMN_NAME, ID_COLUMN_NAME, *PREDICTION_COLUMN_NAMES},
            set(prediction.columns),
        )

        if difference:
            raise ParticipantVisibleError(f"Columns do not match: {difference}")

    with tracer.log("Check for dtypes"):
        if not numpy.issubdtype(prediction.dtypes[MOON_COLUMN_NAME], numpy.integer):
            raise ParticipantVisibleError(f"Column `{MOON_COLUMN_NAME}` must be of type int")

        for prediction_column_name in PREDICTION_COLUMN_NAMES:
            if not numpy.issubdtype(prediction.dtypes[prediction_column_name], numpy.number):
                raise ParticipantVisibleError(f"Column `{prediction_column_name}` must be numeric")

    values = prediction[PREDICTION_COLUMN_NAMES]

    with tracer.log("Check for NaN and infinite values"):
        if values.isna().any().any():
            raise ParticipantVisibleError("Prediction must not contain NaN values")

        if numpy.isinf(values.to_numpy()).any():
            raise ParticipantVisibleError("Prediction must not contain infinite values")

    with tracer.log(f"Check for [{MINIMUM_VALUE}, {MAXIMUM_VALUE}]"):
        for prediction_column_name in PREDICTION_COLUMN_NAMES:
            column = prediction[prediction_column_name]

            if (column < MINIMUM_VALUE).any() or (column > MAXIMUM_VALUE).any():
                raise ParticipantVisibleError(f"Column `{prediction_column_name}` must be between {MINIMUM_VALUE} and {MAXIMUM_VALUE}")

    splits = _load_splits(data_directory_path)
    moons = _get_test_moons(splits)

    with tracer.log("Check for moons"):
        difference = delta_message(
            set(moons),
            set(prediction[MOON_COLUMN_NAME].unique()),
        )

        if difference:
            raise ParticipantVisibleError(f"Moons do not match: {difference}")

    ids = _load_ids(data_directory_path, moons)

    with tracer.log("Check for ids and constant values"):
        ids_by_moon = ids.groupby(MOON_COLUMN_NAME)[ID_COLUMN_NAME]

        for moon, group in prediction.groupby(MOON_COLUMN_NAME):
            moon_ids = group[ID_COLUMN_NAME]

            difference = delta_message(
                set(ids_by_moon.get_group(moon)),
                set(moon_ids),
            )

            if difference:
                raise ParticipantVisibleError(f"IDs do not match for moon {moon}: {difference}")

            if moon_ids.duplicated().any():
                raise ParticipantVisibleError(f"Duplicate IDs found for moon {moon}")

            # the spearman correlation of a constant is undefined
            for prediction_column_name in PREDICTION_COLUMN_NAMES:
                if group[prediction_column_name].nunique() == 1:
                    raise ParticipantVisibleError(f"Constant values found in `{prediction_column_name}` for moon {moon}")


def score(
    prediction_directory_path: str,
    data_directory_path: str,
    target_and_metrics: List[Tuple[Target, List[Metric]]],
):
    prediction = _load_prediction(prediction_directory_path)

    splits = _load_splits(data_directory_path)
    y_raw = _load_y_raw(data_directory_path, _get_scored_moons(splits))

    with tracer.log("Merge prediction with y"):
        merged = y_raw.merge(prediction, on=[MOON_COLUMN_NAME, ID_COLUMN_NAME])

    scored_metrics: Dict[int, ScoredMetric] = {}
    for target, metrics in target_and_metrics:
        target_column_name, prediction_column_name = TARGET_COLUMN_NAMES[target.name]
        spearman_metric = _find_metric_by_name(metrics, "spearman")

        with tracer.log(f"Compute spearman of {target.name}"):
            # the returns that did not resolve (a delisting) are left out, as the platform did
            spearman_by_moon = merged.groupby(MOON_COLUMN_NAME).apply(
                lambda group: group[prediction_column_name].corr(group[target_column_name], method="spearman"),
                include_groups=False,
            )

            if not len(spearman_by_moon):
                raise ValueError("no moon to score")

            scored_metrics[spearman_metric.id] = ScoredMetric(
                value=float(spearman_by_moon.mean()),
            )

    return scored_metrics


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


def _load_ids(
    data_directory_path: str,
    moons: List[int],
) -> pandas.DataFrame:
    with tracer.log("Load ids"):
        return pandas.read_parquet(
            os.path.join(data_directory_path, Y_FILE_NAME),
            columns=[MOON_COLUMN_NAME, ID_COLUMN_NAME],
            filters=[(MOON_COLUMN_NAME, "in", moons)],
            engine="pyarrow",
        )


def _load_y_raw(
    data_directory_path: str,
    moons: List[int],
) -> pandas.DataFrame:
    with tracer.log("Load y raw"):
        return pandas.read_parquet(
            os.path.join(data_directory_path, Y_RAW_FILE_NAME),
            columns=[MOON_COLUMN_NAME, ID_COLUMN_NAME, *(target_column_name for target_column_name, _ in TARGET_COLUMN_NAMES.values())],
            filters=[(MOON_COLUMN_NAME, "in", moons)],
            engine="pyarrow",
        )
