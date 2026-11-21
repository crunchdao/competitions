import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy
import pandas
import scipy.stats
from crunch.api import Metric, Target
from crunch.unstructured import ScoredMetric
from crunch.unstructured.utils import delta_message
from crunch.utils import Tracer

ID_COLUMN_NAME = "id"
MOON_COLUMN_NAME = "moon"
TARGET_COLUMN_NAME = "score_target"
PREDICTION_COLUMN_NAME = "prediction"

SPLITS_FILE_NAME = "splits.json"
PREDICTION_FILE_NAME = "prediction.parquet"
Y_RAW_FILE_NAME = "y_raw.parquet"
ORTHOGONALIZATION_DATA_FILE_NAME = "orthogonalization_data.pickle"

PROJECTOR_COLUMN_NAME_MARKER = "Projector"


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

    # no range: only the ranks are used (the prediction is gaussianized before being orthogonalized)
    with tracer.log("Check for NaN and infinite values"):
        if values.isna().any():
            raise ParticipantVisibleError("Prediction must not contain NaN values")

        if numpy.isinf(values).any():
            raise ParticipantVisibleError("Prediction must not contain infinite values")

    splits = _load_splits(data_directory_path)
    moons = _get_test_moons(splits)

    with tracer.log("Check for moons"):
        difference = delta_message(
            set(moons),
            set(prediction[MOON_COLUMN_NAME].unique()),
        )

        if difference:
            raise ParticipantVisibleError(f"Moons do not match: {difference}")

    y_raw = _load_y_raw(data_directory_path, moons)

    with tracer.log("Check for ids and constant values"):
        y_ids_by_moon = y_raw.groupby(MOON_COLUMN_NAME)[ID_COLUMN_NAME]

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

            # a constant is orthogonalized to zero, which cannot be normalized
            if group[PREDICTION_COLUMN_NAME].nunique() == 1:
                raise ParticipantVisibleError(f"Constant values found for moon {moon}")


def score(
    prediction_directory_path: str,
    data_directory_path: str,
    target_and_metrics: List[Tuple[Target, List[Metric]]],
):
    _, metrics = target_and_metrics[0]
    alpha_score_metric = _find_metric_by_name(metrics, "alpha-score")

    prediction = _load_prediction(prediction_directory_path)

    splits = _load_splits(data_directory_path)
    moons = _get_test_moons(splits)

    y_raw = _load_y_raw(data_directory_path, moons)
    orthogonalization_data = _load_orthogonalization_data(data_directory_path)

    with tracer.log("Orthogonalize prediction"):
        orthogonalized = _orthogonalize(
            prediction[prediction[MOON_COLUMN_NAME].isin(moons)],
            orthogonalization_data,
        )

    with tracer.log("Merge prediction with y_raw"):
        merged = y_raw.merge(orthogonalized, on=[MOON_COLUMN_NAME, ID_COLUMN_NAME])

    with tracer.log("Compute alpha score"):
        dot_product_by_moon = merged.groupby(MOON_COLUMN_NAME).apply(
            lambda group: numpy.dot(group[PREDICTION_COLUMN_NAME], group[TARGET_COLUMN_NAME]),
            include_groups=False,
        )

        if not len(dot_product_by_moon):
            raise ValueError("no moon to score")

        # the returns of every moon, compounded
        score_value = float(numpy.prod(dot_product_by_moon.values + 1) - 1)

    return {
        alpha_score_metric.id: ScoredMetric(
            value=score_value,
        ),
    }


def _orthogonalize(
    prediction: pandas.DataFrame,
    orthogonalization_data: Dict[int, pandas.DataFrame],
) -> pandas.DataFrame:
    """
    As the platform did before scoring: per moon, gaussianize, remove the part explained by the projectors, center, and L1-normalize.
    """

    orthogonalized: List[pandas.DataFrame] = []

    for moon, group in prediction.groupby(MOON_COLUMN_NAME):
        projection = orthogonalization_data[moon]
        projector_column_names = [
            column_name
            for column_name in projection.columns
            if PROJECTOR_COLUMN_NAME_MARKER in column_name
        ]

        gaussianized = group[[ID_COLUMN_NAME]].copy()
        gaussianized[PREDICTION_COLUMN_NAME] = _gaussianize(group[PREDICTION_COLUMN_NAME])

        # "right" keeps the order of the projection matrix's rows
        merged = pandas.merge(
            gaussianized,
            projection.drop(columns=[MOON_COLUMN_NAME]),
            on=ID_COLUMN_NAME,
            how="right",
        )

        values = merged[PREDICTION_COLUMN_NAME].to_numpy()
        values = values - numpy.dot(merged[projector_column_names], values)
        values = values - values.mean()
        values = values / numpy.linalg.norm(values, ord=1)

        orthogonalized.append(pandas.DataFrame({
            MOON_COLUMN_NAME: moon,
            ID_COLUMN_NAME: merged[ID_COLUMN_NAME],
            PREDICTION_COLUMN_NAME: values,
        }))

    return pandas.concat(orthogonalized, ignore_index=True)


def _gaussianize(
    series: pandas.Series,
) -> numpy.ndarray:
    normalized_rank = series.rank(method="average").to_numpy() / (len(series) + 1)
    return scipy.stats.norm.ppf(normalized_rank)


class _Unpickler(pickle.Unpickler):
    """
    The orthogonalization data was pickled with pandas 1, whose `NumericIndex` classes are gone.
    """

    def find_class(self, module: str, name: str):
        if module == "pandas.core.indexes.numeric":
            return pandas.Index

        return super().find_class(module, name)


def _load_orthogonalization_data(
    data_directory_path: str,
) -> Dict[int, pandas.DataFrame]:
    """
    The projection matrix of every moon: `id`, `moon`, and one `Projector_<n>` column per id.
    """

    with tracer.log("Load orthogonalization data"):
        with open(os.path.join(data_directory_path, ORTHOGONALIZATION_DATA_FILE_NAME), "rb") as fd:
            return _Unpickler(fd).load()


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
    The moons the cloud predicts, all of them scored.
    """

    return sorted(
        split["key"]
        for split in splits
        if split["group"] == "TEST"
    )


def _load_prediction(
    prediction_directory_path: str,
) -> pandas.DataFrame:
    path = os.path.join(prediction_directory_path, PREDICTION_FILE_NAME)
    if not os.path.exists(path):
        raise ParticipantVisibleError(f"Prediction file `{PREDICTION_FILE_NAME}` not found")

    with tracer.log("Load prediction"):
        return pandas.read_parquet(path)


def _load_y_raw(
    data_directory_path: str,
    moons: List[int],
) -> pandas.DataFrame:
    with tracer.log("Load y_raw"):
        return pandas.read_parquet(
            os.path.join(data_directory_path, Y_RAW_FILE_NAME),
            columns=[MOON_COLUMN_NAME, ID_COLUMN_NAME, TARGET_COLUMN_NAME],
            filters=[(MOON_COLUMN_NAME, "in", moons)],
            engine="pyarrow",
        )
