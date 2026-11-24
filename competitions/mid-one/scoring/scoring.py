"""
Mid+One, scored as crunch-cli 8.0.0 did (`crunch/scoring/_format/stream.py`, `score.py`, `scorers.py`), the platform's
reduction included: the values returned are the totals (the metric's multiplier still applies).

Per stream, the answers are matched to `y_test`'s messages by their order (the missing values left out). Each moon
(`time`) scores `buy_profit * sign(prediction)` of its **last** message, minus `TRANSACTION_COST` for each of its
messages with a non-zero position: 8.0.0 scored every message, then kept one value per moon, the last one. A stream's
`pnl` is the sum over its scored moons (the test moons that are not public: `reduced` is null in `splits.json`), and
`all-streams` the sum of the streams'. `execution-time` (nanoseconds): per stream, the mean over the scored moons of
the moon's mean answer time; `all-streams` the mean of the streams'.
"""

import json
import os
import statistics
from typing import Any, Dict, List, Set, Tuple

import numpy
import pandas
from crunch.api import Metric, Target
from crunch.unstructured import ScoredMetric
from crunch.utils import Tracer

MOON_COLUMN_NAME = "time"
ID_COLUMN_NAME = "stream"
TARGET_COLUMN_NAME = "buy_profit"
PREDICTION_COLUMN_NAME = "prediction"
DURATION_COLUMN_NAME = "duration_ns"

Y_FILE_NAME = "y_test.parquet"
SPLITS_FILE_NAME = "splits.json"
PREDICTION_FILE_NAME = "prediction.parquet"

ALL_STREAMS_TARGET_NAME = "all-streams"
PROFIT_AND_LOSS_METRIC_NAME = "pnl"
EXECUTION_TIME_METRIC_NAME = "execution-time"

TRANSACTION_COST = 0.0025


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


tracer = Tracer()


def check(
    prediction_directory_path: str,
    data_directory_path: str,
):
    prediction = _load_prediction(prediction_directory_path)
    y_test = _load_y(data_directory_path)

    with tracer.log("Check the columns"):
        missing_column_names = [
            column_name
            for column_name in (MOON_COLUMN_NAME, ID_COLUMN_NAME, PREDICTION_COLUMN_NAME)
            if column_name not in prediction.columns
        ]

        if missing_column_names:
            raise ParticipantVisibleError(f"Missing columns: {missing_column_names}")

    with tracer.log("Check the values"):
        values = pandas.to_numeric(prediction[PREDICTION_COLUMN_NAME], errors="coerce").to_numpy(dtype=numpy.float64)
        if not numpy.isfinite(values).all():
            raise ParticipantVisibleError("Every prediction must be a finite number")

    with tracer.log("Check the messages"):
        expected_names = set(y_test[ID_COLUMN_NAME].unique())
        names = set(prediction[ID_COLUMN_NAME].astype(str).unique())
        if names != expected_names:
            raise ParticipantVisibleError(f"Expected the streams {sorted(expected_names)}, got {sorted(names)}")

        for stream_name, (y_group, prediction_group) in _by_stream(y_test, prediction).items():
            if len(prediction_group) != len(y_group):
                raise ParticipantVisibleError(f"{stream_name}: expected {len(y_group):,} answers, got {len(prediction_group):,}")

            if not numpy.array_equal(prediction_group[MOON_COLUMN_NAME].to_numpy(), y_group[MOON_COLUMN_NAME].to_numpy()):
                raise ParticipantVisibleError(f"{stream_name}: the answers are not in the order of the messages")


def score(
    prediction_directory_path: str,
    data_directory_path: str,
    target_and_metrics: List[Tuple[Target, List[Metric]]],
):
    prediction = _load_prediction(prediction_directory_path)
    y_test = _load_y(data_directory_path)
    scored_moons = _get_scored_moons(_load_splits(data_directory_path))

    profits_and_losses: Dict[str, float] = {}
    execution_times: Dict[str, float] = {}
    has_durations = DURATION_COLUMN_NAME in prediction.columns

    with tracer.log("Score each stream"):
        for stream_name, (y_group, prediction_group) in _by_stream(y_test, prediction).items():
            profits_and_losses[stream_name] = profit_and_loss(
                y_group[TARGET_COLUMN_NAME].to_numpy(dtype=numpy.float64),
                prediction_group[PREDICTION_COLUMN_NAME].to_numpy(dtype=numpy.float64),
                prediction_group[MOON_COLUMN_NAME].to_numpy(),
                scored_moons,
            )

            if has_durations:
                execution_times[stream_name] = execution_time(
                    prediction_group[DURATION_COLUMN_NAME].to_numpy(dtype=numpy.float64),
                    prediction_group[MOON_COLUMN_NAME].to_numpy(),
                    scored_moons,
                )

    profits_and_losses[ALL_STREAMS_TARGET_NAME] = sum(profits_and_losses.values())
    if has_durations:
        execution_times[ALL_STREAMS_TARGET_NAME] = statistics.mean(execution_times.values())

    print(f"pnl: {profits_and_losses[ALL_STREAMS_TARGET_NAME]:.6f} over {len(profits_and_losses) - 1} streams, {len(scored_moons):,} scored moons")

    scores: Dict[int, ScoredMetric] = {}
    for target, metrics in target_and_metrics:
        for metric in metrics:
            if metric.name == PROFIT_AND_LOSS_METRIC_NAME:
                values = profits_and_losses
            elif metric.name == EXECUTION_TIME_METRIC_NAME:
                values = execution_times
            else:
                print(f"unknown metric: {target.name}/{metric.name}")
                continue

            value = values.get(target.name)
            if value is None:
                print(f"no value for {target.name}/{metric.name}")
                continue

            scores[metric.id] = ScoredMetric(value=value)

    if not scores:
        raise ValueError("no known target or metric")

    return scores


def profit_and_loss(
    targets: numpy.ndarray,
    predictions: numpy.ndarray,
    moons: numpy.ndarray,
    scored_moons: Set[int],
) -> float:
    """The sum over the scored moons of the last message's `target * sign(prediction)` minus the moon's costs."""

    frame = pandas.DataFrame({
        "moon": moons,
        "profit": targets * numpy.sign(predictions),
        "transactions": (predictions != 0).astype(numpy.int64),
    })

    # sorted by moon, as 8.0.0 summed them
    grouped = frame.groupby("moon", sort=True).agg(last=("profit", "last"), transactions=("transactions", "sum"))
    values = grouped["last"] - grouped["transactions"] * TRANSACTION_COST

    return sum(value for moon, value in values.items() if moon in scored_moons)


def execution_time(
    durations: numpy.ndarray,
    moons: numpy.ndarray,
    scored_moons: Set[int],
) -> float:
    """The mean over the scored moons of the moon's mean answer time."""

    per_moon = pandas.Series(durations).groupby(moons, sort=True).mean()

    return float(per_moon[per_moon.index.isin(scored_moons)].mean())


def _by_stream(
    y_test: pandas.DataFrame,
    prediction: pandas.DataFrame,
) -> Dict[str, Tuple[pandas.DataFrame, pandas.DataFrame]]:
    """Per stream, the messages (the missing values left out) and the answers, both in their order."""

    stream_names = prediction[ID_COLUMN_NAME].astype(str)

    return {
        stream_name: (
            y_group,
            prediction[(stream_names == stream_name).to_numpy()],
        )
        for stream_name, y_group in y_test.groupby(ID_COLUMN_NAME, sort=False)
    }


def _load_prediction(prediction_directory_path: str) -> pandas.DataFrame:
    path = os.path.join(prediction_directory_path, PREDICTION_FILE_NAME)
    if not os.path.exists(path):
        raise ParticipantVisibleError(f"Prediction file `{PREDICTION_FILE_NAME}` not found")

    with tracer.log("Load the prediction"):
        return pandas.read_parquet(path)


def _load_y(data_directory_path: str) -> pandas.DataFrame:
    with tracer.log("Load y"):
        y_test = pandas.read_parquet(os.path.join(data_directory_path, Y_FILE_NAME), columns=[MOON_COLUMN_NAME, ID_COLUMN_NAME, TARGET_COLUMN_NAME])
        return y_test.dropna(subset=[TARGET_COLUMN_NAME])


def _load_splits(data_directory_path: str) -> List[Dict[str, Any]]:
    """The splits of the data release, as `{"key": <moon>, "group": "TRAIN" | "TEST", "reduced": "XY" | "X" | None}`."""

    with tracer.log("Load splits"):
        with open(os.path.join(data_directory_path, SPLITS_FILE_NAME), "r") as fd:
            return json.load(fd)


def _get_scored_moons(splits: List[Dict[str, Any]]) -> Set[int]:
    """The moons that count: the reduced moons are public (given locally)."""

    return {
        split["key"]
        for split in splits
        if split["group"] == "TEST" and split["reduced"] is None
    }
