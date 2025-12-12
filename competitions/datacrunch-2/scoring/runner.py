import gc
import json
import os
from time import sleep
from typing import TYPE_CHECKING, List

import pandas
from crunch.utils import smart_call

if TYPE_CHECKING:
    from crunch.runner.unstructured import RunnerContext, RunnerExecutorContext, UserModule


EMBARGO = 4


def load_data(
    data_directory_path: str,
):
    splits = _load_splits(
        data_directory_path=data_directory_path,
    )

    moons = splits["reduced_local"]

    (
        x_train,
        y_train,
    ) = _load_train_data(
        data_directory_path=data_directory_path,
        is_local=True,
        last_moon=moons[0],
    )

    x_test = _load_test_data(
        data_directory_path=data_directory_path,
        is_local=True,
        moons=moons,
    )

    return (
        x_train,
        y_train,
        x_test,
    )


def run(
    context: "RunnerContext",
    data_directory_path: str,
    prediction_directory_path: str,
):
    force_first_train = context.force_first_train
    train_frequency = context.train_frequency

    with open(os.path.join(data_directory_path, "moons_split.json"), "r") as fd:
        splits = json.load(fd)

    moons: List[str] = []

    if context.is_local:
        moons.extend(splits["reduced_local"])
    elif context.is_submission_phase:
        moons.extend(splits["reduced_cloud"])
    else:
        moons.extend(splits["oos_private"])
        moons.extend(splits["test"])

    predictions: List[pandas.DataFrame] = []
    prediction_file_path = os.path.join(prediction_directory_path, "prediction.parquet")

    for index, moon in enumerate(moons):
        train, forced_train = False, False
        if train_frequency != 0 and moon % train_frequency == 0:
            train = True
        elif index == 0 and not context.has_model:
            train = True
        elif index == 0 and force_first_train:
            train, forced_train = True, True

        forced_train = " forced=True" if forced_train else ""
        context.log(f"looping moon={moon} train={train}{forced_train} ({index + 1}/{len(moons)})")

        if train:
            context.execute(
                command="train",
                parameters={
                    "moon": moon,
                }
            )

        _delete_if_exists(prediction_file_path)

        context.execute(
            command="infer",
            parameters={
                "moon": moon,
                "prediction_file_path": prediction_file_path,
            }
        )

        predictions.append(pandas.read_parquet(prediction_file_path))

    _delete_if_exists(prediction_file_path)

    dataframe = pandas.concat(predictions)
    dataframe.to_parquet(prediction_file_path, index=False)


def execute(
    context: "RunnerExecutorContext",
    module: "UserModule",
    data_directory_path: str,
    model_directory_path: str,
):
    default_values = {
        "data_directory_path": data_directory_path,
        "model_directory_path": model_directory_path,
        "embargo": EMBARGO,
    }

    def train(
        moon: int,
    ):
        (
            x_train,
            y_train,
        ) = _load_train_data(
            data_directory_path=data_directory_path,
            is_local=context.is_local,
            last_moon=moon,
        )

        context.trip_data_fuse()

        train_function = module.get_function("train")

        smart_call(
            train_function,
            default_values,
            {
                "x_train": x_train,
                "X_train": x_train,
                "y_train": y_train,
                "loop_moon": moon,
            }
        )

        del x_train
        del y_train
        _call_gc()

    def infer(
        moon: int,
        prediction_file_path: str,
    ):
        x_test = _load_test_data(
            data_directory_path=data_directory_path,
            is_local=context.is_local,
            moons=[moon],
        )

        context.trip_data_fuse()

        infer_function = module.get_function("infer")

        prediction = smart_call(
            infer_function,
            default_values,
            {
                "x_test": x_test,
                "X_test": x_test,
                "loop_moon": moon,
            }
        )

        del x_test
        _call_gc()

        if not isinstance(prediction, pandas.DataFrame):
            raise ValueError(f"Expected a `pandas.DataFrame`, but got {type(prediction)}")

        prediction.to_parquet(prediction_file_path, index=False)

    return {
        "train": train,
        "infer": infer,
    }


def _delete_if_exists(path: str):
    if os.path.exists(path):
        os.unlink(path)


def _load_splits(
    *,
    data_directory_path: str,
):
    with open(os.path.join(data_directory_path, "moons_split.json"), "r") as fd:
        return json.load(fd)


def _get_data_path(
    data_directory_path: str,
    file_name: str,
    is_local: bool,
):
    return os.path.join(
        data_directory_path,
        f"{file_name}.reduced.parquet" if is_local else f"{file_name}.parquet"
    )


def _load_train_data(
    *,
    data_directory_path: str,
    is_local: bool,
    last_moon: int,
):
    filters = [
        ("moon", "<", last_moon - EMBARGO)
    ]

    x_train = pandas.read_parquet(
        _get_data_path(
            data_directory_path,
            "X",
            is_local,
        ),
        filters=filters,
        engine="pyarrow",
    )

    y_train = pandas.read_parquet(
        _get_data_path(
            data_directory_path,
            "y",
            is_local,
        ),
        filters=filters,
        engine="pyarrow",
    )

    return x_train, y_train


def _load_test_data(
    *,
    data_directory_path: str,
    is_local: bool,
    moons: List[int],
):
    if len(moons) == 1:
        moon_filter = ("moon", "=", moons[0])
    else:
        moon_filter = ("moon", "in", moons)

    x_test = pandas.read_parquet(
        _get_data_path(
            data_directory_path,
            "X",
            is_local,
        ),
        filters=[moon_filter],
        engine="pyarrow",
    )

    return x_test


def _call_gc():
    gc.collect()
    sleep(0.1)
