import gc
import json
import os
from dataclasses import dataclass
from datetime import timedelta
from statistics import mean
from time import sleep, time
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
        if context.is_first_time:
            moons.extend(splits["oos_private"])

        moons.extend(splits["live_to_predict"])

    predictions: List[pandas.DataFrame] = []
    prediction_file_path = os.path.join(prediction_directory_path, "prediction.parquet")

    moon_infos = MoonInfo.get(
        moons=moons,
        train_frequency=train_frequency,
        has_model=context.has_model,
        force_first_train=force_first_train,
    )

    train_estimator = DurationEstimator(total_count=sum(1 for moon_info in moon_infos if moon_info.train))
    infer_estimator = DurationEstimator(total_count=len(moon_infos))

    def stop_if_estimated_time_exceeds_remaining_time():
        remaining_time = context.remaining_duration_before_timeout
        if remaining_time is None:
            return

        extra_percent = 1.05
        required_time = (train_estimator.estimated_required_time + infer_estimator.estimated_required_time) * extra_percent

        required_time_str = _truncate_milliseconds(required_time)
        train_average = _truncate_milliseconds(train_estimator.average_duration)
        infer_average = _truncate_milliseconds(infer_estimator.average_duration)
        remaining_time_str = _truncate_milliseconds(remaining_time)

        context.log(f"[debug] estimated required time: ~{required_time_str} (train average: {train_average}, infer average: {infer_average}), remaining time: ~{remaining_time_str}")

        if required_time > remaining_time:
            context.log(f"stopping early at moon={moon_info.key} because the estimated required time (~{required_time_str}, at ~{train_average}/train, and ~{infer_average}/infer, +{int((extra_percent - 1) * 100)}%) is greater than the remaining time before timeout (~{remaining_time_str})", error=True)
            exit(1)

    for moon_info in moon_infos:
        forced_train = " forced=True" if moon_info.forced_train else ""
        context.log(f"looping moon={moon_info.key} train={moon_info.train}{forced_train} ({moon_info.number}/{len(moons)})")

        if moon_info.train:
            stop_if_estimated_time_exceeds_remaining_time()

            with train_estimator:
                context.execute(
                    command="train",
                    parameters={
                        "moon": moon_info.key,
                    }
                )

        _delete_if_exists(prediction_file_path)

        if True:
            stop_if_estimated_time_exceeds_remaining_time()

            with infer_estimator:
                context.execute(
                    command="infer",
                    parameters={
                        "moon": moon_info.key,
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

        # pandas.DataFrame().to_parquet(prediction_file_path, index=False)

    return {
        "train": train,
        "infer": infer,
    }


@dataclass
class MoonInfo:

    key: int
    number: int
    train: bool
    forced_train: bool

    @staticmethod
    def get(
        moons: List[int],
        train_frequency: int,
        has_model: bool,
        force_first_train: bool,
    ):
        infos: List["MoonInfo"] = []

        for index, moon in enumerate(moons):
            train, forced_train = False, False
            if train_frequency != 0 and moon % train_frequency == 0:
                train = True
            elif index == 0 and not has_model:
                train = True
            elif index == 0 and force_first_train:
                train, forced_train = True, True

            infos.append(MoonInfo(
                key=moon,
                number=index + 1,
                train=train,
                forced_train=forced_train,
            ))

        return infos


class DurationEstimator:

    def __init__(
        self,
        total_count: int,
    ):
        self.total_count = total_count

        self.durations: List[float] = []

    def __enter__(self):
        self._start_time = time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        duration = time() - self._start_time
        # duration = 3600
        self.durations.append(duration)

    @property
    def average_duration(self) -> float:
        if not self.durations:
            return timedelta(seconds=0)

        return timedelta(seconds=mean(self.durations))

    @property
    def estimated_required_time(self) -> timedelta:
        remaining_count = self.total_count - len(self.durations)

        return timedelta(seconds=self.average_duration.total_seconds() * remaining_count)


def _truncate_milliseconds(duration: timedelta):
    minute, second = divmod(duration.total_seconds(), 60)
    hour, minute = divmod(minute, 60)

    builder = ""
    if hour != 0:
        builder += f"%dh" % int(hour)
    if minute != 0:
        zeros = 2 if minute < 10 and builder != "" else 1
        builder += f"%0{zeros}dm" % int(minute)
    if second != 0 or builder == "":
        zeros = 2 if second < 10 and builder != "" else 1
        builder += f"%0{zeros}ds" % int(second)

    return builder


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
