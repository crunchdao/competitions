import os
import typing

import numpy
import pandas
from crunch.container import GeneratorWrapper
from crunch.utils import smart_call

if typing.TYPE_CHECKING:
    from crunch.runner.custom import (RunnerContext, RunnerExecutorContext,
                                      UserModule)


def load_data(
    data_directory_path: str,
):
    x_train, y_train = _load_train(data_directory_path)
    x_test, _ = _load_x_test(data_directory_path, True)

    return x_train, y_train, x_test


def run(
    context: "RunnerContext",
):
    context.execute(
        command="train",
    )

    prediction = context.execute(
        command="infer",
        return_prediction=True,
    )

    return prediction


def execute(
    context: "RunnerExecutorContext",
    module: "UserModule",
    data_directory_path: str,
    model_directory_path: str,
):
    default_values = {
        "data_directory_path": data_directory_path,
        "model_directory_path": model_directory_path,
    }

    def train():
        x_train = pandas.read_parquet(os.path.join(data_directory_path, "X_train.parquet"))
        y_train = pandas.read_parquet(os.path.join(data_directory_path, "y_train.parquet"))["structural_breakpoint"]

        context.trip_data_fuse()

        train_function = module.get_function("train")

        smart_call(
            train_function,
            default_values,
            {
                "x_train": x_train,
                "X_train": x_train,
                "y_train": y_train,
            }
        )

    def infer():
        x_test_name = "X_test.reduced.parquet" if context.is_local else "X_test.parquet"
        x_test = pandas.read_parquet(os.path.join(data_directory_path, x_test_name))

        datasets, dataset_ids = [], []
        for id, dataset in x_test.groupby(x_test.index.get_level_values("id")):
            dataset.name = id

            datasets.append(dataset)
            dataset_ids.append(id)

        del x_test

        context.trip_data_fuse()

        infer_function = module.get_function("infer")

        wrapper = GeneratorWrapper(
            iter(datasets),
            lambda datasets: smart_call(
                infer_function,
                default_values,
                {
                    "datasets": datasets,
                    "X_test": datasets,
                    "x_test": datasets,
                }
            ),
            post_processor=_post_process_infer_yield_result,
        )

        collected_values, _ = wrapper.collect(len(datasets))
        prediction = pandas.DataFrame(
            collected_values,
            columns=["prediction"],
            index=pandas.Index(dataset_ids, name="id")
        )

        return prediction

    return {
        "train": train,
        "infer": infer,
    }


def _load_train(data_directory_path: str):
    x_train = pandas.read_parquet(os.path.join(data_directory_path, "X_train.parquet"))
    y_train = pandas.read_parquet(os.path.join(data_directory_path, "y_train.parquet"))["structural_breakpoint"]

    return x_train, y_train


def _load_x_test(data_directory_path: str, reduced: bool):
    x_test_name = "X_test.reduced.parquet" if reduced else "X_test.parquet"
    x_test = pandas.read_parquet(os.path.join(data_directory_path, x_test_name))

    datasets, dataset_ids = [], []
    for id, dataset in x_test.groupby(x_test.index.get_level_values("id")):
        dataset.name = id

        datasets.append(dataset)
        dataset_ids.append(id)

    del x_test

    return datasets, dataset_ids


def _post_process_infer_yield_result(result: typing.Any) -> typing.Any:
    if isinstance(result, pandas.Series):
        if len(result) != 1:
            raise ValueError(f"a `pandas.Series` is only allowed if it has a single value, but got a length of {len(result)}")

        result = next(iter(result))

    elif isinstance(result, numpy.ndarray):
        if result.shape != (1,):
            raise ValueError(f"a `numpy.ndarray` is only allowed if it has a single dimension and a single value, but got a shape of {result.shape}")

        result = result[0]

    elif isinstance(result, list):
        if len(result) != 1:
            raise ValueError(f"a `list` is only allowed if it has a single value, but got a length of {len(result)}")

        result = result[0]

    if all(not numpy.issubdtype(type(result), dtype) for dtype in [numpy.floating, numpy.integer, numpy.bool_]):
        raise ValueError(f"value must be a float or an int or a bool, but got {type(result)}")

    else:
        result = float(result)

    return result
