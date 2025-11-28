import inspect
import os
from typing import TYPE_CHECKING, Any, Literal, Union

import numpy
import pandas
from crunch.container import GeneratorWrapper
from crunch.utils import smart_call

if TYPE_CHECKING:
    from crunch.runner.unstructured import RunnerContext, RunnerExecutorContext, UserModule


def load_data(
    data_directory_path: str,
):
    x_train, y_train = _load_train(data_directory_path)
    x_test, _ = _load_x_test(data_directory_path, True)

    return x_train, y_train, x_test


def run(
    context: "RunnerContext",
    prediction_directory_path: str,
):
    if context.force_first_train:
        context.execute(
            command="train",
        )

    if not context.is_local:
        noisy_prediction_parquet_file_path = os.path.join(prediction_directory_path, "prediction.noisy.parquet")

        context.execute(
            command="infer",
            parameters={
                "dataset": "noisy",
                "determinism_check": False,
                "prediction_parquet_file_path": noisy_prediction_parquet_file_path,
            }
        )

    prediction_parquet_file_path = os.path.join(prediction_directory_path, "prediction.parquet")

    context.execute(
        command="infer",
        parameters={
            "dataset": "default",
            "determinism_check": False,
            "prediction_parquet_file_path": prediction_parquet_file_path,
        }
    )

    prediction = pandas.read_parquet(prediction_parquet_file_path)

    if context.is_determinism_check_enabled:
        percentage = 0.3
        tolerance = 1e-8

        context.log(f"checking determinism by executing the inference again with {percentage * 100:.0f}% of the data (tolerance: {tolerance})")

        prediction_determinism_parquet_file_path = os.path.join(prediction_directory_path, "prediction.determinism.parquet")

        context.execute(
            command="infer",
            parameters={
                "dataset": "default",
                "determinism_check": percentage,
                "prediction_parquet_file_path": prediction_determinism_parquet_file_path,
            }
        )

        prediction2 = pandas.read_parquet(prediction_parquet_file_path)
        os.unlink(prediction_determinism_parquet_file_path)

        is_deterministic = numpy.allclose(prediction.loc[prediction2.index], prediction2, atol=tolerance)
        context.report_determinism(is_deterministic)


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

    def infer(
        dataset: Literal["default", "noisy"],
        determinism_check: Union[Literal[False], float],
        prediction_parquet_file_path: str,
    ):
        suffix = ".noisy" if dataset == "noisy" else ""
        x_test_name = "X_test.reduced.parquet" if context.is_local else f"X_test{suffix}.parquet"
        x_test = pandas.read_parquet(os.path.join(data_directory_path, x_test_name))

        context.trip_data_fuse()

        infer_function = module.get_function("infer")

        if inspect.isgeneratorfunction(infer_function):
            print("detected yield-based function, passing iterator")

            datasets, dataset_ids = [], []
            for id, dataset in x_test.groupby(x_test.index.get_level_values("id")):
                dataset.name = id

                datasets.append(dataset)
                dataset_ids.append(id)

            if determinism_check is not False:
                determinism_slice = slice(None, int(len(datasets) * determinism_check))

                datasets = datasets[determinism_slice]
                dataset_ids = dataset_ids[determinism_slice]

            del x_test

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
        else:
            print("detected regular function, passing the full dataframe")

            result = smart_call(
                infer_function,
                default_values,
                {
                    "X_test": x_test,
                    "x_test": x_test,
                }
            )

            prediction = _post_process_infer_prediction_result(
                result,
                x_test,
            )

        prediction.to_parquet(prediction_parquet_file_path)

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


def _post_process_infer_yield_result(result: Any) -> Any:
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


def _post_process_infer_prediction_result(
    result: Any,
    X_test: pandas.DataFrame,
) -> pandas.DataFrame:
    if isinstance(result, pandas.Series):
        result = pandas.DataFrame(
            data={
                "prediction": result.values,
            },
            index=result.index,
        )

    elif isinstance(result, (numpy.ndarray, list)):
        result = pandas.DataFrame(
            data={
                "prediction": result,
            },
            index=pandas.Index(
                data=sorted(X_test.index.get_level_values(0).unique()),
                name="id",
            ),
        )

    if not isinstance(result, pandas.DataFrame):
        raise ValueError(f"must return a pandas.DataFrame, got: {result.__class__}")

    return result
