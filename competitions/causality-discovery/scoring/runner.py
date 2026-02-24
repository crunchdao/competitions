import os
from typing import TYPE_CHECKING

import joblib
import pandas
from crunch.utils import smart_call

if TYPE_CHECKING:
    from crunch.runner.unstructured import RunnerContext, RunnerExecutorContext, UserModule


PREDICTION_FILE_NAME = "prediction.parquet"


def run(
    context: "RunnerContext",
    data_directory_path: str,
    prediction_directory_path: str,
):
    x_train_file_path = os.path.join(data_directory_path, "X_train.pickle")
    y_train_file_path = os.path.join(data_directory_path, "y_train.pickle")

    if context.is_local:
        x_test_file_path = os.path.join(data_directory_path, "X_test_reduced.pickle")
    else:
        x_test_file_path = os.path.join(data_directory_path, "X_test.pickle")

    prediction_file_path = os.path.join(prediction_directory_path, PREDICTION_FILE_NAME)

    if context.force_first_train:
        context.execute(
            command="train",
            parameters={
                "x_train_file_path": x_train_file_path,
                "y_train_file_path": y_train_file_path,
            }
        )

    context.execute(
        command="infer",
        parameters={
            "x_test_file_path": x_test_file_path,
            "prediction_file_path": prediction_file_path,
        }
    )


def execute(
    context: "RunnerExecutorContext",
    module: "UserModule",
    data_directory_path: str,
    model_directory_path: str,
):
    default_values = {
        "data_directory_path": data_directory_path,
        "model_directory_path": model_directory_path,
        "id_column_name": "example_id",
        "moon_column_name": "dataset",
        "prediction_column_name": "prediction",
    }

    def train(
        x_train_file_path: str,
        y_train_file_path: str,
    ):
        x_train = joblib.load(x_train_file_path)
        y_train = joblib.load(y_train_file_path)

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
        x_test_file_path: str,
        prediction_file_path: str,
    ):
        x_test = joblib.load(x_test_file_path)

        context.trip_data_fuse()

        infer_function = module.get_function("infer")

        prediction = smart_call(
            infer_function,
            default_values,
            {
                "x_test": x_test,
                "X_test": x_test,
            }
        )

        if not isinstance(prediction, pandas.DataFrame):
            raise ValueError("infer() must return a pandas.DataFrame()")

        prediction.to_parquet(prediction_file_path, index=False)

    return {
        "train": train,
        "infer": infer,
    }
