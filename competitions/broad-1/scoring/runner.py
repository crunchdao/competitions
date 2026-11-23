"""
The former SPATIAL format of crunch-cli (8.0.0 and before): `train()` once, then `infer()` once per sample, on that
sample's Zarr file. The predictions of every sample are concatenated, with a `sample` column.
"""

import os
from typing import TYPE_CHECKING, List

import pandas
from crunch.utils import smart_call

if TYPE_CHECKING:
    from crunch.runner.unstructured import RunnerContext, RunnerExecutorContext, UserModule


# the targets of the data release, in its order (one Zarr file per sample)
SAMPLE_NAMES = ["DC1", "DC5", "UC1_I", "UC1_NI", "UC6_I", "UC6_NI", "UC7_I", "UC9_I"]
SAMPLE_COLUMN_NAME = "sample"

NUMBER_OF_FEATURES = 8
EMBARGO = 0

PREDICTION_FILE_NAME = "prediction.parquet"
SAMPLE_PREDICTION_FILE_NAME = "sample-prediction.parquet"

# the former platform did not write the index of broad-1's predictions (their cell ids and genes are columns)
WRITE_INDEX = False

# set by the platform on the runner, "true" or "false"
GPU_ENVIRONMENT_VARIABLE_NAME = "GPU"


def run(
    context: "RunnerContext",
    data_directory_path: str,
    prediction_directory_path: str,
):
    # crunch-cli 8.0.0 trained when forced only; without a model, `infer()` could not work either
    if context.force_first_train or not context.has_model:
        context.log(f"train forced={context.force_first_train} has_model={context.has_model}")
        context.execute(
            command="train",
        )

    sample_prediction_file_path = os.path.join(prediction_directory_path, SAMPLE_PREDICTION_FILE_NAME)
    predictions: List[pandas.DataFrame] = []

    for index, sample_name in enumerate(SAMPLE_NAMES):
        context.log(f"looping target=`{sample_name}` ({index + 1}/{len(SAMPLE_NAMES)})")

        _delete_if_exists(sample_prediction_file_path)

        context.execute(
            command="infer",
            parameters={
                "sample_name": sample_name,
                "prediction_file_path": sample_prediction_file_path,
            },
        )

        prediction = pandas.read_parquet(sample_prediction_file_path)
        prediction[SAMPLE_COLUMN_NAME] = sample_name
        predictions.append(prediction)

    _delete_if_exists(sample_prediction_file_path)

    prediction = pandas.concat(predictions)
    prediction.to_parquet(os.path.join(prediction_directory_path, PREDICTION_FILE_NAME), index=WRITE_INDEX)


def execute(
    context: "RunnerExecutorContext",
    module: "UserModule",
    data_directory_path: str,
    model_directory_path: str,
):
    has_gpu = os.environ.get(GPU_ENVIRONMENT_VARIABLE_NAME, "false").lower() == "true"

    # the arguments of crunch-cli 8.0.0, locally and in the cloud
    default_values = {
        "data_directory_path": data_directory_path,
        "model_directory_path": model_directory_path,
        "target_names": list(SAMPLE_NAMES),
        "number_of_features": NUMBER_OF_FEATURES,
        "embargo": EMBARGO,
        "has_gpu": has_gpu,
        "has_trained": True,
    }

    def train():
        smart_call(
            module.get_function("train"),
            default_values,
        )

    def infer(
        sample_name: str,
        prediction_file_path: str,
    ):
        prediction = smart_call(
            module.get_function("infer"),
            default_values,
            {
                "data_file_path": os.path.join(data_directory_path, f"{sample_name}.zarr"),
                "target_name": sample_name,
            },
        )

        if not isinstance(prediction, pandas.DataFrame):
            raise ValueError(f"infer() must return a pandas.DataFrame, but got {type(prediction)}")

        # the index is kept here, the runner decides whether it is written
        prediction.to_parquet(prediction_file_path, index=True)

    return {
        "train": train,
        "infer": infer,
    }


def _delete_if_exists(path: str):
    if os.path.exists(path):
        os.unlink(path)
