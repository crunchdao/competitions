import os
from typing import TYPE_CHECKING

from crunch.utils import smart_call

if TYPE_CHECKING:
    from crunch.runner.unstructured import RunnerContext, UserModule


PREDICTION_FILE_NAME = "prediction.parquet"


def run(
    context: "RunnerContext",
    prediction_directory_path: str,
):
    prediction_file_path = os.path.join(
        prediction_directory_path,
        PREDICTION_FILE_NAME,
    )

    context.execute(
        command="infer",
        parameters={
            "prediction_file_path": prediction_file_path,
        },
    )


def execute(
    module: "UserModule",
    data_directory_path: str,
    model_directory_path: str,
):
    default_values = {
        "data_directory_path": data_directory_path,
        "model_directory_path": model_directory_path,
    }

    def infer(
        prediction_file_path: str,
    ):
        infer_function = module.get_function("infer")

        smart_call(
            infer_function,
            default_values,
            {
                "prediction_file_path": prediction_file_path,
            }
        )

    return {
        "infer": infer,
    }
