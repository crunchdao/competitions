import os
import shutil
from typing import TYPE_CHECKING, List, Literal, Optional

import anndata
import pandas
import scanpy
from crunch.utils import smart_call

if TYPE_CHECKING:
    from crunch.runner.unstructured import RunnerContext, RunnerExecutorContext, UserModule


PREDICTION_FILE_NAME = "prediction.h5ad"
PROGRAM_PROPORTION_FILE_NAME = "predict_program_proportion.csv"


def run(
    context: "RunnerContext",
    data_directory_path: str,
    prediction_directory_path: str,
):
    if context.is_local:
        perturbations_to_score_file_path = os.path.join(data_directory_path, "program_proportion_local_gtruth.csv")
        perturbations_to_score = pandas.read_csv(perturbations_to_score_file_path, usecols=["gene"])["gene"].tolist()
    else:
        perturbations_to_score_file_path = os.path.join(data_directory_path, "perturbations_to_score.txt")
        perturbations_to_score = pandas.read_csv(perturbations_to_score_file_path, header=None)[0].tolist()
        os.unlink(perturbations_to_score_file_path)

    if context.force_first_train:
        context.execute(
            command="train",
        )

    _validate_prediction_files(
        context=context,
        prediction_directory_path=prediction_directory_path,
        location="before",
    )

    prediction_h5ad_file_path = os.path.join(
        prediction_directory_path,
        PREDICTION_FILE_NAME,
    )

    program_proportion_csv_file_path = os.path.join(
        prediction_directory_path,
        PROGRAM_PROPORTION_FILE_NAME,
    )

    context.execute(
        command="infer",
        parameters={
            "prediction_h5ad_file_path": prediction_h5ad_file_path,
            "program_proportion_csv_file_path": program_proportion_csv_file_path,
        }
    )

    prediction = scanpy.read_h5ad(prediction_h5ad_file_path)
    prediction = prediction[prediction.obs["gene"].isin(perturbations_to_score)]
    prediction.write(prediction_h5ad_file_path)

    _validate_prediction_files(
        context=context,
        prediction_directory_path=prediction_directory_path,
        location="after",
    )


def execute(
    context: "RunnerExecutorContext",
    module: "UserModule",
    data_directory_path: str,
    model_directory_path: str,
    prediction_directory_path: str,
):
    default_values = {
        "data_directory_path": data_directory_path,
        "model_directory_path": model_directory_path,
    }

    def train():
        train_function = module.get_function("train")

        smart_call(
            train_function,
            default_values,
        )

    def infer(
        prediction_h5ad_file_path: str,
        program_proportion_csv_file_path: str,
    ):
        infer_function = module.get_function("infer")

        smart_call(
            infer_function,
            default_values,
            {
                "prediction_directory_path": prediction_directory_path,
                "prediction_h5ad_file_path": prediction_h5ad_file_path,
                "program_proportion_csv_file_path": program_proportion_csv_file_path,
                "predict_perturbations": _load_predict_perturbations(data_directory_path, context.is_local),
                "genes_to_predict": _load_genes_to_predict(data_directory_path),
            }
        )

    return {
        "train": train,
        "infer": infer,
    }


def _validate_prediction_files(
    *,
    context: "RunnerContext",
    prediction_directory_path: str,
    location: Literal["before", "after"],
):
    is_specs_compliant = True

    current_files = set(os.listdir(prediction_directory_path))
    expected_files = {
        PREDICTION_FILE_NAME,
        PROGRAM_PROPORTION_FILE_NAME,
    }

    remaining_files = current_files - expected_files

    if location == "after":
        for file_name in expected_files:
            file_path = os.path.join(prediction_directory_path, file_name)

            if not os.path.isfile(file_path):
                context.log(
                    f"missing expected file in prediction directory: {file_name}",
                    error=True,
                )

                is_specs_compliant = False

    for file_name in remaining_files:
        context.log(
            f"unexpected missing file in prediction directory: {file_name}",
            error=True,
        )

        is_specs_compliant = False

    if location == "before":
        for file_name in current_files:
            file_path = os.path.join(prediction_directory_path, file_name)

            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.unlink(file_path)

    if not is_specs_compliant:
        context.log(
            "prediction files are not compliant with the specification",
            error=True,
        )


def _load_predict_perturbations(
    data_directory_path: str,
    is_local: bool,
) -> List[str]:
    if is_local:
        perturbations_to_score_file_path = os.path.join(data_directory_path, "program_proportion_local_gtruth.csv")
        return pandas.read_csv(perturbations_to_score_file_path, usecols=["gene"])["gene"].tolist()
    else:
        txt_file_path = os.path.join(data_directory_path, "predict_perturbations.txt")
        return pandas.read_csv(txt_file_path, header=None)[0].tolist()


def _load_genes_to_predict(
    data_directory_path: str,
) -> List[str]:
    txt_file_path = os.path.join(data_directory_path, "genes_to_predict.txt")
    return pandas.read_csv(txt_file_path, header=None)[0].tolist()
