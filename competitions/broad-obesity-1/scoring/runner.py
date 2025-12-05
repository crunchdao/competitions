import os
import shutil
from typing import TYPE_CHECKING, List, Literal, Optional

import anndata
import pandas
import scanpy
from crunch.utils import smart_call

if TYPE_CHECKING:
    from crunch.runner.unstructured import RunnerContext, RunnerExecutorContext, UserModule


PREDICTION_H5AD_FILE_NAME = "prediction.h5ad"
PREDICTION_PARQUET_FILE_NAME = PREDICTION_H5AD_FILE_NAME + ".parquet"
PROGRAM_PROPORTION_FILE_NAME = "predict_program_proportion.csv"

# Optimization for local testing
shared_local_prediction_instance: Optional[anndata.AnnData] = None


def run(
    context: "RunnerContext",
    data_directory_path: str,
    prediction_directory_path: str,
):
    global shared_local_prediction_instance

    if context.is_local:
        genes_to_score_file_path = os.path.join(data_directory_path, "program_proportion_local_gtruth.csv")
        genes_to_score = pandas.read_csv(genes_to_score_file_path, usecols=["gene"])["gene"].tolist()
    else:
        genes_to_score_file_path = os.path.join(data_directory_path, "genes_to_score.txt")
        genes_to_score = pandas.read_csv(genes_to_score_file_path, header=None)[0].tolist()
        os.unlink(genes_to_score_file_path)

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
        PREDICTION_H5AD_FILE_NAME,
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

    try:
        if context.is_local:
            prediction = shared_local_prediction_instance
            assert prediction
        else:
            prediction = scanpy.read_h5ad(prediction_h5ad_file_path)

        prediction = prediction[prediction.obs["gene"].isin(genes_to_score)]
        # prediction.write(prediction_h5ad_file_path)

        prediction_dataframe = pandas.DataFrame(
            prediction.X,
            columns=prediction.var_names.values,
            index=prediction.obs.index,
        )
        prediction_dataframe.insert(0, "gene", prediction.obs["gene"].values)
        prediction_dataframe.to_parquet(prediction_h5ad_file_path + ".parquet")
    finally:
        if context.is_local:
            shared_local_prediction_instance = None
        elif os.path.exists(prediction_h5ad_file_path):
            os.unlink(prediction_h5ad_file_path)

    # Code to convert back to an AnnData object:
    # genes = prediction_dataframe.columns[1:]
    # restored = anndata.AnnData(
    #     X=prediction_dataframe[genes].values,
    #     obs=prediction_dataframe[["gene"]],
    #     var=pandas.DataFrame(index=genes),
    # )

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
        "predict_perturbations": _load_predict_perturbations(data_directory_path),
        "genes_to_predict": _load_genes_to_predict(data_directory_path),
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

        result = smart_call(
            infer_function,
            default_values,
            {
                "prediction_directory_path": prediction_directory_path,
                # hvg_gene.csv
                # read genes_to_predict.csv ->
                #   generated from taking 600 random genes which are
                #       not in hvg_gene.csv
            }
        )

        assert isinstance(result, tuple), f"infer.result: expected tuple, got {result.__class__.__name__}"
        assert len(result) == 2, f"infer.result: expected tuple of length 2, got {len(result)}"
        assert isinstance(result[0], anndata.AnnData), f"infer.result[0]: expected anndata.AnnData, got {result[0].__class__.__name__}"
        assert isinstance(result[1], pandas.DataFrame), f"infer.result[1]: expected anndata.DataFrame, got {result[1].__class__.__name__}"

        prediction, program_proportion = result

        if context.is_local:
            global shared_local_prediction_instance
            shared_local_prediction_instance = prediction
        else:
            prediction.write(prediction_h5ad_file_path)

        program_proportion.to_csv(program_proportion_csv_file_path, index=False)

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
        PREDICTION_PARQUET_FILE_NAME,
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
) -> List[str]:
    txt_file_path = os.path.join(data_directory_path, "predict_perturbations.txt")
    return pandas.read_csv(txt_file_path, header=None)[0].tolist()


def _load_genes_to_predict(
    data_directory_path: str,
) -> List[str]:
    hvg_genes = pandas.read_csv(os.path.join(data_directory_path, "hvg_gene.csv"))["hvg_gene"].tolist()

    return list({
        *hvg_genes
    })
