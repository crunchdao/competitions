import os

import numpy
import pandas
from crunch.unstructured.utils import delta_message
from crunch.utils import Tracer

PREDICTION_FILE_NAME = "prediction.parquet"


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


tracer = Tracer()


def check(
    prediction_directory_path: str,
    data_directory_path: str,
):
    with tracer.log("Ensure files"):
        difference = delta_message(
            {
                PREDICTION_FILE_NAME,
            },
            set(os.listdir(prediction_directory_path)),
        )

        if difference:
            raise ParticipantVisibleError(f"Prediction files are not valid: {difference}")

    with tracer.log("Load ground truth: thermogenic_signatures"):
        thermogenic_signatures = pandas.read_csv(os.path.join(data_directory_path, "thermogenic_signatures.csv"))
        thermogenic_signatures_names = set(thermogenic_signatures["name"].unique())

    with tracer.log("Load ground truth: predict_perturbations"):
        predict_perturbations = pandas.read_parquet(os.path.join(data_directory_path, "predict_perturbations_3.parquet"))

    with tracer.log("Validating prediction"):
        with tracer.log("Load prediction"):
            prediction = pandas.read_parquet(os.path.join(prediction_directory_path, PREDICTION_FILE_NAME))

        with tracer.log("Check for required columns"):
            difference = delta_message(
                {"GenePairID", "FinalAggScore", "Rank"} | thermogenic_signatures_names,
                set(prediction.columns),
            )

            if difference:
                raise ParticipantVisibleError(f"Columns do not match: {difference}")

        with tracer.log("Check for rows"):
            expected_genes_count = len(predict_perturbations)
            got_rows = len(prediction)

            if expected_genes_count != got_rows:
                raise ParticipantVisibleError(f"Row count is incorrect, expected {expected_genes_count} but got {got_rows}")

        with tracer.log("Check for NaN values"):
            if prediction.isnull().values.any():
                raise ParticipantVisibleError(f"Found NaN value(s)")

        with tracer.log("Check for infinity values"):
            prediction = prediction.replace([-numpy.inf, numpy.inf], numpy.nan)

            if prediction.isnull().values.any():
                raise ParticipantVisibleError(f"Found Inf value(s)")

        with tracer.log(f"Check data types"):
            column_name = "GenePairID"
            with tracer.log(f"...in the '{column_name}' column"):
                expected_values = set(predict_perturbations["gene_1"].astype(str) + "+" + predict_perturbations["gene_2"].astype(str))
                got_values = set(prediction[column_name].values)

                difference = delta_message(
                    expected_values,
                    got_values,
                )

                if difference:
                    raise ParticipantVisibleError(f"Values in column `{column_name}` do not match expected gene pairs: {difference}")

            for column_name in tracer.loop(thermogenic_signatures_names, lambda x: f"...in the '{x}' column"):
                if not pandas.api.types.is_numeric_dtype(prediction[column_name].values):
                    raise ParticipantVisibleError(f"Found non-numeric values for column `{column_name}`")

            column_name = "FinalAggScore"
            with tracer.log(f"...in the '{column_name}' column"):
                if not pandas.api.types.is_float_dtype(prediction[column_name].values):
                    raise ParticipantVisibleError(f"Found non-float values for column `{column_name}`")

            column_name = "Rank"
            with tracer.log(f"...in the '{column_name}' column"):
                if not pandas.api.types.is_integer_dtype(prediction[column_name].values):
                    raise ParticipantVisibleError(f"Found non-integer values for column `{column_name}`")


def score():
    raise NotImplementedError()
