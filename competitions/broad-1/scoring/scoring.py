import contextlib
import datetime
import os
import typing
import zipfile

import crunch
import numpy
import pandas
import pandas as pd
import spatialdata


@contextlib.contextmanager
def log(action: str):
    print(datetime.datetime.now(), action)
    yield True


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


def check(
    prediction: pd.DataFrame,
    data_directory_path: str,
    target_names: typing.List[str],
    phase_type: crunch.api.PhaseType
):
    with log("Check for required columns"):
        required_columns = ['cell_id', 'gene', 'prediction', 'sample']
        missing_columns = [col for col in required_columns if col not in prediction.columns]
        if missing_columns:
            raise ParticipantVisibleError(f"Missing required columns: {', '.join(missing_columns)}")

    with log("Check for missing samples"):
        unique_samples = prediction['sample'].unique()
        missing_samples = [sample for sample in target_names if sample not in unique_samples]
        if missing_samples:
            raise ParticipantVisibleError(f"Missing required samples: {', '.join(missing_samples)}")

    with log("Filter predictions by samples once to avoid filtering in the loop"):
        predictions_by_sample = {
            sample: prediction[prediction['sample'] == sample]
            for sample in target_names
        }

    for target in target_names:
        log(f"Loop through each target -> {target}")

        zar_data = os.path.join(data_directory_path, "test", f"{target}.zarr")
        zip_data = f"{zar_data}.zip"

        with log(f"Only extract if .zarr data does not exist -> {os.path.exists(zar_data)}"):
            if not os.path.exists(zar_data):
                print(zip_data, "exists?", os.path.exists(zip_data))
                if os.path.exists(zip_data):
                    with zipfile.ZipFile(zip_data, "r") as zip_ref:
                        zip_ref.extractall(os.path.dirname(zip_data))
                else:
                    raise FileNotFoundError(f"{zip_data} does not exist and is required for scoring.")

        with log("Read the Zarr data"):
            sdata = spatialdata.read_zarr(zar_data)

        with log("Get predictions for the current sample"):
            predictions = predictions_by_sample[target]

        with log("Determine group type based on phase type"):
            group_type = 'test' if phase_type == crunch.api.PhaseType.SUBMISSION else 'validation'
            cell_ids = sdata['cell_id-group'].obs.query("group == @group_type")['cell_id']
            gene_name_list = sdata['anucleus'].var.index

        with log("Check for NaN values in predictions"):
            if predictions.isnull().values.any():
                raise ParticipantVisibleError("Predictions contain NaN values, which are not allowed.")

        with log("Check that all genes are present in predictions"):
            predictions_genes = set(predictions['gene'].values)
            missing_genes = [gene for gene in gene_name_list if gene not in predictions_genes]
            if missing_genes:
                raise ParticipantVisibleError(f"The following genes are missing in predictions: {', '.join(missing_genes)}.")

        with log("Check that all cell IDs are present in predictions"):
            predictions_cell_ids = set(predictions['cell_id'].values)
            missing_cell_ids = [cell_id for cell_id in cell_ids if cell_id not in predictions_cell_ids]
            if missing_cell_ids:
                raise ParticipantVisibleError(f"The following cell IDs are missing in predictions: {', '.join(map(str, missing_cell_ids[:100]))}.")

        with log("Check data types in the 'prediction' column"):
            if not pd.api.types.is_numeric_dtype(predictions['prediction']):
                raise ParticipantVisibleError("The 'prediction' column should only contain numeric values.")

        with log("Ensure all prediction values are positive"):
            if (predictions['prediction'] < 0).any():
                raise ParticipantVisibleError("Prediction values should be positive.")

        with log("Verify the size of predictions matches expectations"):
            expected_length = len(cell_ids) * len(gene_name_list)
            if len(predictions) != expected_length:
                raise ParticipantVisibleError(f"Predictions should have {expected_length} rows but has {len(predictions)}.")


def score(
    prediction: pandas.DataFrame,
    data_directory_path: str,
    phase_type: crunch.api.PhaseType,
    target_and_metrics: typing.List[typing.Tuple[crunch.api.Target, crunch.api.Metric]],
):
    prediction = prediction

    scores = {}
    for target, metrics in target_and_metrics:
        zar_data = f"{data_directory_path}/test/{target.name}.zarr"
        zip_data = f"{zar_data}.zip"

        # Only extract if .zarr data does not exist
        print(zar_data, "exists?", os.path.exists(zar_data))
        if not os.path.exists(zar_data):
            print(zip_data, "exists?", os.path.exists(zip_data))
            if os.path.exists(zip_data):
                with zipfile.ZipFile(zip_data, "r") as zip_ref:
                    zip_ref.extractall(os.path.dirname(zip_data))
            else:
                raise FileNotFoundError(f"{zip_data} does not exist and is required for scoring.")

        print(zar_data)
        sdata = spatialdata.read_zarr(zar_data)

        group_type = 'test' if phase_type == crunch.api.PhaseType.SUBMISSION else 'validation'

        # Filter cell IDs based on the group type directly
        cell_ids = sdata['cell_id-group'].obs.query("group == @group_type")['cell_id']

        anucleus = sdata['anucleus']

        # TODO: Adjust if you need 'gene_symbols'
        gene_names = anucleus.var.index

        # Filter data in 'anucleus' and construct DataFrame
        expected = pandas.DataFrame(
            anucleus.X[anucleus.obs['cell_id'].isin(cell_ids), :],
            columns=gene_names,
            index=cell_ids.values.flatten()
        )

        group = prediction[prediction['sample'] == "UC1_NI"][['cell_id', 'gene', 'prediction']]
        group = group.pivot(index='cell_id', columns='gene', values='prediction')

        score = _mean_squared_error(group, expected)

        first_metric = metrics[0]
        scores[first_metric.id] = crunch.scoring.ScoredMetric(score)

    return scores


def _mean_squared_error(
    prediction: pandas.DataFrame,
    y_test: pandas.DataFrame
):
    # Ensure the same index and column order
    prediction = prediction.reindex(index=y_test.index, columns=y_test.columns)

    # Extract cell and gene counts directly
    cell_count = len(y_test.index)
    gene_count = len(y_test.columns)

    # Calculate weights for cells
    weight_on_cells = numpy.full(cell_count, 1 / cell_count)

    # Convert y_test and predictions to NumPy arrays
    A = y_test.to_numpy()
    B = prediction.to_numpy()

    # Ensure shape alignment
    assert A.shape == B.shape, "Prediction and Expected gene expression do not match"

    # Calculate mean squared error
    return numpy.sum(weight_on_cells * numpy.mean(numpy.square(A - B), axis=1))
