import os
import zipfile

import crunch
import numpy
import pandas
import spatialdata

import os
import zipfile
import pandas as pd


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


def check(
    df_predictions: pd.DataFrame,
    data_directory_path: str,
    target_names: list[str],
    phase_type: 'crunch.api.PhaseType'  # Assuming PhaseType is imported from crunch.api
):
    # Check for required columns
    required_columns = ['cell_id', 'gene', 'prediction', 'sample']
    missing_columns = [col for col in required_columns if col not in df_predictions.columns]
    if missing_columns:
        raise ParticipantVisibleError(f"Missing required columns: {', '.join(missing_columns)}")

    # Check for missing samples
    unique_samples = df_predictions['sample'].unique()
    missing_samples = [sample for sample in target_names if sample not in unique_samples]
    if missing_samples:
        raise ParticipantVisibleError(f"Missing required samples: {', '.join(missing_samples)}")

    # Filter predictions by samples once to avoid filtering in the loop
    predictions_by_sample = {
        sample: df_predictions[df_predictions['sample'] == sample]
        for sample in target_names
    }

    # Loop through each target
    for target in target_names:
        zar_data = os.path.join(data_directory_path, "test", f"{target}.zarr")
        zip_data = f"{zar_data}.zip"

        # Check if .zarr data exists; if not, ensure zip file exists and extract it
        if not os.path.exists(zar_data):
            if not os.path.exists(zip_data):
                raise FileNotFoundError(f"{zip_data} does not exist and is required for check.")
            with zipfile.ZipFile(zip_data, "r") as zip_ref:
                zip_ref.extractall(data_directory_path)

        # Read the Zarr data
        sdata = sd.read_zarr(zar_data)

        # Get predictions for the current sample
        predictions = predictions_by_sample[target]

        # Determine group type based on phase type
        group_type = 'test' if phase_type == crunch.api.PhaseType.SUBMISSION else 'validation'
        cell_ids = sdata['cell_id-group'].obs.query("group == @group_type")['cell_id']
        gene_name_list = sdata['anucleus'].var.index

        # Check for NaN values in predictions
        if predictions.isnull().values.any():
            raise ParticipantVisibleError("Predictions contain NaN values, which are not allowed.")

        # Check that all genes are present in predictions
        predictions_genes = predictions['gene'].values
        missing_genes = [gene for gene in gene_name_list if gene not in predictions_genes]
        if missing_genes:
            raise ParticipantVisibleError(f"The following genes are missing in predictions: {', '.join(missing_genes)}.")

        # Check that all cell IDs are present in predictions
        predictions_cell_ids = predictions['cell_id'].values
        missing_cell_ids = [cell_id for cell_id in cell_ids if cell_id not in predictions_cell_ids]
        if missing_cell_ids:
            raise ParticipantVisibleError(f"The following cell IDs are missing in predictions: {', '.join(map(str, missing_cell_ids[:100]))}.")

        # Check data types in the 'prediction' column
        if not pd.api.types.is_numeric_dtype(predictions['prediction']):
            raise ParticipantVisibleError("The 'prediction' column should only contain numeric values.")

        # Ensure all prediction values are positive
        if (predictions['prediction'] < 0).any():
            raise ParticipantVisibleError("Prediction values should be positive.")

        # Verify the size of predictions matches expectations
        expected_length = len(cell_ids) * len(gene_name_list)
        if len(predictions) != expected_length:
            raise ParticipantVisibleError(f"Predictions should have {expected_length} rows but has {len(predictions)}.")


def score(
    prediction: pandas.DataFrame,
    data_directory_path: str,
    target_names: list[str],
    phase_type: crunch.api.PhaseType,
):
    df_predictions = prediction

    scores = {}
    for target in target_names:
        zar_data = f"{data_directory_path}/test/{target}.zarr"
        zip_data = f"{zar_data}.zip"

        # Only extract if .zarr data does not exist
        if not os.path.exists(zar_data):
            if os.path.exists(zip_data):
                with zipfile.ZipFile(zip_data, "r") as zip_ref:
                    zip_ref.extractall(data_directory_path)
            else:
                raise FileNotFoundError(f"{zip_data} does not exist and is required for scoring.")

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

        predictions = df_predictions[df_predictions['sample'] == "UC1_NI"][['cell_id', 'gene', 'prediction']]
        predictions = predictions.pivot(index='cell_id', columns='gene', values='prediction')

        scores[target] = _mean_squared_error(predictions, expected)

    return scores


def _mean_squared_error(predictions: pandas.DataFrame, expectations: pandas.DataFrame):
    # Ensure the same index and column order
    predictions = predictions.reindex(index=expectations.index, columns=expectations.columns)

    # Extract cell and gene counts directly
    cell_count = len(expectations.index)
    gene_count = len(expectations.columns)

    # Calculate weights for cells
    weight_on_cells = numpy.full(cell_count, 1 / cell_count)

    # Convert expectations and predictions to NumPy arrays
    A = expectations.to_numpy()
    B = predictions.to_numpy()

    # Ensure shape alignment
    assert A.shape == B.shape, "Prediction and Expected gene expression do not match"

    # Calculate mean squared error
    return numpy.sum(weight_on_cells * numpy.mean(numpy.square(A - B), axis=1))
