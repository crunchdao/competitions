import contextlib
import datetime
import os
import typing
import zipfile

import crunch
import numpy
import pandas
import spatialdata

_LOG_DEPTH = 0


@contextlib.contextmanager
def log(action: str):
    global _LOG_DEPTH

    print(datetime.datetime.now(), "  " * _LOG_DEPTH, action)

    try:
        _LOG_DEPTH += 1

        yield True
    finally:
        _LOG_DEPTH -= 1


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


def check(
    prediction: pandas.DataFrame,
    data_directory_path: str,
    target_names: typing.List[str],
    phase_type: crunch.api.PhaseType
):
    with log("Check for required columns"):
        missing = set(prediction.columns) - {'cell_id', 'gene', 'prediction', 'sample'}

        if missing:
            raise ParticipantVisibleError(f"Missing required columns: {', '.join(missing)}")

    with log("Check for missing samples"):
        missing = set(prediction['sample'].unique()) - set(target_names)

        if missing:
            raise ParticipantVisibleError(f"Missing required samples: {', '.join(missing)}")

    with log("Filter predictions by samples once to avoid filtering in the loop"):
        group_by_sample = {
            sample: group
            for sample, group in prediction.groupby('sample')
        }

    for target in target_names:
        log(f"Loop through each target -> {target}")

        sdata = _read_zarr(data_directory_path, target)

        with log("Get predictions for the current sample"):
            prediction = group_by_sample.get(target)

            if prediction is None:
                raise ParticipantVisibleError(f"No predictions for gene {target.name}.")

        with log("Determine group type based on phase type"):
            group_type = 'test' if phase_type == crunch.api.PhaseType.SUBMISSION else 'validation'

            cell_ids = set(sdata['cell_id-group'].obs.query("group == @group_type")['cell_id'])
            gene_names = set(sdata['anucleus'].var.index)

        with log("Check for NaN values in predictions"):
            if prediction.isnull().values.any():
                raise ParticipantVisibleError("Predictions contain NaN values, which are not allowed.")

        with log("Check that all genes are present in predictions"):
            missing = set(prediction['gene']) - gene_names

            if missing:
                raise ParticipantVisibleError(f"The following genes are missing in predictions: {', '.join(missing)}.")

        with log("Check that all cell IDs are present in predictions"):
            missing = set(prediction['cell_id']) - cell_ids

            if missing:
                raise ParticipantVisibleError(f"The following cell IDs are missing in predictions: {', '.join(missing)}.")

        with log("Check data types in the 'prediction' column"):
            if not pandas.api.types.is_numeric_dtype(prediction['prediction']):
                raise ParticipantVisibleError("The 'prediction' column should only contain numeric values.")

        with log("Ensure all prediction values are positive"):
            if (prediction['prediction'] < 0).any():
                raise ParticipantVisibleError("Prediction values should be positive.")

        with log("Verify the size of predictions matches expectations"):
            expected = len(cell_ids) * len(gene_names)
            got = len(prediction)

            if expected != got:
                raise ParticipantVisibleError(f"Predictions should have {expected} rows but has {got}.")


def score(
    prediction: pandas.DataFrame,
    data_directory_path: str,
    phase_type: crunch.api.PhaseType,
    target_and_metrics: typing.List[typing.Tuple[crunch.api.Target, typing.List[crunch.api.Metric]]],
):
    with log("Filter predictions by samples once to avoid filtering in the loop"):
        group_by_sample = {
            str(sample): group
            for sample, group in prediction.groupby('sample')
        }

    scores = {}
    for target, metrics in target_and_metrics:
        log(f"Loop through each target -> {target.name}")

        sdata = _read_zarr(data_directory_path, target.name)

        with log("Get predictions for the current sample"):
            prediction = group_by_sample.get(target.name)

            if prediction is None:
                raise ParticipantVisibleError(f"No predictions for gene {target.name}.")

        with log("Determine group type based on phase type"):
            group_type = 'test' if phase_type == crunch.api.PhaseType.SUBMISSION else 'validation'

            cell_ids = sdata['cell_id-group'].obs.query("group == @group_type")['cell_id']
            gene_names = sdata['anucleus'].var.index

        with log("Filter data in 'anucleus' and construct DataFrame"):
            anucleus = sdata['anucleus']
            expected = pandas.DataFrame(
                anucleus.X[anucleus.obs['cell_id'].isin(cell_ids), :],
                columns=gene_names,
                index=cell_ids.values.flatten()
            )

        with log("Pivot prediction DataFrame"):
            prediction = prediction.pivot(index='cell_id', columns='gene', values='prediction')

        with log("Score prediction"):
            score = _mean_squared_error(prediction, expected)

        first_metric = metrics[0]
        scores[first_metric.id] = crunch.scoring.ScoredMetric(score)

    return scores


def _mean_squared_error(
    prediction: pandas.DataFrame,
    y_test: pandas.DataFrame
):
    with log("Ensure the same index and column order"):
        prediction = prediction.reindex(index=y_test.index, columns=y_test.columns)

    with log("Extract cell and gene counts directly"):
        cell_count = len(y_test.index)
        gene_count = len(y_test.columns)

    with log("Calculate weights for cells"):
        weight_on_cells = numpy.full(cell_count, 1 / cell_count)

    with log("Convert y_test and predictions to NumPy arrays"):
        A = y_test.to_numpy()
        B = prediction.to_numpy()

        assert A.shape == B.shape, "Prediction and Expected gene expression do not match"

    with log("Calculate mean squared error"):
        return numpy.sum(weight_on_cells * numpy.mean(numpy.square(A - B), axis=1))


def _read_zarr(
    data_directory_path: str,
    target_name: str
):
    zar_data = os.path.join(data_directory_path, "test", f"{target_name}.zarr")
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

    return sdata
