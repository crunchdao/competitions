import contextlib
import datetime
import gc
import os
import typing

import crunch
import numpy
import pandas
import spatialdata

_LOG_DEPTH = 0


@contextlib.contextmanager
def log(action: str):
    global _LOG_DEPTH

    start = datetime.datetime.now()
    print(start, "  " * _LOG_DEPTH, action)

    try:
        _LOG_DEPTH += 1

        yield True
    finally:
        _LOG_DEPTH -= 1

        gc.collect()

        end = datetime.datetime.now()
        print(end, "  " * _LOG_DEPTH, action, "took", end - start)


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
        difference = set(prediction.columns) ^ {'cell_id', 'gene', 'prediction', 'sample'}

        if difference:
            raise ParticipantVisibleError(f"Missing or extra columns: {', '.join(difference)}")

    prediction.set_index("sample", drop=True, inplace=True)
    with log("Check for missing samples"):
        difference = set(prediction.index.unique()) ^ set(target_names)

        if difference:
            raise ParticipantVisibleError(f"Missing or extra samples: {', '.join(difference)}")

    for target_name in target_names:
        with log(f"Filter prediction at target -> {target_name}"):
            prediction_slice = prediction[prediction.index == target_name]

        sdata = _read_zarr(data_directory_path, target_name)

        with log("Extract unique cell IDs where the group is either 'test' or 'validation'"):
            cell_ids = set(sdata['cell_id-group'].obs.query("group == 'test' or group == 'validation'")['cell_id'])
            gene_names = set(sdata['anucleus'].var.index)

        with log("Check for NaN values in predictions"):
            if prediction_slice.isnull().values.any():
                raise ParticipantVisibleError("Predictions contain NaN values, which are not allowed.")

        with log("Check that all genes are present in predictions"):
            missing = set(prediction_slice['gene']) - gene_names

            if missing:
                raise ParticipantVisibleError(f"The following genes are missing in predictions: {', '.join(list(missing)[-10:])}.")

        with log("Check that all cell IDs are present in predictions"):
            missing = set(prediction_slice['cell_id']) - cell_ids

            if missing:
                raise ParticipantVisibleError(f"The following cell IDs are missing in predictions: {', '.join(list(map(str, missing))[-10:])}.")

        with log("Check data types in the 'prediction' column"):
            if not pandas.api.types.is_numeric_dtype(prediction_slice['prediction']):
                raise ParticipantVisibleError("The 'prediction' column should only contain numeric values.")

        with log("Ensure all prediction values are positive"):
            if (prediction_slice['prediction'] < 0).any():
                raise ParticipantVisibleError("Prediction values should be positive.")

        with log("Verify the size of predictions matches expectations"):
            expected = len(cell_ids) * len(gene_names)
            got = len(prediction_slice)

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
            group_type = 'validation' if phase_type == crunch.api.PhaseType.SUBMISSION else 'test'

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
    zar_data = os.path.join(data_directory_path, f"{target_name}.zarr")

    with log("Read the Zarr data"):
        sdata = spatialdata.read_zarr(zar_data, selection=("tables", ))

    return sdata
