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
    group_type = 'validation' if phase_type == crunch.api.PhaseType.SUBMISSION else 'test'

    with log("Filter predictions by samples once to avoid filtering in the loop"):
        group_by_sample = {
            str(sample): group
            for sample, group in prediction.groupby(prediction["sample"])
        }

    scores = {}
    total_cells, mse_weighted_sum = 0, 0
    virtual_mse_metric = None

    for target, metrics in target_and_metrics:
        mse_metric = _find_metric_by_name(metrics, 'mse')

        with log(f'Loop through each target -> {target.name}'):
            if target.name == 'ALL':
                virtual_mse_metric = mse_metric
                continue

            with log(f'Get predictions for the current sample {target.name}'):
                target_predictions = group_by_sample.pop(target.name)

            y_test_file_name = f'{group_type}-{target.name}.csv'
            with log(f"Load y_test from file {y_test_file_name}"):
                y_test_file_path = os.path.join(data_directory_path, y_test_file_name)
                y_test = pandas.read_csv(y_test_file_path, index_col=0)

            with log("Filter prediction who need to be scored"):
                filtered_predictions = target_predictions[target_predictions["cell_id"].isin(y_test.index)]
                target_predictions = None

            with log("Pivoting the dataframe"):
                filtered_predictions = filtered_predictions.pivot(index='cell_id', columns='gene', values='prediction')

            with log("Score prediction"):
                if mse_metric:
                    if not _is_log1p_normalization(filtered_predictions):
                        filtered_predictions = _log1p_normalization(filtered_predictions)

                    mse = _mean_squared_error(filtered_predictions, y_test)

                    scores[mse_metric.id] = crunch.scoring.ScoredMetric(mse)
                    mse_weighted_sum += mse * len(filtered_predictions)

            total_cells += len(filtered_predictions)

    if virtual_mse_metric:
        combine_mse = mse_weighted_sum / total_cells
        scores[virtual_mse_metric.id] = crunch.scoring.ScoredMetric(combine_mse)

    return scores


def _find_metric_by_name(
    metrics: typing.List[crunch.api.Metric],
    name: str
) -> crunch.api.Metric:
    return next((
        metric
        for metric in metrics
        if metric.name == name
    ), None)


def _is_log1p_normalization(arr: pandas.DataFrame):
    ones = (numpy.expm1(arr) / 100).sum()

    return ((ones > 0.999) & (ones < 1.001)).all()


def _log1p_normalization(arr: pandas.DataFrame):
    for column in arr.columns:
        arr[column] /= numpy.sum(arr[column])

    return numpy.log1p(arr * 100)


def _mean_squared_error(
    prediction: pandas.DataFrame,
    y_test: pandas.DataFrame
):
    with log("Ensure the same index and column order"):
        prediction = prediction.reindex(index=y_test.index, columns=y_test.columns)

    cell_count = len(y_test.index)
    log(f"Cell counts is {cell_count}")

    with log("Calculate weights for cells"):
        weight_on_cells = numpy.ones(cell_count) / cell_count

    with log("Convert y_test and predictions to NumPy arrays"):
        A = y_test.to_numpy()
        B = prediction.to_numpy()

        assert A.shape == B.shape, "Prediction and Expected gene expression do not match"

    with log("Calculate mean squared error"):
        return numpy.sum(weight_on_cells * (numpy.square(A - B)).mean(axis=1))


def _read_zarr(
    data_directory_path: str,
    target_name: str
):
    zar_data = os.path.join(data_directory_path, f"{target_name}.zarr")

    with log("Read the Zarr data"):
        sdata = spatialdata.read_zarr(zar_data, selection=("tables",))

    return sdata
