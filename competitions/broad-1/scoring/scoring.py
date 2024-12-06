import contextlib
import datetime
import gc
import os
import typing

import crunch
import numpy
import numpy.typing
import pandas
from scipy import stats

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

    with log("Load region-cell mapping"):
        dataframe = pandas.read_csv(
            os.path.join(data_directory_path, "region-cell_id-map.csv"),
            index_col=0
        )

        dataframe["tissue_id"] = dataframe["tissue_id"].str.replace("Infl", "I").str.replace("Non", "N").astype("category")
        dataframe["region_id"] = dataframe["region_id"].astype("category")

        region_cell_mapping_by_sample = {
            str(sample): group.set_index("region_id", drop=True)["cell_id"]
            for sample, group in dataframe.groupby("tissue_id", observed=False)
        }

        dataframe = None

    with log("Filter predictions by samples once to avoid filtering in the loop"):
        group_by_sample = {
            str(sample): group
            for sample, group in prediction.groupby(prediction["sample"])
        }

    scores = {}
    virtual_mse_metric = None
    virtual_spearman_metric = None

    mse_metric_ids = []
    sperman_metric_ids = []

    for target, metrics in target_and_metrics:
        mse_metric = _find_metric_by_name(metrics, 'mse')
        spearman_metric = _find_metric_by_name(metrics, 'spearman')

        with log(f'Loop through each target -> {target.name}'):
            if target.name == 'ALL':
                virtual_mse_metric = mse_metric
                virtual_spearman_metric = spearman_metric
                continue

            if mse_metric:
                mse_metric_ids.append(mse_metric.id)
            if spearman_metric:
                sperman_metric_ids.append(spearman_metric.id)

            target_predictions = group_by_sample.pop(target.name)
            region_cell_mapping = region_cell_mapping_by_sample.pop(target.name)

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
                mse_score = crunch.scoring.ScoredMetric(None, [])
                spearman_score = crunch.scoring.ScoredMetric(None, [])

                for region_id, cell_ids in region_cell_mapping.groupby("region_id", observed=True):
                    cell_ids = set(cell_ids)

                    with log(f"Score region -> {region_id}"):
                        with log(f"Filter y_test"):
                            region_prediction = filtered_predictions[filtered_predictions.index.isin(cell_ids)]

                        with log(f"Filter y_test"):
                            region_y_test = y_test[y_test.index.isin(cell_ids)]

                        with log("Ensure the same index and column order"):
                            region_prediction = region_prediction.reindex(index=region_y_test.index, columns=region_y_test.columns)

                        with log(f"Calling _mean_squared_error"):
                            region_mse = _mean_squared_error(region_prediction, region_y_test)

                        with log(f"Calling _spearman"):
                            region_spearman = _spearman(region_prediction, region_y_test)

                        mse_score.details.append(crunch.scoring.ScoredMetricDetail(region_id, region_mse, False))
                        spearman_score.details.append(crunch.scoring.ScoredMetricDetail(region_id, region_spearman, False))

                _average_details(mse_metric, mse_score, scores)
                _average_details(spearman_metric, spearman_score, scores)

    _compute_virtual(virtual_mse_metric, mse_metric_ids, scores)
    _compute_virtual(virtual_spearman_metric, sperman_metric_ids, scores)

    return scores


def _average_details(
    metric: typing.Optional[crunch.api.Metric],
    scored_metric: crunch.scoring.ScoredMetric,
    scores: typing.Dict[int, crunch.scoring.ScoredMetric],
):
    if not metric:
        return

    scored_metric.value = numpy.mean([
        detail.value
        for detail in scored_metric.details
    ])

    scores[metric.id] = scored_metric


def _compute_virtual(
    metric: typing.Optional[crunch.api.Metric],
    metric_ids: typing.List[int],
    scores: typing.Dict[int, crunch.scoring.ScoredMetric]
):
    if not metric:
        return

    metric_ids = set(metric_ids)

    mean = numpy.mean([
        score.value
        for metric_id, score in scores.items()
        if metric_id in metric_ids
    ])

    scores[metric.id] = crunch.scoring.ScoredMetric(mean)


def _find_metric_by_name(
    metrics: typing.List[crunch.api.Metric],
    name: str
) -> crunch.api.Metric:
    return next((
        metric
        for metric in metrics
        if metric.name == name
    ), None)


def _mean_squared_error(
    prediction: pandas.DataFrame,
    y_test: pandas.DataFrame,
):
    cell_count = len(y_test.index)
    weight_on_cells = numpy.ones(cell_count) / cell_count

    A = y_test.to_numpy()
    B = prediction.to_numpy()

    return numpy.sum(weight_on_cells * (numpy.square(A - B)).mean(axis=1))


def _spearman(
    prediction: pandas.DataFrame,
    y_test: pandas.DataFrame,
):
    cell_count = len(y_test.index)
    weight_on_cells = numpy.ones(cell_count) / cell_count

    A = y_test.to_numpy()
    B = prediction.to_numpy()

    rank_A = stats.rankdata(A, axis=1)
    rank_B = stats.rankdata(B, axis=1)


    corrs_cell = (
        numpy.multiply(rank_A - numpy.mean(rank_A), rank_B - numpy.mean(rank_B)).mean(axis=1)
        / (numpy.std(rank_A, axis=1) * numpy.std(rank_B, axis=1))
    )

    corrs_cell[numpy.isnan(corrs_cell)] = 0

    return numpy.sum(weight_on_cells * corrs_cell)


def _read_zarr(
    data_directory_path: str,
    target_name: str
):
    zar_data = os.path.join(data_directory_path, f"{target_name}.zarr")

    with log("Importing spatialdata"):
        import spatialdata

    with log("Read the Zarr data"):
        sdata = spatialdata.read_zarr(zar_data, selection=("tables",))

    return sdata
