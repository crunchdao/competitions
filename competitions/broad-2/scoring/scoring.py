import os
import typing

import crunch
import crunch.custom
import crunch.utils
import numpy
import pandas
import scipy.stats


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


tracer = crunch.utils.Tracer()


def check(
    prediction: pandas.DataFrame,
    data_directory_path: str,
    target_names: typing.List[str],
    phase_type: crunch.api.PhaseType
):
    if "ALL" in target_names:
        target_names.remove("ALL")

    with tracer.log("Loading gene list"):
        gene_csv = pandas.read_csv(f'{data_directory_path}/Crunch2_gene_list.csv')
        gene_names = set(gene_csv['gene_symbols'])

    with tracer.log("Check for required columns"):
        difference = crunch.custom.utils.delta_message(
            {'sample'} | gene_names,
            set(prediction.columns),
        )

        if difference:
            raise ParticipantVisibleError(f"Columns do not match: {difference}")

    with tracer.log("Check for missing samples"):
        difference = crunch.custom.utils.delta_message(
            set(target_names),
            set(prediction['sample'].unique()),
        )

        if difference:
            raise ParticipantVisibleError(f"Samples do not match: {difference}")

    for target_name in tracer.loop(target_names, "Checking target -> {value}"):
        with tracer.log(f"Filter prediction at target"):
            prediction_slice = prediction[prediction['sample'] == target_name]
            prediction_slice = prediction_slice.drop(columns=['sample'])

        sdata = _read_zarr(data_directory_path, target_name)

        with tracer.log("Extract unique cell IDs where the group is either 'test' or 'validation'"):
            cell_ids = set(sdata['cell_id-group'].obs.query("group == 'test' or group == 'validation'")['cell_id'])

        with tracer.log("Check for NaN values in predictions"):
            if prediction_slice.isnull().values.any():
                raise ParticipantVisibleError(f"Found NaN values for target `{target_name}`")

        with tracer.log("Check for infinity values in predictions"):
            prediction_slice = prediction_slice.replace([-numpy.inf, numpy.inf], numpy.nan)

        with tracer.log("Check that all cell IDs are present in predictions"):
            difference = crunch.custom.utils.delta_message(
                cell_ids,
                set(prediction_slice.index),
            )

            if difference:
                raise ParticipantVisibleError(f"Cell IDs do not match for target `{target_name}`: {difference}")

        with tracer.log("Check data types in the 'prediction' values"):
            if not pandas.api.types.is_numeric_dtype(prediction_slice.values):
                raise ParticipantVisibleError(f"Found non-numeric values for target `{target_name}`")

        with tracer.log("Ensure all prediction values are positive"):
            if (prediction_slice.values < 0).any():
                raise ParticipantVisibleError(f"Found negative values for target `{target_name}`")

        with tracer.log("Verify the size of predictions matches expectations"):
            expected = len(cell_ids) * len(gene_names)
            got = prediction_slice.size

            if expected != got:
                raise ParticipantVisibleError(f"Row count for target `{target_name}` should be {expected} ({len(cell_ids)} cell ids * {len(gene_names)} gene names), but got {got}")


def score(
    prediction: pandas.DataFrame,
    data_directory_path: str,
    phase_type: crunch.api.PhaseType,
    target_and_metrics: typing.List[typing.Tuple[crunch.api.Target, typing.List[crunch.api.Metric]]],
):
    group_type = 'validation' if phase_type == crunch.api.PhaseType.SUBMISSION else 'test'

    with tracer.log("Load region-cell mapping"):
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

    with tracer.log("Filter predictions by samples once to avoid filtering in the loop"):
        group_by_sample = {
            str(sample): group
            for sample, group in prediction.groupby(prediction["sample"], group_keys=False)
        }

    scores = {}
    virtual_cell_spearman_metric = None
    virtual_gene_spearman_metric = None

    cell_spearman_metric_ids = []
    gene_spearman_metric_ids = []

    for target, metrics in tracer.loop(target_and_metrics, lambda x: f'Loop through each target -> {x[0].name}'):
        cell_spearman_metric = _find_metric_by_name(metrics, 'cell-spearman')
        gene_spearman_metric = _find_metric_by_name(metrics, 'gene-spearman')

        if target.name == 'ALL':
            virtual_cell_spearman_metric = cell_spearman_metric
            virtual_gene_spearman_metric = gene_spearman_metric
            continue

        if cell_spearman_metric:
            cell_spearman_metric_ids.append(cell_spearman_metric.id)
        if gene_spearman_metric:
            gene_spearman_metric_ids.append(gene_spearman_metric.id)

        target_predictions = group_by_sample.pop(target.name)
        region_cell_mapping = region_cell_mapping_by_sample.pop(target.name)

        y_test_file_name = f'{group_type}-{target.name}.csv'
        with tracer.log(f"Load y_test from file {y_test_file_name}"):
            y_test_file_path = os.path.join(data_directory_path, y_test_file_name)
            y_test = pandas.read_csv(y_test_file_path, index_col=0)

        with tracer.log("Filter prediction who need to be scored"):
            filtered_predictions = target_predictions[target_predictions.index.isin(y_test.index)]
            target_predictions = None

        with tracer.log("Score prediction"):
            cell_spearman_score = crunch.scoring.ScoredMetric(None, [])
            gene_spearman_score = crunch.scoring.ScoredMetric(None, [])

            for region_id, cell_ids in tracer.loop(region_cell_mapping.groupby("region_id", observed=True), lambda x: f"Score region -> {x[0]}"):
                cell_ids = set(cell_ids)

                with tracer.log(f"Filter y_test"):
                    region_prediction = filtered_predictions[filtered_predictions.index.isin(cell_ids)]

                with tracer.log(f"Filter y_test"):
                    region_y_test = y_test[y_test.index.isin(cell_ids)]

                with tracer.log(f"Calling _spearman_cell_wise"):
                    region_spearman_cell = _spearman_cell_wise(region_prediction, region_y_test)

                with tracer.log(f"Calling _spearman_gene_wise"):
                    region_spearman_gene = _spearman_gene_wise(region_prediction, region_y_test)

                cell_spearman_score.details.append(crunch.scoring.ScoredMetricDetail(region_id, region_spearman_cell, False))
                gene_spearman_score.details.append(crunch.scoring.ScoredMetricDetail(region_id, region_spearman_gene, False))

            _average_details(cell_spearman_metric, cell_spearman_score, scores)
            _average_details(gene_spearman_metric, gene_spearman_score, scores)

    _compute_virtual(virtual_cell_spearman_metric, cell_spearman_metric_ids, scores)
    _compute_virtual(virtual_gene_spearman_metric, gene_spearman_metric_ids, scores)

    return scores


def _read_zarr(
    data_directory_path: str,
    target_name: str
):
    zar_data = os.path.join(data_directory_path, f"{target_name}.zarr")

    with tracer.log("Importing spatialdata"):
        import spatialdata

    with tracer.log("Read the Zarr data"):
        sdata = spatialdata.read_zarr(zar_data, selection=("tables",))

    return sdata


def _find_metric_by_name(
    metrics: typing.List[crunch.api.Metric],
    name: str
) -> crunch.api.Metric:
    return next((
        metric
        for metric in metrics
        if metric.name == name
    ), None)


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


def _spearman_cell_wise(
    prediction: pandas.DataFrame,
    y_test: pandas.DataFrame,
):
    y_test = y_test[y_test.sum(axis=1) > 0]
    prediction = prediction.reindex(index=y_test.index, columns=y_test.columns)

    cell_count = len(y_test.index)
    weight_on_cells = numpy.ones(cell_count) / cell_count

    A = y_test.to_numpy()
    B = prediction.to_numpy()

    rank_A = scipy.stats.rankdata(A, axis=1)
    rank_B = scipy.stats.rankdata(B, axis=1)

    corrs_cell = (
        numpy.multiply(rank_A - numpy.mean(rank_A), rank_B - numpy.mean(rank_B)).mean(axis=1)
        / (numpy.std(rank_A, axis=1) * numpy.std(rank_B, axis=1))
    )

    corrs_cell[numpy.isnan(corrs_cell)] = 0

    return numpy.sum(weight_on_cells * corrs_cell)


def _spearman_gene_wise(
    prediction: pandas.DataFrame,
    y_test: pandas.DataFrame,
):
    y_test = y_test.loc[:, y_test.sum(axis=0) > 0]
    prediction = prediction.reindex(index=y_test.index, columns=y_test.columns)

    gene_count = len(y_test.columns)
    weight_on_genes = numpy.ones(gene_count) / gene_count

    A = y_test.to_numpy()
    B = prediction.to_numpy()

    rank_A = scipy.stats.rankdata(A, axis=0)
    rank_B = scipy.stats.rankdata(B, axis=0)

    corrs_gene = (
        numpy.multiply(rank_A - numpy.mean(rank_A), rank_B - numpy.mean(rank_B)).mean(axis=0)
        / (numpy.std(rank_A, axis=0) * numpy.std(rank_B, axis=0))
    )

    corrs_gene[numpy.isnan(corrs_gene)] = 0

    return numpy.sum(weight_on_genes * corrs_gene)
