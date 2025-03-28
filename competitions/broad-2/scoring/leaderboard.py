import typing
import os

import crunch
import crunch.custom
import numpy
import pandas
import scipy.stats

tracer = crunch.utils.Tracer()


def compare(
    targets: typing.List[crunch.api.Target],
    predictions: typing.Dict[int, pandas.DataFrame],
    combinations: typing.List[typing.Tuple[int, int]],
    data_directory_path: str,
):
    all_target = _find_target_by_name(targets, "ALL")
    targets = [target for target in targets if target is not all_target]

    # gene_names_per_sample = {
    #     target.name: set(
    #         pandas.read_csv(
    #             os.path.join(data_directory_path, f"validation-{target.name}.csv"),
    #             index_col="cell_id"
    #         ).columns
    #     )
    #     for target in targets
    # }

    predictions_by_sample: typing.Dict[int, typing.Dict[str, pandas.DataFrame]] = {}

    for prediction_id, dataframe in tracer.loop(list(predictions.items()), lambda entry: f"Preparing Prediction #{entry[0]}"):
        prediction_by_sample = {}

        for sample_name, group in tracer.loop(dataframe.groupby("sample"), lambda entry: f"Grouped by {entry[0]}"):
            with tracer.log("Dropping unused gene columns"):
                # useless_columns = set(group.columns) - gene_names_per_sample[sample_name]
                useless_columns = {"sample"}

                group.drop(columns=useless_columns, inplace=True)

            with tracer.log("Sorting the index"):
                group.sort_index(inplace=True)

            prediction_by_sample[sample_name] = group

        predictions_by_sample[prediction_id] = prediction_by_sample
        del predictions[prediction_id]

    similarities: typing.List[crunch.custom.ComparedSimilarity] = []
    for left_id, right_id in tracer.loop(combinations, lambda combination: f"Using Combination {combination[0]} <-> {combination[1]}"):
        left_prediction_by_sample = predictions_by_sample[left_id]
        right_prediction_by_sample = predictions_by_sample[right_id]

        values = []
        for target in tracer.loop(targets, lambda target: f"Correlating {target.name}"):
            if target is all_target:
                continue

            values2 = []
            # for gene_name in tracer.loop(gene_names_per_sample[target.name], lambda gene_name: f"Correlating gene {gene_name}"):
            for gene_name in tracer.loop(left_prediction_by_sample[target.name].columns, lambda gene_name: f"Correlating gene {gene_name}"):
                left = left_prediction_by_sample[target.name][gene_name]
                right = right_prediction_by_sample[target.name][gene_name]

                value2 = left.corr(right, method="spearman")
                values2.append(value2)

            value = numpy.nanmean(values2)
            values.append(value)

        value = numpy.nanmean(values)

        print(f"similarity - left_id={left_id} right_id={right_id} value={value}")
        similarities.append(crunch.custom.ComparedSimilarity(
            left_id,
            right_id,
            all_target.id,
            value,
        ))

    return similarities


def _find_target_by_name(
    targets: typing.List[crunch.api.Target],
    target_name: str,
) -> crunch.api.Metric:
    for target in targets:
        if target.name == target_name:
            return target

    raise ValueError(f"no target found with name=`{target_name}`")


def rank(
    target_and_metrics: typing.List[typing.Tuple[crunch.api.Target, typing.List[crunch.api.Metric]]],
    projects: typing.List[crunch.custom.RankableProject],
):
    cell_spearman_metric = _find_metric_by_name(target_and_metrics, "ALL", "cell-spearman")
    gene_spearman_metric = _find_metric_by_name(target_and_metrics, "ALL", "gene-spearman")

    cell_spearman_column_name = f"metric:{cell_spearman_metric.name}"
    gene_spearman_column_name = f"metric:{gene_spearman_metric.name}"

    dataframe = pandas.DataFrame((
        {
            "project_id": project.id,
            "rewardable": project.rewardable,
            cell_spearman_column_name: project.get_metric(cell_spearman_metric.id).score,
            gene_spearman_column_name: project.get_metric(gene_spearman_metric.id).score,
        }
        for project in projects
    ))

    dataframe["rank_all"] = _rankdata(
        _rankdata(-dataframe[cell_spearman_column_name]) +
        _rankdata(-dataframe[gene_spearman_column_name])
    )

    dataframe.sort_values(
        by=[
            'rank_all',
            'project_id',  # fallback if same `final_score`
        ],
        inplace=True,
    )

    mask = dataframe["rewardable"]
    dataframe.loc[mask, "rank_final"] = _rankdata(dataframe.loc[mask, "rank_all"])

    dataframe.index = range(1, len(dataframe.index) + 1)

    return [
        crunch.custom.RankedProject(
            id=int(row["project_id"]),
            rank=index,
            reward_rank=None if numpy.isnan(row["rank_final"]) else row["rank_final"],
        )
        for index, row in dataframe.iterrows()
    ]


def _rankdata(array: typing.List[typing.Tuple[int, float]]):
    return scipy.stats.rankdata(array, method="min")


def _find_metric_by_name(
    target_and_metrics: typing.List[typing.Tuple[crunch.api.Target, typing.List[crunch.api.Metric]]],
    target_name: str,
    metric_name: str,
) -> crunch.api.Metric:
    for target, metrics in target_and_metrics:
        if target.name != target_name:
            continue

        for metric in metrics:
            if metric.name != metric_name:
                continue

            return metric

    raise ValueError(f"no metric found with name=`{metric_name}` from target with name=`{target_name}`")
