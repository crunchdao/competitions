import typing
import os

import crunch
import crunch.custom
import numpy
import pandas
import glob
import scipy.stats

tracer = crunch.utils.Tracer()


def compare(
    targets: typing.List[crunch.api.Target],
    predictions: typing.Dict[int, pandas.DataFrame],
    combinations: typing.List[typing.Tuple[int, int]],
    data_directory_path: str,
):
    all_target = _find_target_by_name(targets, "ALL")

    # Avoid hardcoding sample names
    validation_file_path_prefix = os.path.join(data_directory_path, "validation-")
    sample_name_to_path = {
        file_path[len(validation_file_path_prefix):-4]: file_path
        for file_path in glob.glob(f"{validation_file_path_prefix}*.csv")
    }

    sample_names = list(sample_name_to_path.keys())
    print(f"found samples - names={sample_names}")

    gene_names_per_sample = {
        sample_name: set(
            pandas.read_csv(
                file_path,
                index_col="cell_id"
            ).columns
        )
        for sample_name, file_path in sample_name_to_path.items()
    }

    predictions_by_sample: typing.Dict[int, typing.Dict[str, pandas.DataFrame]] = {}

    for prediction_id, dataframe in tracer.loop(list(predictions.items()), lambda entry: f"Preparing Prediction #{entry[0]}"):
        prediction_by_sample = {}

        for sample_name, group in tracer.loop(dataframe.groupby("sample"), lambda entry: f"Grouped by {entry[0]}"):
            with tracer.log("Dropping unused gene columns"):
                useless_columns = set(group.columns) - gene_names_per_sample[sample_name]

                group.drop(columns=useless_columns, inplace=True)

            with tracer.log("Melting the dataframe"):
                group = group.melt(
                    var_name="gene",
                    value_name="prediction",
                    ignore_index=False,
                )

            with tracer.log("Setting the index"):
                group.index.name = "cell_id"
                group.set_index([group.index, "gene"], inplace=True)

            with tracer.log("Sorting the index"):
                group.sort_index(inplace=True)

            prediction_by_sample[sample_name] = group["prediction"]

        predictions_by_sample[prediction_id] = prediction_by_sample
        del predictions[prediction_id]

    similarities: typing.List[crunch.custom.ComparedSimilarity] = []
    for left_id, right_id in tracer.loop(combinations, lambda combination: f"Using Combination {combination[0]} <-> {combination[1]}"):
        left_prediction_by_sample = predictions_by_sample[left_id]
        right_prediction_by_sample = predictions_by_sample[right_id]

        values = []
        for sample_name in tracer.loop(sample_names, lambda name: f"Correlating {name}"):
            value = left_prediction_by_sample[sample_name].corr(right_prediction_by_sample[sample_name], method="spearman")
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
