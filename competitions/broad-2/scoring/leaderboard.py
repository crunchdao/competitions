import typing

import crunch
import crunch.custom
import numpy
import pandas
import scipy.stats


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
