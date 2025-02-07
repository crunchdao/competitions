import typing

import crunch
import crunch.custom
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
            cell_spearman_column_name: project.get_metric(cell_spearman_metric.id).score,
            gene_spearman_column_name: project.get_metric(gene_spearman_metric.id).score,
        }
        for project in projects
    ))

    dataframe["final_score"] = scipy.stats.rankdata(
        scipy.stats.rankdata(-dataframe[cell_spearman_column_name])
        + scipy.stats.rankdata(-dataframe[gene_spearman_column_name])
    )

    dataframe.sort_values(
        by=[
            'final_score',
            'project_id',  # fallback if same `final_score`
        ],
        inplace=True
    )

    return dataframe["project_id"].astype(int).tolist()


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
