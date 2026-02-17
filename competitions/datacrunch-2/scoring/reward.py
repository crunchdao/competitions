from typing import List

import pandas
from crunch.api import Metric, Target
from crunch.unstructured import RewardableProject, RewardedProject
from crunch.utils import Tracer

tracer = Tracer()


def compute_bounties(
    metrics: List[Metric],
    projects: List[RewardableProject],
    granted_amount: float,
):
    metric = metrics[0]

    dataframe = pandas.DataFrame((
        {
            "project_id": project.id,
            "group": project.group,
            "rewardable": project.rewardable,
            "metric": project.get_metric(metric.id).score,
            "weight": 0.0,
        }
        for project in projects
    ))

    dataframe.sort_values(by="metric", ascending=False, inplace=True)
    dataframe["percentile_rank"] = dataframe["metric"].rank(pct=True)

    for index, row in dataframe.iterrows():
        percentile_rank = row["percentile_rank"]
        if percentile_rank <= 0.5:
            continue

        e = 2 * (percentile_rank - 0.5)
        weight = e ** 20

        dataframe.at[index, "weight"] = weight

    dataframe["reward"] = dataframe["weight"] / dataframe["weight"].sum() * granted_amount

    return [
        RewardedProject(
            id=int(row["project_id"]),
            amount=row["reward"],
        )
        for _, row in dataframe.iterrows()
    ]
