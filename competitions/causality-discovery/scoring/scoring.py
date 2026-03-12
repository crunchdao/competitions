import os
from typing import Dict, List, Tuple, Union

import crunch
import joblib
import networkx
import numpy
import pandas
from crunch.unstructured import ScoredMetric
from crunch.unstructured.utils import delta_message
from crunch.utils import Tracer
from sklearn.metrics import balanced_accuracy_score

PREDICTION_FILE_NAME = "prediction.parquet"


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


FROM_COLUMN_NAME = "from"
TO_COLUMN_NAME = "to"

MAPPING = numpy.full(16, -1)
MAPPING[0b0000] = 1  # 'Independent'
MAPPING[0b0001] = 2  # 'Cause of X'
MAPPING[0b0010] = 3  # 'Cause of Y'
MAPPING[0b0011] = 4  # 'Confounder'
MAPPING[0b0100] = 5  # 'Consequence of Y'
MAPPING[0b1000] = 6  # 'Consequence of X'
MAPPING[0b1010] = 7  # 'Mediator'
MAPPING[0b1100] = 8  # 'Collider'


class BadGraphError(ParticipantVisibleError):
    pass


tracer = Tracer()


def check(
    prediction_directory_path: str,
    data_directory_path: str,
):
    with tracer.log("Validating prediction file presence"):
        prediction_file_path = os.path.join(prediction_directory_path, PREDICTION_FILE_NAME)
        if not os.path.exists(prediction_file_path):
            raise ParticipantVisibleError(f"Prediction file `{PREDICTION_FILE_NAME}` not found in `{prediction_directory_path}`.")

    with tracer.log("Load prediction"):
        prediction = pandas.read_parquet(prediction_file_path)

    with tracer.log("Load example prediction"):
        example_prediction = pandas.read_parquet(os.path.join(data_directory_path, "example_prediction.parquet"))

    with tracer.log("Check column names"):
        difference = delta_message(example_prediction.columns, prediction.columns)

        if difference:
            raise ParticipantVisibleError(f"Columns do not match: {difference}")

    with tracer.log("Check for NaNs"):
        if prediction.isna().any().any():
            raise ParticipantVisibleError("Prediction contains NaNs values.")

    with tracer.log("Check for infinite values"):
        if prediction.isin([numpy.inf, -numpy.inf]).any().any():
            raise ParticipantVisibleError("Prediction contains infinite values.")

    with tracer.log("Validate IDs"):
        difference = delta_message(example_prediction["example_id"], prediction["example_id"])

        if difference:
            raise ParticipantVisibleError(f"IDs do not match: {difference}")

    with tracer.log("Validate values are either 0 or 1"):
        if not prediction["prediction"].isin([0, 1]).all().all():
            raise ParticipantVisibleError("Prediction contains values other than 0 or 1.")


def score(
    prediction_directory_path: str,
    data_directory_path: str,
    target_and_metrics: List[Tuple[crunch.api.Target, List[crunch.api.Metric]]],
):
    metric = target_and_metrics[0][1][0]

    with tracer.log("Load prediction"):
        prediction = pandas.read_parquet(os.path.join(prediction_directory_path, PREDICTION_FILE_NAME))

    with tracer.log("Process prediction"):
        prediction = _process_prediction(prediction)

    with tracer.log("Load y_test"):
        y_test = joblib.load(os.path.join(data_directory_path, "y_test.pickle"))

    with tracer.log("Process y_test"):
        y_test = _process_y(y_test)

    with tracer.log("Merge prediction and y_test"):
        merged = pandas.merge(
            y_test,
            prediction,
            on=[
                "dataset",
                "example_id",
            ]
        )

    with tracer.log("Compute metric"):
        score_value = balanced_accuracy_score(
            y_true=merged["target"],
            y_pred=merged["prediction"],
        )

    return {
        metric.id: ScoredMetric(
            value=score_value,
        ),
    }


def get_labels(
    key: Union[str, int],
    pivoted: pandas.DataFrame,

    # do not change, local lookups are faster than globals
    mapping=MAPPING,
):
    """
    Classify the nodes of g as "collider", "confounder", etc., wrt the edge X→Y

    For each node i, we look at the role of i wrt X→Y, ignoring all other nodes.
    There are 8 possible cases:
    - Cause of X
    - Consequence of X
    - Confounder
    - Collider
    - Mediator
    - Independent
    - Cause of Y
    - Consequence of Y

    Caveat:
    - The notions of "confounder", "collider", etc. only make sense for small, textbook graphs.

    Input:  g: nx.DiGraph object, with an edge X→Y
    Output: list of tuple, with the edges as keys (excluding 'X' and 'Y'): (dataset_id, node, value)
    """

    nodes = pivoted.columns.to_list()
    graph = networkx.from_pandas_adjacency(pivoted, create_using=networkx.DiGraph)

    if 'X' not in nodes:
        raise BadGraphError(f"X not in nodes for dataset `{key}`")

    if 'Y' not in nodes:
        raise BadGraphError(f"Y not in nodes for dataset `{key}`")

    if ('X', 'Y') not in graph.edges:
        raise BadGraphError(f"X and/or Y not in edges for dataset `{key}`")

    if not networkx.is_directed_acyclic_graph(graph):
        raise BadGraphError(f"not a directed acyclic graph for dataset `{key}`")

    A = pivoted.values

    x_index = nodes.index('X')
    y_index = nodes.index('Y')

    return [
        (
            key,
            node,
            mapping[
                (A[x_index, index] << 3) |
                (A[y_index, index] << 2) |
                (A[index, y_index] << 1) |
                (A[index, x_index])
            ]
        )
        for index, node in enumerate(nodes)
        if node not in "XY"
    ]


def _process_prediction(
    prediction: pandas.DataFrame,
):
    id_column = prediction["example_id"].str.split('_')
    prediction["example_id"] = id_column.str[:-2].str.join('_')
    prediction["from"] = id_column.str[-2]
    prediction["to"] = id_column.str[-1]

    groups = prediction.groupby("example_id")

    labelss: List[tuple] = []
    for key, group in groups:
        pivoted = group.pivot(
            index="from",
            columns="to",
            values="prediction",
        )

        labels = get_labels(key, pivoted)
        labelss.extend(labels)

    return pandas.DataFrame(
        labelss,
        columns=[
            "dataset",
            "example_id",
            "prediction",
        ]
    )


def _process_y(
    y_test: Dict[Union[str, int], pandas.DataFrame],
):
    labelss: List[tuple] = []
    for key, y in y_test.items():
        labels = get_labels(key, y)
        labelss.extend(labels)

    return pandas.DataFrame(
        labelss,
        columns=[
            "dataset",
            "example_id",
            "target"
        ]
    )
