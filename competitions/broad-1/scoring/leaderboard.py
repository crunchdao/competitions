import os
import typing

import crunch
import crunch.api
import crunch.unstructured
import crunch.utils
import pandas

tracer = crunch.utils.Tracer()

PREDICTION_FILE_NAME = "prediction.parquet"


def compare(
    targets: typing.List[crunch.api.Target],
    prediction_directory_path_by_id: typing.Dict[int, str],
    combinations: typing.List[typing.Tuple[int, int]],
):
    all_target = _find_target_by_name(targets, "ALL")

    predictions: typing.Dict[int, pandas.DataFrame] = {}
    for prediction_id, prediction_directory_path in tracer.loop(list(prediction_directory_path_by_id.items()), lambda entry: f"Preparing Prediction #{entry[0]}"):
        with tracer.log("Loading the prediction"):
            dataframe = pandas.read_parquet(os.path.join(prediction_directory_path, PREDICTION_FILE_NAME))

        with tracer.log("Setting the index"):
            dataframe.set_index(["sample", "cell_id", "gene"], inplace=True)

        with tracer.log("Sorting the index"):
            dataframe.sort_index(inplace=True)

        predictions[prediction_id] = dataframe

    similarities: typing.List[crunch.unstructured.ComparedSimilarity] = []
    for left_id, right_id in tracer.loop(combinations, lambda combination: f"Using Combination {combination[0]} <-> {combination[1]}"):
        left = predictions[left_id]
        right = predictions[right_id]

        with tracer.log("Computing correlation"):
            value = left["prediction"].corr(right["prediction"], method="spearman")

        print(f"similarity - left_id={left_id} right_id={right_id} value={value}")
        similarities.append(crunch.unstructured.ComparedSimilarity(
            left_id,
            right_id,
            all_target.id,
            value,
        ))

    return similarities


def _find_target_by_name(
    targets: typing.List[crunch.api.Target],
    target_name: str,
) -> crunch.api.Target:
    for target in targets:
        if target.name == target_name:
            return target

    raise ValueError(f"no target found with name=`{target_name}`")
