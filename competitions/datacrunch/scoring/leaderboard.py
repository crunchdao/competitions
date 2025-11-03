import os
from typing import Dict, List, Tuple

import pandas
from crunch.api import Target
from crunch.unstructured import ComparedSimilarity
from crunch.utils import Tracer

tracer = Tracer()


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


_TARGET_NAME_TO_PREDICTION_COLUMN_MAPPING = {
    "white": "prediction_w",
    "red": "prediction_r",
    "green": "prediction_g",
    "blue": "prediction_b",
}


def compare(
    targets: List[Target],
    combinations: List[Tuple[int, int]],
    prediction_directory_path_by_id: Dict[int, str],
):
    predictions: Dict[int, pandas.DataFrame] = {}

    for prediction_id, prediction_directory_path in tracer.loop(prediction_directory_path_by_id.items(), "Loading prediction -> {value}"):
        dataframe = _try_load_prediction(prediction_directory_path)
        dataframe.set_index([
            "id",
            "moon",
        ], inplace=True)
        dataframe.sort_index(inplace=True)

        predictions[prediction_id] = dataframe

    similarities: List[ComparedSimilarity] = []
    for left_id, right_id in tracer.loop(combinations, "Computing correlation -> {value}"):
        left = predictions[left_id]
        right = predictions[right_id]

        for target in tracer.loop(targets, lambda target: f"Computing correlation for target -> {target.name}"):
            target_id = target.id
            prediction_column_name = _TARGET_NAME_TO_PREDICTION_COLUMN_MAPPING[target.name]

            value = left[prediction_column_name].corr(right[prediction_column_name])

            print(f"similarity - left_id={left_id} right_id={right_id} target_id={target_id} value={value}")
            similarities.append(ComparedSimilarity(
                left_id,
                right_id,
                target_id,
                value,
            ))

    return similarities


def _try_load_prediction(
    prediction_directory_path: str,
) -> pandas.DataFrame:
    (file, *_) = os.listdir(prediction_directory_path)
    prediction_file_path = os.path.join(prediction_directory_path, file)

    if file.endswith(".csv"):
        return pandas.read_csv(prediction_file_path)  # type: ignore

    if file.endswith(".parquet"):
        return pandas.read_parquet(prediction_file_path)  # type: ignore

    raise ParticipantVisibleError(f"no compatible reader for file {os.listdir(file)}")
