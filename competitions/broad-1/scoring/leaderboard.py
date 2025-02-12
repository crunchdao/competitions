import typing

import crunch
import crunch.custom
import crunch.utils
import pandas

tracer = crunch.utils.Tracer()


def compare(
    targets: typing.List[crunch.api.Target],
    predictions: typing.Dict[int, pandas.DataFrame],
    combinations: typing.List[typing.Tuple[int, int]],
):
    all_target = _find_target_by_name(targets, "ALL")

    for _, dataframe in tracer.loop(predictions.items(), lambda entry: f"Preparing Prediction #{entry[0]}"):
        with tracer.log("Setting the index"):
            dataframe.set_index(["sample", "cell_id", "gene"], inplace=True)

        with tracer.log("Sorting the index"):
            dataframe.sort_index(inplace=True)

    similarities: typing.List[crunch.custom.ComparedSimilarity] = []
    for left_id, right_id in tracer.loop(combinations, lambda combination: f"Using Combination {combination[0]} <-> {combination[1]}"):
        left = predictions[left_id]
        right = predictions[right_id]

        with tracer.log("Computing correlation"):
            value = left["prediction"].corr(right["prediction"])

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
