import os
import typing

import crunch
import crunch.utils
import pandas
import spatialdata


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
    with tracer.log("Loading gene list"):
        gene_csv = pandas.read_csv(f'{data_directory_path}/Crunch2_gene_list.csv')
        gene_names = set(gene_csv['gene_symbols'])

    with tracer.log("Check for required columns"):
        difference = set(prediction.columns) ^ ({'sample'} | gene_names)

        if difference:
            raise ParticipantVisibleError(f"Missing or extra columns: {', '.join(difference)}")

    with tracer.log("Check for missing samples"):
        difference = set(prediction['sample'].unique()) ^ set(target_names)

        if difference:
            raise ParticipantVisibleError(f"Missing or extra samples: {', '.join(difference)}")

    for target_name in tracer.loop(target_names, "Checking target -> {value}"):
        with tracer.log(f"Filter prediction at target -> {target_name}"):
            prediction_slice = prediction[prediction['sample'] == target_name]
            prediction_slice = prediction_slice.drop(columns=['sample'])

        sdata = _read_zarr(data_directory_path, target_name)

        with tracer.log("Extract unique cell IDs where the group is either 'test' or 'validation'"):
            cell_ids = set(sdata['cell_id-group'].obs.query("group == 'test' or group == 'validation'")['cell_id'])

        with tracer.log("Check for NaN values in predictions"):
            if prediction_slice.isnull().values.any():
                raise ParticipantVisibleError("Predictions contain NaN values, which are not allowed.")

        with tracer.log("Check that all cell IDs are present in predictions"):
            missing = set(prediction_slice.index) - cell_ids

            if missing:
                raise ParticipantVisibleError(f"The following cell IDs are missing in predictions: {', '.join(list(map(str, missing))[-10:])}.")

        with tracer.log("Check data types in the 'prediction' values"):
            if not pandas.api.types.is_numeric_dtype(prediction_slice.values):
                raise ParticipantVisibleError("The 'prediction' values should only contain numeric values.")  #

        with tracer.log("Ensure all prediction values are positive"):
            if (prediction_slice.values < 0).any():
                raise ParticipantVisibleError("Prediction values should be positive.")

        with tracer.log("Verify the size of predictions matches expectations"):
            expected = len(cell_ids) * len(gene_names)
            got = prediction_slice.size

            if expected != got:
                raise ParticipantVisibleError(f"Predictions should have {expected} values but has {got}.")


def score(
    prediction: pandas.DataFrame,
    data_directory_path: str,
    phase_type: crunch.api.PhaseType,
    target_and_metrics: typing.List[typing.Tuple[crunch.api.Target, typing.List[crunch.api.Metric]]],
):
    """TODO"""
    pass


def _read_zarr(
        data_directory_path: str,
        target_name: str
):
    zar_data = os.path.join(data_directory_path, f"{target_name}.zarr")

    with tracer.log("Read the Zarr data"):
        sdata = spatialdata.read_zarr(zar_data, selection=("tables",))

    return sdata
