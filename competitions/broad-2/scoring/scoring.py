import os
import typing

import crunch
import crunch.custom
import crunch.utils
import numpy
import pandas


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
    if "ALL" in target_names:
        target_names.remove("ALL")

    with tracer.log("Loading gene list"):
        gene_csv = pandas.read_csv(f'{data_directory_path}/Crunch2_gene_list.csv')
        gene_names = set(gene_csv['gene_symbols'])

    with tracer.log("Check for required columns"):
        difference = crunch.custom.utils.delta_message(
            {'sample'} | gene_names,
            set(prediction.columns),
        )

        if difference:
            raise ParticipantVisibleError(f"Columns do not match: {difference}")

    with tracer.log("Check for missing samples"):
        difference = crunch.custom.utils.delta_message(
            set(target_names),
            set(prediction['sample'].unique()),
        )

        if difference:
            raise ParticipantVisibleError(f"Samples do not match: {difference}")

    for target_name in tracer.loop(target_names, "Checking target -> {value}"):
        with tracer.log(f"Filter prediction at target"):
            prediction_slice = prediction[prediction['sample'] == target_name]
            prediction_slice = prediction_slice.drop(columns=['sample'])

        sdata = _read_zarr(data_directory_path, target_name)

        with tracer.log("Extract unique cell IDs where the group is either 'test' or 'validation'"):
            cell_ids = set(sdata['cell_id-group'].obs.query("group == 'test' or group == 'validation'")['cell_id'])

        with tracer.log("Check for NaN values in predictions"):
            if prediction_slice.isnull().values.any():
                raise ParticipantVisibleError(f"Found NaN values for target `{target_name}`")

        with tracer.log("Check for infinity values in predictions"):
            prediction_slice = prediction_slice.replace([-numpy.inf, numpy.inf], numpy.nan)

        with tracer.log("Check that all cell IDs are present in predictions"):
            difference = crunch.custom.utils.delta_message(
                cell_ids,
                set(prediction_slice.index),
            )

            if difference:
                raise ParticipantVisibleError(f"Cell IDs do not match for target `{target_name}`: {difference}")

        with tracer.log("Check data types in the 'prediction' values"):
            if not pandas.api.types.is_numeric_dtype(prediction_slice.values):
                raise ParticipantVisibleError(f"Found non-numeric values for target `{target_name}`")

        with tracer.log("Ensure all prediction values are positive"):
            if (prediction_slice.values < 0).any():
                raise ParticipantVisibleError(f"Found negative values for target `{target_name}`")

        with tracer.log("Verify the size of predictions matches expectations"):
            expected = len(cell_ids) * len(gene_names)
            got = prediction_slice.size

            if expected != got:
                raise ParticipantVisibleError(f"Row count for target `{target_name}` should be {expected} ({len(cell_ids)} cell ids * {len(gene_names)} gene names), but got {got}")


def score(
    prediction: pandas.DataFrame,
    data_directory_path: str,
    phase_type: crunch.api.PhaseType,
    target_and_metrics: typing.List[typing.Tuple[crunch.api.Target, typing.List[crunch.api.Metric]]],
):
    """TODO"""
    raise NotImplementedError()


def _read_zarr(
        data_directory_path: str,
        target_name: str
):
    zar_data = os.path.join(data_directory_path, f"{target_name}.zarr")

    with tracer.log("Importing spatialdata"):
        import spatialdata

    with tracer.log("Read the Zarr data"):
        sdata = spatialdata.read_zarr(zar_data, selection=("tables",))

    return sdata
