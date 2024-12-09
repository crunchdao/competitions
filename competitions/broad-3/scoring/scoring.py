import os
import typing

import crunch
import crunch.custom.utils
import crunch.utils
import numpy
import numpy.typing
import pandas


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


tracer = crunch.utils.Tracer()


def check(
    prediction: pandas.DataFrame,
    data_directory_path: str
):
    with tracer.log("Check for the `Gene Name` column"):
        difference = crunch.custom.utils.delta_message(
            {'Gene Name'},
            set(prediction.columns),
        )

        if difference:
            raise ParticipantVisibleError(f"Columns do not match: {difference}")

    with tracer.log("Loading expected columns"):
        gene_list_path = os.path.join(data_directory_path, 'Crunch3_gene_list.csv')

        expected_genes = pandas.read_csv(gene_list_path)
        expected_genes = set(expected_genes['gene_symbols'])

    with tracer.log("Ensure the DataFrame is sorted by its index before checking"):
        prediction.sort_index(inplace=True)

    with tracer.log("Check that the index runs from 1 to the total number of expected genes"):
        expected_index = numpy.arange(1, len(expected_genes) + 1)

        if not numpy.array_equal(prediction.index.values, expected_index):
            truncated = crunch.custom.utils.truncate(prediction.index.values)
            raise ParticipantVisibleError(f"The prediction DataFrame index must run from 1 to {len(expected_genes)}, but got [{truncated}]")

    with tracer.log("Check that the predicted genes match the expected genes exactly"):
        predicted_genes = set(prediction['Gene Name'])

        if expected_genes != predicted_genes:
            truncated = crunch.custom.utils.delta_message(expected_genes, predicted_genes)
            raise ParticipantVisibleError(f"The predicted genes do not match the expected genes: {truncated}")


def score(
    prediction: pandas.DataFrame,
    data_directory_path: str,
    phase_type: crunch.api.PhaseType,
    target_and_metrics: typing.List[typing.Tuple[crunch.api.Target, typing.List[crunch.api.Metric]]],
):
    """TODO"""
    pass
