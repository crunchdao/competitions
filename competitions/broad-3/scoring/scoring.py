import os
import typing

import crunch
import crunch.utils
import numpy
import numpy.typing
import pandas
import scipy.stats


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


tracer = crunch.utils.Tracer()


def check(
    prediction: pandas.DataFrame,
    data_directory_path: str
):
    with tracer.log("Ensure the DataFrame is sorted by its index before checking"):
        prediction.sort_index(inplace=True)

    with tracer.log("Check for required columns"):
        gene_list_path = os.path.join(data_directory_path, 'Crunch3_gene_list.csv')
        expected_genes_df = pandas.read_csv(gene_list_path)
        expected_genes = expected_genes_df['gene_symbols']

    with tracer.log("Check for the `Gene Name` column"):
        if 'Gene Name' not in prediction.columns:
            raise ParticipantVisibleError("The prediction DataFrame must contain a 'Gene Name' column.")

    with tracer.log("Check that the predicted genes match the expected genes exactly"):

        if set(prediction) != set(expected_genes):
            missing_in_pred = set(expected_genes) - set(prediction)
            extra_in_pred = set(prediction) - set(expected_genes)
            raise ParticipantVisibleError(
                "The predicted genes do not match the expected genes.\n"
                f"Missing genes in prediction: {missing_in_pred}\n"
                f"Extra genes in prediction: {extra_in_pred}"
            )
    with tracer.log("Check that the index runs from 1 to the total number of expected genes"):
        expected_index = numpy.arange(1, len(expected_genes) + 1)
        if not numpy.array_equal(prediction.index.values, expected_index):
            raise ValueError(
                f"The prediction DataFrame index must run from 1 to {len(expected_genes)}, "
                f"but got {prediction.index.values}."
            )

    print("The prediction DataFrame has the correct format and content.")


def score(
    prediction: pandas.DataFrame,
    data_directory_path: str,
    phase_type: crunch.api.PhaseType,
    target_and_metrics: typing.List[typing.Tuple[crunch.api.Target, typing.List[crunch.api.Metric]]],
):
    """TODO"""
    pass
