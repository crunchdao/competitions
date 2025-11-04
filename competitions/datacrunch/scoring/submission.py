from typing import List

from crunch.api import SubmissionType
from crunch.unstructured import File


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


def check(
    submission_type: SubmissionType,
    submission_files: List[File],
):
    if submission_type != SubmissionType.PREDICTION:
        return

    if len(submission_files) != 1:
        raise ParticipantVisibleError("Must only submit one file.")

    submission_file = submission_files[0]

    name = submission_file.name.lower()
    print(name)
    if not name.endswith(".csv") and not name.endswith(".parquet"):
        raise ParticipantVisibleError("Must only submit a .csv or a .parquet file.")
