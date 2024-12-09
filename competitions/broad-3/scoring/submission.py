import typing

import crunch.custom
import crunch.utils
from crunch.external import humanfriendly


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


tracer = crunch.utils.Tracer()


def check(
    submission_files: typing.List[crunch.custom.File],
    model_files: typing.List[crunch.custom.File],
):
    with tracer.log("Finding report file"):
        report_md_file_path = "REPORT.md"
        report_md_file = _find_file_by_path(submission_files, report_md_file_path) or _find_file_by_path(submission_files, report_md_file_path.lower())

        if report_md_file is None:
            raise ParticipantVisibleError(f"Missing `{report_md_file_path}` file.")

        report_md_file_path = report_md_file.path

    with tracer.log("Checking file size"):
        max_size = 4 * 1024 * 1024
        size = report_md_file.size

        if size > max_size:
            max_size = humanfriendly.format_size(max_size)
            size = humanfriendly.format_size(size)

            raise ParticipantVisibleError(f"`{report_md_file_path}` is too big, maximum is {max_size} but file is {size}.")

    with tracer.log("Getting content"):
        content = report_md_file.text

    with tracer.log("Ensure only ASCII"):
        if not content.isascii():
            raise ParticipantVisibleError(f"`{report_md_file_path}` must contain only ASCII characters.")

    with tracer.log("Checking for the default comment"):
        if "<!-- Don't forget to change me -->" in content:
            raise ParticipantVisibleError(f"Default comment found in `{report_md_file_path}`. Has it been edited?")

    with tracer.log("Counting non blank lines"):
        count = 0
        for line in content.splitlines():
            line = line.strip()

            if not line:
                continue

            # TODO Should titles be skipped?
            count += 1

        minimum_line_count = 10
        if count < minimum_line_count:
            raise ParticipantVisibleError(f"`{report_md_file_path}` must be {minimum_line_count} lines long, only found {count} (non blank) line(s).")


def _find_file_by_path(
    files: typing.List[crunch.custom.File],
    path: str
) -> typing.Optional[crunch.custom.File]:
    return next((
        file
        for file in files
        if file.path == path
    ), None)
