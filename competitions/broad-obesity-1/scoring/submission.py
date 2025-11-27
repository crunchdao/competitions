import re
from typing import List, Optional

import crunch.utils
from crunch.external import humanfriendly
from crunch.unstructured import File


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


tracer = crunch.utils.Tracer()


def check(
    submission_files: List[File],
    model_files: List[File],
):
    with tracer.log("Finding report file"):
        report_md_file_path = "Method description.md"
        report_md_file = _find_file_by_path(submission_files, report_md_file_path)

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
        assert content is not None  # should not happen

    with tracer.log("Ensure only ASCII"):
        if not content.isascii():
            raise ParticipantVisibleError(f"`{report_md_file_path}` must contain only ASCII characters.")

    with tracer.log("Checking for the default comment"):
        if "<!-- Don't forget to change me -->" in content:
            raise ParticipantVisibleError(f"Default comment found in `{report_md_file_path}`. Has it been edited?")

    with tracer.log("Remove all comments"):
        content = re.sub(r"<!--.+?-->", r"", content)

    with tracer.log("Finding sections"):
        sections = [
            "Method Description",
            "Rationale",
            "Data and Resources Used",
        ]

        for section in sections:
            matches = re.findall(
                r"^# *" + section + "$",
                content,
                flags=re.IGNORECASE | re.MULTILINE,
            )

            if len(matches) == 0:
                raise ParticipantVisibleError(f"`{report_md_file_path}` must contain a section titled `{section}`.")
            elif len(matches) > 1:
                raise ParticipantVisibleError(f"`{report_md_file_path}` must not contain multiple section titled `{section}`.")

    with tracer.log("Counting non blank lines"):
        line_count = 0
        for line in content.splitlines():
            line = line.strip()

            if not line:
                continue

            # TODO Should titles be skipped?
            line_count += 1

        minimum_line_count = 15
        if line_count < minimum_line_count:
            plural, be = "", "was"
            if line_count > 1:
                plural, be = "s", "were"

            raise ParticipantVisibleError(f"`{report_md_file_path}` must be longer than {minimum_line_count} lines, but only {line_count} non-blank line{plural} {be} found.")


def _find_file_by_path(
    files: List[File],
    path: str,
) -> Optional[File]:
    return next((
        file
        for file in files
        if file.path.casefold() == path.casefold()
    ), None)
