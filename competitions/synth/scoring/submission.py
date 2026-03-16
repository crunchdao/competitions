from typing import List, Optional

from crunch.unstructured.utils import find_requirement_by_name
from crunch.utils import Tracer
from crunch_convert.requirements_txt import NamedRequirement


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


tracer = Tracer()


def check(
    python_requirements: Optional[List[NamedRequirement]],
):
    with tracer.log("Ensure contains package"):
        if python_requirements is None:
            raise ParticipantVisibleError("Missing or invalid `requirements.txt` file.")

        crunch_synth = find_requirement_by_name(python_requirements, "crunch-synth")
        if crunch_synth is None:
            raise ParticipantVisibleError("`requirements.txt` file must include the `crunch-synth` package.")
