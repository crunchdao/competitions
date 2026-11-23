"""
Falcon: the mean log-likelihood of the densities, as the live game computed it.

A density given at a tick is resolved at the first later tick whose time is at least `HORIZON_SECONDS` later, with the
dove's location there (the game's quarantine); the game time restarts with the game, so nothing is resolved across a
restart, and the ticks without a resolving tick are not scored. A missing (`None`), failed or invalid density counts
as a likelihood of 0 (the floor: log(1e-8)), so skipping a tick cannot pay. The likelihood is `densitypdf.density_pdf`,
non-finite values replaced by 0, as in the game.
"""

import json
import math
import os
from typing import List, Tuple

import numpy
import pandas
from crunch.api import Metric, Target
from crunch.unstructured import ScoredMetric
from crunch.utils import Tracer

TICKS_FILE_NAME = "ticks.bin"
PREDICTION_FILE_NAME = "prediction.parquet"

TICK_DTYPE = numpy.dtype([
    ("time", "<f8"),
    ("falcon_location", "<f8"),
    ("falcon_wingspan", "<f8"),
    ("dove_location", "<f8"),
    ("falcon_id", "<u2"),
])

STATUS_SUCCESS = 0
STATUS_NONE = 1
STATUS_FAILED = 2

HORIZON_SECONDS = 3.0
LIKELIHOOD_EPSILON = 1e-8


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


tracer = Tracer()


def check(
    prediction_directory_path: str,
    data_directory_path: str,
):
    prediction = _load_prediction(prediction_directory_path)
    ticks = _load_ticks(data_directory_path)

    with tracer.log("Check the number of answers"):
        if len(prediction) != len(ticks):
            raise ParticipantVisibleError(f"Expected {len(ticks):,} answers, got {len(prediction):,}")

    with tracer.log("Check the statuses"):
        if not prediction["status"].isin([STATUS_SUCCESS, STATUS_NONE, STATUS_FAILED]).all():
            raise ParticipantVisibleError("Unknown answer status")


def score(
    prediction_directory_path: str,
    data_directory_path: str,
    target_and_metrics: List[Tuple[Target, List[Metric]]],
):
    _, metrics = target_and_metrics[0]
    metric = _find_metric(metrics)

    prediction = _load_prediction(prediction_directory_path)
    ticks = _load_ticks(data_directory_path)

    with tracer.log("Resolve each tick"):
        resolving_index = resolve(ticks["time"])
        scored = resolving_index >= 0

        if not scored.any():
            raise ValueError("no tick to score")

    with tracer.log("Compute the likelihoods"):
        statuses = prediction["status"].to_numpy()
        densities = prediction["density"].to_numpy()
        dove_locations = ticks["dove_location"]

        log_likelihoods = numpy.empty(int(scored.sum()))
        for position, index in enumerate(numpy.flatnonzero(scored)):
            likelihood = 0.0
            if statuses[index] == STATUS_SUCCESS:
                likelihood = likelihood_of(densities[index], dove_locations[resolving_index[index]])

            log_likelihoods[position] = math.log(likelihood + LIKELIHOOD_EPSILON)

        score_value = float(log_likelihoods.mean())

    print(f"scored ticks: {len(log_likelihoods):,} of {len(ticks):,}, answered: {int((statuses == STATUS_SUCCESS).sum()):,}, mean log-likelihood: {score_value:.6f}")

    return {
        metric.id: ScoredMetric(
            value=score_value,
        ),
    }


def resolve(times: numpy.ndarray) -> numpy.ndarray:
    """
    For each tick, the index of the tick that resolves it: the first one at least `HORIZON_SECONDS` later in the same
    game (the time goes back when the game restarts), or -1.
    """

    resolving_index = numpy.full(len(times), -1, dtype=numpy.int64)

    restarts = numpy.flatnonzero(numpy.diff(times) < 0) + 1
    for start, end in zip(numpy.concatenate([[0], restarts]), numpy.concatenate([restarts, [len(times)]])):
        segment = times[start:end]
        found = numpy.searchsorted(segment, segment + HORIZON_SECONDS, side="left")
        inside = found < len(segment)
        resolving_index[start:end][inside] = start + found[inside]

    return resolving_index


def likelihood_of(density_json: str, target: float) -> float:
    """As the live game: `density_pdf`, or the legacy gaussian; 0 when invalid or not finite."""

    from densitypdf import density_pdf

    try:
        density = json.loads(density_json)
        if not isinstance(density, dict):
            return 0.0

        try:
            value = density_pdf(density_dict=density, x=target)
        except Exception:
            value = _legacy_density_pdf(density, target)

        value = float(value)
    except Exception:
        return 0.0

    return value if math.isfinite(value) else 0.0


def _legacy_density_pdf(density: dict, target: float) -> float:
    if density.get("model_type") != "gaussian":
        raise ValueError("unsupported density")

    from scipy.stats import norm

    mu = density["model_params"]["mu"]
    sigma = max(0.001, density["model_params"]["sigma"])
    return norm.pdf(target, loc=mu, scale=max(sigma, 1e-8))


def _find_metric(
    metrics: List[Metric],
) -> Metric:
    if len(metrics) == 1:
        return metrics[0]

    for metric in metrics:
        if "likelihood" in metric.name:
            return metric

    raise ValueError(f"no likelihood metric in {[metric.name for metric in metrics]}")


def _load_prediction(
    prediction_directory_path: str,
) -> pandas.DataFrame:
    path = os.path.join(prediction_directory_path, PREDICTION_FILE_NAME)

    with tracer.log("Load the prediction"):
        if not os.path.exists(path):
            raise ParticipantVisibleError(f"No {PREDICTION_FILE_NAME} found")

        return pandas.read_parquet(path, columns=["status", "density"])


def _load_ticks(
    data_directory_path: str,
) -> numpy.ndarray:
    with tracer.log("Load the ticks"):
        return numpy.fromfile(os.path.join(data_directory_path, TICKS_FILE_NAME), dtype=TICK_DTYPE)
