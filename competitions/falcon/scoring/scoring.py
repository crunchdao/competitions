"""
Falcon: the total wealth the tracker would have made as one more player of the recorded 2025 game, with the live wealth
mechanism (birdgame 0.3.1, `birdgame/wealth/wealth_mechanism.py`, checked on the game's history); and, for information,
its mean log-likelihood.

A density given at a tick is resolved at the first later tick whose time is at least `HORIZON_SECONDS` later, with the
dove's location there (the game's quarantine); the game time restarts with the game (it goes back by more than
`RESTART_BACKWARD_SECONDS`), so nothing is resolved across a restart. The likelihood is `densitypdf.density_pdf`
(>= 0.1.4: mixture weights normalized, as the live game), non-finite values replaced by 0.

Total wealth (`pot.bin`: per tick, the sums over the 2025 players who were paid there, from the game's history):
- a player participates with a density answered within `LATE_THRESHOLD_NANOSECONDS`; a missing, failed or late answer
  sits the tick out (no investment, no payout, no EWMA update), as in the live game (an invalid density participates,
  with a likelihood of 0);
- the blended EWMA of the log-likelihood (`ALPHA_SHORT`, `ALPHA_LONG`, `SHORT_WEIGHT`) warms up for `WARM_UP_TICKS`
  after the start and after each game restart, before the player is paid;
- at each tick where the 2025 players were paid, a participant invests `INVESTMENT_FRACTION` of its wealth, the pot is
  inflated by `INFLATION_BASIS_POINTS` and shared in proportion to exp(blended EWMA): the player's over the 2025
  players' (recorded) plus its own;
- every `PRIZE_FREQUENCY_TICKS`, wealth above `THRESHOLD_WEALTH` is withdrawn as a prize;
- the score is the gain of total wealth (the wealth left plus the prizes) over `INITIAL_WEALTH`, as the 2025 players'
  recorded gain over the same window (the game reset every player to `INITIAL_WEALTH` at its start).

Log-likelihood: the mean of log(likelihood + 1e-8) over the resolved ticks, a missing, failed or invalid density
counting as a likelihood of 0.
"""

import json
import math
import os
from importlib.metadata import version
from typing import Dict, List, Tuple

import numpy
import pandas
from crunch.api import Metric, Target
from crunch.unstructured import ScoredMetric
from crunch.utils import Tracer

TICKS_FILE_NAME = "ticks.bin"
POT_FILE_NAME = "pot.bin"
PREDICTION_FILE_NAME = "prediction.parquet"

TICK_DTYPE = numpy.dtype([
    ("time", "<f8"),
    ("falcon_location", "<f8"),
    ("falcon_wingspan", "<f8"),
    ("dove_location", "<f8"),
    ("falcon_id", "<u2"),
])
POT_DTYPE = numpy.dtype([
    ("wealth", "<f8"),
    ("weight", "<f8"),
    ("participants", "<u2"),
])

STATUS_SUCCESS = 0
STATUS_NONE = 1
STATUS_FAILED = 2

HORIZON_SECONDS = 3.0
RESTART_BACKWARD_SECONDS = 60.0  # a game restart: the time goes back by more (smaller steps back are clock jitter)
LIKELIHOOD_EPSILON = 1e-8
LATE_THRESHOLD_NANOSECONDS = 52_000_000

INVESTMENT_FRACTION = 0.0001
INFLATION_BASIS_POINTS = 1
INITIAL_WEALTH = 1000.0
ALPHA_SHORT = 1 / (20 * 1000 / 60)  # 20 seconds (1,000 ticks are about a minute)
ALPHA_LONG = 1 / (20 * 1000)  # 20 minutes
SHORT_WEIGHT = 0.5
LOG_LIKELIHOOD_FLOOR = 1e-12
WARM_UP_TICKS = 20 * 1000
PRIZE_FREQUENCY_TICKS = 1000
THRESHOLD_WEALTH = 2000.0
WEALTH_THRESHOLD = 200.0

TOTAL_WEALTH_METRIC_NAME = "total-wealth"
LOG_LIKELIHOOD_METRIC_NAME = "log-likelihood"


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
    _ensure_density_normalization()

    _, metrics = target_and_metrics[0]
    metrics_by_name = {metric.name: metric for metric in metrics}

    prediction = _load_prediction(prediction_directory_path)
    ticks = _load_ticks(data_directory_path)

    with tracer.log("Resolve each tick"):
        resolving_index = resolve(ticks["time"])
        if not (resolving_index >= 0).any():
            raise ValueError("no tick to score")

    with tracer.log("Compute the likelihoods"):
        likelihoods = likelihoods_of(ticks, resolving_index, prediction)

    scores: Dict[int, ScoredMetric] = {}

    metric = metrics_by_name.get(LOG_LIKELIHOOD_METRIC_NAME)
    if metric is not None:
        resolved = resolving_index >= 0
        answered = likelihoods[resolved]
        value = float(numpy.log(numpy.nan_to_num(answered, nan=0.0) + LIKELIHOOD_EPSILON).mean())
        print(f"log-likelihood: {value:.6f} over {int(resolved.sum()):,} resolved ticks")
        scores[metric.id] = ScoredMetric(value=value)

    metric = metrics_by_name.get(TOTAL_WEALTH_METRIC_NAME)
    if metric is not None:
        pot = _load_pot(data_directory_path)
        if len(pot) != len(ticks):
            raise ValueError(f"the pot has {len(pot):,} ticks, the feed {len(ticks):,}")

        late = (prediction["received_ns"].to_numpy() - prediction["sent_ns"].to_numpy()) > LATE_THRESHOLD_NANOSECONDS
        participating = likelihoods.copy()
        participating[late] = numpy.nan

        with tracer.log("Replay the wealth"):
            result = replay_wealth(ticks["time"], pot, participating)

        value = result["total_wealth"] - INITIAL_WEALTH
        print(f"total wealth: {result['total_wealth']:,.2f} (wealth {result['wealth']:,.2f}, prizes {result['prizes']:,.0f}), paid on {result['paid_ticks']:,} ticks, sat out {result['sat_out_ticks']:,} ({int(late.sum()):,} late)")
        scores[metric.id] = ScoredMetric(value=value)

    if not scores:
        raise ValueError(f"no known metric in {list(metrics_by_name)}")

    return scores


def resolve(times: numpy.ndarray) -> numpy.ndarray:
    """
    For each tick, the index of the tick that resolves it: the first one at least `HORIZON_SECONDS` later in the same
    game (the time goes back by more than `RESTART_BACKWARD_SECONDS` when the game restarts), or -1.
    """

    resolving_index = numpy.full(len(times), -1, dtype=numpy.int64)

    restarts = numpy.flatnonzero(numpy.diff(times) < -RESTART_BACKWARD_SECONDS) + 1
    for start, end in zip(numpy.concatenate([[0], restarts]), numpy.concatenate([restarts, [len(times)]])):
        segment = numpy.maximum.accumulate(times[start:end])  # a step back of clock jitter does not break the search
        found = numpy.searchsorted(segment, times[start:end] + HORIZON_SECONDS, side="left")
        inside = found < len(segment)
        resolving_index[start:end][inside] = start + found[inside]

    return resolving_index


def likelihoods_of(
    ticks: numpy.ndarray,
    resolving_index: numpy.ndarray,
    prediction: pandas.DataFrame,
) -> numpy.ndarray:
    """The likelihood of each resolved tick's density, NaN when there is none (missing, failed, or not resolved)."""

    statuses = prediction["status"].to_numpy()
    densities = prediction["density"].to_numpy()
    dove_locations = ticks["dove_location"]

    likelihoods = numpy.full(len(ticks), numpy.nan)
    for index in numpy.flatnonzero((resolving_index >= 0) & (statuses == STATUS_SUCCESS)):
        likelihoods[index] = likelihood_of(densities[index], dove_locations[resolving_index[index]])

    return likelihoods


def replay_wealth(
    times: numpy.ndarray,
    pot: numpy.ndarray,
    likelihoods: numpy.ndarray,
) -> dict:
    """The player's wealth through the recorded game; `likelihoods` is NaN where the player sits out."""

    times = times.tolist()
    field_wealth = pot["wealth"].tolist()
    field_weight = pot["weight"].tolist()
    field_participants = pot["participants"].tolist()
    likelihoods = likelihoods.tolist()

    wealth = INITIAL_WEALTH
    prizes = 0.0
    short = long = None
    warm_up_left = WARM_UP_TICKS
    paid_ticks = sat_out_ticks = 0
    previous_time = None

    for index in range(len(times)):
        time = times[index]
        if previous_time is not None and time < previous_time - RESTART_BACKWARD_SECONDS:  # a game restart: the EWMAs warm up again
            short = long = None
            warm_up_left = WARM_UP_TICKS
        previous_time = time

        likelihood = likelihoods[index]
        if likelihood != likelihood:  # NaN: sits out
            sat_out_ticks += 1
        else:
            log_likelihood = math.log(max(likelihood, LOG_LIKELIHOOD_FLOOR))
            if long is None:
                short = long = log_likelihood
            else:
                short = ALPHA_SHORT * log_likelihood + (1 - ALPHA_SHORT) * short
                long = ALPHA_LONG * log_likelihood + (1 - ALPHA_LONG) * long

            if warm_up_left <= 0 and field_participants[index] > 0:
                own_weight = math.exp(SHORT_WEIGHT * short + (1 - SHORT_WEIGHT) * long)
                investment = INVESTMENT_FRACTION * wealth
                wealth -= investment
                pot_value = (INVESTMENT_FRACTION * field_wealth[index] + investment) * (1 + INFLATION_BASIS_POINTS / 10000.0)
                wealth += pot_value * own_weight / (field_weight[index] + own_weight)
                paid_ticks += 1

        if warm_up_left > 0:
            warm_up_left -= 1

        if (index + 1) % PRIZE_FREQUENCY_TICKS == 0 and wealth > THRESHOLD_WEALTH:
            prize = round((wealth - THRESHOLD_WEALTH) + 0.3 * (THRESHOLD_WEALTH - WEALTH_THRESHOLD), -2)
            wealth -= prize
            prizes += prize

    return {
        "total_wealth": wealth + prizes,
        "wealth": wealth,
        "prizes": prizes,
        "paid_ticks": paid_ticks,
        "sat_out_ticks": sat_out_ticks,
    }


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

    return value if math.isfinite(value) and value > 0 else 0.0


def _legacy_density_pdf(density: dict, target: float) -> float:
    if density.get("model_type") != "gaussian":
        raise ValueError("unsupported density")

    from scipy.stats import norm

    mu = density["model_params"]["mu"]
    sigma = max(0.001, density["model_params"]["sigma"])
    return norm.pdf(target, loc=mu, scale=max(sigma, 1e-8))


def _ensure_density_normalization():
    """densitypdf < 0.1.4 does not normalize the mixture weights: weights summing over 1 would inflate the score."""

    installed = tuple(int(part) for part in version("densitypdf").split(".")[:3])
    if installed < (0, 1, 4):
        raise RuntimeError(f"densitypdf >= 0.1.4 is required, {version('densitypdf')} is installed")


def _load_prediction(
    prediction_directory_path: str,
) -> pandas.DataFrame:
    path = os.path.join(prediction_directory_path, PREDICTION_FILE_NAME)

    with tracer.log("Load the prediction"):
        if not os.path.exists(path):
            raise ParticipantVisibleError(f"No {PREDICTION_FILE_NAME} found")

        return pandas.read_parquet(path, columns=["status", "density", "sent_ns", "received_ns"])


def _load_ticks(
    data_directory_path: str,
) -> numpy.ndarray:
    with tracer.log("Load the ticks"):
        return numpy.fromfile(os.path.join(data_directory_path, TICKS_FILE_NAME), dtype=TICK_DTYPE)


def _load_pot(
    data_directory_path: str,
) -> numpy.ndarray:
    with tracer.log("Load the pot"):
        return numpy.fromfile(os.path.join(data_directory_path, POT_FILE_NAME), dtype=POT_DTYPE)
