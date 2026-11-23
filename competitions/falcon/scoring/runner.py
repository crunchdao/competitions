"""
Falcon (the bird game), replayed from the recorded feed of the live game.

The participant's code defines a `birdgame.trackers.trackerbase.TrackerBase` subclass, as for the live game: the first
one found is created without arguments, then `tick_and_predict(payload, performance_metrics)` is called once per tick,
in feed order, and returns a density (the `densitypdf` format).

The ticks never live in the participant's process: this file's `run()` runs outside the sandbox, holds the feed, and
serves it one tick at a time over a socket; the sandbox (`execute()`, without any access to the data directory) gets the
next tick only once it has answered the current one. The answers and their timing are recorded here, outside the
sandbox, then scored by `scoring.py`.
"""

import inspect
import json
import os
import shutil
import socket
import struct
import tempfile
import threading
import time
import traceback
from typing import TYPE_CHECKING, Any

import numpy
import pyarrow
import pyarrow.parquet

if TYPE_CHECKING:
    from crunch.runner.unstructured import RunnerContext, RunnerExecutorContext, UserModule


# the ticks of the phase (cloud), and the local history (`crunch test` replays its last `LOCAL_TICK_COUNT` ticks)
TICKS_FILE_NAME = "ticks.bin"
HISTORY_FILE_NAME = "history.bin"
LOCAL_TICK_COUNT = 50_000

# one tick, as recorded: little-endian, packed (34 bytes), `time` in seconds since the game started
TICK_DTYPE = numpy.dtype([
    ("time", "<f8"),
    ("falcon_location", "<f8"),
    ("falcon_wingspan", "<f8"),
    ("dove_location", "<f8"),
    ("falcon_id", "<u2"),
])
TICK_STRUCT = struct.Struct("<ddddH")

# the answer to a tick: a status, then the length of the payload (the density as JSON, or the error)
ANSWER_HEADER_STRUCT = struct.Struct("<BI")
STATUS_SUCCESS = 0
STATUS_NONE = 1
STATUS_FAILED = 2

PREDICTION_FILE_NAME = "prediction.parquet"
PREDICTION_BATCH_SIZE = 100_000

TRACKER_BASE_CLASS_NAME = "birdgame.trackers.trackerbase.TrackerBase"

# what the live game passed to a player who had just joined (no score yet): the replay does not score as it goes
PERFORMANCE_METRICS = {
    "wealth": None,
    "recent_likelihood_ewa": None,
    "steady_likelihood_ewa": None,
    "anchor_likelihood_ewa": None,
    "blend_likelihood_ewa": None,
    "likelihood_ewa": None,
}

# the live game gave each model 51 ms per tick; the round trip adds a little
LATE_THRESHOLD_NANOSECONDS = 52_000_000
VERY_LATE_THRESHOLD_NANOSECONDS = 1_000_000_000
LATE_TICKS_PRINTED_ONE_BY_ONE = 20
LATE_TICKS_PRINTED_EVERY = 1_000
PROGRESS_EVERY = 100_000
ERRORS_PRINTED = 5

CONNECTION_TIMEOUT_SECONDS = 600
SOCKET_PARENT_PATH_LIMIT = 70


def load_data(
    data_directory_path: str,
):
    """
    The local history as a DataFrame, one row per tick in feed order: `time` (seconds since the game started, it goes
    back when the game restarts), `falcon_location`, `falcon_wingspan`, `dove_location`, `falcon_id`.
    """

    import pandas

    ticks = numpy.fromfile(os.path.join(data_directory_path, HISTORY_FILE_NAME), dtype=TICK_DTYPE)
    return pandas.DataFrame(ticks)


def run(
    context: "RunnerContext",
    data_directory_path: str,
    prediction_directory_path: str,
):
    if context.is_local:
        ticks = numpy.fromfile(os.path.join(data_directory_path, HISTORY_FILE_NAME), dtype=TICK_DTYPE)[-LOCAL_TICK_COUNT:]
    else:
        ticks = numpy.fromfile(os.path.join(data_directory_path, TICKS_FILE_NAME), dtype=TICK_DTYPE)

    context.log(f"replaying {len(ticks):,} ticks")

    # outside the prediction directory until the end: the sandbox can write there
    work_directory_path = tempfile.mkdtemp(prefix="falcon-")

    # next to the predictions, which the sandbox can reach; a Unix socket path is limited to ~108 bytes
    socket_parent_path = prediction_directory_path if len(os.path.abspath(prediction_directory_path)) < SOCKET_PARENT_PATH_LIMIT else None
    socket_directory_path = tempfile.mkdtemp(prefix="falcon-socket-", dir=socket_parent_path)
    os.chmod(socket_directory_path, 0o755)

    server, address = _listen(socket_directory_path)
    feeder = _Feeder(server, ticks, os.path.join(work_directory_path, PREDICTION_FILE_NAME), context.log)
    thread = threading.Thread(target=feeder.serve, name="falcon-feeder", daemon=True)
    thread.start()

    try:
        context.execute(
            command="replay",
            parameters={"address": address},
            install_data_fuse=False,
        )
    finally:
        server.close()
        thread.join(timeout=CONNECTION_TIMEOUT_SECONDS)
        shutil.rmtree(socket_directory_path, ignore_errors=True)

    if feeder.error is not None:
        raise RuntimeError(f"the replay stopped: {feeder.error}")

    shutil.move(os.path.join(work_directory_path, PREDICTION_FILE_NAME), os.path.join(prediction_directory_path, PREDICTION_FILE_NAME))
    shutil.rmtree(work_directory_path, ignore_errors=True)


def execute(
    context: "RunnerExecutorContext",
    module: "UserModule",
):
    def replay(
        address: str,
    ):
        tracker = _create_tracker(module.module)
        call = _bind_tick_and_predict(tracker)

        connection = _connect(address)
        errors = 0

        try:
            while True:
                record = _receive_exactly(connection, TICK_STRUCT.size)
                if record is None:
                    break  # the end of the feed

                time_value, falcon_location, falcon_wingspan, dove_location, falcon_id = TICK_STRUCT.unpack(record)
                payload = {
                    "time": time_value,
                    "falcon_location": falcon_location,
                    "falcon_wingspan": falcon_wingspan,
                    "dove_location": dove_location,
                    "falcon_id": falcon_id,
                }

                try:
                    density = call(payload)

                    if density is None:
                        status, body = STATUS_NONE, b""
                    else:
                        status, body = STATUS_SUCCESS, json.dumps(density, default=_to_json).encode()
                except Exception:
                    errors += 1
                    if errors <= ERRORS_PRINTED:
                        traceback.print_exc()

                    status, body = STATUS_FAILED, traceback.format_exc(limit=3).encode()[-1000:]

                connection.sendall(ANSWER_HEADER_STRUCT.pack(status, len(body)) + body)
        finally:
            connection.close()

    return {
        "replay": replay,
    }


class _Feeder:
    """Outside the sandbox: sends the ticks one by one, records the answers and their timing."""

    def __init__(self, server: socket.socket, ticks: numpy.ndarray, prediction_file_path: str, log):
        self.server = server
        self.ticks = ticks
        self.prediction_file_path = prediction_file_path
        self.log = log
        self.error: str | None = None

    def serve(self):
        try:
            self.server.settimeout(CONNECTION_TIMEOUT_SECONDS)
            connection, _ = self.server.accept()
            connection.settimeout(None)
        except OSError as error:
            self.error = f"the tracker never connected ({error})"
            return

        try:
            self._feed(connection)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
        finally:
            connection.close()

    def _feed(self, connection: socket.socket):
        records = self.ticks.tobytes()
        record_size = TICK_DTYPE.itemsize
        tick_count = len(self.ticks)

        statuses, densities, sent_times, received_times = [], [], [], []
        late_count = very_late_count = failed_count = none_count = 0
        slowest = 0

        schema = pyarrow.schema([
            ("status", pyarrow.uint8()),
            ("density", pyarrow.string()),
            ("sent_ns", pyarrow.int64()),
            ("received_ns", pyarrow.int64()),
        ])

        with pyarrow.parquet.ParquetWriter(self.prediction_file_path, schema, compression="zstd") as writer:
            def flush():
                if statuses:
                    writer.write_table(pyarrow.table([statuses, densities, sent_times, received_times], schema=schema))
                    for column in (statuses, densities, sent_times, received_times):
                        column.clear()

            for index in range(tick_count):
                record = records[index * record_size:(index + 1) * record_size]

                sent = time.monotonic_ns()
                connection.sendall(record)

                header = _receive_exactly(connection, ANSWER_HEADER_STRUCT.size)
                if header is None:
                    raise RuntimeError(f"the tracker stopped answering at tick {index:,} of {tick_count:,}")

                status, length = ANSWER_HEADER_STRUCT.unpack(header)
                body = _receive_exactly(connection, length) if length else b""
                received = time.monotonic_ns()

                if body is None or status not in (STATUS_SUCCESS, STATUS_NONE, STATUS_FAILED):
                    raise RuntimeError(f"invalid answer at tick {index:,}")

                statuses.append(status)
                densities.append(body.decode() if status == STATUS_SUCCESS else None)
                sent_times.append(sent)
                received_times.append(received)

                duration = received - sent
                slowest = max(slowest, duration)
                if status == STATUS_FAILED:
                    failed_count += 1
                elif status == STATUS_NONE:
                    none_count += 1
                if duration > VERY_LATE_THRESHOLD_NANOSECONDS:
                    very_late_count += 1
                if duration > LATE_THRESHOLD_NANOSECONDS:
                    late_count += 1
                    if late_count <= LATE_TICKS_PRINTED_ONE_BY_ONE or late_count % LATE_TICKS_PRINTED_EVERY == 0:
                        self.log(f"slow tick {index:,}: {duration / 1e6:.1f} ms (over {LATE_THRESHOLD_NANOSECONDS / 1e6:.0f} ms), {late_count:,} slow ticks so far")

                if (index + 1) % PROGRESS_EVERY == 0:
                    self.log(f"{index + 1:,}/{tick_count:,} ticks, {late_count:,} over {LATE_THRESHOLD_NANOSECONDS / 1e6:.0f} ms, {failed_count:,} failed")

                if len(statuses) >= PREDICTION_BATCH_SIZE:
                    flush()

            flush()

        self.log(f"out of {tick_count:,} ticks:")
        self.log(f"- {late_count:,} ticks were over {LATE_THRESHOLD_NANOSECONDS / 1e6:.0f} ms")
        self.log(f"- {very_late_count:,} ticks were over {VERY_LATE_THRESHOLD_NANOSECONDS / 1e9:.0f} s")
        self.log(f"- the slowest took {slowest / 1e6:.1f} ms")
        self.log(f"- {none_count:,} ticks had no density (None)")
        self.log(f"- {failed_count:,} ticks failed")


def _listen(directory_path: str) -> tuple[socket.socket, str]:
    if hasattr(socket, "AF_UNIX"):
        path = os.path.join(directory_path, "feed.sock")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(path)
        os.chmod(path, 0o777)  # the sandbox runs as another user
        server.listen(1)
        return server, f"unix:{path}"

    # Windows, locally only (the cloud sandbox allows Unix sockets only)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    return server, f"tcp:127.0.0.1:{server.getsockname()[1]}"


def _connect(address: str) -> socket.socket:
    kind, _, target = address.partition(":")
    if kind == "unix":
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.connect(target)
    else:
        host, _, port = target.rpartition(":")
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection.connect((host, int(port)))

    return connection


def _receive_exactly(connection: socket.socket, size: int) -> bytes | None:
    buffer = bytearray()
    while len(buffer) < size:
        chunk = connection.recv(size - len(buffer))
        if not chunk:
            return None
        buffer += chunk

    return bytes(buffer)


def _create_tracker(user_module) -> Any:
    """As the live game's model runner: the first `TrackerBase` subclass of the participant's code, without arguments."""

    from birdgame.trackers.trackerbase import TrackerBase

    for _, candidate in inspect.getmembers(user_module, inspect.isclass):
        if candidate is TrackerBase or getattr(candidate, "__module__", "").split(".")[0] == "birdgame":
            continue

        if issubclass(candidate, TrackerBase) and not inspect.isabstract(candidate):
            return candidate()

    raise ValueError(f"no subclass of `{TRACKER_BASE_CLASS_NAME}` found in your code")


def _bind_tick_and_predict(tracker: Any):
    """As the live game's model runner: `performance_metrics` only if `tick_and_predict` accepts it."""

    method = tracker.tick_and_predict

    if "performance_metrics" in inspect.signature(method).parameters:
        return lambda payload: method(payload, performance_metrics=dict(PERFORMANCE_METRICS))

    return lambda payload: method(payload)


def _to_json(value: Any):
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, numpy.ndarray):
        return value.tolist()

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
