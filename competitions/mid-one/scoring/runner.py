"""
Mid+One (a STREAM competition), as crunch-cli 8.0.0 ran it.

`train(streams)` is called once, when the submission has no model (or the first train is forced): one iterable per
training stream part (each stream is cut at the missing values), each message a `StreamMessage` (`message["x"]`).

Then `infer(stream)` is called once per part of each test stream: a generator that must `yield` once (None) when
ready, then exactly once per message it takes from `stream`, the position to take (its sign: buy, sell, or 0 to
hold).

The test messages never live in the participant's process: this file's `run()` runs outside the sandbox, holds them,
and serves them one at a time over a socket; the sandbox (`execute()`, without any access to the data directory during
the inference) gets a message only when its generator asks for the next one, after it answered the previous one. The
answers and their timing are recorded here, outside the sandbox, then scored by `scoring.py`.
"""

import gc
import os
import shutil
import socket
import struct
import tempfile
import threading
import time
import types
from typing import TYPE_CHECKING, Any, Callable, Iterator, List, Tuple

import numpy
import pandas
import pyarrow
import pyarrow.parquet

if TYPE_CHECKING:
    from crunch.runner.unstructured import RunnerContext, RunnerExecutorContext, UserModule


EMBARGO = 0
NUMBER_OF_FEATURES = 1

MOON_COLUMN_NAME = "time"
ID_COLUMN_NAME = "stream"
SIDE_COLUMN_NAME = "value"
PREDICTION_COLUMN_NAME = "prediction"
DURATION_COLUMN_NAME = "duration_ns"

X_TRAIN_FILE_NAME = "X_train.parquet"
LOCAL_X_TEST_FILE_NAME = "X_test_reduced.parquet"  # locally, only the public test moons are available
CLOUD_X_TEST_FILE_NAME = "X_test.parquet"
PREDICTION_FILE_NAME = "prediction.parquet"

# set by the platform on the runner, "true" or "false"
GPU_ENVIRONMENT_VARIABLE_NAME = "GPU"

# every packet, both ways: a kind, an integer, a float (little-endian, packed: 13 bytes)
PACKET_STRUCT = struct.Struct("<Bqd")

# from the runner: a part of a stream starts (integer: the stream's index, float: the number of messages), a message
# (float: `x`), the part has no more message, every part was served
KIND_PART_BEGIN = 1
KIND_MESSAGE = 2
KIND_PART_END = 3
KIND_FEED_END = 4

# from the sandbox: the generator takes the next message, its answer (float: the position), the part is done
KIND_PULL = 11
KIND_ANSWER = 12
KIND_PART_DONE = 13

# crunch-cli 8.0.0's `GeneratorWrapper` errors, unchanged
ERROR_YIELD_MUST_BE_CALLED_BEFORE = "yield must be called once before the loop"
ERROR_PREVIOUS_VALUE_NOT_YIELD = "previous value not yield-ed"
ERROR_YIELD_NOT_CALLED = "yield not called"
ERROR_FIRST_YIELD_MUST_BE_NONE = "first yield must return None"
ERROR_MULTIPLE_YIELD = "multiple yield detected"
ERROR_WRONG_YIELD_CALL_COUNT_PREFIX = "yield not called enough time"

SLOW_MESSAGE_THRESHOLD_NANOSECONDS = 1_000_000_000
CONNECTION_TIMEOUT_SECONDS = 600
SOCKET_PARENT_PATH_LIMIT = 70


class StreamMessage:
    """crunch-cli 8.0.0's `crunch.StreamMessage`: an immutable message, read as `message.x` or `message["x"]`."""

    x: float

    def __init__(self, x: float):
        object.__setattr__(self, "x", x)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, _):
        raise AttributeError(f"Cannot set key '{key}' - object is immutable")

    def __setattr__(self, key, _):
        raise AttributeError(f"Cannot set attribute '{key}' - object is immutable")

    def __iter__(self):
        return iter(vars(self))

    def __repr__(self):
        return "{" + ", ".join(f"{key}: {value!r}" for key, value in vars(self).items()) + "}"

    def __str__(self):
        return f"{self.__class__.__name__}({self.__repr__()})"

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def keys(self):
        return vars(self).keys()

    def values(self):
        return vars(self).values()

    def items(self):
        return vars(self).items()


class StreamIterable:
    """crunch-cli 8.0.0's `CallableIterable`: can be iterated again and again, and has a length."""

    def __init__(self, values: numpy.ndarray):
        self._values = values

    def __iter__(self) -> Iterator[StreamMessage]:
        return map(StreamMessage, self._values.tolist())

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return f"iterable[{len(self._values)}]"

    def __str__(self):
        return self.__repr__()


def load_data(
    data_directory_path: str,
):
    """
    The training streams and the local test streams, as `crunch.load_streams()` gave them: one iterable of messages
    per part of each stream.
    """

    x_train = pandas.read_parquet(os.path.join(data_directory_path, X_TRAIN_FILE_NAME))
    x_test = pandas.read_parquet(os.path.join(data_directory_path, LOCAL_X_TEST_FILE_NAME))

    return (
        _to_stream_iterables(x_train),
        _to_stream_iterables(x_test),
    )


def run(
    context: "RunnerContext",
    data_directory_path: str,
    prediction_directory_path: str,
):
    if not context.has_model or context.force_first_train:
        context.log("training")
        context.execute(
            command="train",
        )

    x_test_file_name = LOCAL_X_TEST_FILE_NAME if context.is_local else CLOUD_X_TEST_FILE_NAME
    x_test = pandas.read_parquet(os.path.join(data_directory_path, x_test_file_name), columns=[MOON_COLUMN_NAME, ID_COLUMN_NAME, SIDE_COLUMN_NAME])
    feed = _Feed.from_dataframe(x_test)
    del x_test

    context.log(f"replaying: {feed.message_count:,} messages of {len(feed.stream_names)} streams in {len(feed.parts)} parts")

    # outside the prediction directory until the end: the sandbox can write there
    work_directory_path = tempfile.mkdtemp(prefix="mid-one-")

    # next to the predictions, which the sandbox can reach; a Unix socket path is limited to ~108 bytes
    socket_parent_path = prediction_directory_path if len(os.path.abspath(prediction_directory_path)) < SOCKET_PARENT_PATH_LIMIT else None
    socket_directory_path = tempfile.mkdtemp(prefix="mid-one-socket-", dir=socket_parent_path)
    os.chmod(socket_directory_path, 0o755)

    server, address = _listen(socket_directory_path)
    feeder = _Feeder(server, feed, os.path.join(work_directory_path, PREDICTION_FILE_NAME), context.log)
    thread = threading.Thread(target=feeder.serve, name="mid-one-feeder", daemon=True)
    thread.start()

    try:
        context.execute(
            command="infer",
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
    data_directory_path: str,
    model_directory_path: str,
):
    _install_stream_message()

    has_gpu = os.environ.get(GPU_ENVIRONMENT_VARIABLE_NAME, "false").lower() == "true"
    default_values = {
        "number_of_features": NUMBER_OF_FEATURES,
        "model_directory_path": model_directory_path,
        "embargo": EMBARGO,
        "has_gpu": has_gpu,
    }

    def train():
        from crunch.utils import smart_call

        x_train = pandas.read_parquet(os.path.join(data_directory_path, X_TRAIN_FILE_NAME), columns=[MOON_COLUMN_NAME, ID_COLUMN_NAME, SIDE_COLUMN_NAME])

        context.trip_data_fuse()

        streams = _to_stream_iterables(x_train)
        del x_train
        gc.collect()

        smart_call(
            module.get_function("train"),
            default_values,
            {
                "streams": streams,
            },
        )

    def infer(
        address: str,
    ):
        from crunch.utils import smart_call

        infer_function = module.get_function("infer")
        connection = _connect(address)

        try:
            while True:
                kind, stream_index, length = _receive_packet(connection)
                if kind == KIND_FEED_END:
                    break

                if kind != KIND_PART_BEGIN:
                    raise RuntimeError(f"unexpected packet from the runner: {kind}")

                _infer_part(
                    connection,
                    lambda stream: smart_call(
                        infer_function,
                        default_values,
                        {
                            "stream": stream,
                        },
                    ),
                    int(length),
                )

                connection.sendall(PACKET_STRUCT.pack(KIND_PART_DONE, stream_index, 0.0))
        finally:
            connection.close()

    return {
        "train": train,
        "infer": infer,
    }


def _infer_part(
    connection: socket.socket,
    consumer_factory: Callable[[Iterator[StreamMessage]], Any],
    expected_size: int,
):
    """crunch-cli 8.0.0's `GeneratorWrapper`, with the messages taken from the runner one at a time."""

    state = types.SimpleNamespace(ready=None, consumed=True)

    def inner() -> Iterator[StreamMessage]:
        while True:
            if state.ready is None:
                raise RuntimeError(ERROR_YIELD_MUST_BE_CALLED_BEFORE)

            if not state.ready:
                raise RuntimeError(ERROR_PREVIOUS_VALUE_NOT_YIELD)

            connection.sendall(PACKET_STRUCT.pack(KIND_PULL, 0, 0.0))
            kind, _, value = _receive_packet(connection)

            if kind == KIND_PART_END:
                return

            if kind != KIND_MESSAGE:
                raise RuntimeError(f"unexpected packet from the runner: {kind}")

            state.ready = False
            state.consumed = False

            yield StreamMessage(value)

    consumer = consumer_factory(inner())

    if not isinstance(consumer, types.GeneratorType):
        raise RuntimeError(ERROR_YIELD_NOT_CALLED)

    if next(consumer) is not None:
        raise ValueError(ERROR_FIRST_YIELD_MUST_BE_NONE)

    state.ready = True

    size = 0
    for value in consumer:
        state.ready = True

        if state.consumed:
            raise RuntimeError(ERROR_MULTIPLE_YIELD)

        state.consumed = True

        connection.sendall(PACKET_STRUCT.pack(KIND_ANSWER, 0, _to_position(value)))
        size += 1

    if size != expected_size:
        raise ValueError(f"{ERROR_WRONG_YIELD_CALL_COUNT_PREFIX} ({size} / {expected_size})")


class _Feed:
    """The test messages, cut as crunch-cli 8.0.0 did: per stream, then per part (at the missing values)."""

    def __init__(self, stream_names: List[str], parts: List[Tuple[int, numpy.ndarray, numpy.ndarray]]):
        self.stream_names = stream_names
        self.parts = parts  # (stream index, moons, values)
        self.message_count = sum(len(values) for _, _, values in parts)

    @staticmethod
    def from_dataframe(x_test: pandas.DataFrame) -> "_Feed":
        stream_names = _sorted_stream_names(x_test[ID_COLUMN_NAME].unique())

        parts = []
        for stream_index, stream_name in enumerate(stream_names):
            group = x_test[x_test[ID_COLUMN_NAME] == stream_name]
            for moons, values in _split_at_nans(group[MOON_COLUMN_NAME].to_numpy(), group[SIDE_COLUMN_NAME].to_numpy(dtype=numpy.float64)):
                parts.append((stream_index, moons, values))

        return _Feed(stream_names, parts)


class _Feeder:
    """Outside the sandbox: serves the messages one by one, records the answers and their timing."""

    def __init__(self, server: socket.socket, feed: _Feed, prediction_file_path: str, log):
        self.server = server
        self.feed = feed
        self.prediction_file_path = prediction_file_path
        self.log = log
        self.error: str | None = None

    def serve(self):
        try:
            self.server.settimeout(CONNECTION_TIMEOUT_SECONDS)
            connection, _ = self.server.accept()
            connection.settimeout(None)
            if connection.family == socket.AF_INET:
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError as error:
            self.error = f"the participant's code never connected ({error})"
            return

        try:
            self._feed(connection)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
        finally:
            connection.close()

    def _feed(self, connection: socket.socket):
        feed = self.feed
        moons = numpy.empty(feed.message_count, dtype=numpy.int64)
        stream_indexes = numpy.empty(feed.message_count, dtype=numpy.int64)
        positions = numpy.empty(feed.message_count, dtype=numpy.float64)
        durations = numpy.empty(feed.message_count, dtype=numpy.int64)

        offset = 0
        slow_count = 0
        current_stream_index = -1

        for stream_index, part_moons, part_values in feed.parts:
            if stream_index != current_stream_index:
                current_stream_index = stream_index
                self.log(f"replaying: {feed.stream_names[stream_index]} ({stream_index + 1}/{len(feed.stream_names)}), {offset:,}/{feed.message_count:,} messages so far")

            length = len(part_values)
            values = part_values.tolist()
            connection.sendall(PACKET_STRUCT.pack(KIND_PART_BEGIN, stream_index, float(length)))

            delivered = answered = 0
            sent = 0
            while True:
                kind, _, value = _receive_packet(connection, "the participant's code stopped answering")

                if kind == KIND_PULL:
                    if delivered != answered:
                        raise RuntimeError(ERROR_PREVIOUS_VALUE_NOT_YIELD)

                    if delivered == length:
                        connection.sendall(PACKET_STRUCT.pack(KIND_PART_END, 0, 0.0))
                        continue

                    sent = time.monotonic_ns()
                    connection.sendall(PACKET_STRUCT.pack(KIND_MESSAGE, 0, values[delivered]))
                    delivered += 1

                elif kind == KIND_ANSWER:
                    received = time.monotonic_ns()
                    if answered != delivered - 1:
                        raise RuntimeError(ERROR_MULTIPLE_YIELD)

                    index = offset + answered
                    positions[index] = value
                    durations[index] = received - sent
                    if received - sent > SLOW_MESSAGE_THRESHOLD_NANOSECONDS:
                        slow_count += 1
                    answered += 1

                elif kind == KIND_PART_DONE:
                    if answered != length:
                        raise RuntimeError(f"{ERROR_WRONG_YIELD_CALL_COUNT_PREFIX} ({answered} / {length})")
                    break

                else:
                    raise RuntimeError(f"unexpected packet from the participant's code: {kind}")

            moons[offset:offset + length] = part_moons
            stream_indexes[offset:offset + length] = stream_index
            offset += length

        connection.sendall(PACKET_STRUCT.pack(KIND_FEED_END, 0, 0.0))

        stream_names = pyarrow.array(feed.stream_names, type=pyarrow.string())
        table = pyarrow.table({
            MOON_COLUMN_NAME: pyarrow.array(moons),
            ID_COLUMN_NAME: pyarrow.DictionaryArray.from_arrays(pyarrow.array(stream_indexes.astype(numpy.int32)), stream_names),
            PREDICTION_COLUMN_NAME: pyarrow.array(positions),
            DURATION_COLUMN_NAME: pyarrow.array(durations),
        })
        pyarrow.parquet.write_table(table, self.prediction_file_path, compression="zstd")

        self.log(f"result: out of {feed.message_count:,} messages:")
        self.log(f"result: - the mean answer took {durations.mean() / 1e3:.1f} us, the slowest {durations.max() / 1e6:.1f} ms")
        self.log(f"result: - {slow_count:,} answers took over {SLOW_MESSAGE_THRESHOLD_NANOSECONDS / 1e9:.0f} s")
        self.log(f"result: - positions: {int((positions > 0).sum()):,} buy, {int((positions < 0).sum()):,} sell, {int((positions == 0).sum()):,} hold")


def _to_stream_iterables(dataframe: pandas.DataFrame) -> List[StreamIterable]:
    """crunch-cli 8.0.0's `split_streams`: per stream id, the parts between the missing values."""

    streams = []
    for stream_name in _sorted_stream_names(dataframe[ID_COLUMN_NAME].unique()):
        group = dataframe[dataframe[ID_COLUMN_NAME] == stream_name]
        for _, values in _split_at_nans(group[MOON_COLUMN_NAME].to_numpy(), group[SIDE_COLUMN_NAME].to_numpy(dtype=numpy.float64)):
            streams.append(StreamIterable(values))

    return streams


def _split_at_nans(moons: numpy.ndarray, values: numpy.ndarray) -> List[Tuple[numpy.ndarray, numpy.ndarray]]:
    """crunch-cli 8.0.0's `split_at_nans`: the non-empty runs between the missing values."""

    nan_indexes = numpy.flatnonzero(numpy.isnan(values))
    bounds = [-1, *nan_indexes.tolist(), len(values)]

    return [
        (moons[start + 1:end], values[start + 1:end])
        for start, end in zip(bounds[:-1], bounds[1:])
        if end > start + 1
    ]


def _sorted_stream_names(names) -> List[str]:
    """`stream_0`, `stream_1`, ... in number order (the order of the competition's targets)."""

    def key(name: str):
        prefix, _, number = name.rpartition("_")
        return (prefix, int(number)) if number.isdigit() else (name, -1)

    return sorted((str(name) for name in names), key=key)


def _to_position(value: Any) -> float:
    if isinstance(value, (bool, numpy.bool_)):
        return float(value)

    try:
        position = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"infer() must yield a number for each message (its sign is the position), got: {value!r}") from None

    if numpy.isnan(position):
        raise ValueError(f"infer() must yield a number for each message (its sign is the position), got: {value!r}")

    return position


def _install_stream_message():
    """crunch-cli no longer has `crunch.StreamMessage`: the old code importing it gets this file's."""

    import crunch

    if not hasattr(crunch, "StreamMessage"):
        crunch.StreamMessage = StreamMessage


def _listen(directory_path: str) -> Tuple[socket.socket, str]:
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


def _receive_packet(connection: socket.socket, closed_message: str = "the runner closed the connection") -> Tuple[int, int, float]:
    buffer = bytearray()
    while len(buffer) < PACKET_STRUCT.size:
        chunk = connection.recv(PACKET_STRUCT.size - len(buffer))
        if not chunk:
            raise ConnectionError(closed_message)
        buffer += chunk

    return PACKET_STRUCT.unpack(bytes(buffer))
