import os
from dataclasses import dataclass
from enum import Enum, IntEnum
from socket import IPPROTO_TCP, SOCK_STREAM, TCP_NODELAY, socket
from threading import Thread
from types import GeneratorType
from typing import TYPE_CHECKING, Any, Callable, Generator, Generic, Iterator, Literal, Optional, Tuple, TypeVar, Union, cast
from uuid import uuid4

import numpy
import pandas
from crunch.utils import smart_call

if TYPE_CHECKING:
    from crunch.runner.unstructured import RunnerContext, RunnerExecutorContext, UserModule


T = TypeVar("T")


@dataclass
class Box(Generic[T]):
    value: T


_NAN_BYTES = numpy.float32(float("nan")).tobytes()

# the Cloud environment isn't on Windows
if os.name == "nt":
    from socket import AF_INET

    ServerEndpointType = int

    def os_register_at_fork(*, after_in_child: Optional[Callable] = None, **kwargs):
        pass

    def os_create_server_socket() -> Tuple[socket, ServerEndpointType]:
        server = socket(AF_INET, SOCK_STREAM)
        server.bind(("localhost", 0))
        server.listen(1)

        return server, server.getsockname()[1]

    def os_accept(server: socket):
        client, addr = server.accept()
        client.setsockopt(IPPROTO_TCP, TCP_NODELAY, 1)
        return client, addr

    def os_create_client_socket():
        return socket(AF_INET, SOCK_STREAM)

    def os_connect(client: socket, server_port: int):
        client.connect(("localhost", server_port))
        client.setsockopt(IPPROTO_TCP, TCP_NODELAY, 1)

else:
    from socket import AF_UNIX

    ServerEndpointType = str

    os_register_at_fork = os.register_at_fork

    def os_create_server_socket() -> Tuple[socket, ServerEndpointType]:
        from socket import AF_UNIX

        SOCKET_PATH = f"/tmp/runner.{uuid4()}.sock"
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)

        server = socket(AF_UNIX, SOCK_STREAM)
        server.bind(SOCKET_PATH)
        server.listen(1)

        os.chmod(SOCKET_PATH, 0o666)

        return server, SOCKET_PATH

    def os_create_client_socket():
        return socket(AF_UNIX, SOCK_STREAM)

    def os_connect(client: socket, socket_path: str):
        client.connect(socket_path)

    def os_accept(server: socket):
        return server.accept()


def load_data(
    data_directory_path: str,
):
    train = _load_train(data_directory_path)

    if True:
        x_test_name = "X_test.reduced.parquet"
        x_test = pandas.read_parquet(os.path.join(data_directory_path, x_test_name))

        test = []
        for _, dataset in x_test.groupby(x_test.index.get_level_values("id")):
            historical, online = _split_periods(dataset, online_values=True)
            test.append((historical, online))

    return train, test


def run(
    context: "RunnerContext",
    data_directory_path: str,
    prediction_directory_path: str,
):
    if context.force_first_train:
        context.execute(
            command="train",
        )

    prediction = run_infer_with_server(context, data_directory_path, determinism_check=False)

    if context.is_determinism_check_enabled:
        percentage = 0.3
        tolerance = 1e-8

        context.log(f"checking determinism by executing the inference again with {percentage * 100:.0f}% of the data (tolerance: {tolerance})")

        prediction2 = run_infer_with_server(context, data_directory_path, determinism_check=percentage)

        is_deterministic = numpy.allclose(prediction.loc[prediction2.index, "prediction"], prediction2["prediction"], atol=tolerance)
        context.report_determinism(is_deterministic)

    prediction_file_path = os.path.join(prediction_directory_path, "prediction.parquet")
    prediction.to_parquet(prediction_file_path)


def execute(
    context: "RunnerExecutorContext",
    module: "UserModule",
    data_directory_path: str,
    model_directory_path: str,
):
    default_values = {
        "data_directory_path": data_directory_path,
        "model_directory_path": model_directory_path,
    }

    def train():
        datasets = _load_train(data_directory_path)

        context.trip_data_fuse()

        train_function = module.get_function("train")

        smart_call(
            train_function,
            default_values,
            {
                "datasets": datasets,
            }
        )

    def infer(
        determinism_check: Union[Literal[False], float],  # just for logging
        server_endpoint: ServerEndpointType,
    ):
        context.trip_data_fuse()

        with os_create_client_socket() as client:
            os_connect(client, server_endpoint)
            remote = Remote(client)

            infer_function = module.get_function("infer")

            _run_with_double_protection(
                remote=remote,
                consumer_factory=lambda datasets: smart_call(
                    infer_function,
                    default_values,
                    {
                        "datasets": datasets,
                    },
                ),
                post_processor=_post_process_infer_yield_result,
            )

    return {
        "train": train,
        "infer": infer,
    }


def _load_train(data_directory_path: str):
    x_train = pandas.read_parquet(os.path.join(data_directory_path, "X_train.parquet"))
    y_train_index = pandas.read_parquet(os.path.join(data_directory_path, "y_train_index.parquet"))

    datasets = []
    for id, dataset in x_train.groupby(x_train.index.get_level_values("id")):
        historical, online = _split_periods(dataset, online_values=True)

        tau_index = y_train_index.loc[id, "tau_index"]
        if tau_index == -1:
            tau_index = None

        datasets.append((id, historical, online, tau_index))

    return datasets


class Remote:

    def __init__(self, client: socket):
        self._client = client

        self._send_wasted_time_ns = 0.0
        self._receive_wasted_time_ns = 0.0

    def receive(self) -> Tuple["RemoteCommand", Union[float, int, None]]:
        data = self.receive_raw(1 + 4)
        if not data:
            raise RemoteError("connection closed by client")

        return RemoteCommand.decode(data)

    def receive_raw(self, length: int) -> bytes:
        data = bytearray()
        while len(data) < length:
            packet = self._client.recv(length - len(data))
            if not packet:
                raise RemoteError("connection closed by client")
            data.extend(packet)

        return bytes(data)

    def send(self, command: "RemoteCommand", value: Union[float, int] = float("nan")):
        data = command.encode(value)
        self._client.send(data)

    def send_raw(self, data: bytes):
        self._client.send(data)

    def expect(self, expected: "RemoteCommand") -> Union[float, int, None]:
        command, value = self.receive()

        if command != expected:
            raise RemoteError(f"expected command {expected.name} but got {command.name}")

        return value

    def expect2(self, expected: "RemoteCommand", permitted: "RemoteCommand") -> Tuple["RemoteCommand", Union[float, int, None]]:
        command, value = self.receive()

        if command != expected and command != permitted:
            raise RemoteError(f"expected command {expected.name} or {permitted.name} but got {command.name}")

        return command, value


class RemoteError(RuntimeError):
    pass


class RemoteCommand(IntEnum):
    READY = 1
    NEW_TIMESERIES = 2
    HISTORICAL_DATA = 4
    ONLINE_POINT = 5
    END_TIMESERIES = 6
    END = 7
    NEW_PREDICTION = 100

    def encode(self, value: Union[float, int]) -> bytes:
        data = bytearray(1 + 4)
        data[0] = self.value

        if self == RemoteCommand.HISTORICAL_DATA:
            data[1:5] = numpy.int32(value).tobytes()
        elif self == RemoteCommand.ONLINE_POINT or self == RemoteCommand.NEW_PREDICTION:
            data[1:5] = numpy.float32(value).tobytes()
        else:
            data[1:5] = _NAN_BYTES

        return bytes(data)

    @staticmethod
    def decode(data: bytes) -> Tuple["RemoteCommand", Optional[float]]:
        command = RemoteCommand(data[0])

        if command == RemoteCommand.HISTORICAL_DATA:
            value = numpy.frombuffer(data[1:5], dtype=numpy.int32)[0]
        elif command == RemoteCommand.ONLINE_POINT or command == RemoteCommand.NEW_PREDICTION:
            value = numpy.frombuffer(data[1:5], dtype=numpy.float32)[0]
        else:
            value = None

        return command, value


def run_infer_with_server(
    context: "RunnerContext",
    data_directory_path: str,
    determinism_check: Union[Literal[False], float],
) -> pandas.DataFrame:
    x_test_name = "X_test.reduced.parquet" if context.is_local else "X_test.parquet"
    x_test = pandas.read_parquet(os.path.join(data_directory_path, x_test_name))

    if determinism_check is not False:
        ids = sorted(set(x_test.index.get_level_values("id")))
        determinism_slice = slice(None, int(len(ids) * determinism_check))
        x_test = x_test.loc[x_test.index.get_level_values("id").isin(ids[determinism_slice])]

    server, endpoint = os_create_server_socket()

    predicted_values = []
    prediction_index = []

    def run():
        client, _ = os_accept(server)
        with client:
            server.close()

            remote = Remote(client)

            remote.expect(RemoteCommand.READY)

            for _, dataset in x_test.groupby(x_test.index.get_level_values("id")):
                remote.send(RemoteCommand.NEW_TIMESERIES)

                historical, online = _split_periods(dataset, online_values=False)
                remote.send(RemoteCommand.HISTORICAL_DATA, len(historical))
                remote.send_raw(historical.astype(numpy.float32).tobytes())  # pyright: ignore[reportAttributeAccessIssue]

                for index, value in online.items():  # pyright: ignore[reportAttributeAccessIssue]
                    remote.send(RemoteCommand.ONLINE_POINT, value)

                    prediction = remote.expect(RemoteCommand.NEW_PREDICTION)
                    predicted_values.append(prediction)
                    prediction_index.append(index)

                remote.send(RemoteCommand.END_TIMESERIES)

            remote.send(RemoteCommand.END)

    def run_with_catch():
        try:
            run()
        except Exception as error:
            context.log(f"error in data thread: {error}")

    thread = Thread(target=run_with_catch, daemon=True)
    thread.start()

    context.execute(
        command="infer",
        parameters={
            "determinism_check": determinism_check,
            "server_endpoint": endpoint,
        }
    )

    thread.join()

    return pandas.DataFrame(
        data={
            "prediction": predicted_values,
        },
        index=pandas.MultiIndex.from_tuples(
            prediction_index,
            names=["id", "time"],
        ),
    )


class ProtocolError(RuntimeError):
    pass


class NestedForkError(RuntimeError):
    pass


def _run_with_double_protection(
    *,
    remote: "Remote",
    consumer_factory: Callable[[Iterator], Generator],
    post_processor: Callable[[Any], Any],
):
    """
    Run the user's code with a double protection:
     - can only read a single dataset at a time
     - can only read a single value at a time
    """

    class State(Enum):
        BEFORE_FIRST_YIELD = 0
        READY_FOR_DATASET = 1
        READY_FOR_VALUE = 2
        CONSUMED = 3
        NESTED_FORK = 4

    sentinel = object()

    state = State.BEFORE_FIRST_YIELD
    next_value = None

    # 4: provide the datasets
    def dataset_stream():
        nonlocal state, next_value

        while next_value is not None:
            # 5.1: ensure not iterating before first yield
            if state == State.BEFORE_FIRST_YIELD:
                raise ProtocolError("yield must be called once before iterating over datasets")

            # 5.2: ensure either first dataset or ended previous dataset
            if state != State.READY_FOR_DATASET:
                raise ProtocolError("previous dataset not yield-ed properly")

            # 5.3: allow reading a value next
            state = State.READY_FOR_VALUE

            yield next_value

    # 3.1: provide the values
    def online_stream(next_online_value: "Box"):
        nonlocal state

        while next_online_value.value is not None:
            # X: ensure not forked
            if state == State.NESTED_FORK:
                raise NestedForkError("cannot iterate in online stream in a child process after a fork")

            # 6: ensure not iterating a dataset
            if state != State.READY_FOR_VALUE:
                raise ProtocolError(f"previous value not yield-ed")

            # 6.1: mark value as consumed
            state = State.CONSUMED

            yield next_online_value.value

    # X: triggered in the child process after fork, prevent iterations
    def _after_in_child():
        nonlocal state, next_value

        state = State.NESTED_FORK
        next_value = None

    # X: register security measure
    os_register_at_fork(after_in_child=_after_in_child)

    if True:
        # 1: initialize user code
        user_code = consumer_factory(dataset_stream())

        # 1.1: ensure a yield has been used
        if not isinstance(user_code, GeneratorType):
            raise ProtocolError("yield not called")

        # 1.2: detect the first yield is indeed None
        if next(user_code) is not None:
            raise ProtocolError(f"first yield must return None")

        if state == State.NESTED_FORK:
            raise NestedForkError(f"cannot do first yield in a child process after a fork ({os.getpid()})")

        # 1.3: send server ready signal
        remote.send(RemoteCommand.READY)

    while True:
        command, _ = remote.expect2(RemoteCommand.NEW_TIMESERIES, RemoteCommand.END)
        if command == RemoteCommand.END:
            break

        # X: ensure not forked
        if state == State.NESTED_FORK:
            raise NestedForkError(f"cannot iterate in datasets in a child process after a fork ({os.getpid()})")

        # 2: allow reading a dataset next
        state = State.READY_FOR_DATASET

        # 2.1: fetch the historical data
        historical_length = cast(int, remote.expect(RemoteCommand.HISTORICAL_DATA))
        historical_data = remote.receive_raw(historical_length * 4)
        historical = numpy.frombuffer(historical_data, dtype=numpy.float32)
        del historical_data

        # 2.2: prepare the online stream
        next_online_value: Box[Optional[float]] = Box(None)
        online_generator = online_stream(next_online_value)

        # 2.3: prepare the next value of the dataset stream
        next_value = (historical, online_generator)

        while True:
            # 3: get next command
            command, next_online_value.value = remote.expect2(RemoteCommand.ONLINE_POINT, RemoteCommand.END_TIMESERIES)
            if command == RemoteCommand.END_TIMESERIES:
                break

            # 4: re-activate user code
            y = next(user_code, sentinel)
            if y is sentinel:
                raise ProtocolError("yield not called enough times")

            y = post_processor(y)

            # X: ensure not forked
            if state == State.NESTED_FORK:
                raise NestedForkError("cannot yield in a child process after a fork")

            # 7: ensure state has been reset by iterating in online stream
            if state != State.CONSUMED:
                raise ProtocolError("multiple yield detected")

            # 7.1: reset state for next value
            state = State.READY_FOR_VALUE

            if next_online_value.value is None:
                break

            remote.send(RemoteCommand.NEW_PREDICTION, y)

        # 8: clean up for next dataset
        next_value = None

    # 9: ensure user clean up is called
    if True:
        last_check = next(user_code, sentinel)
        if last_check is not sentinel:
            raise ProtocolError("code did not exit after last dataset")


def _split_periods(
    dataset: pandas.DataFrame,
    online_values: bool = False,
):
    historical = dataset[dataset["period"] == 1]["value"].values
    online = dataset[dataset["period"] == 2]["value"]

    if online_values:
        online = online.values

    return historical, online


def _post_process_infer_yield_result(result: Any) -> Any:
    if isinstance(result, pandas.Series):
        if len(result) != 1:
            raise ValueError(f"a `pandas.Series` is only allowed if it has a single value, but got a length of {len(result)}")

        result = next(iter(result))

    elif isinstance(result, numpy.ndarray):
        if result.shape != (1,):
            raise ValueError(f"a `numpy.ndarray` is only allowed if it has a single dimension and a single value, but got a shape of {result.shape}")

        result = result[0]

    elif isinstance(result, list):
        if len(result) != 1:
            raise ValueError(f"a `list` is only allowed if it has a single value, but got a length of {len(result)}")

        result = result[0]

    if all(not numpy.issubdtype(type(result), dtype) for dtype in [numpy.floating, numpy.integer, numpy.bool_]):
        raise ValueError(f"value must be a float or an int or a bool, but got {type(result)}")

    else:
        result = float(result)

    return result
