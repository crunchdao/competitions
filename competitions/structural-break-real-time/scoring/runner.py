import multiprocessing
import os
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from enum import Enum, IntEnum
from socket import IPPROTO_TCP, SOCK_STREAM, TCP_NODELAY, socket
from tempfile import NamedTemporaryFile
from threading import Thread
from types import GeneratorType
from typing import TYPE_CHECKING, Any, Callable, Generator, Generic, Iterator, List, Literal, Optional, Tuple, TypeVar, Union, cast
from uuid import uuid4

import numpy
import pandas
from crunch.runner.tracing import RunnerTracer, to_execute_span_attributes
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
    is_parallelism_supported = False

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
    is_parallelism_supported = True

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
    tracer: "RunnerTracer",
    data_directory_path: str,
    prediction_directory_path: str,
):
    if context.force_first_train:
        context.execute(
            command="train",
        )

    with named_temp_file() as temp_file:
        os.chmod(temp_file.name, 0o666)

        context.execute(
            command="get_parallelism",
            parameters={
                "output_file_path": temp_file.name,
            },
            install_data_fuse=False,
        )

        with open(temp_file.name, "r") as fd:
            parallelism = int(fd.read())

        context.log(f"using a parallelism of {parallelism}")

    prediction = _run_infer(context, tracer, data_directory_path, parallelism, determinism_check=False)

    if context.is_determinism_check_enabled:
        percentage = 0.3
        tolerance = 1e-8

        context.log(f"checking determinism by executing the inference again with {percentage * 100:.0f}% of the data (tolerance: {tolerance})")

        prediction2 = _run_infer(context, tracer, data_directory_path, parallelism, determinism_check=percentage)

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

    def get_parallelism(
        output_file_path: str,
    ):
        def get_value() -> int:
            if not is_parallelism_supported:
                return 1

            key = "INFER_PARALLELISM"
            value = module.get_value(key, default=None)

            if value is None:
                print(f"[parallelism] `{key}` not set, not using parallelism", file=sys.stderr)
                return 1

            if not isinstance(value, int):
                print(f"[parallelism] `{key}` must be an int", file=sys.stderr)
                return 1

            cpu_count = os.cpu_count() or 1
            if value > cpu_count:
                print(f"[parallelism] `{key}` must be at most the number of CPUs ({cpu_count})", file=sys.stderr)
                value = cpu_count
            elif value == 0:
                print(f"[parallelism] using all available CPUs for inference", file=sys.stderr)
                value = cpu_count
            elif value < 1:
                print(f"[parallelism] `{key}` must be at least 1", file=sys.stderr)
                value = 1

            return value

        with open(output_file_path, "w") as output_file:
            value = get_value()
            output_file.write(str(value))

    def infer(
        server_endpoint: ServerEndpointType,
        worker_index: Optional[int],
    ):
        with os_create_client_socket() as client:
            os_connect(client, server_endpoint)
            remote = Remote(client)

            if worker_index is not None:
                from crunch import monkey_patches
                monkey_patches.SHOULD_PRINT_PREFIX_WHEN_POSSIBLE = f"worker:{worker_index}"

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
        "get_parallelism": get_parallelism,
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


@contextmanager
def named_temp_file(**kwargs):
    file = NamedTemporaryFile(delete=False, **kwargs)
    file.close()

    try:
        yield file
    finally:
        os.unlink(file.name)


def _run_infer(
    context: "RunnerContext",
    tracer: "RunnerTracer",
    data_directory_path: str,
    parallelism: int,
    determinism_check: Union[Literal[False], float],
) -> pandas.DataFrame:
    span_attributes = to_execute_span_attributes(
        command="infer",
        parameters={
            "parallelism": parallelism,
        },
        span_attributes={
            "determinism_check": determinism_check,
        }
    )

    with tracer.span("execute", attributes=span_attributes), ExitStack() as stack:
        return _run_infer_with_server(
            context,
            data_directory_path,
            parallelism,
            determinism_check,
            stack,
        )


def _run_infer_with_server(
    context: "RunnerContext",
    data_directory_path: str,
    parallelism: int,
    determinism_check: Union[Literal[False], float],
    stack: ExitStack,
) -> pandas.DataFrame:
    x_test_name = "X_test.reduced.parquet" if context.is_local else "X_test.parquet"
    x_test = pandas.read_parquet(os.path.join(data_directory_path, x_test_name))

    datasets: List[pandas.DataFrame] = []
    for _, dataset in x_test.groupby(x_test.index.get_level_values("id")):
        datasets.append(dataset)

    if determinism_check is not False:
        determinism_slice = slice(None, int(len(datasets) * determinism_check))
        datasets = datasets[determinism_slice]

    slices = _split_into_batches(len(datasets), parallelism)

    def run_server(server: socket, worker_index: int, predicted_values: List[float], prediction_index: List[Tuple[int, int]]):
        try:
            start_index, end_index = slices[worker_index]

            client, _ = os_accept(server)
            with client:
                server.close()

                remote = Remote(client)
                remote.expect(RemoteCommand.READY)

                for dataset in datasets[start_index:end_index]:
                    remote.send(RemoteCommand.NEW_TIMESERIES)

                    historical, online = _split_periods(dataset, online_values=False)
                    remote.send(RemoteCommand.HISTORICAL_DATA, len(historical))
                    remote.send_raw(historical.astype(numpy.float32).tobytes())  # pyright: ignore[reportAttributeAccessIssue]

                    for index, value in online.items():
                        remote.send(RemoteCommand.ONLINE_POINT, value)

                        prediction = remote.expect(RemoteCommand.NEW_PREDICTION)
                        predicted_values.append(prediction)  # pyright: ignore[reportArgumentType]
                        prediction_index.append(index)  # pyright: ignore[reportArgumentType]

                    remote.send(RemoteCommand.END_TIMESERIES)

                remote.send(RemoteCommand.END)
        except Exception as error:
            context.log(f"error in data thread: {error}")
            os._exit(1)  # in a fork, so its fine

    def do_execute(worker_index: int, output_file_path: Optional[str]):
        predicted_values: List[float] = []
        prediction_index: List[Tuple[int, int]] = []

        server, endpoint = os_create_server_socket()

        with server:
            thread = Thread(target=run_server, args=(server, worker_index, predicted_values, prediction_index), daemon=True)
            thread.start()

            context.execute(
                command="infer",
                parameters={
                    "server_endpoint": endpoint,
                    "worker_index": None if output_file_path is None else worker_index,
                },
                trace=False,
                install_data_fuse=False,
            )

            thread.join()

        prediction = pandas.DataFrame(
            data={
                "prediction": predicted_values,
            },
            index=pandas.MultiIndex.from_tuples(
                prediction_index,
                names=["id", "time"],
            ),
        )

        if output_file_path is None:
            return prediction

        prediction.to_parquet(output_file_path)
        os._exit(0)  # in a fork, so its fine

    if parallelism == 1:
        return do_execute(0, None)
    else:
        fork = multiprocessing.get_context("fork")

        processes: List[multiprocessing.Process] = []
        temp_files: List[NamedTemporaryFile] = []  # pyright: ignore[reportGeneralTypeIssues]

        for worker_index in range(len(slices)):
            temp_file = stack.enter_context(NamedTemporaryFile(prefix=f"prediction_{worker_index}_", suffix=".parquet"))

            process = fork.Process(  # pyright: ignore[reportAttributeAccessIssue]
                target=do_execute,
                args=(worker_index, temp_file.name),
            )
            process.start()

            processes.append(process)
            temp_files.append(temp_file)

        prediction_parts = []
        for process, temp_file in zip(processes, temp_files):
            process.join()

            prediction = pandas.read_parquet(temp_file.name)
            prediction_parts.append(prediction)

        return pandas.concat(prediction_parts)


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
                raise ProtocolError("previous value not yield-ed")

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

            # 7.2: send prediction to server
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
) -> Tuple[List[Tuple[int, int]], pandas.Series]:
    historical = dataset[dataset["period"] == 1]["value"].values
    online = dataset[dataset["period"] == 2]["value"]

    if online_values:
        online = online.values

    return historical, online  # pyright: ignore[reportReturnType]


def _split_into_batches(element_count: int, batch_count: int) -> List[Tuple[int, int]]:
    batch_size, remainder = divmod(element_count, batch_count)

    indices = []

    start = 0
    for index in range(batch_count):
        extra = 1 if index < remainder else 0

        end = start + batch_size + extra
        indices.append((start, end))

        start = end

    return indices


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
