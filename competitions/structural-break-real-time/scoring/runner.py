import gc
import os
import traceback
from enum import Enum
from multiprocessing import Process, Queue
from types import GeneratorType
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterator, List, Literal, Optional, Tuple, Union

import numpy
import pandas
from crunch.utils import smart_call

if TYPE_CHECKING:
    from crunch.runner.unstructured import RunnerContext, RunnerExecutorContext, UserModule


is_parallelism_supported = os.name != "nt"
if is_parallelism_supported:
    register_at_fork = os.register_at_fork
else:
    # the Cloud environment isn't on Windows
    def register_at_fork(*, after_in_child: Optional[Callable] = None, **kwargs):
        pass


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
    prediction_directory_path: str,
):
    if context.force_first_train:
        context.execute(
            command="train",
        )

    prediction_parquet_file_path = os.path.join(prediction_directory_path, "prediction.parquet")

    context.execute(
        command="infer",
        parameters={
            "determinism_check": False,
            "prediction_parquet_file_path": prediction_parquet_file_path,
        },
    )

    prediction = pandas.read_parquet(prediction_parquet_file_path)

    if context.is_determinism_check_enabled:
        percentage = 0.3
        tolerance = 1e-8

        context.log(f"checking determinism by executing the inference again with {percentage * 100:.0f}% of the data (tolerance: {tolerance})")

        prediction_determinism_parquet_file_path = os.path.join(prediction_directory_path, "prediction.determinism.parquet")

        context.execute(
            command="infer",
            parameters={
                "determinism_check": percentage,
                "prediction_parquet_file_path": prediction_determinism_parquet_file_path,
            }
        )

        prediction2 = pandas.read_parquet(prediction_determinism_parquet_file_path)
        os.unlink(prediction_determinism_parquet_file_path)

        is_deterministic = numpy.allclose(prediction.loc[prediction2.index, "prediction"], prediction2["prediction"], atol=tolerance)
        context.report_determinism(is_deterministic)


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
        determinism_check: Union[Literal[False], float],
        prediction_parquet_file_path: str,
    ):
        x_test_name = "X_test.reduced.parquet" if context.is_local else "X_test.parquet"
        x_test = pandas.read_parquet(os.path.join(data_directory_path, x_test_name))

        datasets: List[Iterator[pandas.Series]] = []
        for _, dataset in x_test.groupby(x_test.index.get_level_values("id")):
            datasets.append(dataset)

        if determinism_check is not False:
            determinism_slice = slice(None, int(len(datasets) * determinism_check))
            datasets = datasets[determinism_slice]

        del x_test
        gc.collect()

        context.trip_data_fuse()

        infer_function = module.get_function("infer")
        infer_parallelism = _get_infer_parallelism(module)

        prediction = _run_with_double_protection(
            datasets=datasets,
            parallelism=infer_parallelism,
            consumer_factory=lambda datasets: smart_call(
                infer_function,
                default_values,
                {
                    "datasets": datasets,
                },
            ),
            post_processor=_post_process_infer_yield_result,
        )

        prediction.to_parquet(prediction_parquet_file_path)

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


def _get_infer_parallelism(module: "UserModule") -> int:
    if not is_parallelism_supported:
        return 1

    key = "INFER_PARALLELISM"
    value = module.get_value(key, default=1)

    if not isinstance(value, int):
        print(f"[infer] `{key}` must be an int")
        return 1

    cpu_count = os.cpu_count()
    if value > cpu_count:
        print(f"[infer] `{key}` must be at most the number of CPUs ({cpu_count})")
        value = cpu_count
    elif value == 0:
        print(f"[infer] using all available CPUs for inference")
        value = cpu_count
    elif value < 1:
        print(f"[infer] `{key}` must be at least 1")
        value = 1

    return value


class ProtocolError(RuntimeError):
    pass


class NestedForkError(RuntimeError):
    pass


def _run_with_double_protection(
    *,
    datasets: List[pandas.DataFrame],
    parallelism: int,
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

    # 3: provide the values
    def online_stream(online: pandas.Series):
        nonlocal state

        for value in online:
            # X: ensure not forked
            if state == State.NESTED_FORK:
                raise NestedForkError("cannot iterate in online stream in a child process after a fork")

            # 6: ensure not iterating a dataset
            if state != State.READY_FOR_VALUE:
                raise ProtocolError("previous value not yield-ed")

            # 6.1: mark value as consumed
            state = State.CONSUMED

            yield value

    slices = _split_into_batches(len(datasets), parallelism)

    # I: shared variable for both main and worker processes
    prediction_index = []
    prediction_values = []

    # X: triggered in the child process after fork, prevent iterations
    def _after_in_child():
        nonlocal state, next_value

        state = State.NESTED_FORK
        next_value = None

    def worker(worker_index: int, queue: Optional[Queue] = None):
        nonlocal state, next_value

        # X: register security measure
        register_at_fork(after_in_child=_after_in_child)

        start_index, end_index = slices[worker_index]

        if queue is not None:
            print(f"[worker:{worker_index}] started with pid={os.getpid()}, starting at index={start_index} and ending at index={end_index}")

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

        for dataset in datasets[start_index:end_index]:
            # X: ensure not forked
            if state == State.NESTED_FORK:
                raise NestedForkError(f"cannot iterate in datasets in a child process after a fork ({os.getpid()})")

            # 2: allow reading a dataset next
            state = State.READY_FOR_DATASET

            historical, online = _split_periods(dataset, online_values=False)

            # 2.1: set value for dataset stream
            next_value = (historical, online_stream(online))

            for index in online.index:
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

                prediction_index.append(index)
                prediction_values.append(y)

            # 8: clean up for next dataset
            next_value = None

        # 9: ensure user clean up is called
        if True:
            last_check = next(user_code, sentinel)
            if last_check is not sentinel:
                raise ProtocolError("code did not exit after last dataset")

        if queue is not None:
            print(f"[worker:{worker_index}] finished")

    def parallel_worker(worker_index: int, queue: Queue):
        try:
            worker(worker_index, queue)

            queue.put([
                worker_index,
                True,
                (
                    prediction_index,
                    prediction_values,
                )
            ])
        except Exception as exception:
            traceback_str = traceback.format_exception(type(exception), exception, exception.__traceback__)
            print(f"[worker:{worker_index}] encountered an exception:\n" + "".join(traceback_str))

            if not isinstance(exception, NestedForkError):
                queue.put([
                    worker_index,
                    False,
                    None,
                ])

            os._exit(1)

    if parallelism == 1:
        # I: run in the main process
        worker(0)

    else:
        import crunch.monkey_patches as monkey_patches
        monkey_patches.SHOULD_PRINT_PID_WHEN_POSSIBLE = True

        queue = Queue()
        processes = [
            Process(
                target=parallel_worker,
                args=(index, queue)
            )
            for index in range(len(slices))
        ]

        results = _start_processes_and_wait(processes, queue)

        # I: collect results (ordered by worker index)
        for _, _, (batch_prediction_index, batch_prediction_values) in results:
            prediction_index.extend(batch_prediction_index)
            prediction_values.extend(batch_prediction_values)

    return pandas.DataFrame(
        data={
            "prediction": prediction_values,
        },
        index=pandas.MultiIndex.from_tuples(
            prediction_index,
            names=["id", "time"],
        ),
    )


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


def _start_processes_and_wait(
    processes: List[Process],
    queue: Queue,
    dead_check_interval=5.0
):
    results = []

    for process in processes:
        process.start()

    try:
        for _ in range(len(processes)):
            while True:
                try:
                    _, successful, _ = result = queue.get(timeout=dead_check_interval)
                    break
                except Exception:
                    dead = [
                        process
                        for process in processes
                        if not process.is_alive() and process.exitcode != 0
                    ]

                    if dead:
                        pid_to_exit_codes = ", ".join([f"{process.pid}={process.exitcode}" for process in dead])
                        raise ProtocolError(f"{len(dead)} worker(s) died silently (exit codes: {pid_to_exit_codes})")

            if not successful:
                raise ProtocolError("a worker process encountered an exception")

            results.append(result)

    except Exception:
        for process in processes:
            if process.is_alive():
                process.kill()

        raise

    finally:
        for process in processes:
            process.join()

    results.sort(key=lambda x: x[0])
    return results


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
