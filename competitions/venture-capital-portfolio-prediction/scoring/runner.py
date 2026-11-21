import gc
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas
from crunch.utils import smart_call

if TYPE_CHECKING:
    from crunch.runner.unstructured import RunnerContext, RunnerExecutorContext, UserModule


EMBARGO = 0
NUMBER_OF_FEATURES = 229

ID_COLUMN_NAME = "id"
MOON_COLUMN_NAME = "date"
TARGET_NAME = "prediction"
TARGET_COLUMN_NAME = "target_r3"
PREDICTION_COLUMN_NAME = "target_r3"

SPLITS_FILE_NAME = "splits.json"
PREDICTION_FILE_NAME = "prediction.parquet"

# the local files only contain the reduced test moons
LOCAL_X_FILE_NAMES = ["X_train.parquet", "X_test_reduced.parquet"]
LOCAL_Y_FILE_NAMES = ["y_train.parquet", "y_test_reduced.parquet"]
CLOUD_X_FILE_NAMES = ["X.parquet"]
CLOUD_Y_FILE_NAMES = ["y.parquet"]

# set by the platform on the runner, "true" or "false"
GPU_ENVIRONMENT_VARIABLE_NAME = "GPU"


def load_data(
    data_directory_path: str,
):
    x_train = _read(data_directory_path, ["X_train.parquet"])
    y_train = _read(data_directory_path, ["y_train.parquet"])
    x_test = _read(data_directory_path, ["X_test_reduced.parquet"])

    return (
        x_train,
        y_train,
        x_test,
    )


def run(
    context: "RunnerContext",
    data_directory_path: str,
    prediction_directory_path: str,
):
    splits = _load_splits(data_directory_path)
    moons = _get_test_moons(splits, is_local=context.is_local)

    moon_infos = _MoonInfo.get(
        moons=moons,
        train_frequency=context.train_frequency,
        has_model=context.has_model,
        force_first_train=context.force_first_train,
    )

    prediction_file_path = os.path.join(prediction_directory_path, PREDICTION_FILE_NAME)
    predictions: List[pandas.DataFrame] = []

    for moon_info in moon_infos:
        forced_train = " forced=True" if moon_info.forced_train else ""
        context.log(f"looping moon={moon_info.key} train={moon_info.train}{forced_train} ({moon_info.number}/{len(moon_infos)})")

        if moon_info.train:
            context.execute(
                command="train",
                parameters={
                    "moon": moon_info.key,
                },
            )

        _delete_if_exists(prediction_file_path)

        context.execute(
            command="infer",
            parameters={
                "moon": moon_info.key,
                "has_trained": moon_info.train,
                "prediction_file_path": prediction_file_path,
            },
        )

        predictions.append(pandas.read_parquet(prediction_file_path))

    _delete_if_exists(prediction_file_path)

    prediction = pandas.concat(predictions, ignore_index=True)
    prediction.to_parquet(prediction_file_path, index=False)


def execute(
    context: "RunnerExecutorContext",
    module: "UserModule",
    data_directory_path: str,
    model_directory_path: str,
):
    if context.is_local:
        x_file_names, y_file_names = LOCAL_X_FILE_NAMES, LOCAL_Y_FILE_NAMES
    else:
        x_file_names, y_file_names = CLOUD_X_FILE_NAMES, CLOUD_Y_FILE_NAMES

    has_gpu = os.environ.get(GPU_ENVIRONMENT_VARIABLE_NAME, "false").lower() == "true"

    def get_default_values(moon: int, has_trained: bool):
        return {
            "number_of_features": NUMBER_OF_FEATURES,
            "model_directory_path": model_directory_path,
            "id_column_name": ID_COLUMN_NAME,
            "moon_column_name": MOON_COLUMN_NAME,
            "target_column_name": TARGET_COLUMN_NAME,
            "target_column_names": _Columns({TARGET_NAME: TARGET_COLUMN_NAME}),
            "prediction_column_name": PREDICTION_COLUMN_NAME,
            "prediction_column_names": _Columns({TARGET_NAME: PREDICTION_COLUMN_NAME}),
            "column_names": _COLUMN_NAMES,
            "moon": moon,
            "current_moon": moon,
            "embargo": EMBARGO,
            "has_gpu": has_gpu,
            "has_trained": has_trained,
        }

    def train(
        moon: int,
    ):
        # the embargo hides the most recent moons, whose target is not yet known at `moon`
        filters = [(MOON_COLUMN_NAME, "<", moon - EMBARGO)]
        x_train = _read(data_directory_path, x_file_names, filters)
        y_train = _read(data_directory_path, y_file_names, filters)

        context.trip_data_fuse()

        smart_call(
            module.get_function("train"),
            get_default_values(moon, has_trained=True),
            {
                "X_train": x_train,
                "x_train": x_train,
                "Y_train": y_train,
                "y_train": y_train,
            },
        )

        del x_train, y_train
        gc.collect()

    def infer(
        moon: int,
        has_trained: bool,
        prediction_file_path: str,
    ):
        x_test = _read(data_directory_path, x_file_names, [(MOON_COLUMN_NAME, "=", moon)])

        context.trip_data_fuse()

        prediction = smart_call(
            module.get_function("infer"),
            get_default_values(moon, has_trained),
            {
                "X_test": x_test,
                "x_test": x_test,
            },
        )

        del x_test
        gc.collect()

        if not isinstance(prediction, pandas.DataFrame):
            raise ValueError(f"infer() must return a pandas.DataFrame, but got {type(prediction)}")

        missing_column_names = [
            column_name
            for column_name in (MOON_COLUMN_NAME, ID_COLUMN_NAME, PREDICTION_COLUMN_NAME)
            if column_name not in prediction.columns
        ]

        if missing_column_names:
            raise ValueError(f"infer() must return the columns `{MOON_COLUMN_NAME}`, `{ID_COLUMN_NAME}` and `{PREDICTION_COLUMN_NAME}`, missing: {missing_column_names}")

        prediction = prediction[[MOON_COLUMN_NAME, ID_COLUMN_NAME, PREDICTION_COLUMN_NAME]]
        prediction.to_parquet(prediction_file_path, index=False)

    return {
        "train": train,
        "infer": infer,
    }


def _load_splits(
    data_directory_path: str,
) -> List[Dict[str, Any]]:
    """
    The splits of the data release, as `{"key": <moon>, "group": "TRAIN" | "TEST", "reduced": "XY" | "X" | None}`.
    """

    with open(os.path.join(data_directory_path, SPLITS_FILE_NAME), "r") as fd:
        return json.load(fd)


def _get_test_moons(
    splits: List[Dict[str, Any]],
    *,
    is_local: bool,
) -> List[int]:
    """
    The moons to predict: only the reduced ones locally (their files are the only ones available), all of them in the cloud.
    """

    return sorted(
        split["key"]
        for split in splits
        if split["group"] == "TEST" and (not is_local or split["reduced"] is not None)
    )


@dataclass
class _MoonInfo:

    key: int
    number: int
    train: bool
    forced_train: bool

    @staticmethod
    def get(
        moons: List[int],
        train_frequency: int,
        has_model: bool,
        force_first_train: bool,
    ):
        infos: List["_MoonInfo"] = []

        for index, moon in enumerate(moons):
            train, forced_train = False, False
            if train_frequency != 0 and moon % train_frequency == 0:
                train = True
            elif index == 0 and not has_model:
                train = True
            elif index == 0 and force_first_train:
                train, forced_train = True, True

            infos.append(_MoonInfo(
                key=moon,
                number=index + 1,
                train=train,
                forced_train=forced_train,
            ))

        return infos


@dataclass(frozen=True)
class _TargetColumnNames:
    """
    The former `crunch.api.TargetColumnNames`.
    """

    name: str
    side: Optional[str]
    input: Optional[str]
    output: Optional[str]


@dataclass(frozen=True)
class _ColumnNames:
    """
    The former `crunch.api.ColumnNames`, given as `column_names`.
    """

    id: str
    moon: str
    side: Optional[str]
    input: Optional[str]
    output: Optional[str]
    targets: List[_TargetColumnNames]

    @property
    def first_target(self):
        return next(iter(self.targets), None)

    @property
    def inputs(self):
        return [target.input for target in self.targets]

    @property
    def outputs(self):
        return [target.output for target in self.targets]

    @property
    def target_names(self):
        return [target.name for target in self.targets]

    def get_target_by_name(self, name: str):
        return next((target for target in self.targets if target.name == name), None)


_COLUMN_NAMES = _ColumnNames(
    id=ID_COLUMN_NAME,
    moon=MOON_COLUMN_NAME,
    side=None,
    input=None,
    output=None,
    targets=[
        _TargetColumnNames(
            name=TARGET_NAME,
            side=None,
            input=TARGET_COLUMN_NAME,
            output=PREDICTION_COLUMN_NAME,
        ),
    ],
)


class _Columns:
    """
    The column names by target name, as the former `crunch.container.Columns` (`columns.prediction`, `columns["prediction"]`, `list(columns)`).
    """

    def __init__(self, column_name_by_target_name: Dict[str, str]):
        self._column_name_by_target_name = column_name_by_target_name

    def __getattr__(self, name: str):
        try:
            return self.__dict__["_column_name_by_target_name"][name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'") from None

    def __getitem__(self, name: str):
        return self._column_name_by_target_name[name]

    def __iter__(self):
        return iter(self._column_name_by_target_name.values())

    def __repr__(self):
        return f"Columns({self._column_name_by_target_name})"


def _read(
    data_directory_path: str,
    file_names: List[str],
    filters: Optional[List[tuple]] = None,
) -> pandas.DataFrame:
    dataframes = [
        pandas.read_parquet(
            os.path.join(data_directory_path, file_name),
            filters=filters,
            engine="pyarrow",
        )
        for file_name in file_names
    ]

    return pandas.concat(dataframes).reset_index(drop=True)


def _delete_if_exists(path: str):
    if os.path.exists(path):
        os.unlink(path)
