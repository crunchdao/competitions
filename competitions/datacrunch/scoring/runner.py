import gc
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy
import pandas
import pyarrow
import pyarrow.parquet
from crunch.utils import smart_call

if TYPE_CHECKING:
    from crunch.runner.unstructured import RunnerContext, RunnerExecutorContext, UserModule


EMBARGO = 13
NUMBER_OF_FEATURES = 1768

ID_COLUMN_NAME = "id"
MOON_COLUMN_NAME = "moon"

# target name -> (target column name, prediction column name), in the targets' order
TARGET_COLUMN_NAMES = {
    "white": ("target_w", "prediction_w"),
    "red": ("target_r", "prediction_r"),
    "green": ("target_g", "prediction_g"),
    "blue": ("target_b", "prediction_b"),
}

# the value of a target that is not known yet
UNRESOLVED_TARGET_VALUE = -1

DEFAULT_FEATURE_GROUP = "v1"
FEATURE_COLUMN_NAMES_PARAMETER_NAME = "feature_column_names"

SPLITS_FILE_NAME = "splits.json"
FEATURES_FILE_NAME = "features.json"
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
    feature_parameters = _load_feature_parameters(data_directory_path)

    target_column_names = [target_column_name for target_column_name, _ in TARGET_COLUMN_NAMES.values()]
    prediction_column_names = [prediction_column_name for _, prediction_column_name in TARGET_COLUMN_NAMES.values()]

    def get_default_values(moon: int, has_trained: bool):
        return {
            "number_of_features": NUMBER_OF_FEATURES,
            "model_directory_path": model_directory_path,
            "id_column_name": ID_COLUMN_NAME,
            "moon_column_name": MOON_COLUMN_NAME,
            "target_column_name": target_column_names[0],
            "target_column_names": _Columns({target_name: target_column_name for target_name, (target_column_name, _) in TARGET_COLUMN_NAMES.items()}),
            "prediction_column_name": prediction_column_names[0],
            "prediction_column_names": _Columns({target_name: prediction_column_name for target_name, (_, prediction_column_name) in TARGET_COLUMN_NAMES.items()}),
            "column_names": _COLUMN_NAMES,
            "moon": moon,
            "current_moon": moon,
            "embargo": EMBARGO,
            "has_gpu": has_gpu,
            "has_trained": has_trained,
            **feature_parameters,
        }

    def train(
        moon: int,
    ):
        # the embargo hides the most recent moons, whose targets are not yet known at `moon`
        filters = [(MOON_COLUMN_NAME, "<", moon - EMBARGO)]
        y_train = _read(data_directory_path, y_file_names, filters)

        resolved = _get_resolved_rows(y_train, target_column_names)
        if resolved is not None:
            y_train = y_train[resolved].reset_index(drop=True)

        x_train = _read(data_directory_path, x_file_names, filters, rows=resolved)

        context.trip_data_fuse()

        if len(x_train) != len(y_train):
            raise ValueError(f"the X and y files do not have the same rows: {len(x_train)} != {len(y_train)}")

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

        required_column_names = [MOON_COLUMN_NAME, ID_COLUMN_NAME, *prediction_column_names]
        missing_column_names = [
            column_name
            for column_name in required_column_names
            if column_name not in prediction.columns
        ]

        if missing_column_names:
            raise ValueError(f"infer() must return the columns {', '.join(f'`{column_name}`' for column_name in required_column_names)}, missing: {missing_column_names}")

        prediction = prediction[required_column_names]
        prediction.to_parquet(prediction_file_path, index=False)

    return {
        "train": train,
        "infer": infer,
    }


def _get_resolved_rows(
    y_train: pandas.DataFrame,
    target_column_names: List[str],
) -> Optional[numpy.ndarray]:
    """
    The rows to train on, None for all: the rows with a target not known yet are left out (the embargo already does it
    for the most recent moons, this is for the few rows of the replay whose targets could not be recovered).
    The X and y files are row for row.
    """

    resolved = (y_train[target_column_names] != UNRESOLVED_TARGET_VALUE).all(axis=1).to_numpy()
    if resolved.all():
        return None

    return resolved


def _load_splits(
    data_directory_path: str,
) -> List[Dict[str, Any]]:
    """
    The splits of the data release, as `{"key": <moon>, "group": "TRAIN" | "TEST", "reduced": "XY" | "X" | None}`.
    """

    with open(os.path.join(data_directory_path, SPLITS_FILE_NAME), "r") as fd:
        return json.load(fd)


def _load_feature_parameters(
    data_directory_path: str,
) -> Dict[str, List[str]]:
    """
    The feature names of the data release, as `[{"group": "v1" | "v2", "name": <column name>}]`, given as `feature_column_names` (the default group) and `feature_column_names_<group>`.
    """

    with open(os.path.join(data_directory_path, FEATURES_FILE_NAME), "r") as fd:
        features = json.load(fd)

    names_by_group: Dict[str, List[str]] = {DEFAULT_FEATURE_GROUP: []}
    for feature in features:
        names = names_by_group.setdefault(feature["group"], [])

        if feature["name"] not in names:
            names.append(feature["name"])

    parameters = {
        FEATURE_COLUMN_NAMES_PARAMETER_NAME: names_by_group[DEFAULT_FEATURE_GROUP],
    }

    for group, names in names_by_group.items():
        parameters[f"{FEATURE_COLUMN_NAMES_PARAMETER_NAME}_{group}"] = names

    return parameters


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
            name=target_name,
            side=None,
            input=target_column_name,
            output=prediction_column_name,
        )
        for target_name, (target_column_name, prediction_column_name) in TARGET_COLUMN_NAMES.items()
    ],
)


class _Columns:
    """
    The column names by target name, as the former `crunch.container.Columns` (`columns.blue`, `columns["blue"]`, `list(columns)`).
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

    def __len__(self):
        return len(self._column_name_by_target_name)

    def __repr__(self):
        return f"Columns({self._column_name_by_target_name})"


def _read(
    data_directory_path: str,
    file_names: List[str],
    filters: Optional[List[tuple]] = None,
    rows: Optional[numpy.ndarray] = None,
) -> pandas.DataFrame:
    """
    The features alone take a few GB: the rows are selected before the conversion to pandas, so that there is never
    more than the table and the DataFrame in memory.
    """

    tables = [
        pyarrow.parquet.read_table(
            os.path.join(data_directory_path, file_name),
            filters=filters,
        )
        for file_name in file_names
    ]

    table = tables[0] if len(tables) == 1 else pyarrow.concat_tables(tables)
    del tables

    if rows is not None:
        table = table.filter(pyarrow.array(rows))

    return table.to_pandas()


def _delete_if_exists(path: str):
    if os.path.exists(path):
        os.unlink(path)
