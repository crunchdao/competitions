# Organizer Testing Tools

- [Organizer Testing Tools](#organizer-testing-tools)
- [CLI](#cli)
	- [Location Options](#location-options)
- [Leaderboard Module](#leaderboard-module)
	- [Compare Function](#compare-function)
		- [API](#api)
	- [Rank Function](#rank-function)
		- [Score File Format](#score-file-format)
		- [API](#api-1)
- [Scoring Module](#scoring-module)
	- [Check Function](#check-function)
		- [API](#api-2)
	- [Score Function](#score-function)
		- [API](#api-3)
- [Submission Module](#submission-module)
	- [Check Function](#check-function-1)
		- [API](#api-4)

# CLI

When organizing a competition, there are multiple tools available to test scripts without running the full infrastructure.

```bash
crunch [location options] organizer <competition name> test <module> <function> [options]
```

## Location Options

It is important to be able to run an alternative version of the code when debugging.

The CLI can load modules from GitHub (any repository and branch) or a local directory, provided the directory layout is followed.

```bash
/(root)
  /competitions
    /<competition name>
      /scoring
        /<module>.py
```

You can change the location using the following options:

| Option                                                    | Description                            | Default                    |
| --------------------------------------------------------- | -------------------------------------- | -------------------------- |
| `--competitions-repository <user name>/<repository name>` | Load the code from a GitHub repository | `"crunchdao/competitions"` |
| `--competitions-branch <branch name>`                     | Load the code from a specific branch   | `"master"`                 |
| `--competitions-directory-path <directory path>`          | Load the code from a local directory   | `(unset)`                  |

> [!NOTE]
> The GitHub options are ignored if the `--competitions-directory-path` is set.

# Leaderboard Module

The code must be in the file named `leaderboard.py`.

## Compare Function

```bash
crunch organizer <competition name> test leaderboard compare \
    --data-directory <data directory> \
    --prediction-file <prediction file 1> \
    --prediction-file <prediction file 2> \
    --prediction-file <prediction file n>
```

### API

```python
from crunch.api import Target
from crunch.unstructured import ComparedSimilarity

import pandas

def compare(
    targets: list[Target],
    predictions: dict[int, pandas.DataFrame],
    combinations: list[tuple[int, int]],
) -> list[ComparedSimilarity]:
    """
    Parameters:
        targets: List of targets for comparing columns.
        predictions: Mapping from a prediction's id to its dataframe.
        combinations: List of ids to be compared with each other (left and right).

    Return:
        Compared similarities of every combination in `combinations`.
    """

    return []
```

<details>

<summary>Example Implementation</summary>

```python
from crunch.api import Target
from crunch.unstructured import ComparedSimilarity

import pandas

def compare(
    targets: list[Target],
    predictions: dict[int, pandas.DataFrame],
    combinations: list[tuple[int, int]],
) -> list[ComparedSimilarity]:
    # TODO Do as needed
    target = targets[0]

    similarities: list[ComparedSimilarity] = []
    for left_id, right_id in combinations:
        left = predictions[left_id]
        right = predictions[right_id]

        value = left["prediction"].corr(right["prediction"], method="spearman")

        similarities.append(ComparedSimilarity(
            left_id,
            right_id,
            target.id,
            value,
        ))

    return similarities
```

</details>

## Rank Function

```bash
crunch organizer <competition name> test leaderboard rank \
    --scores-file <score file> \
    --rank-pass <"PRE_DUPLICATE" or "FINAL"> \
    --shuffle
```

> [!NOTE]
> To ensure the ranking function is deterministic, it is recommended to use the `--shuffle` option.

> [!WARNING]
> During the `PRE_DUPLICATE` pass, `rewardable` is always set to `false`.

### Score File Format

The score file must use the following schema:

```typescript
type RankableProject = {
    /** Project ID. */
    id: number; 

    /** Current project group. Format: `(user|team)-(id)` */
    group: string;

    /* Is the project rewardable? (only if not a duplicate and deterministic and ...) */
    rewardable: boolean;

    /* Scored metrics. */
    metrics: Array<RankableProjectMetric>;
}

type RankableProjectMetric = {
    /** Metric ID. */
    id: number;

    /** Scored value. */
    score: number;
}

type Root = Array<RankableProject>;
```

### API

```python
from crunch.api import Target, Metric
from crunch.unstructured import RankableProject, RankPass, RankedProject

import numpy
import scipy.stats

def rank(
    target_and_metrics: list[tuple[Target, list[Metric]]],
    projects: list[RankableProject],
    rank_pass: RankPass
) -> list[RankedProject]:
    """
    Parameters:
        target_and_metrics: List of targets and metrics that can be used for ranking.
        projects: List of projects to rank.
        rank_pass: Current ranking pass.

    Return:
        The ranked projects.
    """

    return []
```

<details>

<summary>Example Implementation</summary>

```python
from crunch.api import Target, Metric
from crunch.unstructured import RankableProject, RankPass, RankedProject

import numpy
import scipy.stats

def rank(
    target_and_metrics: list[tuple[Target, list[Metric]]],
    projects: list[RankableProject],
    rank_pass: RankPass
) -> list[RankedProject]:
    # TODO Do as needed
    target, metrics = target_and_metrics[0]
    metric = metrics[0]

    metric_column_name = f"metric:{metric.name}"

    dataframe = pandas.DataFrame((
        {
            "project_id": project.id,
            "group": project.group,
            "rewardable": project.rewardable,
            metric_column_name: project.get_metric(metric.id).score,
        }
        for project in projects
    ))
    
    def _rankdata(array: list[tuple[int, float]]):
        return scipy.stats.rankdata(array, method="min")

    dataframe["rank"] = _rankdata(dataframe[metric_column_name])

    dataframe.sort_values(
        by=[
            'rank',
            'project_id',  # fallback if same `rank_all`
        ],
        inplace=True,
    )

    dataframe.index = range(1, len(dataframe.index) + 1)

    # ignore non rewardable, and only keep first from each group
    mask = dataframe["rewardable"] & ~dataframe["group"].duplicated(keep="first")
    dataframe.loc[mask, "rank"] = numpy.nan

    return [
        RankedProject(
            id=int(row["project_id"]),
            rank=index,
            reward_rank=None if numpy.isnan(row["rank"]) else row["rank"],
        )
        for index, row in dataframe.iterrows()
    ]
```

</details>

# Scoring Module

The code must be in the file named `scoring.py`.

## Check Function

Check to see if the prediction is valid. <br />
Raise an error that will be displayed to the user if not.

```bash
crunch organizer <competition name> test scoring check \
    --data-directory <data directory> \
    --prediction-file <prediction file> \
    --phase-type <"SUBMISSION" or "OUT_OF_SAMPLE">
```

### API

```python
from crunch.api import PhaseType

import pandas


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


def check(
    prediction: pandas.DataFrame,
    data_directory_path: str,
    target_names: list[str],
    phase_type: PhaseType
) -> None:
    """
    Parameters:
        prediction: The dataframe to check.
        data_directory_path: Directory containing the data.
        target_names: List of the name of the targets.
        phase_type: Current phase type.

    Return:
        None.

    Raises:
        ParticipantVisibleError: If the file is invalid for a given reason.
        (extends) BaseException: If the file is invalid, but the reason will be hidden to avoid leaks.
    """

	pass
```

<details>

<summary>Example Implementation</summary>

```python
from crunch.api import PhaseType

import pandas


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


def check(
    prediction: pandas.DataFrame,
    data_directory_path: str,
    target_names: list[str],
    phase_type: PhaseType
) -> None:
    if True:
        difference = crunch.custom.utils.delta_message(
            target_names,
            set(prediction.columns),
        )

        if difference:
            raise ParticipantVisibleError(f"Columns do not match: {difference}")

    if True:
        # TODO Add more checks
        # Ideas:
        # - column names
        # - column dtypes
        # - row count

        pass
```

</details>

## Score Function

Give a score to a prediction. <br />
Any check can be ignored because the prediction should already be valid thanks to the [Check Function](#check-function).

```bash
crunch organizer <competition name> test scoring score \
    --data-directory <data directory> \
    --prediction-file <prediction file> \
    --phase-type <"SUBMISSION" or "OUT_OF_SAMPLE">
```

### API

```python
from crunch.api import PhaseType, Target, Metric
from crunch.scoring import ScoredMetric

import pandas


def score(
    prediction: pandas.DataFrame,
    data_directory_path: str,
    phase_type: PhaseType,
    target_and_metrics: list[tuple[Target, list[Metric]]],
) -> dict[int, ScoredMetric]:
    """
    Parameters:
        prediction: The dataframe to score.
        data_directory_path: Directory containing the data.
        phase_type: Current phase type.
        target_and_metrics: List of targets and metrics.

    Return:
        A mapping from a metric id to a scored metric.
    """

    return {}
```

<details>

<summary>Example Implementation</summary>

```python
from crunch.api import PhaseType, Target, Metric
from crunch.scoring import ScoredMetric

import pandas


def score(
    prediction: pandas.DataFrame,
    data_directory_path: str,
    phase_type: PhaseType,
    target_and_metrics: list[tuple[Target, list[Metric]]],
) -> dict[int, ScoredMetric]:
    # TODO Do as needed
    target, metrics = target_and_metrics[0]
    metric = metrics[0]

    y_test_path = os.path.join(data_directory_path, "y_test.parquet")
    y_test = pandas.read_parquet(y_test_path)

    score = prediction["prediction"].corr(y_test["target"])

    return {
        metric.id: ScoredMetric(score)
    }
```

</details>

# Submission Module

The code must be in the file named `submission.py`.

## Check Function

Validate the content of a submission.

```bash
crunch organizer <competition name> test submission check \
    --root-directory <root directory> \
    --model-directory <resources directory>
```

> [!WARNING]
> Only the content of text files is accessible. <br />
> Binary files are only listed.

### API

```python
from crunch.unstructured import File


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


def check(
    submission_files: list[File],
    model_files: list[File],
) -> None:
    """
    Parameters:
        submission_files: File of the submissions.
        model_files: File of the resources directory. (empty if none)

    Return:
        None.

    Raises:
        ParticipantVisibleError: If the submission is invalid for a given reason.
    """

    pass
```

<details>

<summary>Example Implementation</summary>

```python
from crunch.unstructured import File


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


def check(
    submission_files: list[File],
    model_files: list[File],
) -> None:
    report_md_file = _find_file_by_path(submission_files, "REPORT.md")
    if report_md_file is None:
        raise ParticipantVisibleError(f"Missing `{report_md_file_path}` file.")

    content = report_md_file.text
    if "<!-- Don't forget to change me -->" in content:
        raise ParticipantVisibleError(f"Default comment found in `{report_md_file_path}`. Has it been edited?")


def _find_file_by_path(files: list[File], path: str) -> File | None:
    return next((file for file in files if file.path == path ), None)
```

</details>
