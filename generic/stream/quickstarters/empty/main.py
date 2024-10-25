"""
This is a basic example of what you need to do to participate to the tournament.
The code will not have access to the internet (or any socket related operation).
"""

# Imports
import os
import typing

import crunch

# dont forget to update the `requirements.txt`
import joblib


# Uncomment what you need!
def train(
    streams: typing.List[typing.Iterable[crunch.StreamMessage]],
    model_directory_path: str,
    # has_gpu: bool,
) -> None:
    """
    Do your model training here.
    This function will only be called if the model_directiory_path is empty.
    Note: You can use other serialization methods than joblib.dump(), as
    long as it matches what reads the model in infer().

    Args:
        streams: a collection of streams to read (can be read multiple times from the beginning)
        model_directory_path: the path to save your updated model
        has_gpu: if the runner has a gpu

    Returns:
        None
    """

    # TODO: EDIT ME
    model_or_parameters = ...

    for stream in streams:
        for message in stream:
            x = message.x

    model_file_path = os.path.join(model_directory_path, f"model.joblib")
    joblib.dump(model_or_parameters, model_file_path)


# Uncomment what you need!
def infer(
    stream: typing.Iterator[crunch.StreamMessage],
    model_directory_path: str,
    # has_gpu: bool,
    # has_trained: bool,
):
    """
    Do your inference here.
    This function will load the model/parameters saved in the infer function.
    It is mandatory to send your conclusions to the system with a `yield`.
    The first yield tells the system that your model has been loaded and
    is ready to receive further messages from the stream.
    The stream cannot be read twice from the beginning.

    Args:
        stream: a non-repeatable iterator of values to tick and predict one.
        model_directory_path: the path to the directory to the directory in wich we will be saving your updated model
        has_gpu: if the runner has a gpu
        has_trained: if the moon will train

    Returns:
        Yielding values, one after the other.
    """

    # load your model
    # TODO: EDIT ME
    model = ...

    yield  # mark as ready

    for message in stream:
        x = message.x

        y = model.tick_and_predict(x)

        # return the response
        yield y
