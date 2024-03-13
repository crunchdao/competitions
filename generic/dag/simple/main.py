"""
This is a basic example of what you need to do to participate to the tournament.
The code will not have access to the internet (or any socket related operation).
"""

# Imports
import os
import typing

# dont forget to update the `requirements.txt`
import joblib
import pandas as pd


# Uncomment what you need!
def train(
    X_train: typing.Dict[str, pd.DataFrame],
    y_train: typing.Dict[str, pd.DataFrame],
    # number_of_features: int,
    model_directory_path: str,
    # id_column_name: str,
    # prediction_column_name: str,
    # embargo: int,
    # has_gpu: bool,
    # has_trained: bool,
) -> None:
    """
    Do your model training here.
    At each retrain this function will have to save an updated version of
    the model under the model_directiory_path, as in the example below.
    Note: You can use other serialization methods than joblib.dump(), as
    long as it matches what reads the model in infer().

    Args:
        X_train, y_train: the data to train the model.
        number_of_features: the number of features of the dataset
        model_directory_path: the path to save your updated model
        id_column_name: the name of the id column
        prediction_column_name: the name of the prediction column
        embargo: data embrago
        has_gpu: if the runner has a gpu
        has_trained: if the moon will train

    Returns:
        None
    """
    
    # TODO: EDIT ME
    model = ...

    joblib.dump(
        model,
        os.path.join(model_directory_path, "model.joblib")
    )


# Uncomment what you need!
def infer(
    X_test: typing.Dict[str, pd.DataFrame],
    # number_of_features: int,
    model_directory_path: str,
    id_column_name: str,
    prediction_column_name: str,
    # embargo: int,
    # has_gpu: bool,
    # has_trained: bool,
) -> pd.DataFrame:
    """
    Do your inference here.
    This function will load the model saved at the previous iteration and use
    it to produce your inference on the current date.
    It is mandatory to send your inferences with the ids so the system
    can match it correctly.

    Args:
        X_test: the independant variables of the current date passed to your model.
        number_of_features: the number of features of the dataset
        model_directory_path: the path to the directory to the directory in wich we will be saving your updated model.
        id_column_name: the name of the id column
        prediction_column_name: the name of the prediction column
        has_gpu: if the runner has a gpu
        has_trained: if the moon will train

    Returns:
        A dataframe (id, value) with the inferences of your model for the current date.
    """

    # loading the model saved by the train function at previous iteration
    model = joblib.load(os.path.join(model_directory_path, "model.joblib"))

    # creating the predicted dataframe
    prediction = ...

    return prediction
