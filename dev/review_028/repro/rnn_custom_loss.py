"""
BUG-rnn-custom-loss: a Keras model compiled with a custom loss that is not
registered (`keras.saving.register_keras_serializable`) makes `ForecasterRnn`
fail when it copies the model. Pre-existing: same behavior in 0.27.x. A
registered loss works.

Needs keras and torch (KERAS_BACKEND=torch). Usage: python rnn_custom_loss.py
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import warnings

import keras
import numpy as np
import pandas as pd
from _common import attempt, versions

from skforecast.deep_learning import ForecasterRnn, create_and_compile_model

warnings.filterwarnings("ignore")
print(versions())
n = 120
series = pd.DataFrame(
    {"a": np.sin(np.arange(n) / 5), "b": np.cos(np.arange(n) / 7)},
    index=pd.date_range("2020-01-01", periods=n, freq="D"),
)


def pinball_09(y_true, y_pred):
    error = y_true - y_pred
    return keras.ops.mean(keras.ops.maximum(0.9 * error, -0.1 * error))


@keras.saving.register_keras_serializable(package="review028")
def pinball_09_registered(y_true, y_pred):
    return pinball_09(y_true, y_pred)


for loss in [pinball_09, pinball_09_registered]:
    keras.utils.set_random_seed(0)
    model = create_and_compile_model(
        series=series, levels=["a"], lags=6, steps=2, recurrent_units=4,
        dense_units=4, compile_kwargs={"optimizer": "adam", "loss": loss},
    )

    def create_and_fit():
        forecaster = ForecasterRnn(
            model, levels=["a"], lags=6, fit_kwargs={"epochs": 2, "verbose": 0}
        )
        forecaster.fit(series)
        return forecaster.predict().shape

    attempt(loss.__name__, create_and_fit)
