# Unit test predict method using PyTorch backend
# ==============================================================================
import os
import re
import pytest
import numpy as np
import pandas as pd
os.environ["KERAS_BACKEND"] = "torch"
import keras
from skforecast.deep_learning.utils import create_and_compile_model
from skforecast.deep_learning import ForecasterRnn

series = pd.DataFrame(
    {
        "1": np.arange(50),
        "2": np.arange(50),
        "3": np.arange(50),
    },
    index=pd.date_range("2020-01-01", periods=50, freq="D")
)

exog = pd.DataFrame(
    {
        "exog1": np.arange(50),
        "exog2": np.arange(50),
    },
    index=pd.date_range("2020-01-01", periods=50, freq="D")
)

exog_pred = pd.DataFrame(
    {
        "exog1": np.arange(50, 60),
        "exog2": np.arange(50, 60),
    },
    index=pd.date_range("2020-02-20", periods=10, freq="D")
)

model = create_and_compile_model(
            series=series, 
            levels=["1", "2"],    
            lags=3,           
            steps=4,              
            recurrent_layer="LSTM",
            recurrent_units=128,
            dense_units=64,
        )

model_exog = create_and_compile_model(
            series=series, 
            exog=exog,
            levels=["1", "2", "3"],    
            lags=10,           
            steps=8,              
            recurrent_layer="LSTM",
            recurrent_units=128,
            dense_units=64,
        )


def test_predict_3_steps_ahead():
    """
    Test case for predicting 3 steps ahead
    """
    forecaster = ForecasterRnn(estimator=model, levels=["1", "2"], lags=3)
    forecaster.fit(series=series)
    predictions = forecaster.predict(steps=3)

    assert predictions.shape == (6, 2)


def test_predict_specific_levels():
    """
    Test case for predicting with specific levels
    """
    forecaster = ForecasterRnn(estimator=model, levels=["1", "2"], lags=3)
    forecaster.fit(series=series)
    predictions = forecaster.predict(steps=None, levels=["1"])

    assert predictions.shape == (4, 2)


def test_predict_exog():
    """
    Test case for predicting with exogenous variables
    """
    forecaster = ForecasterRnn(
        estimator=model_exog, levels=["1", "2", "3"], lags=10
    )
    forecaster.fit(series=series, exog=exog)
    predictions = forecaster.predict(steps=None, exog=exog_pred)

    assert predictions.shape == (24, 2)


def test_predict_specific_levels_with_exog():
    """
    Test case for predicting with specific levels
    """
    forecaster = ForecasterRnn(
        estimator=model_exog, levels=["1", "2", "3"], lags=10
    )
    forecaster.fit(series=series, exog=exog)
    predictions = forecaster.predict(steps=5, exog=exog_pred, levels=["1", "2"])

    assert predictions.shape == (10, 2)


def test_predict_ValueError_when_last_window_without_series_not_in_levels():
    """
    Test ValueError is raised when `last_window` does not contain a series used
    as input that is not predicted (not in `levels`). Before, the missing series
    was replaced by another column of `last_window`.
    """
    forecaster = ForecasterRnn(estimator=model, levels=["1", "2"], lags=3)
    forecaster.fit(series=series)
    last_window = series[["1", "2"]].iloc[-3:]

    err_msg = re.escape(
        "`last_window` columns must be the same as the `series` "
        "column names used to create the X_train matrix.\n"
        "    `last_window` columns    : ['1', '2']\n"
        "    `series` columns X train : ['1', '2', '3']"
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict(steps=3, last_window=last_window)
