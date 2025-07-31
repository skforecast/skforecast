# Unit test predict method using PyTorch backend
# ==============================================================================
import os
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
    forecaster = ForecasterRnn(model, levels=["1", "2"], lags=3)
    forecaster.fit(series=series)
    predictions = forecaster.predict(steps=3)

    assert predictions.shape == (6, 2)


def test_predict_specific_levels():
    """
    Test case for predicting with specific levels
    """
    forecaster = ForecasterRnn(model, levels=["1", "2"], lags=3)
    forecaster.fit(series=series)
    predictions = forecaster.predict(steps=None, levels=["1"])

    assert predictions.shape == (4, 2)


def test_predict_exog():
    """
    Test case for predicting with exogenous variables
    """
    forecaster = ForecasterRnn(model_exog, levels=["1", "2", "3"], lags=10)
    forecaster.fit(series=series, exog=exog)
    predictions = forecaster.predict(steps=None, exog=exog_pred)

    assert predictions.shape == (24, 2)


def test_predict_specific_levels_with_exog():
    """
    Test case for predicting with specific levels
    """
    forecaster = ForecasterRnn(model_exog, levels=["1", "2", "3"], lags=10)
    forecaster.fit(series=series, exog=exog)
    predictions = forecaster.predict(steps=5, exog=exog_pred, levels=["1", "2"])

    assert predictions.shape == (10, 2)
