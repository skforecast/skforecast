# Unit test backtesting_forecaster_multiseries ForecasterRnn using PyTorch backend
# ==============================================================================
import os
import numpy as np
import pandas as pd
os.environ["KERAS_BACKEND"] = "torch"
import keras
from skforecast.deep_learning.utils import create_and_compile_model
from skforecast.deep_learning import ForecasterRnn
from skforecast.model_selection._split import TimeSeriesFold
from skforecast.model_selection import backtesting_forecaster_multiseries

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
            lags=4,           
            steps=5,              
            recurrent_layer="LSTM",
            recurrent_units=128,
            dense_units=64,
        )


def test_backtesting_forecaster_multiseries_ForecasterRnn():
    """
    Test case for backtesting ForecasterRnn with multiseries data.
    """
    forecaster = ForecasterRnn(model, levels=["1", "2"], lags=3)
    cv = TimeSeriesFold(
            initial_train_size = len(series) - 15,
            steps              = forecaster.max_step,
            refit              = False
         )

    metrics, preds = backtesting_forecaster_multiseries(
                         forecaster = forecaster,
                         series     = series,
                         cv         = cv,
                         levels     = ['1'],
                         metric     = 'mean_absolute_error', 
                         verbose    = False
                     )

    assert metrics.shape == (1, 2)
    assert preds.shape == (15, 3)


def test_backtesting_forecaster_multiseries_ForecasterRnn_with_exog():
    """
    Test case for backtesting ForecasterRnn with multiseries data with
    exogenous variables.
    """
    forecaster = ForecasterRnn(model_exog, levels=["1", "2", "3"], lags=4)
    cv = TimeSeriesFold(
            initial_train_size = len(series) - 18,
            steps              = forecaster.max_step,
            refit              = True
         )

    metrics, preds = backtesting_forecaster_multiseries(
                         forecaster = forecaster,
                         series     = series,
                         exog       = exog,
                         cv         = cv,
                         levels     = None,
                         metric     = 'mean_absolute_error', 
                         verbose    = False
                     )

    assert metrics.shape == (6, 2)
    assert preds.shape == (54, 3)


def test_backtesting_forecaster_multiseries_ForecasterRnn_with_exog_and_interval():
    """
    Test case for backtesting ForecasterRnn with multiseries data with 
    exogenous variables and interval predictions.
    """
    forecaster = ForecasterRnn(model_exog, levels=["1", "2", "3"], lags=4)
    cv = TimeSeriesFold(
            initial_train_size = len(series) - 20,
            steps              = forecaster.max_step - 2,
            gap                = 2,
            refit              = False
         )

    metrics, preds = backtesting_forecaster_multiseries(
                         forecaster      = forecaster,
                         series          = series,
                         exog            = exog,
                         cv              = cv,
                         levels          = ['1', '2'],
                         metric          = 'mean_absolute_error', 
                         interval        = [5, 95],
                         interval_method = "conformal",
                         verbose         = False
                     )

    assert metrics.shape == (5, 2)
    assert preds.shape == (48, 5)
