# Unit test predict interval method with PyTorch backend
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


def test_predict_interval_ValueError_when_method_not_valid():
    """
    Test ValueError is raised when an invalid method is passed to predict_interval.
    """
    forecaster = ForecasterRnn(model, levels=["1", "2"], lags=3)
    forecaster.fit(series, store_in_sample_residuals=True)

    err_msg = re.escape(
        "Invalid `method` 'not_conformal'. Only 'conformal' is available."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.predict_interval(method="not_conformal")


def test_predict_interval_output_size_with_steps_by_default():
    """
    Test output sizes for predicting steps defined by default with intervals
    """
    # Create a ForecasterRnn object
    forecaster = ForecasterRnn(model, levels=["1", "2"], lags=3)
    forecaster.fit(series, store_in_sample_residuals=True)

    # Call the predict method
    int_preds = forecaster.predict_interval()

    # Check the shape and values of the predictions
    assert int_preds.shape == (4 * len(["1", "2"]), 4)


def test_predict_interval_output_size_3_steps_ahead():
    """
    Test output sizes for predicting 3 steps ahead with intervals
    """
    # Create a ForecasterRnn object
    forecaster = ForecasterRnn(model, levels=["1", "2"], lags=3)
    forecaster.fit(series, store_in_sample_residuals=True)

    # Call the predict method
    int_preds = forecaster.predict_interval(steps=3, interval=0.9)

    # Check the shape and values of the predictions
    assert int_preds.shape == (3 * len(["1", "2"]), 4)


def test_predict_interval_output_size_2_steps_ahead_specific_levels():
    """
    Test output sizes for predicting 2 steps ahead with intervals and specific levels
    """
    # Create a ForecasterRnn object
    forecaster = ForecasterRnn(model, levels=["1", "2"], lags=3)
    forecaster.fit(series, store_in_sample_residuals=True)

    # Call the predict method
    int_preds = forecaster.predict_interval(steps=2, levels="1")

    # Check the shape and values of the predictions
    assert int_preds.shape == (2 * 1, 4)


def test_predict_interval_exog_and_out_sample_residuals():
    """
    Test case for predicting with exogenous variables
    """
    forecaster = ForecasterRnn(model_exog, levels=["1", "2", "3"], lags=10)
    forecaster.fit(series=series, exog=exog, store_in_sample_residuals=True)
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_

    predictions = forecaster.predict_interval(
        steps=None, exog=exog_pred, use_in_sample_residuals=False
    )

    assert predictions.shape == (24, 4)


def test_predict_interval_specific_levels_with_exog():
    """
    Test case for predicting with specific levels
    """
    forecaster = ForecasterRnn(model_exog, levels=["1", "2", "3"], lags=10)
    forecaster.fit(series=series, exog=exog, store_in_sample_residuals=True)
    predictions = forecaster.predict_interval(steps=5, exog=exog_pred, levels=["1", "2"])

    assert predictions.shape == (10, 4)
