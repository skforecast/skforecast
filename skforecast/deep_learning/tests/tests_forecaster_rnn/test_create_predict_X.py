# Unit test create_predict_X method using PyTorch backend
# ==============================================================================
import os
import re
import numpy as np
import pandas as pd
import pytest
os.environ["KERAS_BACKEND"] = "torch"
import keras
from skforecast.exceptions import DataTransformationWarning
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


def test_create_predict_X_3_steps_ahead():
    """
    Test case for create_predict_X 3 steps ahead
    """
    forecaster = ForecasterRnn(estimator=model, levels=["1", "2"], lags=3)
    forecaster.fit(series=series)

    warn_msg = re.escape(
        "The output matrix is in the transformed scale due to the "
        "inclusion of transformations in the Forecaster. "
        "As a result, any predictions generated using this matrix will also "
        "be in the transformed scale. Please refer to the documentation "
        "for more details: "
        "https://skforecast.org/latest/user_guides/training-and-prediction-matrices.html"
    )
    with pytest.warns(DataTransformationWarning, match = warn_msg):
        X_predict, exog_predict = forecaster.create_predict_X(steps=3)

    expected = (
        pd.DataFrame(
            data=np.array([
                    [0.95918367, 0.95918367, 0.95918367],
                    [0.97959184, 0.97959184, 0.97959184],
                    [1.        , 1.        , 1.        ]
                ]),
            index=['lag_3', 'lag_2', 'lag_1'],
            columns=["1", "2", "3"]
        ),
        None
    )

    pd.testing.assert_frame_equal(X_predict, expected[0])
    assert exog_predict is None


def test_create_predict_X_specific_levels():
    """
    Test case for create_predict_X with specific levels
    """
    forecaster = ForecasterRnn(estimator=model, levels=["1", "2"], lags=3)
    forecaster.fit(series=series)

    warn_msg = re.escape(
        "The output matrix is in the transformed scale due to the "
        "inclusion of transformations in the Forecaster. "
        "As a result, any predictions generated using this matrix will also "
        "be in the transformed scale. Please refer to the documentation "
        "for more details: "
        "https://skforecast.org/latest/user_guides/training-and-prediction-matrices.html"
    )
    with pytest.warns(DataTransformationWarning, match = warn_msg):
        X_predict, exog_predict = forecaster.create_predict_X(steps=None, levels=["1"])

    expected = (
        pd.DataFrame(
            data=np.array([
                    [0.95918367, 0.95918367, 0.95918367],
                    [0.97959184, 0.97959184, 0.97959184],
                    [1.        , 1.        , 1.        ]
                ]),
            index=['lag_3', 'lag_2', 'lag_1'],
            columns=["1", "2", "3"]
        ),
        None
    )

    pd.testing.assert_frame_equal(X_predict, expected[0])
    assert exog_predict is None


def test_create_predict_X_exog():
    """
    Test case for create_predict_X with exogenous variables
    """
    forecaster = ForecasterRnn(
        estimator=model_exog, levels=["1", "2", "3"], lags=10
    )
    forecaster.fit(series=series, exog=exog)

    warn_msg = re.escape(
        "The output matrix is in the transformed scale due to the "
        "inclusion of transformations in the Forecaster. "
        "As a result, any predictions generated using this matrix will also "
        "be in the transformed scale. Please refer to the documentation "
        "for more details: "
        "https://skforecast.org/latest/user_guides/training-and-prediction-matrices.html"
    )
    with pytest.warns(DataTransformationWarning, match = warn_msg):
        X_predict, exog_predict = forecaster.create_predict_X(steps=None, exog=exog_pred)

    expected = (
        pd.DataFrame(
            data=np.array([
                    [0.81632653, 0.81632653, 0.81632653],
                    [0.83673469, 0.83673469, 0.83673469],
                    [0.85714286, 0.85714286, 0.85714286],
                    [0.87755102, 0.87755102, 0.87755102],
                    [0.89795918, 0.89795918, 0.89795918],
                    [0.91836735, 0.91836735, 0.91836735],
                    [0.93877551, 0.93877551, 0.93877551],
                    [0.95918367, 0.95918367, 0.95918367],
                    [0.97959184, 0.97959184, 0.97959184],
                    [1.        , 1.        , 1.        ]
                ]),
            index=[f"lag_{i}" for i in range(10, 0, -1)],
            columns=["1", "2", "3"]
        ),
        pd.DataFrame(
            data=np.array([
                    [1.02040816, 1.02040816],
                    [1.04081633, 1.04081633],
                    [1.06122449, 1.06122449],
                    [1.08163265, 1.08163265],
                    [1.10204082, 1.10204082],
                    [1.12244898, 1.12244898],
                    [1.14285714, 1.14285714],
                    [1.16326531, 1.16326531]
                ]),
            index=[f"step_{i}" for i in range(1, 9)],
            columns=["exog1", "exog2"]
        )
    )

    pd.testing.assert_frame_equal(X_predict, expected[0])
    pd.testing.assert_frame_equal(exog_predict, expected[1])


def test_create_predict_X_specific_levels_with_exog():
    """
    Test case for create_predict_X with specific levels
    """
    forecaster = ForecasterRnn(
        estimator=model_exog, levels=["1", "2", "3"], lags=10
    )
    forecaster.fit(series=series, exog=exog)

    warn_msg = re.escape(
        "The output matrix is in the transformed scale due to the "
        "inclusion of transformations in the Forecaster. "
        "As a result, any predictions generated using this matrix will also "
        "be in the transformed scale. Please refer to the documentation "
        "for more details: "
        "https://skforecast.org/latest/user_guides/training-and-prediction-matrices.html"
    )
    with pytest.warns(DataTransformationWarning, match = warn_msg):
        X_predict, exog_predict = forecaster.create_predict_X(
            steps=5, exog=exog_pred, levels=["1", "2"]
        )

    expected = (
        pd.DataFrame(
            data=np.array([
                    [0.81632653, 0.81632653, 0.81632653],
                    [0.83673469, 0.83673469, 0.83673469],
                    [0.85714286, 0.85714286, 0.85714286],
                    [0.87755102, 0.87755102, 0.87755102],
                    [0.89795918, 0.89795918, 0.89795918],
                    [0.91836735, 0.91836735, 0.91836735],
                    [0.93877551, 0.93877551, 0.93877551],
                    [0.95918367, 0.95918367, 0.95918367],
                    [0.97959184, 0.97959184, 0.97959184],
                    [1.        , 1.        , 1.        ]
                ]),
            index=[f"lag_{i}" for i in range(10, 0, -1)],
            columns=["1", "2", "3"]
        ),
        pd.DataFrame(
            data=np.array([
                    [1.02040816, 1.02040816],
                    [1.04081633, 1.04081633],
                    [1.06122449, 1.06122449],
                    [1.08163265, 1.08163265],
                    [1.10204082, 1.10204082],
                    [1.12244898, 1.12244898],
                    [1.14285714, 1.14285714],
                    [1.16326531, 1.16326531]
                ]),
            index=[f"step_{i}" for i in range(1, 9)],
            columns=["exog1", "exog2"]
        )
    )

    pd.testing.assert_frame_equal(X_predict, expected[0])
    pd.testing.assert_frame_equal(exog_predict, expected[1])
