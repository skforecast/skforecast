# Unit test create_train_X_y ForecasterRnn using PyTorch backend
# ==============================================================================
import os
import numpy as np
import pandas as pd
os.environ["KERAS_BACKEND"] = "torch"
import keras
from sklearn.preprocessing import StandardScaler
from skforecast.deep_learning.utils import create_and_compile_model
from skforecast.deep_learning import ForecasterRnn


def test_create_train_X_y_output_when_series_10_and_transformer_series_is_StandardScaler():
    """
    Test the output of create_train_X_y when exog is None and transformer_series
    is StandardScaler.
    """
    series = pd.DataFrame(
        {
            "l1": pd.Series(np.arange(10, dtype=float)),
            "l2": pd.Series(np.arange(10, dtype=float)),
        }
    )
    model = create_and_compile_model(
                series=series, 
                levels="l1",    
                lags=5,           
                steps=2,              
                recurrent_layer="LSTM",
                recurrent_units=128,
                dense_units=64,
            )
    forecaster = ForecasterRnn(
        model, levels="l1", transformer_series=StandardScaler(), lags=5
    )

    results = forecaster.create_train_X_y(series=series)
    expected = [
        np.array([[[-1.5666989 , -1.5666989 ],
        [-1.21854359, -1.21854359],
        [-0.87038828, -0.87038828],
        [-0.52223297, -0.52223297],
        [-0.17407766, -0.17407766]],

       [[-1.21854359, -1.21854359],
        [-0.87038828, -0.87038828],
        [-0.52223297, -0.52223297],
        [-0.17407766, -0.17407766],
        [ 0.17407766,  0.17407766]],

       [[-0.87038828, -0.87038828],
        [-0.52223297, -0.52223297],
        [-0.17407766, -0.17407766],
        [ 0.17407766,  0.17407766],
        [ 0.52223297,  0.52223297]],

       [[-0.52223297, -0.52223297],
        [-0.17407766, -0.17407766],
        [ 0.17407766,  0.17407766],
        [ 0.52223297,  0.52223297],
        [ 0.87038828,  0.87038828]]])
    ,
    None,
        [
            [[0.17407766], [0.52223297]],
            [[0.52223297], [0.87038828]],
            [[0.87038828], [1.21854359]],
            [[1.21854359], [1.5666989]],
        ],
    ]

    expected_dimension_names = {
        "X_train": {
            0: pd.RangeIndex(start=5, stop=9, step=1),
            1: ['lag_5', 'lag_4', 'lag_3', 'lag_2', 'lag_1'],
            2: ["l1", "l2"],
        },
        "y_train": {
            0: pd.RangeIndex(start=5, stop=9, step=1),
            1: ["step_1", "step_2"],
            2: ["l1"]
        },
        "exog_train": {0: None, 1: None, 2: None}
    }
    for i in [0, 2]:
        np.testing.assert_almost_equal(results[i], expected[i])
    results[1] == expected[1]

    results[3]['X_train'][0].equals(expected_dimension_names['X_train'][0])
    results[3]['X_train'][1] == (expected_dimension_names['X_train'][1])
    results[3]['X_train'][2] == (expected_dimension_names['X_train'][2])
    results[3]['y_train'][0].equals(expected_dimension_names['y_train'][0])
    results[3]['y_train'][1] == expected_dimension_names['y_train'][1]
    results[3]['y_train'][2] == expected_dimension_names['y_train'][2]
    results[3]['exog_train'] == expected_dimension_names['exog_train']


def test_create_train_X_y_output_when_series_10_and_transformer_series_is_StandardScaler_and_exog():
    """
    Test the output of create_train_X_y when exog is None and transformer_series
    is StandardScaler and exog is provided.
    """
    series = pd.DataFrame(
        {
            "l1": pd.Series(np.arange(10, dtype=float)),
            "l2": pd.Series(np.arange(10, dtype=float)),
        }
    )
    exog = pd.DataFrame(
        {
            "exog1": pd.Series(np.arange(10, dtype=float)),
            "exog2": pd.Series(np.arange(10, dtype=float)),
        }
    )
    model = create_and_compile_model(
                series=series,
                exog=exog,
                levels="l1",
                lags=5,
                steps=2,
                recurrent_layer="LSTM",
                recurrent_units=128,
                dense_units=64,
            )
    forecaster = ForecasterRnn(
        model, levels="l1", transformer_series=StandardScaler(), lags=5
    )

    (
        X_train,
        exog_train,
        y_train,
        dimension_names
    ) = forecaster.create_train_X_y(series=series, exog=exog)

    expected_X_train = np.array(
        [[[-1.5666989 , -1.5666989 ],
        [-1.21854359, -1.21854359],
        [-0.87038828, -0.87038828],
        [-0.52223297, -0.52223297],
        [-0.17407766, -0.17407766]],

       [[-1.21854359, -1.21854359],
        [-0.87038828, -0.87038828],
        [-0.52223297, -0.52223297],
        [-0.17407766, -0.17407766],
        [ 0.17407766,  0.17407766]],

       [[-0.87038828, -0.87038828],
        [-0.52223297, -0.52223297],
        [-0.17407766, -0.17407766],
        [ 0.17407766,  0.17407766],
        [ 0.52223297,  0.52223297]],

       [[-0.52223297, -0.52223297],
        [-0.17407766, -0.17407766],
        [ 0.17407766,  0.17407766],
        [ 0.52223297,  0.52223297],
        [ 0.87038828,  0.87038828]]]
    )
    expected_exog_train = np.array(
        [[[0.55555556, 0.55555556],
        [0.66666667, 0.66666667]],
       [[0.66666667, 0.66666667],
        [0.77777778, 0.77777778]],
       [[0.77777778, 0.77777778],
        [0.88888889, 0.88888889]],
       [[0.88888889, 0.88888889],
        [1.        , 1.        ]]]
    )
    expected_y_train = np.array(
        [
            [[0.17407766], [0.52223297]],
            [[0.52223297], [0.87038828]],
            [[0.87038828], [1.21854359]],
            [[1.21854359], [1.5666989]],
        ]
    )

    expected_dimension_names = {
        "X_train": {
            0: pd.RangeIndex(start=5, stop=9, step=1),
            1: ['lag_5', 'lag_4', 'lag_3', 'lag_2', 'lag_1'],
            2: ["l1", "l2"],
        },
        "y_train": {
            0: pd.RangeIndex(start=5, stop=9, step=1),
            1: ["step_1", "step_2"],
            2: ["l1"]
        },
        "exog_train": {
            0: pd.RangeIndex(start=5, stop=9, step=1),
            1: ['step_1', 'step_2'],
            2: ['exog1', 'exog2']
        }
    }
    
    np.testing.assert_almost_equal(X_train, expected_X_train)
    np.testing.assert_almost_equal(exog_train, expected_exog_train)
    np.testing.assert_almost_equal(y_train, expected_y_train)
    assert dimension_names['X_train'][0].equals(expected_dimension_names['X_train'][0])
    assert dimension_names['X_train'][1] == expected_dimension_names['X_train'][1]
    assert dimension_names['X_train'][2] == expected_dimension_names['X_train'][2]
    assert dimension_names['y_train'][0].equals(expected_dimension_names['y_train'][0])
    assert dimension_names['y_train'][1] == expected_dimension_names['y_train'][1]
    assert dimension_names['y_train'][2] == expected_dimension_names['y_train'][2]
    assert dimension_names['exog_train'][0].equals(expected_dimension_names['exog_train'][0])
    assert dimension_names['exog_train'][1] == expected_dimension_names['exog_train'][1]
    assert dimension_names['exog_train'][2] == expected_dimension_names['exog_train'][2]
