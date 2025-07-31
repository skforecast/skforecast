# Unit test _create_train_X_y ForecasterRnn using PyTorch backend
# ==============================================================================
import os
import re
import pytest
import numpy as np
import pandas as pd
os.environ["KERAS_BACKEND"] = "torch"
import keras
from sklearn.preprocessing import StandardScaler
from skforecast.deep_learning.utils import create_and_compile_model
from skforecast.deep_learning import ForecasterRnn

series = pd.DataFrame(np.random.randn(100, 3), columns=['l1', 'l2', 'l3'])


def test_create_train_X_y_TypeError_when_series_not_dataframe():
    """
    Test TypeError is raised when series is not a pandas DataFrame.
    """
    series = pd.DataFrame(np.random.randn(100, 3), columns=['l1', 'l2', 'l3'])
    model = create_and_compile_model(
                series=series, 
                levels="l1",    
                lags=3,           
                steps=1,              
                recurrent_layer="LSTM",
                recurrent_units=100,
                dense_units=[128, 64],
            )
    forecaster = ForecasterRnn(model, lags=3, levels="l1")
    series = pd.Series(np.arange(7))

    err_msg = (
        "`series` must be a pandas DataFrame. Got <class 'pandas.core.series.Series'>."
    )
    with pytest.raises(TypeError, match=err_msg):
        forecaster._create_train_X_y(series=series)


def test_create_train_X_y_UserWarning_when_levels_of_transformer_series_not_equal_to_series_col_names():
    """
    Test UserWarning is raised when `transformer_series` is a dict and its keys are
    not the same as forecaster.series_col_names.
    """
    series = pd.DataFrame({"1": pd.Series(np.arange(5)), "2": pd.Series(np.arange(5))})
    dict_transformers = {"1": StandardScaler(), "3": StandardScaler()}
    model = create_and_compile_model(
                series=series, 
                levels="1",    
                lags=3,           
                steps=1,              
                recurrent_layer="LSTM",
                recurrent_units=100,
                dense_units=[128, 64],
            )
    forecaster = ForecasterRnn(
        model, levels="1", transformer_series=dict_transformers, lags=3
    )

    series_not_in_transformer_series = set(series.columns) - set(
        forecaster.transformer_series.keys()
    )

    warn_msg = re.escape(
        f"{series_not_in_transformer_series} not present in `transformer_series`."
        f" No transformation is applied to these series."
    )
    with pytest.warns(UserWarning, match=warn_msg):
        forecaster._create_train_X_y(series=series)


def test_create_train_X_y_ValueError_when_all_series_values_are_missing():
    """
    Test ValueError is raised when all series values are missing.
    """
    series = pd.DataFrame({"1": pd.Series(np.arange(7)), "2": pd.Series([np.nan] * 7)})
    series.index = pd.date_range(start="2022-01-01", periods=7, freq="1D")
    model = create_and_compile_model(
                series=series, 
                levels="1",    
                lags=3,           
                steps=1,              
                recurrent_layer="LSTM",
                recurrent_units=100,
                dense_units=[128, 64],
            )
    forecaster = ForecasterRnn(model, levels="1", lags=3)

    err_msg = re.escape("`y` has missing values.")
    with pytest.raises(ValueError, match=err_msg):
        forecaster._create_train_X_y(series=series)


@pytest.mark.parametrize(
    "values",
    [
        [0, 1, 2, 3, 4, 5, np.nan],
        [0, 1] + [np.nan] * 5,
        [np.nan, 1, 2, 3, 4, 5, np.nan],
        [0, 1, np.nan, 3, np.nan, 5, 6],
        [np.nan, np.nan, np.nan, 3, np.nan, 5, 6],
    ],
)
def test_create_train_X_y_ValueError_when_series_values_are_missing(values):
    """
    Test ValueError is raised when series values are missing in different
    locations.
    """
    series = pd.DataFrame({"1": pd.Series(values), "2": pd.Series(np.arange(7))})
    series.index = pd.date_range(start="2022-01-01", periods=7, freq="1D")
    model = create_and_compile_model(
                series=series, 
                levels="1",    
                lags=3,           
                steps=1,              
                recurrent_layer="LSTM",
                recurrent_units=100,
                dense_units=[128, 64],
            )
    forecaster = ForecasterRnn(model, levels="1", lags=3)

    err_msg = re.escape("`y` has missing values.")
    with pytest.raises(ValueError, match=err_msg):
        forecaster._create_train_X_y(series=series)


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

    (
        X_train,
        exog_train,
        y_train,
        dimension_names,
        exog_names_in_,
        exog_dtypes_in_,
        exog_dtypes_out_
    ) = forecaster._create_train_X_y(series=series)

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
    expected_exog_train = None
    expected_y_train = np.array(
        [
            [[0.17407766], [0.52223297]],
            [[0.52223297], [0.87038828]],
            [[0.87038828], [1.21854359]],
            [[1.21854359], [1.5666989]],
        ]
    )
    expected_exog_names_in_ = None
    expected_exog_dtypes_in_ = None
    expected_exog_dtypes_out_ = None

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
    
    np.testing.assert_almost_equal(X_train, expected_X_train)
    assert exog_train is expected_exog_train
    np.testing.assert_almost_equal(y_train, expected_y_train)
    assert exog_names_in_ is expected_exog_names_in_
    assert exog_dtypes_in_ is expected_exog_dtypes_in_
    assert exog_dtypes_out_ is expected_exog_dtypes_out_
    assert dimension_names['X_train'][0].equals(expected_dimension_names['X_train'][0])
    assert dimension_names['X_train'][1] == expected_dimension_names['X_train'][1]
    assert dimension_names['X_train'][2] == expected_dimension_names['X_train'][2]
    assert dimension_names['y_train'][0].equals(expected_dimension_names['y_train'][0])
    assert dimension_names['y_train'][1] == expected_dimension_names['y_train'][1]
    assert dimension_names['y_train'][2] == expected_dimension_names['y_train'][2]
    assert dimension_names['exog_train'] == expected_dimension_names['exog_train']


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
        dimension_names,
        exog_names_in_,
        exog_dtypes_in_,
        exog_dtypes_out_
    ) = forecaster._create_train_X_y(series=series, exog=exog)

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
    expected_exog_names_in_ = ['exog1', 'exog2']
    expected_exog_dtypes_in_ = {'exog1': np.dtype('float64'), 'exog2': np.dtype('float64')}
    expected_exog_dtypes_out_ = {'exog1': np.dtype('float64'), 'exog2': np.dtype('float64')}

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
    assert exog_names_in_ == expected_exog_names_in_
    assert exog_dtypes_in_ == expected_exog_dtypes_in_
    assert exog_dtypes_out_ == expected_exog_dtypes_out_
    assert dimension_names['X_train'][0].equals(expected_dimension_names['X_train'][0])
    assert dimension_names['X_train'][1] == expected_dimension_names['X_train'][1]
    assert dimension_names['X_train'][2] == expected_dimension_names['X_train'][2]
    assert dimension_names['y_train'][0].equals(expected_dimension_names['y_train'][0])
    assert dimension_names['y_train'][1] == expected_dimension_names['y_train'][1]
    assert dimension_names['y_train'][2] == expected_dimension_names['y_train'][2]
    assert dimension_names['exog_train'][0].equals(expected_dimension_names['exog_train'][0])
    assert dimension_names['exog_train'][1] == expected_dimension_names['exog_train'][1]
    assert dimension_names['exog_train'][2] == expected_dimension_names['exog_train'][2]
