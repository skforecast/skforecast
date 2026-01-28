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
    forecaster = ForecasterRnn(estimator=model, lags=3, levels="l1")
    series = pd.Series(np.arange(7))

    err_msg = (
        "`series` must be a pandas DataFrame. Got <class 'pandas.core.series.Series'>."
    )
    with pytest.raises(TypeError, match=err_msg):
        forecaster._create_train_X_y(series=series)


def test_create_train_X_y_ValueError_when_number_of_series_in_series_not_equal_to_forecaster():
    """
    Test ValueError is raised when number of series in `series` is not equal to
    the number of series expected by the forecaster.
    """
    series = pd.DataFrame(np.random.randn(100, 3), columns=['l1', 'l2', 'l3'])
    model = create_and_compile_model(
                series=series[['l1', 'l2']], 
                levels="l1",    
                lags=3,           
                steps=1,              
                recurrent_layer="LSTM",
                recurrent_units=100,
                dense_units=[128, 64],
            )
    forecaster = ForecasterRnn(estimator=model, lags=3, levels="l1")

    err_msg = re.escape(
        "Number of series in `series` (3) "
        "does not match the number of series expected by the model "
        "architecture (2)."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster._create_train_X_y(series=series)


def test_create_train_X_y_ValueError_when_series_not_include_levels():
    """
    Test ValueError is raised when `levels` defined when initializing the forecaster
    is not included in `series` used for training.
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
    forecaster = ForecasterRnn(estimator=model, lags=3, levels="l1")
    
    series_not_levels = series.copy()
    series_not_levels.columns = ['l4', 'l5', 'l6']

    err_msg = (
        f"`levels` defined when initializing the forecaster must be "
        f"included in `series` used for training. "
        f"{set(forecaster.levels) - set(list(series_not_levels.columns))} not found."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster._create_train_X_y(series=series_not_levels)


def test_create_train_X_y_ValueError_when_len_series_is_lower_than_maximum_window_size_plus_steps():
    """
    Test ValueError is raised when length of series is lower than maximum window_size 
    plus number of steps included in the forecaster.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(5)),  
                           'l2': pd.Series(np.arange(5))})
    model = create_and_compile_model(
                series=series, 
                levels="l1",    
                lags=3,           
                steps=3,              
                recurrent_layer="LSTM",
                recurrent_units=100,
                dense_units=[128, 64],
            )
    forecaster = ForecasterRnn(estimator=model, lags=3, levels="l1")

    err_msg = re.escape(
        "Minimum length of `series` for training this forecaster is "
        "6. Reduce the number of "
        "predicted steps, 3, or the maximum "
        "lag, 3, if no more data is available.\n"
        "    Length `series`: 5.\n"
        "    Max step : 3.\n"
        "    Lags window size: 3."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(series=series)


def test_create_train_X_y_ValueError_when_exog_not_included_but_model_requires_exog():
    """
    Test ValueError is raised when `exog` is not included but the model
    requires `exog` to be present.
    """
    series = pd.DataFrame(np.random.randn(100, 3), columns=['l1', 'l2', 'l3'])
    exog = pd.DataFrame(np.random.randn(100, 2), columns=['exog1', 'exog2'])
    model = create_and_compile_model(
                series=series,
                exog=exog, 
                levels="l1",    
                lags=3,           
                steps=1,              
                recurrent_layer="LSTM",
                recurrent_units=100,
                dense_units=[128, 64],
            )
    forecaster = ForecasterRnn(estimator=model, lags=3, levels="l1")

    err_msg = re.escape(
        "The estimator architecture expects exogenous variables during "
        "training. Please provide the `exog` argument. If this is "
        "unexpected, check your estimator architecture or the "
        "initialization parameters of the forecaster."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster._create_train_X_y(series=series)


def test_create_train_X_y_ValueError_when_exog_included_but_model_no_exog():
    """
    Test ValueError is raised when `exog` is included but the model
    does not require `exog` to be present.
    """
    series = pd.DataFrame(np.random.randn(100, 3), columns=['l1', 'l2', 'l3'])
    exog = pd.DataFrame(np.random.randn(100, 2), columns=['exog1', 'exog2'])
    model = create_and_compile_model(
                series=series,
                levels="l1",    
                lags=3,           
                steps=1,              
                recurrent_layer="LSTM",
                recurrent_units=100,
                dense_units=[128, 64],
            )
    forecaster = ForecasterRnn(estimator=model, lags=3, levels="l1")

    err_msg = re.escape(
        "Exogenous variables (`exog`) were provided, but the model "
        "architecture was not built to expect exogenous variables. Please "
        "remove the `exog` argument or rebuild the model to include "
        "exogenous inputs."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster._create_train_X_y(series=series, exog=exog)


def test_create_train_X_y_ValueError_when_number_of_exog_columns_not_equal_to_model():
    """
    Test ValueError is raised when number of columns in `exog` is not equal to
    the number of columns expected by the model.
    """
    series = pd.DataFrame(np.random.randn(100, 3), columns=['l1', 'l2', 'l3'])
    exog = pd.DataFrame(np.random.randn(100, 2), columns=['exog1', 'exog2'])
    model = create_and_compile_model(
                series=series,
                exog=exog, 
                levels="l1",    
                lags=3,    
                steps=1,              
                recurrent_layer="LSTM",
                recurrent_units=100,
                dense_units=[128, 64],
            )
    forecaster = ForecasterRnn(estimator=model, lags=3, levels="l1")

    err_msg = re.escape(
        "Number of columns in `exog` (1) "
        "does not match the number of exogenous variables expected "
        "by the model architecture (2)."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster._create_train_X_y(series=series, exog=exog.iloc[:, :1])


@pytest.mark.parametrize("exog", 
    [pd.Series(np.arange(15), name='exog'), 
     pd.DataFrame(np.arange(50).reshape(25, 2)),
     pd.Series(np.arange(50), index=pd.date_range(start='2022-01-01', periods=50, freq='1D'), name='exog')
])
def test_create_train_X_y_ValueError_when_series_and_exog_have_different_length(exog):
    """
    Test ValueError is raised when length of series is not equal to length exog.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10))})
    if isinstance(exog.index, pd.DatetimeIndex):
        series.index = exog.index[:10]
    
    model = create_and_compile_model(
                series=series,
                exog=exog, 
                levels="l1",    
                lags=3,    
                steps=3,              
                recurrent_layer="LSTM",
                recurrent_units=100,
                dense_units=[128, 64],
            )
    forecaster = ForecasterRnn(estimator=model, lags=3, levels="l1")

    len_exog = len(exog)
    len_series = len(series)
    series_index_no_ws = series.index[forecaster.window_size:]
    len_series_no_ws = len(series_index_no_ws)
    err_msg = re.escape(
        f"Length of `exog` must be equal to the length of `series` (if "
        f"index is fully aligned) or length of `series` - `window_size` "
        f"(if `exog` starts after the first `window_size` values).\n"
        f"    `exog`                   : ({exog.index[0]} -- {exog.index[-1]})  (n={len_exog})\n"
        f"    `series`                 : ({series.index[0]} -- {series.index[-1]})  (n={len_series})\n"
        f"    `series` - `window_size` : ({series_index_no_ws[0]} -- {series_index_no_ws[-1]})  (n={len_series_no_ws})"
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(series=series, exog=exog)


def test_create_train_X_y_ValueError_when_exog_columns_same_as_series_col_names():
    """
    Test ValueError is raised when an exog column is named the same as
    the series columns.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10))})
    exog = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                         'exog2': pd.Series(np.arange(10))})

    model = create_and_compile_model(
                series=series,
                exog=exog, 
                levels="l1",    
                lags=3,    
                steps=3,              
                recurrent_layer="LSTM",
                recurrent_units=100,
                dense_units=[128, 64],
            )
    forecaster = ForecasterRnn(estimator=model, lags=3, levels="l1")

    series_col_names = list(series.columns)
    exog_col_names = list(exog.columns)

    err_msg = re.escape(
        f"`exog` cannot contain a column named the same as one of "
        f"the series (column names of series).\n"
        f"  `series` columns : {series_col_names}.\n"
        f"  `exog`   columns : {exog_col_names}."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(series=series, exog=exog)


def test_create_train_X_y_ValueError_when_transformer_exog_alter_shape_of_exog():
    """
    Test ValueError is raised when transformer_exog alters the shape of `exog`.
    """
    series = pd.DataFrame(
        {"l1": pd.Series(np.arange(10)), "l2": pd.Series(np.arange(10))}
    )
    series.index = pd.date_range(start="2022-01-01", periods=10, freq="D")
    exog = pd.DataFrame(
        {"exog_1": pd.Series(np.arange(10)), "exog_2": pd.Series(np.arange(10))}
    )
    exog.index = pd.date_range(start="2022-01-01", periods=10, freq="D")

    model = create_and_compile_model(
                series=series,
                exog=exog, 
                levels="l1",    
                lags=3,    
                steps=3,              
                recurrent_layer="LSTM",
                recurrent_units=100,
                dense_units=[128, 64],
            )
    
    class CustomTransformer:  # pragma: no cover
        """
        Custom transformer that alters the shape of `exog`.
        """
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            self.categories_ = ['exog_1', 'exog_2', 'exog_3']
            return X.assign(exog_3=X['exog_1'] + X['exog_2'])
        
        def fit_transform(self, X, y=None):
            return self.transform(X)
    
    forecaster = ForecasterRnn(
        estimator=model, lags=3, levels="l1", transformer_exog=CustomTransformer()
    )

    err_msg = re.escape(
        "Number of columns in `exog` after transformation (3) "
        "does not match the number of columns before transformation (2). "
        "The ForecasterRnn does not support transformations that "
        "change the number of columns in `exog`. Preprocess `exog` "
        "before passing it to the `create_and_compile_model` function."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster._create_train_X_y(series=series, exog=exog)


def test_create_train_X_y_ValueError_when_series_and_exog_have_different_index_but_same_length():
    """
    Test ValueError is raised when series and exog have different index but same length.
    """
    series = pd.DataFrame(
        {"l1": pd.Series(np.arange(10)), "l2": pd.Series(np.arange(10))}
    )
    series.index = pd.date_range(start="2022-01-01", periods=10, freq="1D")
    exog = pd.Series(
        np.arange(10), index=pd.RangeIndex(start=0, stop=10, step=1), name="exog"
    )

    model = create_and_compile_model(
                series=series,
                exog=exog, 
                levels="l1",    
                lags=3,    
                steps=3,              
                recurrent_layer="LSTM",
                recurrent_units=100,
                dense_units=[128, 64],
            )
    forecaster = ForecasterRnn(estimator=model, lags=3, levels="l1")

    err_msg = re.escape(
        "When `exog` has the same length as `series`, the index "
        "of `exog` must be aligned with the index of `series` "
        "to ensure the correct alignment of values."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster._create_train_X_y(series=series, exog=exog)


def test_create_train_X_y_ValueError_when_series_and_exog_have_different_index_and_length_exog_no_window_size():
    """
    Test ValueError is raised when y and exog have different index and
    length exog no window_size.
    """
    series = pd.DataFrame(
        {"l1": pd.Series(np.arange(10)), "l2": pd.Series(np.arange(10))}
    )
    series.index = pd.date_range(start="2022-01-01", periods=10, freq="1D")
    exog = pd.Series(
        np.arange(3, 10), index=pd.RangeIndex(start=3, stop=10, step=1), name="exog"
    )

    model = create_and_compile_model(
                series=series,
                exog=exog, 
                levels="l1",    
                lags=3,    
                steps=3,              
                recurrent_layer="LSTM",
                recurrent_units=100,
                dense_units=[128, 64],
            )
    forecaster = ForecasterRnn(estimator=model, lags=3, levels="l1")

    err_msg = re.escape(
        "When `exog` doesn't contain the first `window_size` "
        "observations, the index of `exog` must be aligned with "
        "the index of `series` minus the first `window_size` "
        "observations to ensure the correct alignment of values."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster._create_train_X_y(series=series, exog=exog)


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
        estimator=model, levels="1", transformer_series=dict_transformers, lags=3
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
    forecaster = ForecasterRnn(estimator=model, levels="1", lags=3)

    err_msg = re.escape("series '2' has missing values.")
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
    forecaster = ForecasterRnn(estimator=model, levels="1", lags=3)

    err_msg = re.escape("series '1' has missing values.")
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
        estimator=model, levels="l1", transformer_series=StandardScaler(), lags=5
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
        estimator=model, levels="l1", transformer_series=StandardScaler(), lags=5
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
