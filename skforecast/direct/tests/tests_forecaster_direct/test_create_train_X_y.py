# Unit test _create_train_X_y ForecasterDirect
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.exceptions import MissingValuesWarning
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from skforecast.preprocessing import TimeSeriesDifferentiator
from skforecast.preprocessing import RollingFeatures
from skforecast.direct import ForecasterDirect

# Fixtures
from .fixtures_forecaster_direct import data  # to test results when using differentiation


def test_create_train_X_y_ValueError_when_len_y_is_lower_than_maximum_window_size_plus_steps():
    """
    Test ValueError is raised when length of y is lower than maximum window_size 
    plus number of steps included in the forecaster.
    """
    y = pd.Series(np.arange(5), name='y')

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=3
    )
    err_msg = re.escape(
        "Minimum length of `y` for training this forecaster is "
        "6. Reduce the number of "
        "predicted steps, 3, or the maximum "
        "window_size, 3, if no more data is available.\n"
        "    Length `y`: 5.\n"
        "    Max step : 3.\n"
        "    Max window size: 3.\n"
        "    Lags window size: 3.\n"
        "    Window features window size: None."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(y=y)

    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=3)
    forecaster = ForecasterDirect(
        estimator=LinearRegression(),  steps=3, lags=2, window_features=rolling
    )
    err_msg = re.escape(
        "Minimum length of `y` for training this forecaster is "
        "6. Reduce the number of "
        "predicted steps, 3, or the maximum "
        "window_size, 3, if no more data is available.\n"
        "    Length `y`: 5.\n"
        "    Max step : 3.\n"
        "    Max window size: 3.\n"
        "    Lags window size: 2.\n"
        "    Window features window size: 3."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(y=y)


def test_create_train_X_y_ValueError_when_categorical_features_columns_not_in_exog():
    """
    Test ValueError is raised when explicit categorical_features list contains
    columns not present in exog after transformer_exog.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=float)})
    forecaster = ForecasterDirect(
        LinearRegression(), lags=5, steps=1,
        categorical_features=['exog_1', 'non_existent']
    )
    err_msg = re.escape(
        "The following columns specified in `categorical_features` "
        "are not present in `exog` after `transformer_exog`: "
        "{'non_existent'}."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster._create_train_X_y(y=y, exog=exog)


def test_create_train_X_y_TypeError_when_exog_is_categorical_of_no_int():
    """
    Test TypeError is raised when exog is categorical with no int values
    and categorical_features=None.
    """
    y = pd.Series(np.arange(5))
    exog = pd.Series(['A', 'B', 'C', 'D', 'E'], name='exog', dtype='category')
    forecaster = ForecasterDirect(
        LinearRegression(), lags=2, steps=1, categorical_features=None
    )

    err_msg = re.escape(
        "Categorical dtypes in exog must contain only integer values. "
        "See skforecast docs for more info about how to include "
        "categorical features https://skforecast.org/"
        "latest/user_guides/categorical-features.html"
    )
    with pytest.raises(TypeError, match=err_msg):
        forecaster._create_train_X_y(y=y, exog=exog)


def test_create_train_X_y_MissingValuesWarning_when_exog_has_missing_values():
    """
    Test _create_train_X_y is issues a MissingValuesWarning when exog has missing values.
    """
    y = pd.Series(np.arange(6))
    exog = pd.Series([1, 2, 3, np.nan, 5, 6], name='exog')
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=2, steps=2
    )

    warn_msg = re.escape(
        "`exog` has missing values. Most machine learning models do "
        "not allow missing values. Fitting the forecaster may fail."
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):
        forecaster._create_train_X_y(y=y, exog=exog)


@pytest.mark.parametrize(
    "y                        , exog", 
    [(pd.Series(np.arange(50), name='y'), pd.Series(np.arange(10), name='exog')), 
     (pd.Series(np.arange(10), name='y'), pd.Series(np.arange(50), name='exog')), 
     (pd.Series(np.arange(10), name='y'), pd.DataFrame(np.arange(50).reshape(25, 2), columns=['exog_1', 'exog_2'])),
     (pd.Series(np.arange(10), index=pd.date_range(start='2022-01-01', periods=10, freq='1D'), name='y'), 
      pd.Series(np.arange(50), index=pd.date_range(start='2022-01-01', periods=50, freq='1D'), name='exog'))
])
def test_create_train_X_y_ValueError_when_len_y_is_different_from_len_exog(y, exog):
    """
    Test ValueError is raised when length of y is not equal to length exog.
    """
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=3
    )

    len_exog = len(exog)
    len_y = len(y)
    y_index_no_ws = y.index[forecaster.window_size:]
    len_y_no_ws = len(y_index_no_ws)
    err_msg = re.escape(
        f"Length of `exog` must be equal to the length of `y` (if index is "
        f"fully aligned) or length of `y` - `window_size` (if `exog` "
        f"starts after the first `window_size` values).\n"
        f"    `exog`              : ({exog.index[0]} -- {exog.index[-1]})  (n={len_exog})\n"
        f"    `y`                 : ({y.index[0]} -- {y.index[-1]})  (n={len_y})\n"
        f"    `y` - `window_size` : ({y_index_no_ws[0]} -- {y_index_no_ws[-1]})  (n={len_y_no_ws})"
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(y=y, exog=exog)


def test_create_train_X_y_ValueError_when_y_and_exog_have_different_index_but_same_length():
    """
    Test ValueError is raised when y and exog have different index but same length.
    """
    y = pd.Series(np.arange(10), name='y')
    y.index = pd.date_range(start='2022-01-01', periods=10, freq='1D')
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=3
    )

    err_msg = re.escape(
        "When `exog` has the same length as `y`, the index of "
        "`exog` must be aligned with the index of `y` "
        "to ensure the correct alignment of values." 
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.fit(
            y=y,
            exog=pd.Series(
                np.arange(10),
                index=pd.RangeIndex(start=0, stop=10, step=1),
                name="exog",
            ),
        )


def test_create_train_X_y_ValueError_when_y_and_exog_have_different_index_and_length_exog_no_window_size():
    """
    Test ValueError is raised when y and exog have different index and
    length exog no window_size.
    """
    y = pd.Series(np.arange(10), name='y')
    y.index = pd.date_range(start='2022-01-01', periods=10, freq='1D')
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=3
    )

    err_msg = re.escape(
        "When `exog` doesn't contain the first `window_size` observations, "
        "the index of `exog` must be aligned with the index of `y` minus "
        "the first `window_size` observations to ensure the correct "
        "alignment of values."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.fit(
            y=y,
            exog=pd.Series(
                np.arange(3, 10),
                index=pd.RangeIndex(start=3, stop=10, step=1),
                name="exog",
            ),
        )


def test_create_train_X_y_output_when_lags_3_steps_1_and_exog_is_None():
    """
    Test output of _create_train_X_y when estimator is LinearRegression, 
    lags is 3 and steps is 1.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = None

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=1
    )
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        np.array([[2., 1., 0.],
                  [3., 2., 1.],
                  [4., 3., 2.],
                  [5., 4., 3.],
                  [6., 5., 4.],
                  [7., 6., 5.],
                  [8., 7., 6.]], dtype=float),
        {1: np.array([3., 4., 5., 6., 7., 8., 9.], dtype=float)},
        {1: pd.RangeIndex(start=3, stop=10, step=1)},
        None,
        None,
        None,
        ['lag_1', 'lag_2', 'lag_3'],
        None,
        None
    )

    forecaster.exog_in_ is False
    forecaster.X_train_direct_exog_names_out_ is None
    forecaster.X_train_window_features_names_out_ is None

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    assert results[8] == expected[8]


def test_create_train_X_y_output_when_interspersed_lags_steps_2_and_exog_is_None():
    """
    Test output of _create_train_X_y when estimator is LinearRegression, 
    interspersed_lags and steps is 2.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = None

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=[1, 3], steps=2
    )
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        np.array([[2., 0.],
                  [3., 1.],
                  [4., 2.],
                  [5., 3.],
                  [6., 4.],
                  [7., 5.]], dtype=float),
        {1: np.array([3., 4., 5., 6., 7., 8.], dtype=float),
         2: np.array([4., 5., 6., 7., 8., 9.], dtype=float)},
        {1: pd.RangeIndex(start=3, stop=9, step=1),
         2: pd.RangeIndex(start=4, stop=10, step=1)},
        None,
        None,
        None,
        ['lag_1', 'lag_3'],
        None,
        None
    )

    forecaster.exog_in_ is False
    forecaster.X_train_direct_exog_names_out_ is None
    forecaster.X_train_window_features_names_out_ is None

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    assert results[8] == expected[8]


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt: f'dtype: {dt}')
def test_create_train_X_y_output_when_y_is_series_10_steps_1_and_exog_is_series_of_float_int(dtype):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)), steps=1 
    and exog is a pandas Series of floats or ints.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=dtype)

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=1
    )
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        np.array([[4., 3., 2., 1., 0., 105.],
                  [5., 4., 3., 2., 1., 106.],
                  [6., 5., 4., 3., 2., 107.],
                  [7., 6., 5., 4., 3., 108.],
                  [8., 7., 6., 5., 4., 109.]], dtype=float),
        {1: np.array([5., 6., 7., 8., 9.], dtype=float)},
        {1: pd.RangeIndex(start=5, stop=10, step=1)},
        ['exog'],
        [],
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_step_1'],
        {'exog': exog.dtypes},
        {'exog': exog.dtypes}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1']
    forecaster.X_train_window_features_names_out_ is None

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt: f'dtype: {dt}')
def test_create_train_X_y_output_when_y_is_series_10_steps_2_and_exog_is_series_of_float_int(dtype):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)), steps=2 
    and exog is a pandas Series of floats or ints.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=dtype)
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=2
    )
    results = forecaster._create_train_X_y(y=y, exog=exog)
    expected = (
        np.array([[4., 3., 2., 1., 0., 105., 106.],
                  [5., 4., 3., 2., 1., 106., 107.],
                  [6., 5., 4., 3., 2., 107., 108.],
                  [7., 6., 5., 4., 3., 108., 109.]], dtype=float),
        {1: np.array([5., 6., 7., 8.], dtype=float),
         2: np.array([6., 7., 8., 9.], dtype=float)},
        {1: pd.RangeIndex(start=5, stop=9, step=1),
         2: pd.RangeIndex(start=6, stop=10, step=1)},
        ['exog'],
        [],
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_step_1', 'exog_step_2'],
        {'exog': exog.dtypes},
        {'exog': exog.dtypes}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1', 'exog_step_2']
    forecaster.X_train_window_features_names_out_ is None

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]


@pytest.mark.parametrize("datetime_index", 
                         [True, False], 
                         ids = lambda dt: f'datetime_index: {dt}')
@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt: f'dtype: {dt}')
def test_create_train_X_y_output_when_exog_as_float_int_with_no_window_size(datetime_index, dtype):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)), steps=1 
    and exog is a pandas Series of floats or ints and no initial window_size
    observations.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(
        np.arange(105, 110), index=pd.RangeIndex(start=5, stop=10, step=1), 
        name='exog', dtype=dtype
    )

    expected_index = pd.RangeIndex(start=5, stop=10, step=1)
    if datetime_index:
        y.index = pd.date_range(start='2022-01-01', periods=10, freq='D')
        exog.index = pd.date_range(start='2022-01-06', periods=5, freq='D')
        expected_index = pd.date_range(start='2022-01-06', periods=5, freq='D')

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=1
    )
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        np.array([[4., 3., 2., 1., 0., 105.],
                  [5., 4., 3., 2., 1., 106.],
                  [6., 5., 4., 3., 2., 107.],
                  [7., 6., 5., 4., 3., 108.],
                  [8., 7., 6., 5., 4., 109.]], dtype=float),
        {1: np.array([5., 6., 7., 8., 9.], dtype=float)},
        {1: expected_index},
        ['exog'],
        [],
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_step_1'],
        {'exog': exog.dtypes},
        {'exog': exog.dtypes}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1']
    forecaster.X_train_window_features_names_out_ is None

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]


@pytest.mark.parametrize("datetime_index", 
                         [True, False], 
                         ids = lambda dt: f'datetime_index: {dt}')
@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt: f'dtype: {dt}')
def test_create_train_X_y_output_when_steps_2_and_exog_as_float_int_with_no_window_size(datetime_index, dtype):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)), steps=2 
    and exog is a pandas Series of floats or ints and no initial window_size
    observations.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(
        np.arange(105, 110), index=pd.RangeIndex(start=5, stop=10, step=1), 
        name='exog', dtype=dtype
    )

    expected_index_1 = pd.RangeIndex(start=5, stop=9, step=1)
    expected_index_2 = pd.RangeIndex(start=6, stop=10, step=1)
    if datetime_index:
        y.index = pd.date_range(start='2022-01-01', periods=10, freq='D')
        exog.index = pd.date_range(start='2022-01-06', periods=5, freq='D')
        expected_index_1 = pd.date_range(start='2022-01-06', periods=4, freq='D')
        expected_index_2 = pd.date_range(start='2022-01-07', periods=4, freq='D')

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=2
    )
    results = forecaster._create_train_X_y(y=y, exog=exog)
    expected = (
        np.array([[4., 3., 2., 1., 0., 105., 106.],
                  [5., 4., 3., 2., 1., 106., 107.],
                  [6., 5., 4., 3., 2., 107., 108.],
                  [7., 6., 5., 4., 3., 108., 109.]], dtype=float),
        {1: np.array([5., 6., 7., 8.], dtype=float),
         2: np.array([6., 7., 8., 9.], dtype=float)},
        {1: expected_index_1,
         2: expected_index_2},
        ['exog'],
        [],
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_step_1', 'exog_step_2'],
        {'exog': exog.dtypes},
        {'exog': exog.dtypes}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1', 'exog_step_2']
    forecaster.X_train_window_features_names_out_ is None

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt: f'dtype: {dt}')
def test_create_train_X_y_output_when_steps_1_and_exog_is_df_of_float_int(dtype):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)), steps=1 
    and exog is a pandas DataFrame of floats or ints.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=dtype),
                         'exog_2': np.arange(1000, 1010, dtype=dtype)})
    
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=1
    )
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        np.array([[4., 3., 2., 1., 0., 105., 1005.],
                  [5., 4., 3., 2., 1., 106., 1006.],
                  [6., 5., 4., 3., 2., 107., 1007.],
                  [7., 6., 5., 4., 3., 108., 1008.],
                  [8., 7., 6., 5., 4., 109., 1009.]], dtype=float),
        {1: np.array([5., 6., 7., 8., 9.], dtype=float)},
        {1: pd.RangeIndex(start=5, stop=10, step=1)},
        ['exog_1', 'exog_2'],
        [],
        ['exog_1', 'exog_2'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_1_step_1', 'exog_2_step_1'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes},
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1', 'exog_step_2']
    forecaster.X_train_window_features_names_out_ is None

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt: f'dtype: {dt}')
def test_create_train_X_y_output_when_steps_3_and_exog_is_df_of_float_int(dtype):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)), steps=3 
    and exog is a pandas DataFrame of floats or ints.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=dtype),
                         'exog_2': np.arange(1000, 1010, dtype=dtype)})
    
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=3
    )
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        np.array([[4, 3, 2, 1, 0, 105, 1005, 106, 1006, 107, 1007],
                  [5, 4, 3, 2, 1, 106, 1006, 107, 1007, 108, 1008],
                  [6, 5, 4, 3, 2, 107, 1007, 108, 1008, 109, 1009]],
                 dtype=float),
        {1: np.array([5., 6., 7.], dtype=float),
         2: np.array([6., 7., 8.], dtype=float),
         3: np.array([7., 8., 9.], dtype=float)},
        {1: pd.RangeIndex(start=5, stop=8, step=1),
         2: pd.RangeIndex(start=6, stop=9, step=1),
         3: pd.RangeIndex(start=7, stop=10, step=1)},
        ['exog_1', 'exog_2'],
        [],
        ['exog_1', 'exog_2'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
         'exog_1_step_1', 'exog_2_step_1', 
         'exog_1_step_2', 'exog_2_step_2', 
         'exog_1_step_3', 'exog_2_step_3'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes},
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == [
        'exog_1_step_1', 'exog_2_step_1', 
        'exog_1_step_2', 'exog_2_step_2', 
        'exog_1_step_3', 'exog_2_step_3'
    ]
    forecaster.X_train_window_features_names_out_ is None

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]


@pytest.mark.parametrize("exog_values, dtype, expected_exog_val, expected_cat_names, expected_dtype_out", 
                         [([True]    , bool, 1.0, []      , bool),
                          (['string'], str , 0.0, ['exog'], np.float64)], 
                         ids = lambda dt: f'values, dtype: {dt}')
def test_create_train_X_y_output_when_steps_1_and_exog_is_series_of_bool_str(exog_values, dtype, expected_exog_val, expected_cat_names, expected_dtype_out):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)), steps=1 
    and exog is a pandas Series of bool or str.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(exog_values * 10, name='exog', dtype=dtype)

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=1
    )
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        np.array([[4., 3., 2., 1., 0., expected_exog_val],
                  [5., 4., 3., 2., 1., expected_exog_val],
                  [6., 5., 4., 3., 2., expected_exog_val],
                  [7., 6., 5., 4., 3., expected_exog_val],
                  [8., 7., 6., 5., 4., expected_exog_val]], dtype=float),
        {1: np.array([5., 6., 7., 8., 9.], dtype=float)},
        {1: pd.RangeIndex(start=5, stop=10, step=1)},
        ['exog'],
        expected_cat_names,
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_step_1'],
        {'exog': exog.dtypes},
        {'exog': expected_dtype_out}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1']
    forecaster.X_train_window_features_names_out_ is None

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]


@pytest.mark.parametrize("exog_values, dtype, expected_exog_val, expected_cat_names, expected_dtype_out", 
                         [([True]    , bool, 1.0, []      , bool),
                          (['string'], str , 0.0, ['exog'], np.float64)], 
                         ids = lambda dt: f'values, dtype: {dt}')
def test_create_train_X_y_output_when_steps_2_and_exog_is_series_of_bool_str(exog_values, dtype, expected_exog_val, expected_cat_names, expected_dtype_out):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)), steps=2
    and exog is a pandas Series of bool or str.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(exog_values * 10, name='exog', dtype=dtype)

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=2
    )
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        np.array([[4., 3., 2., 1., 0., expected_exog_val, expected_exog_val],
                  [5., 4., 3., 2., 1., expected_exog_val, expected_exog_val],
                  [6., 5., 4., 3., 2., expected_exog_val, expected_exog_val],
                  [7., 6., 5., 4., 3., expected_exog_val, expected_exog_val]], dtype=float),
        {1: np.array([5., 6., 7., 8.], dtype=float),
         2: np.array([6., 7., 8., 9.], dtype=float)},
        {1: pd.RangeIndex(start=5, stop=9, step=1),
         2: pd.RangeIndex(start=6, stop=10, step=1)},
        ['exog'],
        expected_cat_names,
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_step_1', 'exog_step_2'],
        {'exog': exog.dtypes},
        {'exog': expected_dtype_out}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1', 'exog_step_2']
    forecaster.X_train_window_features_names_out_ is None

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]


@pytest.mark.parametrize("v_exog_1   , v_exog_2  , dtype, exp_val_1, exp_val_2, expected_cat_names          , expected_dtype_out", 
                         [([True]    , [False]   , bool , 1.0      , 0.0      , []                          , bool),
                          (['string'], ['string'], str  , 0.0      , 0.0      , ['exog_1', 'exog_2'], np.float64)], 
                         ids = lambda dt: f'values, dtype: {dt}')
def test_create_train_X_y_output_when_steps_1_and_exog_is_df_of_bool_str(v_exog_1, v_exog_2, dtype, exp_val_1, exp_val_2, expected_cat_names, expected_dtype_out):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)), steps=1 
    and exog is a pandas dataframe with two columns of bool or str.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.DataFrame({'exog_1': v_exog_1 * 10,
                         'exog_2': v_exog_2 * 10})
    
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=1
    )
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        np.array([[4., 3., 2., 1., 0., exp_val_1, exp_val_2],
                  [5., 4., 3., 2., 1., exp_val_1, exp_val_2],
                  [6., 5., 4., 3., 2., exp_val_1, exp_val_2],
                  [7., 6., 5., 4., 3., exp_val_1, exp_val_2],
                  [8., 7., 6., 5., 4., exp_val_1, exp_val_2]], dtype=float),
        {1: np.array([5., 6., 7., 8., 9.], dtype=float)},
        {1: pd.RangeIndex(start=5, stop=10, step=1)},
        ['exog_1', 'exog_2'],
        expected_cat_names,
        ['exog_1', 'exog_2'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_1_step_1', 'exog_2_step_1'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes},
        {'exog_1': expected_dtype_out, 'exog_2': expected_dtype_out}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_1_step_1', 'exog_2_step_1']
    forecaster.X_train_window_features_names_out_ is None

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]


@pytest.mark.parametrize("v_exog_1   , v_exog_2  , dtype, exp_val_1, exp_val_2, expected_cat_names          , expected_dtype_out", 
                         [([True]    , [False]   , bool , 1.0      , 0.0      , []                          , bool),
                          (['string'], ['string'], str  , 0.0      , 0.0      , ['exog_1', 'exog_2'], np.float64)], 
                         ids = lambda dt: f'values, dtype: {dt}')
def test_create_train_X_y_output_when_steps_3_and_exog_is_df_of_bool_str(v_exog_1, v_exog_2, dtype, exp_val_1, exp_val_2, expected_cat_names, expected_dtype_out):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)), steps=3 
    and exog is a pandas dataframe with two columns of bool or str.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.DataFrame({'exog_1': v_exog_1 * 10,
                         'exog_2': v_exog_2 * 10})
    
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=3
    )
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        np.array([[4., 3., 2., 1., 0., exp_val_1, exp_val_2, exp_val_1, exp_val_2, exp_val_1, exp_val_2],
                  [5., 4., 3., 2., 1., exp_val_1, exp_val_2, exp_val_1, exp_val_2, exp_val_1, exp_val_2],
                  [6., 5., 4., 3., 2., exp_val_1, exp_val_2, exp_val_1, exp_val_2, exp_val_1, exp_val_2]],
                 dtype=float),
        {1: np.array([5., 6., 7.], dtype=float),
         2: np.array([6., 7., 8.], dtype=float),
         3: np.array([7., 8., 9.], dtype=float)},
        {1: pd.RangeIndex(start=5, stop=8, step=1),
         2: pd.RangeIndex(start=6, stop=9, step=1),
         3: pd.RangeIndex(start=7, stop=10, step=1)},
        ['exog_1', 'exog_2'],
        expected_cat_names,
        ['exog_1', 'exog_2'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
         'exog_1_step_1', 'exog_2_step_1', 
         'exog_1_step_2', 'exog_2_step_2', 
         'exog_1_step_3', 'exog_2_step_3'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes},
        {'exog_1': expected_dtype_out, 'exog_2': expected_dtype_out}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == [
        'exog_1_step_1', 'exog_2_step_1', 
        'exog_1_step_2', 'exog_2_step_2', 
        'exog_1_step_3', 'exog_2_step_3'
    ]
    forecaster.X_train_window_features_names_out_ is None

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]


@pytest.mark.parametrize(
    "categorical_features",
    [None, 'auto', ['exog']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_train_X_y_output_when_y_is_series_10_steps_1_and_exog_is_series_of_category(categorical_features):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)), steps=1 
    and exog is a pandas Series of category.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(range(100, 110), name='exog', dtype='category')

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=1,
        categorical_features=categorical_features
    )
    results = forecaster._create_train_X_y(y=y, exog=exog)

    if categorical_features is None:
        expected = (
            np.array([[  4.,   3.,   2.,   1.,   0., 105.],
                      [  5.,   4.,   3.,   2.,   1., 106.],
                      [  6.,   5.,   4.,   3.,   2., 107.],
                      [  7.,   6.,   5.,   4.,   3., 108.],
                      [  8.,   7.,   6.,   5.,   4., 109.]], dtype=float),
            {1: np.array([5., 6., 7., 8., 9.], dtype=float)},
            {1: pd.RangeIndex(start=5, stop=10, step=1)},
            ['exog'],
            None,
            ['exog'],
            ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_step_1'],
            {'exog': exog.dtypes},
            {'exog': exog.dtypes}
        )
    else:
        expected = (
            np.array([[4., 3., 2., 1., 0., 5.],
                      [5., 4., 3., 2., 1., 6.],
                      [6., 5., 4., 3., 2., 7.],
                      [7., 6., 5., 4., 3., 8.],
                      [8., 7., 6., 5., 4., 9.]], dtype=float),
            {1: np.array([5., 6., 7., 8., 9.], dtype=float)},
            {1: pd.RangeIndex(start=5, stop=10, step=1)},
            ['exog'],
            ['exog'],
            ['exog'],
            ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_step_1'],
            {'exog': exog.dtypes},
            {'exog': np.float64}
        )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    if categorical_features is not None:
        assert len(forecaster.categorical_encoder.categories_) == 1
        np.testing.assert_array_equal(
            forecaster.categorical_encoder.categories_[0],
            np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        )


def test_create_train_X_y_output_when_y_is_series_10_steps_2_and_exog_is_series_of_category():
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)), steps=2 
    and exog is a pandas Series of category.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(range(10), name='exog', dtype='category')

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=2
    )
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        np.array([[4., 3., 2., 1., 0., 5., 6.],
                  [5., 4., 3., 2., 1., 6., 7.],
                  [6., 5., 4., 3., 2., 7., 8.],
                  [7., 6., 5., 4., 3., 8., 9.]], dtype=float),
        {1: np.array([5., 6., 7., 8.], dtype=float),
         2: np.array([6., 7., 8., 9.], dtype=float)},
        {1: pd.RangeIndex(start=5, stop=9, step=1),
         2: pd.RangeIndex(start=6, stop=10, step=1)},
        ['exog'],
        ['exog'],
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_step_1', 'exog_step_2'],
        {'exog': exog.dtypes},
        {'exog': np.float64}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1', 'exog_step_2']
    forecaster.X_train_window_features_names_out_ is None

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]


@pytest.mark.parametrize(
    "categorical_features",
    [None, 'auto', ['exog_1', 'exog_2']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_train_X_y_output_when_y_is_series_10_steps_1_and_exog_is_dataframe_of_category(categorical_features):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)), steps=1 
    and exog is a pandas DataFrame of category.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.DataFrame({'exog_1': pd.Categorical(range(10)),
                         'exog_2': pd.Categorical(range(100, 110))})
    
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=1,
        categorical_features=categorical_features
    )
    results = forecaster._create_train_X_y(y=y, exog=exog)

    if categorical_features is None:
        expected = (
            np.array([[4., 3., 2., 1., 0., 5., 105.],
                      [5., 4., 3., 2., 1., 6., 106.],
                      [6., 5., 4., 3., 2., 7., 107.],
                      [7., 6., 5., 4., 3., 8., 108.],
                      [8., 7., 6., 5., 4., 9., 109.]], dtype=float),
            {1: np.array([5., 6., 7., 8., 9.], dtype=float)},
            {1: pd.RangeIndex(start=5, stop=10, step=1)},
            ['exog_1', 'exog_2'],
            None,
            ['exog_1', 'exog_2'],
            ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_1_step_1', 'exog_2_step_1'],
            {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes},
            {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes}
        )
    else:
        expected = (
            np.array([[4., 3., 2., 1., 0., 5., 5.],
                      [5., 4., 3., 2., 1., 6., 6.],
                      [6., 5., 4., 3., 2., 7., 7.],
                      [7., 6., 5., 4., 3., 8., 8.],
                      [8., 7., 6., 5., 4., 9., 9.]], dtype=float),
            {1: np.array([5., 6., 7., 8., 9.], dtype=float)},
            {1: pd.RangeIndex(start=5, stop=10, step=1)},
            ['exog_1', 'exog_2'],
            ['exog_1', 'exog_2'],
            ['exog_1', 'exog_2'],
            ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_1_step_1', 'exog_2_step_1'],
            {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes},
            {'exog_1': np.float64, 'exog_2': np.float64}
        )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    if categorical_features is not None:
        assert len(forecaster.categorical_encoder.categories_) == 2
        np.testing.assert_array_equal(
            forecaster.categorical_encoder.categories_[0],
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        )
        np.testing.assert_array_equal(
            forecaster.categorical_encoder.categories_[1],
            np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        )


def test_create_train_X_y_output_when_y_is_series_10_steps_3_and_exog_is_dataframe_of_category():
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)), steps=3 
    and exog is a pandas DataFrame of category.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.DataFrame({'exog_1': pd.Categorical(range(10)),
                         'exog_2': pd.Categorical(range(100, 110))})
    
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=3
    )
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        np.array([[4., 3., 2., 1., 0., 5., 5., 6., 6., 7., 7.],
                  [5., 4., 3., 2., 1., 6., 6., 7., 7., 8., 8.],
                  [6., 5., 4., 3., 2., 7., 7., 8., 8., 9., 9.]],
                 dtype=float),
        {1: np.array([5., 6., 7.], dtype=float),
         2: np.array([6., 7., 8.], dtype=float),
         3: np.array([7., 8., 9.], dtype=float)},
        {1: pd.RangeIndex(start=5, stop=8, step=1),
         2: pd.RangeIndex(start=6, stop=9, step=1),
         3: pd.RangeIndex(start=7, stop=10, step=1)},
        ['exog_1', 'exog_2'],
        ['exog_1', 'exog_2'],
        ['exog_1', 'exog_2'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
         'exog_1_step_1', 'exog_2_step_1', 
         'exog_1_step_2', 'exog_2_step_2', 
         'exog_1_step_3', 'exog_2_step_3'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes},
        {'exog_1': np.float64, 'exog_2': np.float64}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == [
        'exog_1_step_1', 'exog_2_step_1', 
        'exog_1_step_2', 'exog_2_step_2', 
        'exog_1_step_3', 'exog_2_step_3'
    ]
    forecaster.X_train_window_features_names_out_ is None

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]


@pytest.mark.parametrize(
    "categorical_features",
    [None, 'auto', ['exog_3', 'exog_4']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_dataframe_of_float_int_category_steps_1(categorical_features):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)), steps=1 
    and exog is a pandas DataFrame with columns of float, int, category, and int
    (forced as categorical in explicit list). The explicit list includes
    exog_4 (int) to verify that numeric columns can be forced as categorical.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.DataFrame({
        'exog_1': pd.Series(np.arange(100, 110), dtype=float),
        'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
        'exog_3': pd.Categorical(range(100, 110)),
        'exog_4': pd.Series(np.arange(10, 20), dtype=int),
    })
    
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=1,
        categorical_features=categorical_features
    )
    results = forecaster._create_train_X_y(y=y, exog=exog)

    exog_dtypes_in = {
        'exog_1': exog['exog_1'].dtypes,
        'exog_2': exog['exog_2'].dtypes,
        'exog_3': exog['exog_3'].dtypes,
        'exog_4': exog['exog_4'].dtypes,
    }

    if categorical_features is None:
        expected = (
            np.array([[4., 3., 2., 1., 0., 105., 1005., 105., 15.],
                      [5., 4., 3., 2., 1., 106., 1006., 106., 16.],
                      [6., 5., 4., 3., 2., 107., 1007., 107., 17.],
                      [7., 6., 5., 4., 3., 108., 1008., 108., 18.],
                      [8., 7., 6., 5., 4., 109., 1009., 109., 19.]],
                     dtype=float),
            {1: np.array([5., 6., 7., 8., 9.], dtype=float)},
            {1: pd.RangeIndex(start=5, stop=10, step=1)},
            ['exog_1', 'exog_2', 'exog_3', 'exog_4'],
            None,
            ['exog_1', 'exog_2', 'exog_3', 'exog_4'],
            ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
             'exog_1_step_1', 'exog_2_step_1', 'exog_3_step_1', 'exog_4_step_1'],
            exog_dtypes_in,
            exog_dtypes_in
        )
    elif categorical_features == 'auto':
        expected = (
            np.array([[4., 3., 2., 1., 0., 105., 1005., 5., 15.],
                      [5., 4., 3., 2., 1., 106., 1006., 6., 16.],
                      [6., 5., 4., 3., 2., 107., 1007., 7., 17.],
                      [7., 6., 5., 4., 3., 108., 1008., 8., 18.],
                      [8., 7., 6., 5., 4., 109., 1009., 9., 19.]],
                     dtype=float),
            {1: np.array([5., 6., 7., 8., 9.], dtype=float)},
            {1: pd.RangeIndex(start=5, stop=10, step=1)},
            ['exog_1', 'exog_2', 'exog_3', 'exog_4'],
            ['exog_3'],
            ['exog_1', 'exog_2', 'exog_3', 'exog_4'],
            ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
             'exog_1_step_1', 'exog_2_step_1', 'exog_3_step_1', 'exog_4_step_1'],
            exog_dtypes_in,
            {'exog_1': np.float64, 'exog_2': np.int64, 'exog_3': np.float64, 'exog_4': np.int64}
        )
    else:
        expected = (
            np.array([[4., 3., 2., 1., 0., 105., 1005., 5., 5.],
                      [5., 4., 3., 2., 1., 106., 1006., 6., 6.],
                      [6., 5., 4., 3., 2., 107., 1007., 7., 7.],
                      [7., 6., 5., 4., 3., 108., 1008., 8., 8.],
                      [8., 7., 6., 5., 4., 109., 1009., 9., 9.]],
                     dtype=float),
            {1: np.array([5., 6., 7., 8., 9.], dtype=float)},
            {1: pd.RangeIndex(start=5, stop=10, step=1)},
            ['exog_1', 'exog_2', 'exog_3', 'exog_4'],
            ['exog_3', 'exog_4'],
            ['exog_1', 'exog_2', 'exog_3', 'exog_4'],
            ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
             'exog_1_step_1', 'exog_2_step_1', 'exog_3_step_1', 'exog_4_step_1'],
            exog_dtypes_in,
            {'exog_1': np.float64, 'exog_2': np.int64, 'exog_3': np.float64, 'exog_4': np.float64}
        )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    if categorical_features == 'auto':
        assert len(forecaster.categorical_encoder.categories_) == 1
        np.testing.assert_array_equal(
            forecaster.categorical_encoder.categories_[0],
            np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        )
    elif categorical_features is not None:
        assert len(forecaster.categorical_encoder.categories_) == 2
        np.testing.assert_array_equal(
            forecaster.categorical_encoder.categories_[0],
            np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        )
        np.testing.assert_array_equal(
            forecaster.categorical_encoder.categories_[1],
            np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        )


def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_dataframe_of_float_int_category_steps_3():
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)), steps=3 
    and exog is a pandas DataFrame of float, int and category.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.DataFrame({'exog_1': pd.Series(np.arange(100, 110), dtype=float),
                         'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
                         'exog_3': pd.Categorical(range(100, 110))})
    
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=3
    )
    results = forecaster._create_train_X_y(y=y, exog=exog)        
    expected = (
        np.array([[4., 3., 2., 1., 0., 105., 1005., 5., 106., 1006., 6., 107., 1007., 7.],
                  [5., 4., 3., 2., 1., 106., 1006., 6., 107., 1007., 7., 108., 1008., 8.],
                  [6., 5., 4., 3., 2., 107., 1007., 7., 108., 1008., 8., 109., 1009., 9.]],
                 dtype=float),
        {1: np.array([5., 6., 7.], dtype=float),
         2: np.array([6., 7., 8.], dtype=float),
         3: np.array([7., 8., 9.], dtype=float)},
        {1: pd.RangeIndex(start=5, stop=8, step=1),
         2: pd.RangeIndex(start=6, stop=9, step=1),
         3: pd.RangeIndex(start=7, stop=10, step=1)},
        ['exog_1', 'exog_2', 'exog_3'],
        ['exog_3'],
        ['exog_1', 'exog_2', 'exog_3'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
         'exog_1_step_1', 'exog_2_step_1', 'exog_3_step_1', 
         'exog_1_step_2', 'exog_2_step_2', 'exog_3_step_2', 
         'exog_1_step_3', 'exog_2_step_3', 'exog_3_step_3'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes, 'exog_3': exog['exog_3'].dtypes},
        {'exog_1': np.float64, 'exog_2': np.int64, 'exog_3': np.float64}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == [
        'exog_1_step_1', 'exog_2_step_1', 'exog_3_step_1', 
        'exog_1_step_2', 'exog_2_step_2', 'exog_3_step_2', 
        'exog_1_step_3', 'exog_2_step_3', 'exog_3_step_3'
    ]
    forecaster.X_train_window_features_names_out_ is None

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]


def test_create_train_X_y_output_when_y_is_series_10_and_transformer_y_is_StandardScaler():
    """
    Test the output of _create_train_X_y when exog is None and transformer_y
    is StandardScaler with steps=1.
    """
    forecaster = ForecasterDirect(
                     estimator     = LinearRegression(),
                     lags          = 5,
                     steps         = 1,
                     transformer_y = StandardScaler()
                 )
    results = forecaster._create_train_X_y(y=pd.Series(np.arange(10), name='y', dtype=float))
    expected = (
        np.array([[-0.17407766, -0.52223297, -0.87038828, -1.21854359, -1.5666989],
                  [0.17407766, -0.17407766, -0.52223297, -0.87038828, -1.21854359],
                  [0.52223297,  0.17407766, -0.17407766, -0.52223297, -0.87038828],
                  [0.87038828,  0.52223297,  0.17407766, -0.17407766, -0.52223297],
                  [1.21854359,  0.87038828,  0.52223297,  0.17407766, -0.17407766]],
                 dtype=float),
        {1: np.array([0.17407766, 0.52223297, 0.87038828, 
                      1.21854359, 1.5666989], dtype=float)},
        {1: pd.RangeIndex(start=5, stop=10, step=1)},
        None,
        None,
        None,
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5'],
        None,
        None
    )

    forecaster.exog_in_ is False
    forecaster.X_train_direct_exog_names_out_ is None
    forecaster.X_train_window_features_names_out_ is None

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    assert results[8] == expected[8]


def test_create_train_X_y_output_when_lags_3_steps_1_and_exog_is_None_and_transformer_exog_is_not_None():
    """
    Test output of _create_train_X_y when estimator is LinearRegression, lags is 3
    and steps is 1 and transformer_exog is not None.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)

    forecaster = ForecasterDirect(
                     estimator        = LinearRegression(),
                     lags             = 3,
                     steps            = 1,
                     transformer_exog = StandardScaler()
                 )
    results = forecaster._create_train_X_y(y=y, exog=None)
    
    expected = (
        np.array([[2., 1., 0.],
                  [3., 2., 1.],
                  [4., 3., 2.],
                  [5., 4., 3.],
                  [6., 5., 4.],
                  [7., 6., 5.],
                  [8., 7., 6.]], dtype=float),
        {1: np.array([3., 4., 5., 6., 7., 8., 9.], dtype=float)},
        {1: pd.RangeIndex(start=3, stop=10, step=1)},
        None,
        None,
        None,
        ['lag_1', 'lag_2', 'lag_3'],
        None,
        None
    )

    forecaster.exog_in_ is False
    forecaster.X_train_direct_exog_names_out_ is None
    forecaster.X_train_window_features_names_out_ is None

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    assert results[8] == expected[8]


def test_create_train_X_y_output_when_transformer_y_and_transformer_exog_steps_2():
    """
    Test the output of _create_train_X_y when using transformer_y and transformer_exog 
    with steps=2.
    """
    y = pd.Series(np.arange(10), dtype = float)
    y.index = pd.date_range("1990-01-01", periods=10, freq='D')
    exog = pd.DataFrame({
               'col_1': [7.5, 24.4, 54.2, 12.1, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4],
               'col_2': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b']},
               index = pd.date_range("1990-01-01", periods=10, freq='D')
           )

    transformer_y = StandardScaler()
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['col_1']),
                            ('onehot', OneHotEncoder(), ['col_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )

    forecaster = ForecasterDirect(
                     estimator        = LinearRegression(),
                     lags             = 5,
                     steps            = 2,
                     transformer_y    = transformer_y,
                     transformer_exog = transformer_exog
                 )
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        np.array([[-0.17407766, -0.52223297, -0.87038828, -1.21854359, -1.5666989,
                   0.57176024, 0., 1., 0.28259414, 1., 0.],
                  [0.17407766, -0.17407766, -0.52223297, -0.87038828, -1.21854359,
                   0.28259414, 1., 0., -0.12486718, 0., 1.],
                  [0.52223297,  0.17407766, -0.17407766, -0.52223297, -0.87038828,
                   -0.12486718, 0., 1., 1.88177028, 1., 0.],
                  [0.87038828,  0.52223297,  0.17407766, -0.17407766, -0.52223297,
                   1.88177028, 1., 0., 0.13801109, 0., 1.]],
                 dtype=float),
        {1: np.array([0.17407766, 0.52223297, 0.87038828, 1.21854359], dtype=float),
         2: np.array([0.52223297, 0.87038828, 1.21854359, 1.5666989], dtype=float)},
        {1: pd.date_range("1990-01-06", periods=4, freq='D'),
         2: pd.date_range("1990-01-07", periods=4, freq='D')},
        ['col_1', 'col_2'],
        [],
        ['col_1', 'col_2_a', 'col_2_b'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
         'col_1_step_1', 'col_2_a_step_1', 'col_2_b_step_1', 
         'col_1_step_2', 'col_2_a_step_2', 'col_2_b_step_2'],
        {'col_1': exog['col_1'].dtypes, 'col_2': exog['col_2'].dtypes},
        {'col_1': np.float64, 'col_2_a': np.float64, 'col_2_b': np.float64}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == [
        'col_1_step_1', 'col_2_a_step_1', 'col_2_b_step_1', 
        'col_1_step_2', 'col_2_a_step_2', 'col_2_b_step_2'
    ]
    forecaster.X_train_window_features_names_out_ is None

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]


@pytest.mark.parametrize("fit_forecaster", 
                         [True, False], 
                         ids = lambda fitted: f'fit_forecaster: {fitted}')
def test_create_train_X_y_output_when_pandas_series_and_differentiation_is_1(fit_forecaster):
    """
    Test the output of _create_train_X_y when using differentiation=1. Comparing 
    the matrix created with and without differentiating the series.
    """
    # Data differentiated
    differentiator = TimeSeriesDifferentiator(order=1)
    data_diff = differentiator.fit_transform(data.to_numpy())
    data_diff = pd.Series(data_diff, index=data.index).dropna()

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff = exog.iloc[1:]
    end_train = '2003-03-01 23:59:00'

    forecaster_1 = ForecasterDirect(
        estimator=LinearRegression(), steps=1, lags=5
    )
    forecaster_2 = ForecasterDirect(
        estimator=LinearRegression(), steps=1, lags=5, differentiation=1
    )
    
    if fit_forecaster:
        forecaster_2.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])

    output_1 = forecaster_1._create_train_X_y(
                   y    = data_diff.loc[:end_train],
                   exog = exog_diff.loc[:end_train]
               )
    output_2 = forecaster_2._create_train_X_y(
                   y    = data.loc[:end_train],
                   exog = exog.loc[:end_train]
               )

    forecaster_1.exog_in_ is True
    forecaster_1.X_train_direct_exog_names_out_ == ['exog_step_1']
    forecaster_1.X_train_window_features_names_out_ is None

    forecaster_2.exog_in_ is True
    forecaster_2.X_train_direct_exog_names_out_ == ['exog_step_1']
    forecaster_2.X_train_window_features_names_out_ is None
    
    np.testing.assert_array_almost_equal(output_1[0], output_2[0])
    for key in output_1[1]: 
        np.testing.assert_array_almost_equal(output_1[1][key], output_2[1][key]) 
    assert isinstance(output_1[2], dict)
    for key in output_1[2]:
        pd.testing.assert_index_equal(output_1[2][key], output_2[2][key])
    assert output_1[3] == output_2[3]
    assert output_1[4] == output_2[4]
    assert output_1[5] == output_2[5]
    assert output_1[6] == output_2[6]
    for k in output_1[7].keys():
        assert output_1[7][k] == output_2[7][k]
    for k in output_1[8].keys():
        assert output_1[8][k] == output_2[8][k]


@pytest.mark.parametrize("fit_forecaster", 
                         [True, False], 
                         ids = lambda fitted: f'fit_forecaster: {fitted}')
def test_create_train_X_y_output_when_pandas_series_and_differentiation_is_1_steps_3(fit_forecaster):
    """
    Test the output of _create_train_X_y when using differentiation=1. Comparing 
    the matrix created with and without differentiating the series with steps=3.
    """
    # Data differentiated
    differentiator = TimeSeriesDifferentiator(order=1)
    data_diff = differentiator.fit_transform(data.to_numpy())
    data_diff = pd.Series(data_diff, index=data.index).dropna()

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff = exog.iloc[1:]
    end_train = '2003-03-01 23:59:00'

    forecaster_1 = ForecasterDirect(
        estimator=LinearRegression(), steps=3, lags=5
    )
    forecaster_2 = ForecasterDirect(
        estimator=LinearRegression(), steps=3, lags=5, differentiation=1
    )
    
    if fit_forecaster:
        forecaster_2.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])

    output_1 = forecaster_1._create_train_X_y(
                   y    = data_diff.loc[:end_train],
                   exog = exog_diff.loc[:end_train]
               )
    output_2 = forecaster_2._create_train_X_y(
                   y    = data.loc[:end_train],
                   exog = exog.loc[:end_train]
               )

    forecaster_1.exog_in_ is True
    forecaster_1.X_train_direct_exog_names_out_ == ['exog_step_1', 'exog_step_2', 'exog_step_3']
    forecaster_1.X_train_window_features_names_out_ is None

    forecaster_2.exog_in_ is True
    forecaster_2.X_train_direct_exog_names_out_ == ['exog_step_1', 'exog_step_2', 'exog_step_3']
    forecaster_2.X_train_window_features_names_out_ is None
    
    np.testing.assert_array_almost_equal(output_1[0], output_2[0])
    for key in output_1[1]: 
        np.testing.assert_array_almost_equal(output_1[1][key], output_2[1][key]) 
    assert isinstance(output_1[2], dict)
    for key in output_1[2]:
        pd.testing.assert_index_equal(output_1[2][key], output_2[2][key])
    assert output_1[3] == output_2[3]
    assert output_1[4] == output_2[4]
    assert output_1[5] == output_2[5]
    assert output_1[6] == output_2[6]
    for k in output_1[7].keys():
        assert output_1[7][k] == output_2[7][k]
    for k in output_1[8].keys():
        assert output_1[8][k] == output_2[8][k]


def test_create_train_X_y_output_when_y_is_series_exog_is_series_and_differentiation_is_2():
    """
    Test the output of _create_train_X_y when using differentiation=2. Comparing 
    the matrix created with and without differentiating the series.
    """

    # Data differentiated
    differentiator = TimeSeriesDifferentiator(order=2)
    data_diff_2 = differentiator.fit_transform(data.to_numpy())
    data_diff_2 = pd.Series(data_diff_2, index=data.index).dropna()

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff_2 = exog.iloc[2:]
    end_train = '2003-03-01 23:59:00'

    forecaster_1 = ForecasterDirect(
        estimator=LinearRegression(), steps=1, lags=5
    )
    forecaster_2 = ForecasterDirect(
        estimator=LinearRegression(), steps=1, lags=5, differentiation=2
    )

    output_1 = forecaster_1._create_train_X_y(
                   y    = data_diff_2.loc[:end_train],
                   exog = exog_diff_2.loc[:end_train]
               )
    output_2 = forecaster_2._create_train_X_y(
                   y    = data.loc[:end_train],
                   exog = exog.loc[:end_train]
               )

    forecaster_1.exog_in_ is True
    forecaster_1.X_train_direct_exog_names_out_ == ['exog_step_1']
    forecaster_1.X_train_window_features_names_out_ is None

    forecaster_2.exog_in_ is True
    forecaster_2.X_train_direct_exog_names_out_ == ['exog_step_1']
    forecaster_2.X_train_window_features_names_out_ is None
    
    np.testing.assert_array_almost_equal(output_1[0], output_2[0])
    for key in output_1[1]: 
        np.testing.assert_array_almost_equal(output_1[1][key], output_2[1][key]) 
    assert isinstance(output_1[2], dict)
    for key in output_1[2]:
        pd.testing.assert_index_equal(output_1[2][key], output_2[2][key])
    assert output_1[3] == output_2[3]
    assert output_1[4] == output_2[4]
    assert output_1[5] == output_2[5]
    assert output_1[6] == output_2[6]
    for k in output_1[7].keys():
        assert output_1[7][k] == output_2[7][k]
    for k in output_1[8].keys():
        assert output_1[8][k] == output_2[8][k]


def test_create_train_X_y_output_when_window_features_and_exog_steps_1():
    """
    Test the output of _create_train_X_y when using window_features and exog 
    with datetime index and steps=1.
    """
    y_datetime = pd.Series(
        np.arange(15), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='y', dtype=float
    )
    exog_datetime = pd.Series(
        np.arange(100, 115), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='exog', dtype=float
    )
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6]
    )

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), steps=1, lags=5, window_features=rolling
    )
    results = forecaster._create_train_X_y(y=y_datetime, exog=exog_datetime)
    
    expected = (
        np.array([[5., 4., 3., 2., 1., 3., 3., 15., 106.],
                  [6., 5., 4., 3., 2., 4., 4., 21., 107.],
                  [7., 6., 5., 4., 3., 5., 5., 27., 108.],
                  [8., 7., 6., 5., 4., 6., 6., 33., 109.],
                  [9., 8., 7., 6., 5., 7., 7., 39., 110.],
                  [10., 9., 8., 7., 6., 8., 8., 45., 111.],
                  [11., 10., 9., 8., 7., 9., 9., 51., 112.],
                  [12., 11., 10., 9., 8., 10., 10., 57., 113.],
                  [13., 12., 11., 10., 9., 11., 11., 63., 114.]],
                 dtype=float),
        {1: np.array([6., 7., 8., 9., 10., 11., 12., 13., 14.], dtype=float)},
        {1: pd.date_range('2000-01-07', periods=9, freq='D')},
        ['exog'],
        [],
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
         'roll_mean_5', 'roll_median_5', 'roll_sum_6', 'exog_step_1'],
        {'exog': exog_datetime.dtypes},
        {'exog': exog_datetime.dtypes}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1']
    forecaster.window_features_names == ['roll_mean_5', 'roll_median_5', 'roll_sum_6']
    forecaster.X_train_window_features_names_out_ == ['roll_mean_5', 'roll_median_5', 'roll_sum_6']

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]


def test_create_train_X_y_output_when_window_features_and_exog_steps_2():
    """
    Test the output of _create_train_X_y when using window_features and exog 
    with datetime index and steps=2.
    """
    y_datetime = pd.Series(
        np.arange(15), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='y', dtype=float
    )
    exog_datetime = pd.Series(
        np.arange(100, 115), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='exog', dtype=float
    )
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6]
    )

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), steps=2, lags=5, window_features=rolling
    )
    results = forecaster._create_train_X_y(y=y_datetime, exog=exog_datetime)
    
    expected = (
        np.array([[5., 4., 3., 2., 1., 3., 3., 15., 106., 107.],
                  [6., 5., 4., 3., 2., 4., 4., 21., 107., 108.],
                  [7., 6., 5., 4., 3., 5., 5., 27., 108., 109.],
                  [8., 7., 6., 5., 4., 6., 6., 33., 109., 110.],
                  [9., 8., 7., 6., 5., 7., 7., 39., 110., 111.],
                  [10., 9., 8., 7., 6., 8., 8., 45., 111., 112.],
                  [11., 10., 9., 8., 7., 9., 9., 51., 112., 113.],
                  [12., 11., 10., 9., 8., 10., 10., 57., 113., 114.]],
                 dtype=float),
        {1: np.array([6., 7., 8., 9., 10., 11., 12., 13.], dtype=float),
         2: np.array([7., 8., 9., 10., 11., 12., 13., 14.], dtype=float)},
        {1: pd.date_range('2000-01-07', periods=8, freq='D'),
         2: pd.date_range('2000-01-08', periods=8, freq='D')},
        ['exog'],
        [],
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
         'roll_mean_5', 'roll_median_5', 'roll_sum_6', 
         'exog_step_1', 'exog_step_2'],
        {'exog': exog_datetime.dtypes},
        {'exog': exog_datetime.dtypes}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1', 'exog_step_2']
    forecaster.window_features_names == ['roll_mean_5', 'roll_median_5', 'roll_sum_6']
    forecaster.X_train_window_features_names_out_ == ['roll_mean_5', 'roll_median_5', 'roll_sum_6']

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]


def test_create_train_X_y_output_when_two_window_features_and_exog_steps_2():
    """
    Test the output of _create_train_X_y when using 2 window_features and exog 
    with datetime index and steps=2.
    """
    y_datetime = pd.Series(
        np.arange(15), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='y', dtype=float
    )
    exog_datetime = pd.Series(
        np.arange(100, 115), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='exog', dtype=float
    )
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[5, 5])
    rolling_2 = RollingFeatures(stats='sum', window_sizes=[6])

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), steps=2, lags=5, window_features=[rolling, rolling_2]
    )
    results = forecaster._create_train_X_y(y=y_datetime, exog=exog_datetime)
    
    expected = (
        np.array([[5., 4., 3., 2., 1., 3., 3., 15., 106., 107.],
                  [6., 5., 4., 3., 2., 4., 4., 21., 107., 108.],
                  [7., 6., 5., 4., 3., 5., 5., 27., 108., 109.],
                  [8., 7., 6., 5., 4., 6., 6., 33., 109., 110.],
                  [9., 8., 7., 6., 5., 7., 7., 39., 110., 111.],
                  [10., 9., 8., 7., 6., 8., 8., 45., 111., 112.],
                  [11., 10., 9., 8., 7., 9., 9., 51., 112., 113.],
                  [12., 11., 10., 9., 8., 10., 10., 57., 113., 114.]],
                 dtype=float),
        {1: np.array([6., 7., 8., 9., 10., 11., 12., 13.], dtype=float),
         2: np.array([7., 8., 9., 10., 11., 12., 13., 14.], dtype=float)},
        {1: pd.date_range('2000-01-07', periods=8, freq='D'),
         2: pd.date_range('2000-01-08', periods=8, freq='D')},
        ['exog'],
        [],
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
         'roll_mean_5', 'roll_median_5', 'roll_sum_6', 
         'exog_step_1', 'exog_step_2'],
        {'exog': exog_datetime.dtypes},
        {'exog': exog_datetime.dtypes}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1', 'exog_step_2']
    forecaster.window_features_names == ['roll_mean_5', 'roll_median_5', 'roll_sum_6']
    forecaster.X_train_window_features_names_out_ == ['roll_mean_5', 'roll_median_5', 'roll_sum_6']

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]


def test_create_train_X_y_output_when_window_features_lags_None_and_exog():
    """
    Test the output of _create_train_X_y when using window_features and exog 
    with datetime index and lags=None and steps=1.
    """
    y_datetime = pd.Series(
        np.arange(15), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='y', dtype=float
    )
    exog_datetime = pd.Series(
        np.arange(100, 115), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='exog', dtype=float
    )
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6]
    )

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), steps=1, lags=None, window_features=rolling
    )
    results = forecaster._create_train_X_y(y=y_datetime, exog=exog_datetime)
    
    expected = (
        np.array([[3., 3., 15., 106.],
                  [4., 4., 21., 107.],
                  [5., 5., 27., 108.],
                  [6., 6., 33., 109.],
                  [7., 7., 39., 110.],
                  [8., 8., 45., 111.],
                  [9., 9., 51., 112.],
                  [10., 10., 57., 113.],
                  [11., 11., 63., 114.]],
                 dtype=float),
        {1: np.array([6., 7., 8., 9., 10., 11., 12., 13., 14.], dtype=float)},
        {1: pd.date_range('2000-01-07', periods=9, freq='D')},
        ['exog'],
        [],
        ['exog'],
        ['roll_mean_5', 'roll_median_5', 'roll_sum_6', 'exog_step_1'],
        {'exog': exog_datetime.dtypes},
        {'exog': exog_datetime.dtypes}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1']
    forecaster.window_features_names == ['roll_mean_5', 'roll_median_5', 'roll_sum_6']
    forecaster.X_train_window_features_names_out_ == ['roll_mean_5', 'roll_median_5', 'roll_sum_6']

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]


def test_create_train_X_y_output_when_window_features_and_exog_transformers_diff():
    """
    Test the output of _create_train_X_y when using window_features, exog, 
    transformers and differentiation with steps=1.
    """
    y_datetime = pd.Series(
        [25.3, 29.1, 27.5, 24.3, 2.1, 46.5, 31.3, 87.1, 133.5, 4.3],
        index=pd.date_range('2000-01-01', periods=10, freq='D'),
        name='y', dtype=float
    )
    exog = pd.DataFrame({
        'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4, 14.6, 73.5],
        'col_2': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']},
        index = pd.date_range('2000-01-01', periods=10, freq='D')
    )

    transformer_y = StandardScaler()
    transformer_exog = ColumnTransformer(
                            [('scale', StandardScaler(), ['col_1']),
                             ('onehot', OneHotEncoder(), ['col_2'])],
                            remainder = 'passthrough',
                            verbose_feature_names_out = False
                        )
    rolling = RollingFeatures(
        stats=['ratio_min_max', 'median'], window_sizes=4
    )

    forecaster = ForecasterDirect(
                     estimator        = LinearRegression(), 
                     steps            = 1,
                     lags             = [1, 5], 
                     window_features  = rolling,
                     transformer_y    = transformer_y,
                     transformer_exog = transformer_exog,
                     differentiation  = 2
                 )
    results = forecaster._create_train_X_y(y=y_datetime, exog=exog)
    
    expected = (
        np.array([[-1.56436158, -0.14173746, -0.89489489, -0.27035108,  0.04040264,
                    0.        ,  1.        ],
                  [ 1.8635851 , -0.04199628, -0.83943662,  0.62469472, -1.32578962,
                    0.        ,  1.        ],
                  [-0.24672817, -0.49870587, -0.83943662,  0.75068358,  1.12752513,
                    0.        ,  1.        ]],
                 dtype=float),
        {1: np.array([1.8635851, -0.24672817, -4.60909217], dtype=float)},
        {1: pd.date_range('2000-01-08', periods=3, freq='D')},
        ['col_1', 'col_2'],
        [],
        ['col_1', 'col_2_a', 'col_2_b'],
        ['lag_1', 'lag_5', 'roll_ratio_min_max_4', 'roll_median_4',
         'col_1_step_1', 'col_2_a_step_1', 'col_2_b_step_1'],
        {'col_1': exog['col_1'].dtypes, 'col_2': exog['col_2'].dtypes},
        {'col_1': np.float64, 'col_2_a': np.float64, 'col_2_b': np.float64}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['col_1_step_1', 'col_2_a_step_1', 'col_2_b_step_1']
    forecaster.window_features_names == ['roll_ratio_min_max_4', 'roll_median_4']
    forecaster.X_train_window_features_names_out_ == ['roll_ratio_min_max_4', 'roll_median_4']

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]


def test_create_train_X_y_output_when_window_features_and_exog_transformers_diff_steps_2():
    """
    Test the output of _create_train_X_y when using window_features, exog, 
    transformers and differentiation with steps=2.
    """
    y_datetime = pd.Series(
        [25.3, 29.1, 27.5, 24.3, 2.1, 46.5, 31.3, 87.1, 133.5, 4.3],
        index=pd.date_range('2000-01-01', periods=10, freq='D'),
        name='y', dtype=float
    )
    exog = pd.DataFrame({
               'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4, 14.6, 73.5],
               'col_2': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']},
               index = pd.date_range('2000-01-01', periods=10, freq='D')
           )

    transformer_y = StandardScaler()
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['col_1']),
                            ('onehot', OneHotEncoder(), ['col_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )
    rolling = RollingFeatures(
        stats=['ratio_min_max', 'median'], window_sizes=4
    )

    forecaster = ForecasterDirect(
                     estimator        = LinearRegression(), 
                     steps            = 2,
                     lags             = [1, 5], 
                     window_features  = rolling,
                     transformer_y    = transformer_y,
                     transformer_exog = transformer_exog,
                     differentiation  = 1
                 )
    results = forecaster._create_train_X_y(y=y_datetime, exog=exog)
    
    expected = (
        np.array([[ 1.16539688,  0.09974117, -0.5       , -0.06299443,  1.69816032,
                    0.        ,  1.        ,  0.04040264,  0.        ,  1.        ],
                  [-0.3989647 , -0.04199628, -0.5       , -0.24147863,  0.04040264,
                    0.        ,  1.        , -1.32578962,  0.        ,  1.        ],
                  [ 1.46462041, -0.08399257, -0.39784946,  0.38321609, -1.32578962,
                    0.        ,  1.        ,  1.12752513,  0.        ,  1.        ]],
                 dtype=float),
        {1: np.array([-0.3989647, 1.46462041, 1.21789224], dtype=float),
         2: np.array([1.46462041, 1.21789224, -3.39119993], dtype=float)},
        {1: pd.date_range('2000-01-07', periods=3, freq='D'),
         2: pd.date_range('2000-01-08', periods=3, freq='D')},
        ['col_1', 'col_2'],
        [],
        ['col_1', 'col_2_a', 'col_2_b'],
        ['lag_1', 'lag_5', 'roll_ratio_min_max_4', 'roll_median_4',
         'col_1_step_1', 'col_2_a_step_1', 'col_2_b_step_1', 
         'col_1_step_2', 'col_2_a_step_2', 'col_2_b_step_2'],
        {'col_1': exog['col_1'].dtypes, 'col_2': exog['col_2'].dtypes},
        {'col_1': np.float64, 'col_2_a': np.float64, 'col_2_b': np.float64}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == [
        'col_1_step_1', 'col_2_a_step_1', 'col_2_b_step_1', 
        'col_1_step_2', 'col_2_a_step_2', 'col_2_b_step_2'
    ]
    forecaster.window_features_names == ['roll_ratio_min_max_4', 'roll_median_4']
    forecaster.X_train_window_features_names_out_ == ['roll_ratio_min_max_4', 'roll_median_4']

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]


@pytest.mark.parametrize(
    "categorical_features",
    ['auto', ['exog']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_train_X_y_output_when_exog_is_series_of_string_category(categorical_features):
    """
    Test the output of _create_train_X_y when exog is a pandas series of
    string categories. OrdinalEncoder maps ['a'..'j'] -> [0.0..9.0].
    None is not parametrized because it raises TypeError (tested separately).
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(
        pd.Categorical(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']),
        name='exog'
    )
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=1,
        categorical_features=categorical_features
    )
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        np.array([[4., 3., 2., 1., 0., 5.],
                  [5., 4., 3., 2., 1., 6.],
                  [6., 5., 4., 3., 2., 7.],
                  [7., 6., 5., 4., 3., 8.],
                  [8., 7., 6., 5., 4., 9.]], dtype=float),
        {1: np.array([5., 6., 7., 8., 9.], dtype=float)},
        {1: pd.RangeIndex(start=5, stop=10, step=1)},
        ['exog'],
        ['exog'],
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_step_1'],
        {'exog': exog.dtypes},
        {'exog': np.float64}
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    assert len(forecaster.categorical_encoder.categories_) == 1
    np.testing.assert_array_equal(
        forecaster.categorical_encoder.categories_[0],
        np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'], dtype=object)
    )


@pytest.mark.parametrize(
    "categorical_features",
    ['auto', ['exog_1', 'exog_2']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_train_X_y_output_when_exog_is_dataframe_of_string_category(categorical_features):
    """
    Test the output of _create_train_X_y when exog is a pandas DataFrame with
    two string category columns. OrdinalEncoder maps each column alphabetically.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.DataFrame({
        'exog_1': pd.Categorical(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']),
        'exog_2': pd.Categorical(['k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'])
    })
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=1,
        categorical_features=categorical_features
    )
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        np.array([[4., 3., 2., 1., 0., 5., 5.],
                  [5., 4., 3., 2., 1., 6., 6.],
                  [6., 5., 4., 3., 2., 7., 7.],
                  [7., 6., 5., 4., 3., 8., 8.],
                  [8., 7., 6., 5., 4., 9., 9.]], dtype=float),
        {1: np.array([5., 6., 7., 8., 9.], dtype=float)},
        {1: pd.RangeIndex(start=5, stop=10, step=1)},
        ['exog_1', 'exog_2'],
        ['exog_1', 'exog_2'],
        ['exog_1', 'exog_2'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_1_step_1', 'exog_2_step_1'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes},
        {'exog_1': np.float64, 'exog_2': np.float64}
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    assert len(forecaster.categorical_encoder.categories_) == 2
    np.testing.assert_array_equal(
        forecaster.categorical_encoder.categories_[0],
        np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'], dtype=object)
    )
    np.testing.assert_array_equal(
        forecaster.categorical_encoder.categories_[1],
        np.array(['k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'], dtype=object)
    )


@pytest.mark.parametrize(
    "categorical_features",
    ['auto', ['exog_3']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_train_X_y_output_when_exog_is_dataframe_of_float_int_string_category(categorical_features):
    """
    Test the output of _create_train_X_y when exog is a pandas DataFrame with
    float, int, and string category columns. Only the string category column
    is detected/encoded. auto correctly ignores float and int columns.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.DataFrame({
        'exog_1': pd.Series(np.arange(100, 110), dtype=float),
        'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
        'exog_3': pd.Categorical(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    })
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=1,
        categorical_features=categorical_features
    )
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        np.array([[4., 3., 2., 1., 0., 105., 1005., 5.],
                  [5., 4., 3., 2., 1., 106., 1006., 6.],
                  [6., 5., 4., 3., 2., 107., 1007., 7.],
                  [7., 6., 5., 4., 3., 108., 1008., 8.],
                  [8., 7., 6., 5., 4., 109., 1009., 9.]], dtype=float),
        {1: np.array([5., 6., 7., 8., 9.], dtype=float)},
        {1: pd.RangeIndex(start=5, stop=10, step=1)},
        ['exog_1', 'exog_2', 'exog_3'],
        ['exog_3'],
        ['exog_1', 'exog_2', 'exog_3'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
         'exog_1_step_1', 'exog_2_step_1', 'exog_3_step_1'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes,
         'exog_3': exog['exog_3'].dtypes},
        {'exog_1': np.float64, 'exog_2': np.int64, 'exog_3': np.float64}
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    assert len(forecaster.categorical_encoder.categories_) == 1
    np.testing.assert_array_equal(
        forecaster.categorical_encoder.categories_[0],
        np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'], dtype=object)
    )


def test_create_train_X_y_output_when_is_fitted_uses_transform_not_fit_transform():
    """
    Test that when is_fitted=True (after fit), _create_train_X_y uses
    transform (not fit_transform) on the categorical encoder and produces
    the same result as the first call.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    y.index = pd.date_range('2000-01-01', periods=10, freq='D')
    exog = pd.DataFrame({
        'exog_1': np.arange(100, 110, dtype=float),
        'exog_2': pd.Categorical(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    }, index=y.index)

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=1,
        categorical_features='auto'
    )
    forecaster.fit(y=y, exog=exog)

    # Second call with is_fitted=True triggers transform branch
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        np.array([[4., 3., 2., 1., 0., 105., 5.],
                  [5., 4., 3., 2., 1., 106., 6.],
                  [6., 5., 4., 3., 2., 107., 7.],
                  [7., 6., 5., 4., 3., 108., 8.],
                  [8., 7., 6., 5., 4., 109., 9.]], dtype=float),
        {1: np.array([5., 6., 7., 8., 9.], dtype=float)},
        {1: pd.DatetimeIndex(['2000-01-06', '2000-01-07', '2000-01-08',
                              '2000-01-09', '2000-01-10'], freq='D')},
        ['exog_1', 'exog_2'],
        ['exog_2'],
        ['exog_1', 'exog_2'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_1_step_1', 'exog_2_step_1'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes},
        {'exog_1': np.float64, 'exog_2': np.float64}
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    assert len(forecaster.categorical_encoder.categories_) == 1
    np.testing.assert_array_equal(
        forecaster.categorical_encoder.categories_[0],
        np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'], dtype=object)
    )


@pytest.mark.parametrize(
    "categorical_features",
    ['auto', ['col_2']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_train_X_y_output_when_transformer_exog_is_make_column_transformer_and_categorical(categorical_features):
    """
    Test the output of _create_train_X_y when using make_column_transformer
    with StandardScaler only for numeric columns and a string categorical
    column passed through as remainder. With set_output(transform='pandas'),
    the category dtype is preserved and 'auto' correctly detects col_2.
    OrdinalEncoder maps ['a'..'c'] -> [0.0..2.0].
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    y.index = pd.date_range('1990-01-01', periods=10, freq='D')
    exog = pd.DataFrame({
               'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4, 30.1, 22.3],
               'col_2': pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'c'])},
               index=pd.date_range('1990-01-01', periods=10, freq='D')
           )

    transformer_exog = make_column_transformer(
                           (StandardScaler(), ['col_1']),
                           remainder='passthrough',
                           verbose_feature_names_out=False
                       ).set_output(transform='pandas')

    forecaster = ForecasterDirect(
                     estimator        = LinearRegression(),
                     lags             = 5,
                     steps            = 1,
                     transformer_exog = transformer_exog,
                     categorical_features = categorical_features
                 )
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        np.array(
            [[4., 3., 2., 1., 0., -0.06706325, 2.],
             [5., 4., 3., 2., 1.,  2.03670162, 0.],
             [6., 5., 4., 3., 2.,  0.20853914, 1.],
             [7., 6., 5., 4., 3., -0.5861144 , 2.],
             [8., 7., 6., 5., 4., -0.9443975 , 2.]]
        ),
        {1: np.array([5., 6., 7., 8., 9.], dtype=float)},
        {1: pd.DatetimeIndex(['1990-01-06', '1990-01-07', '1990-01-08',
                              '1990-01-09', '1990-01-10'], freq='D')},
        ['col_1', 'col_2'],
        ['col_2'],
        ['col_1', 'col_2'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'col_1_step_1', 'col_2_step_1'],
        {'col_1': exog['col_1'].dtypes, 'col_2': exog['col_2'].dtypes},
        {'col_1': np.float64, 'col_2': np.float64}
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, np.ndarray) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        np.testing.assert_array_almost_equal(results[1][key], expected[1][key]) 
    assert isinstance(results[2], dict)
    assert results[2].keys() == expected[2].keys()
    for key in expected[2]:
        pd.testing.assert_index_equal(results[2][key], expected[2][key])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
