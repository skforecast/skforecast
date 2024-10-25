# Unit test _create_train_X_y ForecasterDirectMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.exceptions import MissingValuesWarning
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from skforecast.preprocessing import TimeSeriesDifferentiator
from skforecast.preprocessing import RollingFeatures
from skforecast.direct import ForecasterDirectMultiVariate

# Fixtures
from .fixtures_forecaster_direct_multivariate import data  # to test results when using differentiation


def test_create_train_X_y_TypeError_when_series_not_DataFrame():
    """
    Test TypeError is raised when series is not a pandas DataFrame.
    """
    series = pd.Series(np.arange(7))
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1', lags=3, steps=2)

    err_msg = re.escape(
        f"`series` must be a pandas DataFrame. Got {type(series)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster._create_train_X_y(series=series)


def test_create_train_X_y_ValueError_when_len_series_is_lower_than_maximum_window_size_plus_steps():
    """
    Test ValueError is raised when length of series is lower than maximum window_size 
    plus number of steps included in the forecaster.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(5)),  
                           'l2': pd.Series(np.arange(5))})
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', lags=3, steps=3
    )

    err_msg = re.escape(
        ("Minimum length of `series` for training this forecaster is "
         "6. Reduce the number of "
         "predicted steps, 3, or the maximum "
         "window_size, 3, if no more data is available.\n"
         "    Length `series`: 5.\n"
         "    Max step : 3.\n"
         "    Max window size: 3.\n"
         "    Lags window size: 3.\n"
         "    Window features window size: None.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(series=series)


def test_create_train_X_y_ValueError_when_level_not_in_series():
    """
    Test ValueError is raised when level not in series column names.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)),  
                           'l2': pd.Series(np.arange(5))})
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1', lags=3, steps=2)

    series_col_names = list(series.columns)

    err_msg = re.escape(
        (f"One of the `series` columns must be named as the `level` of the forecaster.\n"
         f"  Forecaster `level` : {forecaster.level}.\n"
         f"  `series` columns   : {series_col_names}.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(series=series)


def test_create_train_X_y_IgnoredArgumentWarning_when_levels_of_transformer_series_not_equal_to_series_col_names():
    """
    Test IgnoredArgumentWarning is raised when `transformer_series` is a dict and its keys 
    are not the same as series column names.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)),  
                           'l2': pd.Series(np.arange(10))})
    dict_transformers = {'l1': StandardScaler(), 
                         'l3': StandardScaler()}
    
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(), 
                     level              = 'l1',
                     lags               = 3,
                     steps              = 3,
                     transformer_series = dict_transformers
                 )
    series_not_in_transformer_series = set(series.columns) - set(forecaster.transformer_series.keys())

    warn_msg = re.escape(
        (f"{series_not_in_transformer_series} not present in `transformer_series`."
         f" No transformation is applied to these series.")
    )
    with pytest.warns(UserWarning, match = warn_msg):
        forecaster._create_train_X_y(series=series)


def test_create_train_X_y_TypeError_when_exog_is_categorical_of_no_int():
    """
    Test TypeError is raised when exog is categorical with no int values.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(6)),  
                           'l2': pd.Series(np.arange(6))})
    exog = pd.Series(['A', 'B', 'C', 'A', 'B', 'C'], name='exog', dtype='category')
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1', lags=2, steps=2)

    err_msg = re.escape(
        ("Categorical dtypes in exog must contain only integer values. "
         "See skforecast docs for more info about how to include "
         "categorical features https://skforecast.org/"
         "latest/user_guides/categorical-features.html")
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster._create_train_X_y(series=series, exog=exog)


def test_create_train_X_y_MissingValuesWarning_when_exog_has_missing_values():
    """
    Test _create_train_X_y is issues a MissingValuesWarning when exog has missing values.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(6)),  
                           'l2': pd.Series(np.arange(6))})
    exog = pd.Series([1, 2, 3, np.nan, 5, 6], name='exog')
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l2', lags=2, steps=2)

    warn_msg = re.escape(
        ("`exog` has missing values. Most machine learning models do "
         "not allow missing values. Fitting the forecaster may fail.")  
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):
        forecaster._create_train_X_y(series=series, exog=exog)


def test_create_train_X_y_TypeError_when_exog_is_not_pandas_Series_or_DataFrame():
    """
    Test TypeError is raised when exog is not a pandas Series or DataFrame.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10))})
    exog = np.arange(10)
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1', 
                                               lags=3, steps=2)
    
    err_msg = (
        f"`exog` must be a pandas Series or DataFrame. Got {type(exog)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster._create_train_X_y(series=series, exog=exog)


@pytest.mark.parametrize("exog", 
                         [pd.Series(np.arange(15), name='exog'), 
                          pd.DataFrame(np.arange(50).reshape(25, 2))])
def test_create_train_X_y_ValueError_when_series_and_exog_have_different_length(exog):
    """
    Test ValueError is raised when length of series is not equal to length exog.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10))})
    forecaster = forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                                            lags=3, steps=3)

    err_msg = re.escape(
        (f"`exog` must have same number of samples as `series`. "
         f"length `exog`: ({len(exog)}), length `series`: ({len(series)})")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(series=series, exog=exog)


@pytest.mark.parametrize('exog', ['l1', ['l1'], ['l1', 'l2']])
def test_create_train_X_y_ValueError_when_exog_columns_same_as_series_col_names(exog):
    """
    Test ValueError is raised when an exog column is named the same as
    the series columns.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10))})

    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1', lags=3, steps=2)
    series_col_names = list(series.columns)
    exog_col_names = exog if isinstance(exog, list) else [exog]

    err_msg = re.escape(
        (f"`exog` cannot contain a column named the same as one of "
         f"the series (column names of series).\n"
         f"  `series` columns : {series_col_names}.\n"
         f"  `exog`   columns : {exog_col_names}.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(series=series, exog=series[exog])


def test_create_train_X_y_ValueError_when_series_and_exog_have_different_index():
    """
    Test ValueError is raised when series and exog have different index.
    """
    series = pd.DataFrame(
        {"l1": pd.Series(np.arange(10)), "l2": pd.Series(np.arange(10))}
    )
    series.index = pd.date_range(start="2022-01-01", periods=10, freq="1D")
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level="l1", lags=3, steps=3
    )

    exog = pd.Series(
        np.arange(10), index=pd.RangeIndex(start=0, stop=10, step=1), name="exog"
    )

    err_msg = re.escape(
        ("Different index for `series` and `exog`. They must be equal "
         "to ensure the correct alignment of values.")
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.fit(series=series, exog=exog)


@pytest.mark.parametrize("level, expected_y_values", 
                         [('l1', [-0.52223297, -0.17407766,  0.17407766,  0.52223297,  0.87038828, 1.21854359,  1.5666989 ]), 
                          ('l2', [-0.52223297, -0.17407766,  0.17407766,  0.52223297,  0.87038828, 1.21854359,  1.5666989 ])])
def test_create_train_X_y_output_when_lags_3_steps_1_and_exog_is_None(level, expected_y_values):
    """
    Test output of _create_train_X_y when regressor is LinearRegression, 
    lags is 3 and steps is 1.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(100, 110), dtype=float)})
    exog = None

    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level=level,
                                               lags=3, steps=1)
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[-0.87038828, -1.21854359, -1.5666989 , -0.87038828, -1.21854359, -1.5666989 ],
                             [-0.52223297, -0.87038828, -1.21854359, -0.52223297, -0.87038828, -1.21854359],
                             [-0.17407766, -0.52223297, -0.87038828, -0.17407766, -0.52223297, -0.87038828],
                             [ 0.17407766, -0.17407766, -0.52223297,  0.17407766, -0.17407766, -0.52223297],
                             [ 0.52223297,  0.17407766, -0.17407766,  0.52223297,  0.17407766, -0.17407766],
                             [ 0.87038828,  0.52223297,  0.17407766,  0.87038828,  0.52223297, 0.17407766],
                             [ 1.21854359,  0.87038828,  0.52223297,  1.21854359,  0.87038828, 0.52223297]], 
                            dtype=float),
            index   = pd.RangeIndex(start=3, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3',
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3']
        ),
        {1: pd.Series(
                data  = np.array(expected_y_values, dtype=float), 
                index = pd.RangeIndex(start=3, stop=10, step=1),
                name  = f'{level}_step_1'
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        None,
        None,
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l2_lag_1', 'l2_lag_2', 'l2_lag_3'],
        None
    )

    forecaster.exog_in_ is False
    forecaster.X_train_direct_exog_names_out_ is None
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key])
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]


@pytest.mark.parametrize("level, expected_y_values", 
                         [('l1', ([3., 4., 5., 6., 7., 8.], 
                                  [4., 5., 6., 7., 8., 9.])), 
                          ('l2', ([103., 104., 105., 106., 107., 108.], 
                                  [104., 105., 106., 107., 108., 109.]))])
def test_create_train_X_y_output_when_interspersed_lags_steps_2_and_exog_is_None(level, expected_y_values):
    """
    Test output of _create_train_X_y when regressor is LinearRegression, 
    interspersed lags and steps is 2.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(100, 110), dtype=float)})
    exog = None

    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level=level,
                                               lags=[1, 3], steps=2, transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[2., 0., 102., 100.],
                             [3., 1., 103., 101.],
                             [4., 2., 104., 102.],
                             [5., 3., 105., 103.],
                             [6., 4., 106., 104.],
                             [7., 5., 107., 105.]], dtype=float),
            index   = pd.RangeIndex(start=4, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_3',
                       'l2_lag_1', 'l2_lag_3']
        ),
        {1: pd.Series(
                data  = np.array(expected_y_values[0], dtype=float), 
                index = pd.RangeIndex(start=3, stop=9, step=1),
                name  = f'{level}_step_1'
            ),
         2: pd.Series(
                data  = np.array(expected_y_values[1], dtype=float), 
                index = pd.RangeIndex(start=4, stop=10, step=1),
                name  = f'{level}_step_2'
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        None,
        None,
        ['l1_lag_1', 'l1_lag_3', 'l2_lag_1', 'l2_lag_3'],
        None
    )

    forecaster.exog_in_ is False
    forecaster.X_train_direct_exog_names_out_ is None
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key])
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]


@pytest.mark.parametrize("level, expected_y_values", 
                         [('l1', ([5., 6., 7., 8.], 
                                  [6., 7., 8., 9.])), 
                          ('l2', ([105., 106., 107., 108.], 
                                  [106., 107., 108., 109.]))])
def test_create_train_X_y_output_when_lags_dict_steps_2_and_exog_is_None(level, expected_y_values):
    """
    Test output of _create_train_X_y when regressor is LinearRegression, 
    different lags and steps is 2.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(100, 110))})
    exog = None

    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level=level,
                                               lags={'l1': 3, 'l2': [1, 5]}, 
                                               steps=2, transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 104., 100.],
                             [5., 4., 3., 105., 101.],
                             [6., 5., 4., 106., 102.],
                             [7., 6., 5., 107., 103.]], dtype=float),
            index   = pd.RangeIndex(start=6, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3',
                       'l2_lag_1', 'l2_lag_5']
        ),
        {1: pd.Series(
                data  = np.array(expected_y_values[0], dtype=float), 
                index = pd.RangeIndex(start=5, stop=9, step=1),
                name  = f'{level}_step_1'
            ),
         2: pd.Series(
                data  = np.array(expected_y_values[1], dtype=float), 
                index = pd.RangeIndex(start=6, stop=10, step=1),
                name  = f'{level}_step_2'
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        None,
        None,
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l2_lag_1', 'l2_lag_5'],
        None
    )

    forecaster.exog_in_ is False
    forecaster.X_train_direct_exog_names_out_ is None
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]


@pytest.mark.parametrize("level, expected_y_values", 
                         [('l1', ([3., 4., 5., 6., 7., 8.], 
                                  [4., 5., 6., 7., 8., 9.])), 
                          ('l2', ([103., 104., 105., 106., 107., 108.], 
                                  [104., 105., 106., 107., 108., 109.]))])
def test_create_train_X_y_output_when_lags_dict_with_None_steps_2_and_exog_is_None(level, expected_y_values):
    """
    Test output of _create_train_X_y when regressor is LinearRegression, 
    lags is a dict with None and steps is 2.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(100, 110))})
    exog = None

    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level=level,
                                               lags={'l1': 3, 'l2': None}, 
                                               steps=2, transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[2., 1., 0.],
                             [3., 2., 1.],
                             [4., 3., 2.],
                             [5., 4., 3.],
                             [6., 5., 4.],
                             [7., 6., 5.]], dtype=float),
            index   = pd.RangeIndex(start=4, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3']
        ),
        {1: pd.Series(
                data  = np.array(expected_y_values[0], dtype=float), 
                index = pd.RangeIndex(start=3, stop=9, step=1),
                name  = f'{level}_step_1'
            ),
         2: pd.Series(
                data  = np.array(expected_y_values[1], dtype=float), 
                index = pd.RangeIndex(start=4, stop=10, step=1),
                name  = f'{level}_step_2'
            )
        },
        ['l1', 'l2'],
        ['l1'],
        None,
        None,
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3'],
        None
    )

    forecaster.exog_in_ is False
    forecaster.X_train_direct_exog_names_out_ is None
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]


def test_create_train_X_y_output_when_y_and_exog_no_pandas_index():
    """
    Test the output of _create_train_X_y when y and exog have no pandas index 
    that doesn't start at 0.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    series.index = np.arange(6, 16)
    exog = pd.Series(np.arange(100, 110), name='exog', 
                     index=np.arange(6, 16), dtype=float)

    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=5, steps=1, transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50., 105.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51., 106.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52., 107.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53., 108.],
                             [8., 7., 6., 5., 4., 58., 57., 56., 55., 54., 109.]], 
                             dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                       'exog_step_1']
        ),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "l1_step_1"
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        ['exog'],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'exog_step_1'],
        {'exog': exog.dtype}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1']
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt: f'dtype: {dt}')
def test_create_train_X_y_output_when_lags_5_steps_1_and_exog_is_series_of_float_int(dtype):
    """
    Test output of _create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 1 and exog is pandas Series of floats or ints.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=dtype)

    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=5, steps=1, transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50., 105.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51., 106.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52., 107.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53., 108.],
                             [8., 7., 6., 5., 4., 58., 57., 56., 55., 54., 109.]], 
                             dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                       'exog_step_1']
        ).astype({'exog_step_1': float}),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "l1_step_1"
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        ['exog'],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'exog_step_1'],
        {'exog': exog.dtype}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1']
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt: f'dtype: {dt}')
def test_create_train_X_y_output_when_lags_5_steps_2_and_exog_is_series_of_float_int(dtype):
    """
    Test output of _create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 2 and exog is pandas Series of floats or ints.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=dtype)

    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=5, steps=2, transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog)
 
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50., 105., 106.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51., 106., 107.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52., 107., 108.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53., 108., 109.]], 
                             dtype=float),
            index = pd.RangeIndex(start=6, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                       'exog_step_1', 'exog_step_2']
        ).astype({'exog_step_1': float, 'exog_step_2': float}),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=9, step=1),
                name  = "l1_step_1"
            ),
         2: pd.Series(
                data  = np.array([6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=6, stop=10, step=1),
                name  = "l1_step_2"
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        ['exog'],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'exog_step_1', 'exog_step_2'],
        {'exog': exog.dtype}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1', 'exog_step_2']
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt: f'dtype: {dt}')
def test_create_train_X_y_output_when_lags_5_steps_1_and_exog_is_dataframe_of_float_int(dtype):
    """
    Test output of _create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 1 and exog is pandas DataFrame of floats or ints.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog   = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=dtype),
                           'exog_2': np.arange(1000, 1010, dtype=dtype)})
    
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l2',
                                               lags=5, steps=1, transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50., 105., 1005.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51., 106., 1006.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52., 107., 1007.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53., 108., 1008.],
                             [8., 7., 6., 5., 4., 58., 57., 56., 55., 54., 109., 1009.]], 
                             dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                       'exog_1_step_1', 'exog_2_step_1']
        ).astype({'exog_1_step_1': float, 'exog_2_step_1': float}),
        {1: pd.Series(
                data  = np.array([55., 56., 57., 58., 59.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "l2_step_1"
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog_1', 'exog_2'],
        ['exog_1', 'exog_2'],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'exog_1_step_1', 'exog_2_step_1'],
        {'exog_1': exog['exog_1'].dtype, 'exog_2': exog['exog_2'].dtype}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_1_step_1', 'exog_2_step_1']
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt: f'dtype: {dt}')
def test_create_train_X_y_output_when_lags_5_steps_3_and_exog_is_dataframe_of_float_int(dtype):
    """
    Test output of _create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 3 and exog is pandas DataFrame of floats or ints.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=dtype),
                         'exog_2': np.arange(1000, 1010, dtype=dtype)})
    
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l2',
                                               lags=5, steps=3, transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50., 105., 1005., 106., 1006., 107., 1007.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51., 106., 1006., 107., 1007., 108., 1008.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52., 107., 1007., 108., 1008., 109., 1009.]],
                            dtype=float),
            index = pd.RangeIndex(start=7, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                       'exog_1_step_1', 'exog_2_step_1', 
                       'exog_1_step_2', 'exog_2_step_2', 
                       'exog_1_step_3', 'exog_2_step_3']
        ).astype({'exog_1_step_1': float, 'exog_2_step_1': float, 
                  'exog_1_step_2': float, 'exog_2_step_2': float, 
                  'exog_1_step_3': float, 'exog_2_step_3': float}),
        {1: pd.Series(
                data  = np.array([55., 56., 57.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=8, step=1),
                name  = "l2_step_1"
            ),
         2: pd.Series(
                data  = np.array([56., 57., 58.], dtype=float), 
                index = pd.RangeIndex(start=6, stop=9, step=1),
                name  = "l2_step_2"
            ),
         3: pd.Series(
                data  = np.array([57., 58., 59.], dtype=float), 
                index = pd.RangeIndex(start=7, stop=10, step=1),
                name  = "l2_step_3"
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog_1', 'exog_2'],
        ['exog_1', 'exog_2'],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'exog_1_step_1', 'exog_2_step_1', 
         'exog_1_step_2', 'exog_2_step_2', 
         'exog_1_step_3', 'exog_2_step_3'],
        {'exog_1': exog['exog_1'].dtype, 'exog_2': exog['exog_2'].dtype}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == [
        'exog_1_step_1', 'exog_2_step_1', 
        'exog_1_step_2', 'exog_2_step_2', 
        'exog_1_step_3', 'exog_2_step_3'
    ]
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]


@pytest.mark.parametrize("exog_values, dtype", 
                         [([True]    , bool), 
                          (['string'], str)], 
                         ids = lambda dt: f'values, dtype: {dt}')
def test_create_train_X_y_output_when_lags_5_steps_1_and_exog_is_series_of_bool_str(exog_values, dtype):
    """
    Test output of _create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 1 and exog is pandas Series of bool or str.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.Series(exog_values * 10, name='exog', dtype=dtype)

    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=5, steps=1, transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53.],
                             [8., 7., 6., 5., 4., 58., 57., 56., 55., 54.]], 
                             dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5']
        ).assign(exog_step_1=exog_values * 5).astype({'exog_step_1': dtype}),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "l1_step_1"
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        ['exog'],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'exog_step_1'],
        {'exog': exog.dtype}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1']
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]


@pytest.mark.parametrize("exog_values, dtype", 
                         [([True]    , bool), 
                          (['string'], str)], 
                         ids = lambda dt: f'values, dtype: {dt}')
def test_create_train_X_y_output_when_lags_5_steps_2_and_exog_is_series_of_bool_str(exog_values, dtype):
    """
    Test output of _create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 2 and exog is pandas Series of bool or str.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.Series(exog_values * 10, name='exog', dtype=dtype)

    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=5, steps=2, transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog)
 
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53.]], 
                             dtype=float),
            index = pd.RangeIndex(start=6, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5']
        ).assign(exog_step_1=exog_values * 4,
                 exog_step_2=exog_values * 4).astype({'exog_step_1': dtype, 
                                                      'exog_step_2': dtype}),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=9, step=1),
                name  = "l1_step_1"
            ),
         2: pd.Series(
                data  = np.array([6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=6, stop=10, step=1),
                name  = "l1_step_2"
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        ['exog'],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'exog_step_1', 'exog_step_2'],
        {'exog': exog.dtype}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1', 'exog_step_2']
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key])
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]


@pytest.mark.parametrize("v_exog_1   , v_exog_2  , dtype", 
                         [([True]    , [False]   , bool), 
                          (['string'], ['string'], str)], 
                         ids = lambda dt: f'values, dtype: {dt}')
def test_create_train_X_y_output_when_lags_5_steps_1_and_exog_is_dataframe_of_bool_str(v_exog_1, v_exog_2, dtype):
    """
    Test output of _create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 1 and exog is pandas DataFrame of bool or str.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.DataFrame({'exog_1': v_exog_1 * 10,
                         'exog_2': v_exog_2 * 10})
    
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l2',
                                               lags=5, steps=1, transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53.],
                             [8., 7., 6., 5., 4., 58., 57., 56., 55., 54.]], 
                             dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5']
        ).assign(exog_1_step_1=v_exog_1 * 5,
                 exog_2_step_1=v_exog_2 * 5).astype({'exog_1_step_1': dtype, 
                                                     'exog_2_step_1': dtype}),
        {1: pd.Series(
                data  = np.array([55., 56., 57., 58., 59.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "l2_step_1"
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog_1', 'exog_2'],
        ['exog_1', 'exog_2'],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'exog_1_step_1', 'exog_2_step_1'],
        {'exog_1': exog['exog_1'].dtype, 'exog_2': exog['exog_2'].dtype}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1', 'exog_step_2']
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]


@pytest.mark.parametrize("v_exog_1   , v_exog_2  , dtype", 
                         [([True]    , [False]   , bool), 
                          (['string'], ['string'], str)], 
                         ids = lambda dt: f'values, dtype: {dt}')
def test_create_train_X_y_output_when_lags_5_steps_3_and_exog_is_dataframe_of_bool_str(v_exog_1, v_exog_2, dtype):
    """
    Test output of _create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 3 and exog is pandas DataFrame of bool or str.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.DataFrame({'exog_1': v_exog_1 * 10,
                         'exog_2': v_exog_2 * 10})
    
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l2',
                                               lags=5, steps=3, transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52.]],
                            dtype=float),
            index = pd.RangeIndex(start=7, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5']
        ).assign(exog_1_step_1=v_exog_1 * 3,
                 exog_2_step_1=v_exog_2 * 3,
                 exog_1_step_2=v_exog_1 * 3,
                 exog_2_step_2=v_exog_2 * 3,
                 exog_1_step_3=v_exog_1 * 3,
                 exog_2_step_3=v_exog_2 * 3
        ).astype({'exog_1_step_1': dtype, 'exog_2_step_1': dtype, 
                  'exog_1_step_2': dtype, 'exog_2_step_2': dtype, 
                  'exog_1_step_3': dtype, 'exog_2_step_3': dtype}
        ),
        {1: pd.Series(
                data  = np.array([55., 56., 57.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=8, step=1),
                name  = "l2_step_1"
            ),
         2: pd.Series(
                data  = np.array([56., 57., 58.], dtype=float), 
                index = pd.RangeIndex(start=6, stop=9, step=1),
                name  = "l2_step_2"
            ),
         3: pd.Series(
                data  = np.array([57., 58., 59.], dtype=float), 
                index = pd.RangeIndex(start=7, stop=10, step=1),
                name  = "l2_step_3"
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog_1', 'exog_2'],
        ['exog_1', 'exog_2'],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'exog_1_step_1', 'exog_2_step_1',
         'exog_1_step_2', 'exog_2_step_2',
         'exog_1_step_3', 'exog_2_step_3'],
        {'exog_1': exog['exog_1'].dtype, 'exog_2': exog['exog_2'].dtype}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == [
        'exog_1_step_1', 'exog_2_step_1',
        'exog_1_step_2', 'exog_2_step_2',
        'exog_1_step_3', 'exog_2_step_3'
    ]
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]


def test_create_train_X_y_output_when_lags_5_steps_1_and_exog_is_series_of_category():
    """
    Test output of _create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 1 and exog is pandas Series of category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.Series(range(10), name='exog', dtype='category')

    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=5, steps=1, transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53.],
                             [8., 7., 6., 5., 4., 58., 57., 56., 55., 54.]], 
                             dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5']
        ).assign(exog_step_1=pd.Categorical(range(5, 10), categories=range(10))),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "l1_step_1"
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        ['exog'],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'exog_step_1'],
        {'exog': exog.dtype}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1']
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]


def test_create_train_X_y_output_when_lags_5_steps_2_and_exog_is_series_of_category():
    """
    Test output of _create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 2 and exog is pandas Series of category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.Series(range(10), name='exog', dtype='category')

    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=5, steps=2, transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog)
 
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53.]], 
                             dtype=float),
            index = pd.RangeIndex(start=6, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5']
        ).assign(exog_step_1=pd.Categorical(range(5, 9), categories=range(10)),
                 exog_step_2=pd.Categorical(range(6, 10), categories=range(10))),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=9, step=1),
                name  = "l1_step_1"
            ),
         2: pd.Series(
                data  = np.array([6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=6, stop=10, step=1),
                name  = "l1_step_2"
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        ['exog'],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'exog_step_1', 'exog_step_2'],
        {'exog': exog.dtype}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1', 'exog_step_2']
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]


def test_create_train_X_y_output_when_lags_5_steps_1_and_exog_is_dataframe_of_category():
    """
    Test output of _create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 1 and exog is pandas DataFrame of category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.DataFrame({'exog_1': pd.Categorical(range(10)),
                         'exog_2': pd.Categorical(range(100, 110))})
    
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l2',
                                               lags=5, steps=1, transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53.],
                             [8., 7., 6., 5., 4., 58., 57., 56., 55., 54.]], 
                             dtype=float),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5']
        ).assign(
            exog_1_step_1=pd.Categorical(range(5, 10), categories=range(10)),
            exog_2_step_1=pd.Categorical(range(105, 110), categories=range(100, 110))
        ),
        {1: pd.Series(
                data  = np.array([55., 56., 57., 58., 59.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "l2_step_1"
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog_1', 'exog_2'],
        ['exog_1', 'exog_2'],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'exog_1_step_1', 'exog_2_step_1'],
        {'exog_1': exog['exog_1'].dtype, 'exog_2': exog['exog_2'].dtype}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1', 'exog_step_2']
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]


def test_create_train_X_y_output_when_lags_5_steps_3_and_exog_is_dataframe_of_category():
    """
    Test output of _create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 3 and exog is pandas DataFrame of category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.DataFrame({'exog_1': pd.Categorical(range(10)),
                         'exog_2': pd.Categorical(range(100, 110))})
    
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l2',
                                               lags=5, steps=3, transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52.]],
                            dtype=float),
            index = pd.RangeIndex(start=7, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5']
        ).assign(
            exog_1_step_1=pd.Categorical(range(5, 8), categories=range(10)),
            exog_2_step_1=pd.Categorical(range(105, 108), categories=range(100, 110)),
            exog_1_step_2=pd.Categorical(range(6, 9), categories=range(10)),
            exog_2_step_2=pd.Categorical(range(106, 109), categories=range(100, 110)),
            exog_1_step_3=pd.Categorical(range(7, 10), categories=range(10)),
            exog_2_step_3=pd.Categorical(range(107, 110), categories=range(100, 110))
        ),
        {1: pd.Series(
                data  = np.array([55., 56., 57.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=8, step=1),
                name  = "l2_step_1"
            ),
         2: pd.Series(
                data  = np.array([56., 57., 58.], dtype=float), 
                index = pd.RangeIndex(start=6, stop=9, step=1),
                name  = "l2_step_2"
            ),
         3: pd.Series(
                data  = np.array([57., 58., 59.], dtype=float), 
                index = pd.RangeIndex(start=7, stop=10, step=1),
                name  = "l2_step_3"
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog_1', 'exog_2'],
        ['exog_1', 'exog_2'],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'exog_1_step_1', 'exog_2_step_1',
         'exog_1_step_2', 'exog_2_step_2',
         'exog_1_step_3', 'exog_2_step_3'],
        {'exog_1': exog['exog_1'].dtype, 'exog_2': exog['exog_2'].dtype}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == [
        'exog_1_step_1', 'exog_2_step_1',
        'exog_1_step_2', 'exog_2_step_2',
        'exog_1_step_3', 'exog_2_step_3'
    ]
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]


def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_dataframe_of_float_int_category_steps_1():
    """
    Test output of _create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 1 and exog is pandas DataFrame of float, int and category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.DataFrame({'exog_1': pd.Series(np.arange(100, 110), dtype=float),
                         'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
                         'exog_3': pd.Categorical(range(100, 110))})
    
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=5, steps=1, transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog)        
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50., 105., 1005.],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51., 106., 1006.],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52., 107., 1007.],
                             [7., 6., 5., 4., 3., 57., 56., 55., 54., 53., 108., 1008.],
                             [8., 7., 6., 5., 4., 58., 57., 56., 55., 54., 109., 1009.]], 
                             dtype=float),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                       'exog_1_step_1', 'exog_2_step_1']
        ).astype({'exog_1_step_1': float, 
                  'exog_2_step_1': int}
        ).assign(exog_3_step_1=pd.Categorical(range(105, 110), categories=range(100, 110))
        ),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "l1_step_1"
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog_1', 'exog_2', 'exog_3'],
        ['exog_1', 'exog_2', 'exog_3'],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'exog_1_step_1', 'exog_2_step_1', 'exog_3_step_1'],
        {'exog_1': exog['exog_1'].dtype, 'exog_2': exog['exog_2'].dtype, 'exog_3': exog['exog_3'].dtype}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_1_step_1', 'exog_2_step_1', 'exog_3_step_1']
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]


def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_dataframe_of_float_int_category_steps_3():
    """
    Test output of _create_train_X_y when regressor is LinearRegression, 
    lags is 5, steps is 3 and exog is pandas DataFrame of float, int and category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.DataFrame({'exog_1': pd.Series(np.arange(100, 110), dtype=float),
                         'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
                         'exog_3': pd.Categorical(range(100, 110))})
    
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=5, steps=3, transformer_series=None)
    results = forecaster._create_train_X_y(series=series, exog=exog)        
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 54., 53., 52., 51., 50., 105, 1005, 105, 106, 1006, 106, 107, 1007, 107],
                             [5., 4., 3., 2., 1., 55., 54., 53., 52., 51., 106, 1006, 106, 107, 1007, 107, 108, 1008, 108],
                             [6., 5., 4., 3., 2., 56., 55., 54., 53., 52., 107, 1007, 107, 108, 1008, 108, 109, 1009, 109]],
                            dtype=float),
            index = pd.RangeIndex(start=7, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                       'exog_1_step_1', 'exog_2_step_1', 'exog_3_step_1', 
                       'exog_1_step_2', 'exog_2_step_2', 'exog_3_step_2', 
                       'exog_1_step_3', 'exog_2_step_3', 'exog_3_step_3']
        ).astype({'exog_1_step_1': float, 'exog_2_step_1': int,
                  'exog_1_step_2': float, 'exog_2_step_2': int,
                  'exog_1_step_3': float, 'exog_2_step_3': int}
        ).assign(exog_3_step_1=pd.Categorical(range(105, 108), categories=range(100, 110)),
                 exog_3_step_2=pd.Categorical(range(106, 109), categories=range(100, 110)),
                 exog_3_step_3=pd.Categorical(range(107, 110), categories=range(100, 110))
        ),
        {1: pd.Series(
                data  = np.array([5., 6., 7.], dtype=float), 
                index = pd.RangeIndex(start=5, stop=8, step=1),
                name  = "l1_step_1"
            ),
         2: pd.Series(
                data  = np.array([6., 7., 8.], dtype=float), 
                index = pd.RangeIndex(start=6, stop=9, step=1),
                name  = "l1_step_2"
            ),
         3: pd.Series(
                data  = np.array([7., 8., 9.], dtype=float), 
                index = pd.RangeIndex(start=7, stop=10, step=1),
                name  = "l1_step_3"
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog_1', 'exog_2', 'exog_3'],
        ['exog_1', 'exog_2', 'exog_3'],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'exog_1_step_1', 'exog_2_step_1', 'exog_3_step_1', 
         'exog_1_step_2', 'exog_2_step_2', 'exog_3_step_2', 
         'exog_1_step_3', 'exog_2_step_3', 'exog_3_step_3'],
        {'exog_1': exog['exog_1'].dtype, 'exog_2': exog['exog_2'].dtype, 'exog_3': exog['exog_3'].dtype}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == [
        'exog_1_step_1', 'exog_2_step_1', 'exog_3_step_1', 
        'exog_1_step_2', 'exog_2_step_2', 'exog_3_step_2', 
        'exog_1_step_3', 'exog_2_step_3', 'exog_3_step_3'
    ]
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'l1': StandardScaler(), 'l2': StandardScaler()}], 
                         ids = lambda tr: f'transformer_series: {tr}')
def test_create_train_X_y_output_when_lags_5_steps_1_and_transformer_series_StandardScaler(transformer_series):
    """
    Test the output of _create_train_X_y when exog is None and transformer_series
    is StandardScaler with steps=1.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    
    forecaster = ForecasterDirectMultiVariate(
                    regressor          = LinearRegression(),
                    lags               = 5,
                    level              = 'l1',
                    steps              = 1,
                    transformer_series = transformer_series
                )
    results = forecaster._create_train_X_y(series=series)

    expected = (
        pd.DataFrame(
            data = np.array([[-0.17407766, -0.52223297, -0.87038828, -1.21854359, -1.5666989 ,
                              -0.17407766, -0.52223297, -0.87038828, -1.21854359, -1.5666989 ],
                             [ 0.17407766, -0.17407766, -0.52223297, -0.87038828, -1.21854359,
                               0.17407766, -0.17407766, -0.52223297, -0.87038828, -1.21854359],
                             [ 0.52223297,  0.17407766, -0.17407766, -0.52223297, -0.87038828,
                               0.52223297,  0.17407766, -0.17407766, -0.52223297, -0.87038828],
                             [ 0.87038828,  0.52223297,  0.17407766, -0.17407766, -0.52223297,
                               0.87038828,  0.52223297,  0.17407766, -0.17407766, -0.52223297],
                             [ 1.21854359,  0.87038828,  0.52223297,  0.17407766, -0.17407766,
                               1.21854359,  0.87038828,  0.52223297,  0.17407766, -0.17407766]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5']
        ),
        {1: pd.Series(
                data  = np.array([0.17407766, 0.52223297, 0.87038828, 
                                  1.21854359, 1.5666989]), 
                index = pd.RangeIndex(start=5, stop=10, step=1),
                name  = "l1_step_1"
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        None,
        None,
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5'],
        None
    )

    forecaster.exog_in_ is False
    forecaster.X_train_direct_exog_names_out_ is None
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]


@pytest.mark.parametrize("level, expected_y_values", 
                         [('l1', [3., 4., 5., 6., 7., 8., 9.]), 
                          ('l2', [103., 104., 105., 106., 107., 108., 109.])])
def test_create_train_X_y_output_when_lags_3_steps_1_and_exog_is_None_and_transformer_exog_is_not_None(level, expected_y_values):
    """
    Test output of _create_train_X_y when regressor is LinearRegression, 
    lags is 3 and steps is 1 and transformer_exog is not None.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(100, 110), dtype=float)})
    exog = None

    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level=level,
                                               lags=3, steps=1, transformer_series=None, 
                                               transformer_exog = StandardScaler())
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[2., 1., 0., 102., 101., 100.],
                             [3., 2., 1., 103., 102., 101.],
                             [4., 3., 2., 104., 103., 102.],
                             [5., 4., 3., 105., 104., 103.],
                             [6., 5., 4., 106., 105., 104.],
                             [7., 6., 5., 107., 106., 105.],
                             [8., 7., 6., 108., 107., 106.]], dtype=float),
            index   = pd.RangeIndex(start=3, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3',
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3']
        ),
        {1: pd.Series(
                data  = np.array(expected_y_values, dtype=float), 
                index = pd.RangeIndex(start=3, stop=10, step=1),
                name  = f'{level}_step_1'
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        None,
        None,
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l2_lag_1', 'l2_lag_2', 'l2_lag_3'],
        None
    )

    forecaster.exog_in_ is False
    forecaster.X_train_direct_exog_names_out_ is None
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'l1': StandardScaler(), 'l2': StandardScaler()}], 
                         ids = lambda tr: f'transformer_series: {tr}')
def test_create_train_X_y_output_when_transformer_series_and_transformer_exog_steps_2(transformer_series):
    """
    Test the output of _create_train_X_y when using transformer_series and 
    transformer_exog with steps=2.
    """
    series = pd.DataFrame({'l1': np.arange(10, dtype=float), 
                           'l2': np.arange(10, dtype=float)},
                           index = pd.date_range("1990-01-01", periods=10, freq='D'))
    exog = pd.DataFrame({
               'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 24.4, 87.2, 47.4, 23.8],
               'col_2': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b']},
                index = pd.date_range("1990-01-01", periods=10, freq='D'))

    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['col_1']),
                            ('onehot', OneHotEncoder(), ['col_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )

    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     level              = 'l1',
                     steps              = 2,
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog
                 )
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[-0.17407766, -0.52223297, -0.87038828, -1.21854359, -1.5666989 ,
                              -0.17407766, -0.52223297, -0.87038828, -1.21854359, -1.5666989 ,
                              -0.04719331,  0.        ,   1.        ,-0.81862236,  1.        ,
                               0.        ],
                             [ 0.17407766, -0.17407766, -0.52223297, -0.87038828, -1.21854359,
                               0.17407766, -0.17407766, -0.52223297, -0.87038828, -1.21854359,
                              -0.81862236,  1.        ,  0.        ,  2.03112731,  0.        ,
                               1.        ],
                             [ 0.52223297,  0.17407766, -0.17407766, -0.52223297, -0.87038828,
                               0.52223297,  0.17407766, -0.17407766, -0.52223297, -0.87038828,
                               2.03112731,  0.        ,  1.        ,  0.22507577,  1.        ,
                               0.        ],
                             [ 0.87038828,  0.52223297,  0.17407766, -0.17407766, -0.52223297,
                               0.87038828,  0.52223297,  0.17407766, -0.17407766, -0.52223297,
                               0.22507577,  1.        ,  0.        , -0.84584926,  0.        ,
                               1.        ]]),
            index   = pd.date_range("1990-01-07", periods=4, freq='D'),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5', 
                       'col_1_step_1', 'col_2_a_step_1', 'col_2_b_step_1',
                       'col_1_step_2', 'col_2_a_step_2', 'col_2_b_step_2']
        ),
        {1: pd.Series(
                data  = np.array([0.17407766, 0.52223297, 0.87038828, 1.21854359]), 
                index = pd.date_range("1990-01-06", periods=4, freq='D'),
                name  = "l1_step_1"
            ),
         2: pd.Series(
                data  = np.array([0.52223297, 0.87038828, 1.21854359, 1.5666989]), 
                index = pd.date_range("1990-01-07", periods=4, freq='D'),
                name  = "l1_step_2"
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['col_1', 'col_2'],
        ['col_1', 'col_2_a', 'col_2_b'],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5', 
         'col_1_step_1', 'col_2_a_step_1', 'col_2_b_step_1',
         'col_1_step_2', 'col_2_a_step_2', 'col_2_b_step_2'],
        {'col_1': exog['col_1'].dtype, 'col_2': exog['col_2'].dtype}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1']
    forecaster.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    

@pytest.mark.parametrize("fit_forecaster", 
                         [True, False], 
                         ids = lambda fitted: f'fit_forecaster: {fitted}')
def test_create_train_X_y_output_when_y_is_series_exog_is_series_and_differentiation_is_1_steps_3(fit_forecaster):
    """
    Test the output of _create_train_X_y when using differentiation=1. Comparing 
    the matrix created with and without differentiating the series with steps=3.
    """
    # Data differentiated
    arr = data.to_numpy(copy=True)
    diferenciator = TimeSeriesDifferentiator(order=1)
    arr_diff = diferenciator.fit_transform(arr)

    series = pd.DataFrame(
        {'l1': arr,
         'l2': arr},
        index=data.index
    )

    series_diff = pd.DataFrame(
        {'l1': arr_diff,
         'l2': arr_diff},
        index=data.index
    ).dropna()

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff = exog.iloc[1:]
    end_train = '2003-03-01 23:59:00'

    forecaster_1 = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', steps=3, lags=5, transformer_series=None
    )
    forecaster_2 = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', steps=3, lags=5, transformer_series=None, differentiation=1
    )
    
    if fit_forecaster:
        forecaster_2.fit(series=series.loc[:end_train], exog=exog.loc[:end_train])

    output_1 = forecaster_1._create_train_X_y(
                   series = series_diff.loc[:end_train],
                   exog   = exog_diff.loc[:end_train]
               )
    output_2 = forecaster_2._create_train_X_y(
                   series = series.loc[:end_train],
                   exog   = exog.loc[:end_train]
               )

    forecaster_1.exog_in_ is True
    forecaster_1.X_train_direct_exog_names_out_ == ['exog_step_1', 'exog_step_2', 'exog_step_3']
    forecaster_1.X_train_window_features_names_out_ is None

    forecaster_2.exog_in_ is True
    forecaster_2.X_train_direct_exog_names_out_ == ['exog_step_1', 'exog_step_2', 'exog_step_3']
    forecaster_2.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(output_1[0], output_2[0])
    assert output_1[1].keys() == output_2[1].keys()
    for key in output_1[1]: 
        pd.testing.assert_series_equal(output_1[1][key], output_2[1][key]) 
    assert output_1[2] == output_2[2]
    assert output_1[3] == output_2[3]
    assert output_1[4] == output_2[4]
    assert output_1[5] == output_2[5]
    assert output_1[6] == output_2[6]
    for k in output_1[7].keys():
        assert output_1[7][k] == output_2[7][k]


def test_create_train_X_y_output_when_y_is_series_exog_is_series_and_differentiation_is_2():
    """
    Test the output of _create_train_X_y when using differentiation=2. Comparing 
    the matrix created with and without differentiating the series.
    """
    # Data differentiated
    arr = data.to_numpy(copy=True)
    diferenciator = TimeSeriesDifferentiator(order=2)
    data_diff_2 = diferenciator.fit_transform(arr)

    series = pd.DataFrame(
        {'l1': arr,
         'l2': arr},
        index=data.index
    )

    series_diff_2 = pd.DataFrame(
        {'l1': data_diff_2,
         'l2': data_diff_2},
        index=data.index
    ).dropna()

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff_2 = exog.iloc[2:]
    end_train = '2003-03-01 23:59:00'

    forecaster_1 = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', steps=2, lags=5, transformer_series=None
    )
    forecaster_2 = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', steps=2, lags=5, transformer_series=None, differentiation=2
    )

    output_1 = forecaster_1._create_train_X_y(
                   series = series_diff_2.loc[:end_train],
                   exog   = exog_diff_2.loc[:end_train]
               )
    output_2 = forecaster_2._create_train_X_y(
                   series = series.loc[:end_train],
                   exog   = exog.loc[:end_train]
               )

    forecaster_1.exog_in_ is True
    forecaster_1.X_train_direct_exog_names_out_ == ['exog_step_1', 'exog_step_2']
    forecaster_1.X_train_window_features_names_out_ is None

    forecaster_2.exog_in_ is True
    forecaster_2.X_train_direct_exog_names_out_ == ['exog_step_1', 'exog_step_2']
    forecaster_2.X_train_window_features_names_out_ is None

    pd.testing.assert_frame_equal(output_1[0], output_2[0])
    assert output_1[1].keys() == output_2[1].keys()
    for key in output_1[1]: 
        pd.testing.assert_series_equal(output_1[1][key], output_2[1][key]) 
    assert output_1[2] == output_2[2]
    assert output_1[3] == output_2[3]
    assert output_1[4] == output_2[4]
    assert output_1[5] == output_2[5]
    assert output_1[6] == output_2[6]
    for k in output_1[7].keys():
        assert output_1[7][k] == output_2[7][k]


def test_create_train_X_y_output_when_window_features_and_exog_steps_1():
    """
    Test the output of _create_train_X_y when using window_features and exog 
    with datetime index and steps=1.
    """
    series = pd.DataFrame(
        {'l1': np.arange(15, dtype=float), 
         'l2': np.arange(50, 65, dtype=float)},
        index=pd.date_range('2000-01-01', periods=15, freq='D')
    )
    exog = pd.Series(
        np.arange(100, 115), name='exog', 
        index=pd.date_range('2000-01-01', periods=15, freq='D')
    )
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6]
    )

    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), steps=1, level='l1', lags=5, window_features=rolling, transformer_series=None
    )
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[5., 4., 3., 2., 1., 3., 3., 15., 55., 54., 53., 52., 51., 53., 53., 315., 106.],
                             [6., 5., 4., 3., 2., 4., 4., 21., 56., 55., 54., 53., 52., 54., 54., 321., 107.],
                             [7., 6., 5., 4., 3., 5., 5., 27., 57., 56., 55., 54., 53., 55., 55., 327., 108.],
                             [8., 7., 6., 5., 4., 6., 6., 33., 58., 57., 56., 55., 54., 56., 56., 333., 109.],
                             [9., 8., 7., 6., 5., 7., 7., 39., 59., 58., 57., 56., 55., 57., 57., 339., 110.],
                             [10., 9., 8., 7., 6., 8., 8., 45., 60., 59., 58., 57., 56., 58., 58., 345., 111.],
                             [11., 10., 9., 8., 7., 9., 9., 51., 61., 60., 59., 58., 57., 59., 59., 351., 112.],
                             [12., 11., 10., 9., 8., 10., 10., 57., 62., 61., 60., 59., 58., 60., 60., 357., 113.],
                             [13., 12., 11., 10., 9., 11., 11., 63., 63., 62., 61., 60., 59., 61., 61., 363., 114.]],
                             dtype=float),
            index = pd.date_range('2000-01-07', periods=9, freq='D'),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l1_roll_mean_5', 'l1_roll_median_5', 'l1_roll_sum_6',
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                       'l2_roll_mean_5', 'l2_roll_median_5', 'l2_roll_sum_6',
                       'exog_step_1']
        ),
        {1: pd.Series(
                data  = np.array([6., 7., 8., 9., 10., 11., 12., 13., 14.], dtype=float), 
                index = pd.date_range('2000-01-07', periods=9, freq='D'),
                name  = "l1_step_1"
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        ['exog'],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l1_roll_mean_5', 'l1_roll_median_5', 'l1_roll_sum_6',
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'l2_roll_mean_5', 'l2_roll_median_5', 'l2_roll_sum_6',
         'exog_step_1'],
        {'exog': exog.dtype}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1']
    forecaster.window_features_names == ['roll_mean_5', 'roll_median_5', 'roll_sum_6']
    forecaster.X_train_window_features_names_out_ == [
        'l1_roll_mean_5', 'l1_roll_median_5', 'l1_roll_sum_6',
        'l2_roll_mean_5', 'l2_roll_median_5', 'l2_roll_sum_6'
    ]
    
    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]


def test_create_train_X_y_output_when_window_features_and_exog_steps_2():
    """
    Test the output of _create_train_X_y when using window_features and exog 
    with datetime index and steps=2.
    """
    series = pd.DataFrame(
        {'l1': np.arange(15, dtype=float), 
         'l2': np.arange(50, 65, dtype=float)},
        index=pd.date_range('2000-01-01', periods=15, freq='D')
    )
    exog = pd.Series(
        np.arange(100, 115), name='exog', 
        index=pd.date_range('2000-01-01', periods=15, freq='D')
    )
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6]
    )

    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), steps=2, level='l1', lags=5, 
        window_features=rolling, transformer_series=None
    )
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[5., 4., 3., 2., 1., 3., 3., 15., 55., 54., 53., 52., 51., 53., 53., 315., 106., 107.],
                             [6., 5., 4., 3., 2., 4., 4., 21., 56., 55., 54., 53., 52., 54., 54., 321., 107., 108.],
                             [7., 6., 5., 4., 3., 5., 5., 27., 57., 56., 55., 54., 53., 55., 55., 327., 108., 109.],
                             [8., 7., 6., 5., 4., 6., 6., 33., 58., 57., 56., 55., 54., 56., 56., 333., 109., 110.],
                             [9., 8., 7., 6., 5., 7., 7., 39., 59., 58., 57., 56., 55., 57., 57., 339., 110., 111.],
                             [10., 9., 8., 7., 6., 8., 8., 45., 60., 59., 58., 57., 56., 58., 58., 345., 111., 112.],
                             [11., 10., 9., 8., 7., 9., 9., 51., 61., 60., 59., 58., 57., 59., 59., 351., 112., 113.],
                             [12., 11., 10., 9., 8., 10., 10., 57., 62., 61., 60., 59., 58., 60., 60., 357., 113., 114.]],
                             dtype=float),
            index = pd.date_range('2000-01-08', periods=8, freq='D'),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l1_roll_mean_5', 'l1_roll_median_5', 'l1_roll_sum_6',
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                       'l2_roll_mean_5', 'l2_roll_median_5', 'l2_roll_sum_6',
                       'exog_step_1', 'exog_step_2']
        ),
        {1: pd.Series(
                data  = np.array([6., 7., 8., 9., 10., 11., 12., 13.], dtype=float), 
                index = pd.date_range('2000-01-07', periods=8, freq='D'),
                name  = "l1_step_1"
            ),
         2: pd.Series(
                data  = np.array([7., 8., 9., 10., 11., 12., 13., 14.], dtype=float), 
                index = pd.date_range('2000-01-08', periods=8, freq='D'),
                name  = "l1_step_2"
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        ['exog'],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l1_roll_mean_5', 'l1_roll_median_5', 'l1_roll_sum_6',
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'l2_roll_mean_5', 'l2_roll_median_5', 'l2_roll_sum_6',
         'exog_step_1', 'exog_step_2'],
        {'exog': exog.dtype}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1', 'exog_step_2']
    forecaster.window_features_names == ['roll_mean_5', 'roll_median_5', 'roll_sum_6']
    forecaster.X_train_window_features_names_out_ == [
        'l1_roll_mean_5', 'l1_roll_median_5', 'l1_roll_sum_6',
        'l2_roll_mean_5', 'l2_roll_median_5', 'l2_roll_sum_6'
    ]

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]


def test_create_train_X_y_output_when_two_window_features_and_exog_steps_2():
    """
    Test the output of _create_train_X_y when using 2 window_features and exog 
    with datetime index and steps=2.
    """
    series = pd.DataFrame(
        {'l1': np.arange(15, dtype=float), 
         'l2': np.arange(50, 65, dtype=float)},
        index=pd.date_range('2000-01-01', periods=15, freq='D')
    )
    exog = pd.Series(
        np.arange(100, 115), name='exog', 
        index=pd.date_range('2000-01-01', periods=15, freq='D')
    )
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[5, 5])
    rolling_2 = RollingFeatures(stats='sum', window_sizes=[6])

    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), steps=2, level='l1', lags=5, 
        window_features=[rolling, rolling_2], transformer_series=None
    )
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[5., 4., 3., 2., 1., 3., 3., 15., 55., 54., 53., 52., 51., 53., 53., 315., 106., 107.],
                             [6., 5., 4., 3., 2., 4., 4., 21., 56., 55., 54., 53., 52., 54., 54., 321., 107., 108.],
                             [7., 6., 5., 4., 3., 5., 5., 27., 57., 56., 55., 54., 53., 55., 55., 327., 108., 109.],
                             [8., 7., 6., 5., 4., 6., 6., 33., 58., 57., 56., 55., 54., 56., 56., 333., 109., 110.],
                             [9., 8., 7., 6., 5., 7., 7., 39., 59., 58., 57., 56., 55., 57., 57., 339., 110., 111.],
                             [10., 9., 8., 7., 6., 8., 8., 45., 60., 59., 58., 57., 56., 58., 58., 345., 111., 112.],
                             [11., 10., 9., 8., 7., 9., 9., 51., 61., 60., 59., 58., 57., 59., 59., 351., 112., 113.],
                             [12., 11., 10., 9., 8., 10., 10., 57., 62., 61., 60., 59., 58., 60., 60., 357., 113., 114.]],
                             dtype=float),
            index = pd.date_range('2000-01-08', periods=8, freq='D'),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l1_roll_mean_5', 'l1_roll_median_5', 'l1_roll_sum_6',
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                       'l2_roll_mean_5', 'l2_roll_median_5', 'l2_roll_sum_6',
                       'exog_step_1', 'exog_step_2']
        ),
        {1: pd.Series(
                data  = np.array([6., 7., 8., 9., 10., 11., 12., 13.], dtype=float), 
                index = pd.date_range('2000-01-07', periods=8, freq='D'),
                name  = "l1_step_1"
            ),
         2: pd.Series(
                data  = np.array([7., 8., 9., 10., 11., 12., 13., 14.], dtype=float), 
                index = pd.date_range('2000-01-08', periods=8, freq='D'),
                name  = "l1_step_2"
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        ['exog'],
        ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
         'l1_roll_mean_5', 'l1_roll_median_5', 'l1_roll_sum_6',
         'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
         'l2_roll_mean_5', 'l2_roll_median_5', 'l2_roll_sum_6',
         'exog_step_1', 'exog_step_2'],
        {'exog': exog.dtype}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1', 'exog_step_2']
    forecaster.window_features_names == ['roll_mean_5', 'roll_median_5', 'roll_sum_6']
    forecaster.X_train_window_features_names_out_ == [
        'l1_roll_mean_5', 'l1_roll_median_5', 'l1_roll_sum_6',
        'l2_roll_mean_5', 'l2_roll_median_5', 'l2_roll_sum_6'
    ]

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]


def test_create_train_X_y_output_when_window_features_lags_None_and_exog():
    """
    Test the output of _create_train_X_y when using window_features and exog 
    with datetime index and lags=None and steps=1.
    """
    series = pd.DataFrame(
        {'l1': np.arange(15, dtype=float), 
         'l2': np.arange(50, 65, dtype=float)},
        index=pd.date_range('2000-01-01', periods=15, freq='D')
    )
    exog = pd.Series(
        np.arange(100, 115), name='exog', 
        index=pd.date_range('2000-01-01', periods=15, freq='D')
    )
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6]
    )

    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), steps=1, level='l1', lags=None, window_features=rolling, transformer_series=None
    )
    results = forecaster._create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[3., 3., 15., 53., 53., 315., 106.],
                             [4., 4., 21., 54., 54., 321., 107.],
                             [5., 5., 27., 55., 55., 327., 108.],
                             [6., 6., 33., 56., 56., 333., 109.],
                             [7., 7., 39., 57., 57., 339., 110.],
                             [8., 8., 45., 58., 58., 345., 111.],
                             [9., 9., 51., 59., 59., 351., 112.],
                             [10., 10., 57., 60., 60., 357., 113.],
                             [11., 11., 63., 61., 61., 363., 114.]],
                             dtype=float),
            index = pd.date_range('2000-01-07', periods=9, freq='D'),
            columns = ['l1_roll_mean_5', 'l1_roll_median_5', 'l1_roll_sum_6',
                       'l2_roll_mean_5', 'l2_roll_median_5', 'l2_roll_sum_6',
                       'exog_step_1']
        ),
        {1: pd.Series(
                data  = np.array([6., 7., 8., 9., 10., 11., 12., 13., 14.], dtype=float), 
                index = pd.date_range('2000-01-07', periods=9, freq='D'),
                name  = "l1_step_1"
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['exog'],
        ['exog'],
        ['l1_roll_mean_5', 'l1_roll_median_5', 'l1_roll_sum_6',
         'l2_roll_mean_5', 'l2_roll_median_5', 'l2_roll_sum_6',
         'exog_step_1'],
        {'exog': exog.dtype}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['exog_step_1']
    forecaster.window_features_names == ['roll_mean_5', 'roll_median_5', 'roll_sum_6']
    forecaster.X_train_window_features_names_out_ == [
        'l1_roll_mean_5', 'l1_roll_median_5', 'l1_roll_sum_6',
        'l2_roll_mean_5', 'l2_roll_median_5', 'l2_roll_sum_6'
    ]

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]


def test_create_train_X_y_output_when_window_features_and_exog_transformers_diff():
    """
    Test the output of _create_train_X_y when using window_features, exog, 
    transformers and differentiation with steps=1.
    """

    series = pd.DataFrame(
        {'l1': [25.3, 29.1, 27.5, 24.3, 2.1, 46.5, 31.3, 87.1, 133.5, 4.3],
         'l2': [25.3, 29.1, 27.5, 24.3, 2.1, 46.5, 31.3, 87.1, 133.5, 4.3]},
        index = pd.date_range('2000-01-01', periods=10, freq='D')
    )
    exog = pd.DataFrame({
        'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4, 14.6, 73.5],
        'col_2': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']},
        index = pd.date_range('2000-01-01', periods=10, freq='D')
    )

    transformer_series = StandardScaler()
    transformer_exog = ColumnTransformer(
                            [('scale', StandardScaler(), ['col_1']),
                             ('onehot', OneHotEncoder(), ['col_2'])],
                            remainder = 'passthrough',
                            verbose_feature_names_out = False
                        )
    rolling = RollingFeatures(
        stats=['ratio_min_max', 'median'], window_sizes=4
    )

    forecaster = ForecasterDirectMultiVariate(
                     LinearRegression(), 
                     level              = 'l1',
                     steps              = 1,
                     lags               = [1, 5], 
                     window_features    = rolling,
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog,
                     differentiation    = 2
                 )
    results = forecaster._create_train_X_y(series=series, exog=exog)
    
    expected = (
        pd.DataFrame(
            data = np.array([[-1.56436158, -0.14173746, -0.89489489, -0.27035108,  
                              -1.56436158, -0.14173746, -0.89489489, -0.27035108,
                               0.04040264,  0.        ,  1.        ],
                             [ 1.8635851 , -0.04199628, -0.83943662,  0.62469472, 
                               1.8635851 , -0.04199628, -0.83943662,  0.62469472, 
                              -1.32578962,  0.        ,  1.        ],
                             [-0.24672817, -0.49870587, -0.83943662,  0.75068358,  
                              -0.24672817, -0.49870587, -0.83943662,  0.75068358,
                               1.12752513,  0.        ,  1.        ]]),
            index   = pd.date_range('2000-01-08', periods=3, freq='D'),
            columns = ['l1_lag_1', 'l1_lag_5', 'l1_roll_ratio_min_max_4', 'l1_roll_median_4',
                       'l2_lag_1', 'l2_lag_5', 'l2_roll_ratio_min_max_4', 'l2_roll_median_4',
                       'col_1_step_1', 'col_2_a_step_1', 'col_2_b_step_1']
        ),
        {1: pd.Series(
                data  = np.array([1.8635851, -0.24672817, -4.60909217]),
                index = pd.date_range('2000-01-08', periods=3, freq='D'),
                name  = 'l1_step_1',
                dtype = float
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['col_1', 'col_2'],
        ['col_1', 'col_2_a', 'col_2_b'],
        ['l1_lag_1', 'l1_lag_5', 'l1_roll_ratio_min_max_4', 'l1_roll_median_4',
         'l2_lag_1', 'l2_lag_5', 'l2_roll_ratio_min_max_4', 'l2_roll_median_4',
         'col_1_step_1', 'col_2_a_step_1', 'col_2_b_step_1'],
        {'col_1': exog['col_1'].dtypes, 'col_2': exog['col_2'].dtypes}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == ['col_1_step_1', 'col_2_a_step_1', 'col_2_b_step_1']
    forecaster.window_features_names == ['roll_ratio_min_max_4', 'roll_median_4']
    forecaster.X_train_window_features_names_out_ == [
        'l1_roll_ratio_min_max_4', 'l1_roll_median_4', 
        'l2_roll_ratio_min_max_4', 'l2_roll_median_4'
    ]

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]


def test_create_train_X_y_output_when_window_features_and_exog_transformers_diff_steps_2():
    """
    Test the output of _create_train_X_y when using window_features, exog, 
    transformers and differentiation with steps=2.
    """

    series = pd.DataFrame(
        {'l1': [25.3, 29.1, 27.5, 24.3, 2.1, 46.5, 31.3, 87.1, 133.5, 4.3],
         'l2': [25.3, 29.1, 27.5, 24.3, 2.1, 46.5, 31.3, 87.1, 133.5, 4.3]},
        index = pd.date_range('2000-01-01', periods=10, freq='D')
    )
    exog = pd.DataFrame({
        'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4, 14.6, 73.5],
        'col_2': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']},
        index = pd.date_range('2000-01-01', periods=10, freq='D')
    )

    transformer_series = StandardScaler()
    transformer_exog = ColumnTransformer(
                            [('scale', StandardScaler(), ['col_1']),
                             ('onehot', OneHotEncoder(), ['col_2'])],
                            remainder = 'passthrough',
                            verbose_feature_names_out = False
                        )
    rolling = RollingFeatures(
        stats=['ratio_min_max', 'median'], window_sizes=4
    )

    forecaster = ForecasterDirectMultiVariate(
                     LinearRegression(), 
                     level              = 'l1',
                     steps              = 2,
                     lags               = [1, 5], 
                     window_features    = rolling,
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog,
                     differentiation    = 1
                 )
    results = forecaster._create_train_X_y(series=series, exog=exog)
    
    expected = (
        pd.DataFrame(
            data = np.array([[ 1.16539688,  0.09974117, -0.5       , -0.06299443,  
                               1.16539688,  0.09974117, -0.5       , -0.06299443,
                               1.69816032,  0.        ,  1.        ,  0.04040264,  0.        ,  1.        ],
                             [-0.3989647 , -0.04199628, -0.5       , -0.24147863, 
                              -0.3989647 , -0.04199628, -0.5       , -0.24147863, 
                               0.04040264,  0.        ,  1.        , -1.32578962,  0.        ,  1.        ],
                             [ 1.46462041, -0.08399257, -0.39784946,  0.38321609,  
                               1.46462041, -0.08399257, -0.39784946,  0.38321609,
                              -1.32578962,  0.        ,  1.        ,  1.12752513,  0.        ,  1.        ]]),
            index   = pd.date_range('2000-01-08', periods=3, freq='D'),
            columns = ['l1_lag_1', 'l1_lag_5', 'l1_roll_ratio_min_max_4', 'l1_roll_median_4',
                       'l2_lag_1', 'l2_lag_5', 'l2_roll_ratio_min_max_4', 'l2_roll_median_4',
                       'col_1_step_1', 'col_2_a_step_1', 'col_2_b_step_1', 
                       'col_1_step_2', 'col_2_a_step_2', 'col_2_b_step_2']
        ),
        {1: pd.Series(
                data  = np.array([-0.3989647, 1.46462041, 1.21789224]),
                index = pd.date_range('2000-01-07', periods=3, freq='D'),
                name  = 'l1_step_1',
                dtype = float
            ),
         2: pd.Series(
                data  = np.array([1.46462041, 1.21789224, -3.39119993]),
                index = pd.date_range('2000-01-08', periods=3, freq='D'),
                name  = 'l1_step_2',
                dtype = float
            )
        },
        ['l1', 'l2'],
        ['l1', 'l2'],
        ['col_1', 'col_2'],
        ['col_1', 'col_2_a', 'col_2_b'],
        ['l1_lag_1', 'l1_lag_5', 'l1_roll_ratio_min_max_4', 'l1_roll_median_4',
         'l2_lag_1', 'l2_lag_5', 'l2_roll_ratio_min_max_4', 'l2_roll_median_4',
         'col_1_step_1', 'col_2_a_step_1', 'col_2_b_step_1', 
         'col_1_step_2', 'col_2_a_step_2', 'col_2_b_step_2'],
        {'col_1': exog['col_1'].dtypes, 'col_2': exog['col_2'].dtypes}
    )

    forecaster.exog_in_ is True
    forecaster.X_train_direct_exog_names_out_ == [
        'col_1_step_1', 'col_2_a_step_1', 'col_2_b_step_1', 
        'col_1_step_2', 'col_2_a_step_2', 'col_2_b_step_2'
    ]
    forecaster.window_features_names == ['roll_ratio_min_max_4', 'roll_median_4']
    forecaster.X_train_window_features_names_out_ == [
        'l1_roll_ratio_min_max_4', 'l1_roll_median_4', 
        'l2_roll_ratio_min_max_4', 'l2_roll_median_4'
    ]

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert isinstance(results[1], dict)
    assert all(isinstance(x, pd.Series) for x in results[1].values())
    assert results[1].keys() == expected[1].keys()
    for key in expected[1]: 
        pd.testing.assert_series_equal(results[1][key], expected[1][key]) 
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
