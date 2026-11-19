# Unit test _create_predict_inputs ForecasterRecursiveMultiSeries
# ==============================================================================
import re
import pytest
import warnings
import numpy as np
import pandas as pd
from pandas.errors import InvalidIndexError
from sklearn.exceptions import NotFittedError
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor

from ....preprocessing import TimeSeriesDifferentiator, CalendarFeatures
from ....recursive import ForecasterRecursiveMultiSeries

# Fixtures
from .fixtures_forecaster_recursive_multiseries import (
    series_wide_range,
    series_wide_dt,
    series_long_dt,
    series_dict_range,
    series_dict_dt,
    exog_wide_range,
    exog_long_dt,
    exog_pred_wide_range,
    exog_pred_long_dt,
    series_dict_nans_train,
    exog_dict_nans_train,
    exog_dict_nans_test
)


def test_create_predict_inputs_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5)

    err_msg = re.escape(
        "This Forecaster instance is not fitted yet. Call `fit` with "
        "appropriate arguments before using predict."
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster._create_predict_inputs(steps=5)


def test_create_predict_inputs_ValueError_when_last_window_MultiIndex_is_used():
    """
    Test ValueError is raised when last_window is a MultiIndex DataFrame.
    """

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5)
    forecaster.fit(series=series_long_dt, exog=exog_long_dt)

    idx = pd.MultiIndex.from_product(
        [forecaster.series_names_in_, forecaster.last_window_["1"].index],
        names=["series_id", "datetime"],
    )
    last_window_long = pd.DataFrame(
        {
            "values": np.concatenate(
                [
                    forecaster.last_window_["1"].to_numpy(),
                    forecaster.last_window_["2"].to_numpy(),
                ]
            )
        },
        index=idx,
    )

    err_msg = re.escape(
        "`last_window` must be a pandas DataFrame with one column "
        "per series and a single-level index (MultiIndex is not "
        "supported)."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_predict_inputs(steps=5, last_window=last_window_long)


def test_create_predict_inputs_TypeError_when_exog_MultiIndex_no_DatetimeIndex():
    """
    Test NotFittedError is raised when fitted is False.
    """
    exog_long_no_index = exog_long_dt.copy()
    exog_long_no_index = exog_long_no_index.reset_index(level=1, drop=True)
    exog_long_no_index['idx'] = pd.RangeIndex(
        start=0, stop=len(exog_long_no_index), step=1
    )
    exog_long_no_index = exog_long_no_index.set_index("idx", append=True)

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5)
    forecaster.fit(series=series_long_dt, exog=exog_long_dt)

    err_msg = re.escape(
        f"When `exog` is a pandas MultiIndex DataFrame, its index "
        f"must be a pandas DatetimeIndex. If you want to use a pandas "
        f"RangeIndex, use a dictionary instead. Found `exog` index "
        f"type: {type(exog_long_no_index.index.levels[1])}."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster._create_predict_inputs(steps=5, exog=exog_long_no_index)


@pytest.mark.parametrize(
    "check_inputs", [True, False], ids=lambda ci: f"check_inputs: {ci}"
)
@pytest.mark.parametrize(
    "levels", [None, ['1', '2']], ids=lambda lv: f"levels: {lv}"
)
def test_create_predict_inputs_InvalidIndexError_when_last_window_has_duplicated_columns(
    levels, check_inputs
):
    """
    Test the pandas InvalidIndexError is raised when `last_window` has
    duplicated column names, also when `levels` is taken from its columns
    (`levels=None`) and without checking the inputs (as in backtesting).
    """
    series = {
        '1': pd.Series(np.arange(start=0, stop=50, dtype=float)),
        '2': pd.Series(np.arange(start=50, stop=100, dtype=float))
    }
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5)
    forecaster.fit(series=series)
    last_window = pd.DataFrame(
        np.arange(15, dtype=float).reshape(5, 3),
        columns = ['1', '2', '1'],
        index   = pd.RangeIndex(start=45, stop=50, step=1)
    )

    err_msg = re.escape("Reindexing only valid with uniquely valued Index objects")
    with pytest.raises(InvalidIndexError, match = err_msg):
        forecaster._create_predict_inputs(
            steps=5, levels=levels, last_window=last_window, check_inputs=check_inputs
        )


def test_output_create_predict_inputs_when_estimator_is_LinearRegression():
    """
    Test output _create_predict_inputs when using LinearRegression as estimator.
    """
    series = {
        '1': pd.Series(np.arange(start=0, stop=50, dtype=float)), 
        '2': pd.Series(np.arange(start=50, stop=100, dtype=float))
    }
    
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, transformer_series=None
    )
    forecaster.fit(series=series)
    results = forecaster._create_predict_inputs(steps=5)

    expected = (
        pd.DataFrame(
            {'1': np.array([45., 46., 47., 48., 49.]),
             '2': np.array([95., 96., 97., 98., 99.])},
            index = pd.RangeIndex(start=45, stop=50, step=1)
        ),
        None,
        None,
        ['1', '2'],
        pd.RangeIndex(start=50, stop=55, step=1)
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    pd.testing.assert_index_equal(results[4], expected[4])
    assert results[5] == {}


def test_create_predict_inputs_output_when_with_transform_series():
    """
    Test _create_predict_inputs output when using LinearRegression as estimator 
    and StandardScaler.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = 5,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series_long_dt)
    results = forecaster._create_predict_inputs(steps=5, levels='1')
    
    expected = (
        pd.DataFrame(
            {'1': np.array([
                      -1.0814183452563133, -0.08097053361357116, 2.0806640273868244, 
                       0.07582436346198025, 0.4776288428854555])},
            index = pd.date_range(start='2000-02-15', periods=5, freq='D')
        ),
        None,
        None,
        ['1'],
        pd.date_range(start='2000-02-20', periods=5, freq='D')
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    pd.testing.assert_index_equal(results[4], expected[4])
    assert results[5] == {}


@pytest.mark.parametrize(
    "levels, n_rows, expected_last_window",
    [
        (None, 5, {'1': [1000., 1001., 1002., 1003., 1004.],
                   '2': [2000., 2001., 2002., 2003., 2004.],
                   '3': [3000., 3001., 3002., 3003., 3004.]}),
        (None, 8, {'1': [1003., 1004., 1005., 1006., 1007.],
                   '2': [2003., 2004., 2005., 2006., 2007.],
                   '3': [3003., 3004., 3005., 3006., 3007.]}),
        (['1', '2', '3'], 5, {'1': [1000., 1001., 1002., 1003., 1004.],
                              '2': [2000., 2001., 2002., 2003., 2004.],
                              '3': [3000., 3001., 3002., 3003., 3004.]}),
        (['3', '2', '1'], 8, {'3': [3003., 3004., 3005., 3006., 3007.],
                              '2': [2003., 2004., 2005., 2006., 2007.],
                              '1': [1003., 1004., 1005., 1006., 1007.]}),
        (['3', '1'], 5, {'3': [3000., 3001., 3002., 3003., 3004.],
                         '1': [1000., 1001., 1002., 1003., 1004.]}),
        (None, None, {'1': [45., 46., 47., 48., 49.],
                      '2': [145., 146., 147., 148., 149.],
                      '3': [245., 246., 247., 248., 249.]}),
        (['3', '2', '1'], None, {'3': [245., 246., 247., 248., 249.],
                                 '2': [145., 146., 147., 148., 149.],
                                 '1': [45., 46., 47., 48., 49.]}),
    ],
    ids=["levels_None", "levels_None_more_rows", "levels_columns_order",
         "levels_other_order_more_rows", "levels_subset",
         "stored_levels_None", "stored_levels_other_order"]
)
def test_create_predict_inputs_output_last_window_selection(
    levels, n_rows, expected_last_window
):
    """
    Test the last window returned by _create_predict_inputs has the last
    `window_size` rows of `last_window` and the columns of `levels`, in their
    order, when `levels` are its columns (in the same order or taken from them)
    and when they are a subset or in another order. Also with the last window
    stored in `fit` (`n_rows=None`), whose columns follow the order of the
    series in `fit`, not the one of `levels`. The returned last window does not
    share memory with `last_window`, which is not modified.
    """
    series = {
        '1': pd.Series(np.arange(start=0, stop=50, dtype=float)),
        '2': pd.Series(np.arange(start=100, stop=150, dtype=float)),
        '3': pd.Series(np.arange(start=200, stop=250, dtype=float))
    }
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=5, transformer_series=None
    )
    forecaster.fit(series=series)
    last_window = None
    if n_rows is not None:
        last_window = pd.DataFrame(
            {'1': 1000. + np.arange(n_rows),
             '2': 2000. + np.arange(n_rows),
             '3': 3000. + np.arange(n_rows)},
            index = pd.RangeIndex(start=50 - n_rows, stop=50, step=1)
        )
        last_window_copy = last_window.copy()

    results = forecaster._create_predict_inputs(
        steps=3, levels=levels, last_window=last_window
    )

    expected = (
        pd.DataFrame(
            expected_last_window,
            index = pd.RangeIndex(start=45, stop=50, step=1)
        ),
        None,
        None,
        list(expected_last_window.keys()),
        pd.RangeIndex(start=50, stop=53, step=1)
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert results[1] == expected[1]
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    pd.testing.assert_index_equal(results[4], expected[4])
    assert results[5] == {}
    if last_window is not None:
        pd.testing.assert_frame_equal(last_window, last_window_copy)
        assert not np.shares_memory(results[0].to_numpy(), last_window.to_numpy())


@pytest.mark.parametrize("transformer_series",
                         [StandardScaler(),
                          {'l1': StandardScaler(), 'l2': StandardScaler(), '_unknown_level': StandardScaler()}], 
                         ids = lambda tr: f'transformer_series: {type(tr)}')
def test_create_predict_inputs_when_transform_series_and_transform_exog(transformer_series):
    """
    Test _create_predict_inputs output when using LinearRegression as estimator, 
    StandardScaler as transformer_series and transformer_exog as transformer_exog.
    """
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = 5,
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series_dict_dt, exog=exog_long_dt)
    results = forecaster._create_predict_inputs(
        steps=5, levels='l1', exog=exog_pred_long_dt
    )
    
    expected = (
        pd.DataFrame(
            {'l1': np.array([
                      -1.0814183452563133, -0.08097053361357116, 2.0806640273868244, 
                       0.07582436346198025, 0.4776288428854555])},
            index = pd.date_range(start='2000-02-15', periods=5, freq='D')
        ),
        {1: np.array([[-0.09362908,  1., 0.]]),
         2: np.array([[0.45144522,  1., 0.]]),
         3: np.array([[-1.53968887,  1., 0.]]),
         4: np.array([[-1.45096055,  1., 0.]]),
         5: np.array([[-0.77240468,  1., 0.]])
        },
        None,
        ['l1'],
        pd.date_range(start='2000-02-20', periods=5, freq='D')
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    for k in results[1].keys():
        np.testing.assert_array_almost_equal(results[1][k], expected[1][k])
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    pd.testing.assert_index_equal(results[4], expected[4])
    assert results[5] == {}


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'l1': StandardScaler(), 'l2': StandardScaler(), '_unknown_level': StandardScaler()}], 
                         ids = lambda tr: f'transformer_series: {type(tr)}')
def test_create_predict_inputs_when_transform_series_and_transform_exog_different_length_series(transformer_series):
    """
    Test _create_predict_inputs output when using LinearRegression as estimator, StandardScaler
    as transformer_series and transformer_exog as transformer_exog with series 
    of different lengths.
    """
    series_dict_range_nan = {
        'l1': series_dict_range['l1'].copy(),
        'l2': series_dict_range['l2'].copy()
    }
    series_dict_range_nan['l2'].iloc[:10] = np.nan

    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = 5,
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series_dict_range_nan, exog=exog_wide_range)
    results = forecaster._create_predict_inputs(steps=5, exog=exog_pred_wide_range)
    
    expected = (
        pd.DataFrame(
            {'l1': np.array([-1.08141835, -0.08097053, 2.08066403, 0.07582436, 0.47762884]),
             'l2': np.array([0.73973379, -1.29936753, -0.36483376, -0.96090271, -0.57388468])},
            index = pd.RangeIndex(start=45, stop=50, step=1)
        ),
        {1: np.array([[-0.09575486,  1., 0.],
                      [-0.09575486,  1., 0.]]),
         2: np.array([[0.44024531,  1., 0.],
                      [0.44024531,  1., 0.]]),
         3: np.array([[-1.51774136,  1., 0.],
                      [-1.51774136,  1., 0.]]),
         4: np.array([[-1.43049014,  1., 0.],
                      [-1.43049014,  1., 0.]]),
         5: np.array([[-0.76323054,  1., 0.],
                      [-0.76323054,  1., 0.]])
        },
        None,
        ['l1', 'l2'],
        pd.RangeIndex(start=50, stop=55, step=1)
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    for k in results[1].keys():
        np.testing.assert_array_almost_equal(results[1][k], expected[1][k])
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    pd.testing.assert_index_equal(results[4], expected[4])
    assert results[5] == {}


def test_create_predict_inputs_output_when_categorical_features_native_implementation_HistGradientBoostingRegressor():
    """
    Test _create_predict_inputs output when using HistGradientBoostingRegressor and categorical variables.
    """
    df_exog = pd.DataFrame(
        {'exog_1': exog_wide_range['exog_1'],
         'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
         'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J'] * 10)}
    )
    
    exog_predict = df_exog.copy()
    exog_predict.index = pd.RangeIndex(start=50, stop=100)

    categorical_features = df_exog.select_dtypes(exclude=[np.number]).columns.tolist()
    transformer_exog = make_column_transformer(
                           (
                               OrdinalEncoder(
                                   dtype=int,
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1,
                                   encoded_missing_value=-1
                               ),
                               categorical_features
                           ),
                           remainder="passthrough",
                           verbose_feature_names_out=False,
                       ).set_output(transform="pandas")
    
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = HistGradientBoostingRegressor(
                                              categorical_features = categorical_features,
                                              random_state         = 123
                                          ),
                     lags               = 5,
                     transformer_series = None,
                     transformer_exog   = transformer_exog
                 )
    forecaster.fit(series=series_dict_range, exog=df_exog)
    results = forecaster._create_predict_inputs(steps=10, exog=exog_predict)
    
    expected = (
        pd.DataFrame(
            {'l1': np.array([0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453]),
             'l2': np.array([0.6917018 , 0.15112745, 0.39887629, 0.2408559 , 0.34345601])},
            index = pd.RangeIndex(start=45, stop=50, step=1)
        ),
        {1: np.array([[0, 0, 0.51312815],
                      [0, 0, 0.51312815]]),
         2: np.array([[1, 1, 0.66662455],
                      [1, 1, 0.66662455]]),
         3: np.array([[2, 2, 0.10590849],
                      [2, 2, 0.10590849]]),
         4: np.array([[3, 3, 0.13089495],
                      [3, 3, 0.13089495]]),
         5: np.array([[4, 4, 0.32198061],
                      [4, 4, 0.32198061]]),
         6: np.array([[0, 0, 0.66156434],
                      [0, 0, 0.66156434]]),
         7: np.array([[1, 1, 0.84650623],
                      [1, 1, 0.84650623]]),
         8: np.array([[2, 2, 0.55325734],
                      [2, 2, 0.55325734]]),
         9: np.array([[3, 3, 0.85445249],
                      [3, 3, 0.85445249]]),
         10: np.array([[4, 4, 0.38483781],
                       [4, 4, 0.38483781]])

        },
        None,
        ['l1', 'l2'],
        pd.RangeIndex(start=50, stop=60, step=1)
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    for k in results[1].keys():
        np.testing.assert_array_almost_equal(results[1][k], expected[1][k])
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    pd.testing.assert_index_equal(results[4], expected[4])
    assert results[5] == {}


@pytest.mark.parametrize(
    "categorical_features",
    ['auto', ['exog_2', 'exog_3']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_predict_inputs_when_categorical_features_auto_and_explicit_no_transformer_exog(
    categorical_features,
):
    """
    Test _create_predict_inputs when using internal categorical encoding
    (`categorical_features='auto'` and explicit list) without `transformer_exog`.
    This exercises the copy guard branch (`transformer_exog is None`).
    """
    series = {
        'l1': series_wide_range['1'].copy(),
        'l2': series_wide_range['2'].copy()
    }
    df_exog = pd.DataFrame({
        'exog_1': exog_wide_range['exog_1'],
        'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
        'exog_3': pd.Categorical(['F', 'F', 'G', 'G', 'H'] * 10)
    })

    exog_predict = df_exog.copy()
    exog_predict.index = pd.RangeIndex(start=50, stop=100)

    forecaster = ForecasterRecursiveMultiSeries(
                     estimator            = LinearRegression(),
                     lags                 = 5,
                     encoding             = 'ordinal',
                     transformer_series   = None,
                     transformer_exog     = None,
                     categorical_features = categorical_features
                 )
    forecaster.fit(series=series, exog=df_exog, suppress_warnings=True)
    results = forecaster._create_predict_inputs(steps=5, exog=exog_predict)

    expected = (
        pd.DataFrame(
            {'l1': np.array([0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453]),
             'l2': np.array([0.6917018 , 0.15112745, 0.39887629, 0.2408559 , 0.34345601])},
            index = pd.RangeIndex(start=45, stop=50, step=1)
        ),
        {1: np.array([[0.51312815, 0., 0.],
                      [0.51312815, 0., 0.]]),
         2: np.array([[0.66662455, 1., 0.],
                      [0.66662455, 1., 0.]]),
         3: np.array([[0.10590849, 2., 1.],
                      [0.10590849, 2., 1.]]),
         4: np.array([[0.13089495, 3., 1.],
                      [0.13089495, 3., 1.]]),
         5: np.array([[0.32198061, 4., 2.],
                      [0.32198061, 4., 2.]])
        },
        None,
        ['l1', 'l2'],
        pd.RangeIndex(start=50, stop=55, step=1)
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    for k in results[1].keys():
        np.testing.assert_array_almost_equal(results[1][k], expected[1][k])
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    pd.testing.assert_index_equal(results[4], expected[4])
    assert results[5] == {}


def test_create_predict_inputs_when_categorical_features_auto_with_transformer_exog():
    """
    Test _create_predict_inputs when using internal categorical encoding
    (`categorical_features='auto'`) together with `transformer_exog`
    (StandardScaler on numeric columns). This exercises the branch where
    copy is NOT needed because `transformer_exog` already returns a new
    DataFrame.
    """
    series = {
        'l1': series_wide_range['1'].copy(),
        'l2': series_wide_range['2'].copy()
    }
    df_exog = pd.DataFrame({
        'exog_1': exog_wide_range['exog_1'],
        'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
        'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J'] * 10)
    })

    exog_predict = df_exog.copy()
    exog_predict.index = pd.RangeIndex(start=50, stop=100)

    transformer_exog = make_column_transformer(
                           (StandardScaler(), make_column_selector(dtype_include=np.number)),
                           remainder='passthrough',
                           verbose_feature_names_out=False,
                       ).set_output(transform='pandas')

    forecaster = ForecasterRecursiveMultiSeries(
                     estimator            = LinearRegression(),
                     lags                 = 5,
                     encoding             = 'ordinal',
                     transformer_series   = None,
                     transformer_exog     = transformer_exog,
                     categorical_features = 'auto'
                 )
    forecaster.fit(series=series, exog=df_exog, suppress_warnings=True)
    results = forecaster._create_predict_inputs(steps=5, exog=exog_predict)

    expected = (
        pd.DataFrame(
            {'l1': np.array([0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453]),
             'l2': np.array([0.6917018 , 0.15112745, 0.39887629, 0.2408559 , 0.34345601])},
            index = pd.RangeIndex(start=45, stop=50, step=1)
        ),
        {1: np.array([[-0.09362908, 0., 0.],
                      [-0.09362908, 0., 0.]]),
         2: np.array([[0.45144522, 1., 1.],
                      [0.45144522, 1., 1.]]),
         3: np.array([[-1.53968887, 2., 2.],
                      [-1.53968887, 2., 2.]]),
         4: np.array([[-1.45096055, 3., 3.],
                      [-1.45096055, 3., 3.]]),
         5: np.array([[-0.77240468, 4., 4.],
                      [-0.77240468, 4., 4.]])
        },
        None,
        ['l1', 'l2'],
        pd.RangeIndex(start=50, stop=55, step=1)
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    for k in results[1].keys():
        np.testing.assert_array_almost_equal(results[1][k], expected[1][k])
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    pd.testing.assert_index_equal(results[4], expected[4])
    assert results[5] == {}


def test_create_predict_inputs_output_when_series_and_exog_dict():
    """
    Test output ForecasterRecursiveMultiSeries _create_predict_inputs method when 
    series and exog are dictionaries.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        estimator          = LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags               = 5,
        encoding           = 'ordinal',
        dropna_from_series = False,
        transformer_series = StandardScaler(),
        transformer_exog   = StandardScaler(),
    )
    forecaster.fit(
        series=series_dict_nans_train, exog=exog_dict_nans_train, suppress_warnings=True
    )
    results = forecaster._create_predict_inputs(steps=5, exog=exog_dict_nans_test)
    
    expected = (
        pd.DataFrame(
            {'id_1000': np.array([-0.3525861, -0.457091, -0.49618465, -1.07810218, -1.77580056]),
             'id_1001': np.array([0.21800529, 0.36936645, 0.67896814, 1.16332175, 1.1523137]),
             'id_1003': np.array([-0.62651976, -0.74685959, -1.03823091, -0.37837738, 3.39980134]),
             'id_1004': np.array([0.61495753, 0.8322971, 0.6719899, -0.24309812, -0.75073762])},
            index = pd.date_range(start='2016-07-27', periods=5, freq='D', unit='ns')
        ),
        {1: np.array([[0.00821644, 1.42962482, np.nan, np.nan],
                      [0.00821644, 1.42962482, 1.11141113, -0.87943526],
                      [0.00821644, np.nan, 1.11141113, -0.87943526],
                      [0.00821644, 1.42962482, 1.11141113, -0.87943526]]),
         2: np.array([[1.11220226, 0.89634375, np.nan, np.nan],
                      [1.11220226, 0.89634375, 1.1327558 , 0.0058948 ],
                      [1.11220226, np.nan, 1.1327558 , 0.0058948 ],
                      [1.11220226, 0.89634375, 1.1327558 , 0.0058948 ]]),
         3: np.array([[1.38486425, -0.30192795, np.nan, np.nan],
                      [1.38486425, -0.30192795, 1.1775869 , -0.3532584 ],
                      [1.38486425, np.nan, 1.1775869 , -0.3532584 ],
                      [1.38486425, -0.30192795, 1.1775869 , -0.3532584 ]]),
         4: np.array([[0.62088235, -1.26286725, np.nan, np.nan],
                      [0.62088235, -1.26286725, 1.0428337 , 0.84287284],
                      [0.62088235, np.nan, 1.0428337 , 0.84287284],
                      [0.62088235, -1.26286725, 1.0428337 , 0.84287284]]),
         5: np.array([[-0.60444947, -1.26286725, np.nan, np.nan],
                      [-0.60444947, -1.26286725, 1.00599776, -0.62314633],
                      [-0.60444947, np.nan, 1.00599776, -0.62314633],
                      [-0.60444947, -1.26286725, 1.00599776, -0.62314633]])
        },
        None,
        ['id_1000', 'id_1001', 'id_1003', 'id_1004'],
        pd.date_range(start='2016-08-01', periods=5, freq='D', unit='ns')
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    for k in results[1].keys():
        np.testing.assert_array_almost_equal(results[1][k], expected[1][k])
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    pd.testing.assert_index_equal(results[4], expected[4])
    assert results[5] == {}


@pytest.mark.parametrize("differentiation", 
                         [1, {'1': 1, '2': 1, '_unknown_level': 1}], 
                         ids = lambda diff: f'differentiation: {diff}')
def test_create_predict_inputs_when_exog_differentiation_1_and_transformer_series(differentiation):
    """
    Test _create_predict_inputs output when using LinearRegression as estimator and differentiation=1,
    and transformer_series is StandardScaler.
    """
    end_train = '2003-01-30 23:59:00'

    # Data scaled and differentiated
    series_datetime = series_wide_dt.copy()
    series_datetime.index = pd.date_range(
        start='2003-01-01', periods=len(series_wide_dt), freq='D'
    )
    series_dict_datetime = {
        "1": series_datetime['1'].loc[:end_train],
        "2": series_datetime['2'].loc[:end_train]
    }
    
    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(series_wide_dt)), name='exog'
    )
    exog.index = pd.date_range(start='2003-01-01', periods=len(series_wide_dt), freq='D')
    exog_dict_datetime = {
        "1": exog.loc[:end_train],
        "2": exog.loc[:end_train]
    }
    exog_pred = {
        '1': exog.loc[end_train:],
        '2': exog.loc[end_train:]
    }

    steps = len(series_datetime.loc[end_train:])

    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(), 
                     lags               = 15, 
                     transformer_series = StandardScaler(),    
                     differentiation    = differentiation
                 )
    forecaster.fit(
        series=series_dict_datetime, exog=exog_dict_datetime, store_in_sample_residuals=True
    )
    results = forecaster._create_predict_inputs(
        steps=steps, exog=exog_pred, predict_probabilistic=True
    )
    
    expected = (
        pd.DataFrame(
            {'1': np.array([
                      np.nan, 1.53099926e+00, -2.50175863e+00, -3.17051107e-02,
                      1.60372524e+00, 1.24398054e-03, 4.61947999e-01, 9.68409847e-01,
                      -5.62842287e-01, -5.10849917e-01, 5.01788973e-01, -1.79911269e+00,
                      1.74873121e-01, -6.01343264e-01, 2.94763400e-01, 1.51888875e+00]
                  ),
             '2': np.array([
                      np.nan, -2.93045496, 2.6268138, -2.00736345, -0.19086106,
                      1.46005589, -1.8403251, 3.04806289, -0.99623011, 0.3712246,
                      -2.73026424, 2.23235938, -0.14532345, -1.53568303, -0.02273313,
                      2.09399597]
                  )},
            index = pd.date_range(start='2003-01-15', periods=16, freq='D')
        ),
        {1: np.array([[-0.78916779],
                      [-0.78916779]]),
         2: np.array([[0.07769117],
                      [0.07769117]]),
         3: np.array([[0.27283009],
                      [0.27283009]]),
         4: np.array([[0.35075195],
                      [0.35075195]]),
         5: np.array([[0.31597336],
                      [0.31597336]]),
         6: np.array([[0.14091339],
                      [0.14091339]]),
         7: np.array([[1.28618144],
                      [1.28618144]]),
         8: np.array([[0.09164865],
                      [0.09164865]]),
         9: np.array([[-0.50744682],
                      [-0.50744682]]),
         10: np.array([[0.01522573],
                       [0.01522573]]),
         11: np.array([[0.82813767],
                       [0.82813767]]),
         12: np.array([[0.15290495],
                       [0.15290495]]),
         13: np.array([[0.61532804],
                       [0.61532804]]),
         14: np.array([[-0.06287136],
                       [-0.06287136]]),
         15: np.array([[-1.1896156],
                       [-1.1896156]]),
         16: np.array([[0.28674823],
                       [0.28674823]]),
         17: np.array([[-0.64581528],
                       [-0.64581528]]),
         18: np.array([[0.20879998],
                       [0.20879998]]),
         19: np.array([[1.80029302],
                       [1.80029302]]),
         20: np.array([[0.14269745],
                       [0.14269745]])
        },
        None,
        ['1', '2'],
        pd.date_range(start='2003-01-31', periods=steps, freq='D')
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    for k in results[1].keys():
        np.testing.assert_array_almost_equal(results[1][k], expected[1][k])
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    pd.testing.assert_index_equal(results[4], expected[4])
    assert set(results[5].keys()) == {'1', '2'}
    for level in results[5]:
        assert isinstance(results[5][level], TimeSeriesDifferentiator)


@pytest.mark.parametrize("levels", 
                         [None, ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005']], 
                         ids = lambda levels: f'levels: {levels}')
def test_create_predict_inputs_when_series_and_exog_dict_unknown_level_calendar_features(levels):
    """
    Test output ForecasterRecursiveMultiSeries _create_predict_inputs method when 
    series and exog are dictionaries and unknown level.
    """

    calendar = CalendarFeatures(
        features=['day_of_week', 'weekend'], encoding="cyclical"
    )

    forecaster = ForecasterRecursiveMultiSeries(
        estimator          = LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags               = 5,
        calendar_features  = calendar,
        encoding           = 'ordinal',
        dropna_from_series = False,
        transformer_series = StandardScaler(),
        transformer_exog   = StandardScaler(),
    )
    forecaster.fit(
        series=series_dict_nans_train, exog=exog_dict_nans_train, suppress_warnings=True
    )

    series_to_predict = ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005']
    last_window = pd.DataFrame(
        {k: v for k, v in forecaster.last_window_.items() if k in series_to_predict}
    )
    last_window['id_1005'] = last_window['id_1004']
    exog_dict_nans_test_2 = exog_dict_nans_test.copy()
    exog_dict_nans_test_2['id_1005'] = exog_dict_nans_test_2['id_1004']
    results = forecaster._create_predict_inputs(
        steps=5, levels=levels, last_window=last_window, exog=exog_dict_nans_test_2
    )
    
    expected = (
        pd.DataFrame(
            {'id_1000': np.array([-0.3525861, -0.457091, -0.49618465, -1.07810218, -1.77580056]),
             'id_1001': np.array([0.21800529, 0.36936645, 0.67896814, 1.16332175, 1.1523137]),
             'id_1003': np.array([-0.62651976, -0.74685959, -1.03823091, -0.37837738, 3.39980134]),
             'id_1004': np.array([0.61495753, 0.8322971, 0.6719899, -0.24309812, -0.75073762]),
             'id_1005': np.array([2.38367241, 2.56743042, 2.4318926, 1.65819658, 1.22899343])},
            index = pd.date_range(start='2016-07-27', periods=5, freq='D', unit='ns')
        ),
        {1: np.array([[0.00821644, 1.42962482, np.nan, np.nan],
                      [0.00821644, 1.42962482, 1.11141113, -0.87943526],
                      [0.00821644, np.nan, 1.11141113, -0.87943526],
                      [0.00821644, 1.42962482, 1.11141113, -0.87943526],
                      [0.00821644, 1.42962482, 1.11141113, -0.87943526]]),
         2: np.array([[1.11220226, 0.89634375, np.nan, np.nan],
                      [1.11220226, 0.89634375, 1.1327558 , 0.0058948 ],
                      [1.11220226, np.nan, 1.1327558 , 0.0058948 ],
                      [1.11220226, 0.89634375, 1.1327558 , 0.0058948 ],
                      [1.11220226, 0.89634375, 1.1327558 , 0.0058948 ]]),
         3: np.array([[1.38486425, -0.30192795, np.nan, np.nan],
                      [1.38486425, -0.30192795, 1.1775869 , -0.3532584 ],
                      [1.38486425, np.nan, 1.1775869 , -0.3532584 ],
                      [1.38486425, -0.30192795, 1.1775869 , -0.3532584 ],
                      [1.38486425, -0.30192795, 1.1775869 , -0.3532584 ]]),
         4: np.array([[0.62088235, -1.26286725, np.nan, np.nan],
                      [0.62088235, -1.26286725, 1.0428337 , 0.84287284],
                      [0.62088235, np.nan, 1.0428337 , 0.84287284],
                      [0.62088235, -1.26286725, 1.0428337 , 0.84287284],
                      [0.62088235, -1.26286725, 1.0428337 , 0.84287284]]),
         5: np.array([[-0.60444947, -1.26286725, np.nan, np.nan],
                      [-0.60444947, -1.26286725, 1.00599776, -0.62314633],
                      [-0.60444947, np.nan, 1.00599776, -0.62314633],
                      [-0.60444947, -1.26286725, 1.00599776, -0.62314633],
                      [-0.60444947, -1.26286725, 1.00599776, -0.62314633]])
        },
        np.array([[ 0.        ,  0.        ,  1.        ],
                  [ 0.        ,  0.78183148,  0.6234898 ],
                  [ 0.        ,  0.97492791, -0.22252093],
                  [ 0.        ,  0.43388374, -0.90096887],
                  [ 0.        , -0.43388374, -0.90096887]]),
        ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005'],
        pd.date_range(start='2016-08-01', periods=5, freq='D', unit='ns')
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    for k in results[1].keys():
        np.testing.assert_array_almost_equal(results[1][k], expected[1][k])
    np.testing.assert_array_almost_equal(results[2], expected[2])
    assert results[3] == expected[3]
    pd.testing.assert_index_equal(results[4], expected[4])
    assert results[5] == {}


@pytest.mark.parametrize("levels", 
                         [None, ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005']], 
                         ids = lambda levels: f'levels: {levels}')
def test_create_predict_inputs_when_series_and_exog_dict_unknown_level_encoding_None(levels):
    """
    Test output ForecasterRecursiveMultiSeries _create_predict_inputs method when 
    series and exog are dictionaries and unknown level with encoding=None.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        estimator          = LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags               = 5,
        encoding           = None,
        dropna_from_series = False,
        transformer_series = StandardScaler(),
        transformer_exog   = StandardScaler(),
    )
    forecaster.fit(
        series=series_dict_nans_train, exog=exog_dict_nans_train, suppress_warnings=True
    )

    series_to_predict = ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005']
    last_window = pd.DataFrame(
        {k: v for k, v in forecaster.last_window_.items() if k in series_to_predict}
    )
    last_window['id_1005'] = last_window['id_1004']
    results = forecaster._create_predict_inputs(
        steps=5, levels=levels, last_window=last_window, exog=exog_dict_nans_test
    )
    
    expected = (
        pd.DataFrame(
            {'id_1000': np.array([-0.77768307, -0.79277912, -0.79842631, -0.88248605, -0.98327068]),
             'id_1001': np.array([-0.39572285, -0.34793695, -0.25019327, -0.09727905, -0.10075438]),
             'id_1003': np.array([-0.56650418, -0.59023234, -0.64768383, -0.51757641,  0.22739054]),
             'id_1004': np.array([2.38367241, 2.56743042, 2.4318926, 1.65819658, 1.22899343]),
             'id_1005': np.array([2.38367241, 2.56743042, 2.4318926, 1.65819658, 1.22899343])},
            index = pd.date_range(start='2016-07-27', periods=5, freq='D', unit='ns')
        ),
        {1: np.array([[0.00821644, 1.42962482, np.nan, np.nan],
                      [0.00821644, 1.42962482, 1.11141113, -0.87943526],
                      [0.00821644, np.nan, 1.11141113, -0.87943526],
                      [0.00821644, 1.42962482, 1.11141113, -0.87943526],
                      [np.nan, np.nan, np.nan, np.nan]]),
         2: np.array([[1.11220226, 0.89634375, np.nan, np.nan],
                      [1.11220226, 0.89634375, 1.1327558 , 0.0058948 ],
                      [1.11220226, np.nan, 1.1327558 , 0.0058948 ],
                      [1.11220226, 0.89634375, 1.1327558 , 0.0058948 ],
                      [np.nan, np.nan, np.nan, np.nan]]),
         3: np.array([[1.38486425, -0.30192795, np.nan, np.nan],
                      [1.38486425, -0.30192795, 1.1775869 , -0.3532584 ],
                      [1.38486425, np.nan, 1.1775869 , -0.3532584 ],
                      [1.38486425, -0.30192795, 1.1775869 , -0.3532584 ],
                      [np.nan, np.nan, np.nan, np.nan]]),
         4: np.array([[0.62088235, -1.26286725, np.nan, np.nan],
                      [0.62088235, -1.26286725, 1.0428337 , 0.84287284],
                      [0.62088235, np.nan, 1.0428337 , 0.84287284],
                      [0.62088235, -1.26286725, 1.0428337 , 0.84287284],
                      [np.nan, np.nan, np.nan, np.nan]]),
         5: np.array([[-0.60444947, -1.26286725, np.nan, np.nan],
                      [-0.60444947, -1.26286725, 1.00599776, -0.62314633],
                      [-0.60444947, np.nan, 1.00599776, -0.62314633],
                      [-0.60444947, -1.26286725, 1.00599776, -0.62314633],
                      [np.nan, np.nan, np.nan, np.nan]])
        },
        None,
        ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005'],
        pd.date_range(start='2016-08-01', periods=5, freq='D', unit='ns')
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    for k in results[1].keys():
        np.testing.assert_array_almost_equal(results[1][k], expected[1][k])
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    pd.testing.assert_index_equal(results[4], expected[4])
    assert results[5] == {}


@pytest.mark.parametrize("differentiation", 
                         [1, {'id_1000': 1, 'id_1001': 1, 'id_1003': 1, 'id_1004': 1, '_unknown_level': 1}], 
                         ids = lambda diff: f'differentiation: {diff}')
def test_create_predict_inputs_when_unknown_level_and_differentiation(differentiation):
    """
    Test output _create_predict_inputs when series and exog are dictionaries,
    unknown level is included, and differentiation is set. Verifies that
    results[4] (differentiators) contains entries for all levels including
    unknown ones (copied from '_unknown_level').\n    """
    forecaster = ForecasterRecursiveMultiSeries(
        estimator          = LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags               = 5,
        encoding           = 'ordinal',
        dropna_from_series = False,
        transformer_series = StandardScaler(),
        transformer_exog   = StandardScaler(),
        differentiation    = differentiation
    )
    forecaster.fit(
        series=series_dict_nans_train, exog=exog_dict_nans_train, suppress_warnings=True
    )

    levels = ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005']
    last_window = pd.DataFrame(
        {k: v for k, v in forecaster.last_window_.items() if k in levels}
    )
    last_window['id_1005'] = last_window['id_1004']
    exog_dict_nans_test_2 = exog_dict_nans_test.copy()
    exog_dict_nans_test_2['id_1005'] = exog_dict_nans_test_2['id_1004']
    results = forecaster._create_predict_inputs(
        steps=5, levels=levels, last_window=last_window, exog=exog_dict_nans_test_2
    )

    expected = (
        pd.DataFrame(
            {'id_1000': np.array([np.nan, -0.1939735 , -0.1045049 , -0.03909364, -0.58191753, -0.69769838]),
             'id_1001': np.array([np.nan,  0.08806471,  0.15136117,  0.30960169,  0.48435361, -0.01100806]),
             'id_1003': np.array([np.nan, -0.36050495, -0.12033983, -0.29137132,  0.65985353,  3.77817872]),
             'id_1004': np.array([np.nan,  0.05600424,  0.21733957, -0.16030719, -0.91508802, -0.5076395 ]),
             'id_1005': np.array([np.nan,  0.04735092,  0.18375801, -0.13553782, -0.77369602, -0.42920315])},
            index = pd.date_range(start='2016-07-26', periods=6, freq='D', unit='ns')
        ),
        {1: np.array([[ 0.01311204,  1.42216832,         np.nan,         np.nan],
                      [ 0.01311204,  1.42216832,  1.11184323, -0.90109483],
                      [ 0.01311204,         np.nan,  1.11184323, -0.90109483],
                      [ 0.01311204,  1.42216832,  1.11184323, -0.90109483],
                      [ 0.01311204,  1.42216832,  1.11184323, -0.90109483]]),
         2: np.array([[ 1.11949358,  0.89003132,         np.nan,         np.nan],
                      [ 1.11949358,  0.89003132,  1.13329737, -0.00734789],
                      [ 1.11949358,         np.nan,  1.13329737, -0.00734789],
                      [ 1.11949358,  0.89003132,  1.13329737, -0.00734789],
                      [ 1.11949358,  0.89003132,  1.13329737, -0.00734789]]),
         3: np.array([[ 1.39274726, -0.30566966,         np.nan,         np.nan],
                      [ 1.39274726, -0.30566966,  1.1783584 , -0.36991557],
                      [ 1.39274726,         np.nan,  1.1783584 , -0.36991557],
                      [ 1.39274726, -0.30566966,  1.1783584 , -0.36991557],
                      [ 1.39274726, -0.30566966,  1.1783584 , -0.36991557]]),
         4: np.array([[ 0.62710747, -1.2645474 ,         np.nan,         np.nan],
                      [ 0.62710747, -1.2645474 ,  1.04291408,  0.83758735],
                      [ 0.62710747,         np.nan,  1.04291408,  0.83758735],
                      [ 0.62710747, -1.2645474 ,  1.04291408,  0.83758735],
                      [ 0.62710747, -1.2645474 ,  1.04291408,  0.83758735]]),
         5: np.array([[-0.6008834 , -1.2645474 ,         np.nan,         np.nan],
                      [-0.6008834 , -1.2645474 ,  1.00588921, -0.64236935],
                      [-0.6008834 ,         np.nan,  1.00588921, -0.64236935],
                      [-0.6008834 , -1.2645474 ,  1.00588921, -0.64236935],
                      [-0.6008834 , -1.2645474 ,  1.00588921, -0.64236935]])
        },
        None,
        ['id_1000', 'id_1001', 'id_1003', 'id_1004', 'id_1005'],
        pd.date_range(start='2016-08-01', periods=5, freq='D', unit='ns')
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    for k in results[1].keys():
        np.testing.assert_array_almost_equal(results[1][k], expected[1][k])
    assert results[2] == expected[2]
    assert results[3] == expected[3]
    pd.testing.assert_index_equal(results[4], expected[4])
    assert set(results[5].keys()) == set(levels)
    for level in levels:
        assert isinstance(results[5][level], TimeSeriesDifferentiator)
    # Unknown level 'id_1005' should get a copy from '_unknown_level', not modify self.differentiator_
    assert 'id_1005' not in forecaster.differentiator_


@pytest.mark.parametrize(
    "extra_rows", [0, 3], ids=lambda n: f"extra_rows: {n}"
)
@pytest.mark.parametrize(
    "levels", [None, ['2', '1']], ids=lambda lv: f"levels: {lv}"
)
@pytest.mark.parametrize(
    "forecaster_kwargs",
    [
        {"estimator": LinearRegression(), "lags": 5},
        {"estimator": LinearRegression(), "lags": 5,
         "transformer_series": StandardScaler(copy=False)},
        {"estimator": LinearRegression(), "lags": 5,
         "transformer_series": StandardScaler(copy=False), "differentiation": 1},
    ],
    ids=["base", "in_place_transformer", "in_place_transformer_differentiation"]
)
def test_create_predict_inputs_does_not_modify_last_window(
    forecaster_kwargs, levels, extra_rows
):
    """
    Test _create_predict_inputs does not modify `last_window`, neither the one
    passed by the user nor the one stored in `fit`, also with in-place
    transformers. Its values are only read, not copied, when its columns are
    `levels`.
    """
    series = {
        '1': pd.Series(np.arange(start=0, stop=50, dtype=float)),
        '2': pd.Series(np.arange(start=100, stop=150, dtype=float))
    }
    forecaster = ForecasterRecursiveMultiSeries(**forecaster_kwargs)
    forecaster.fit(series=series)
    n_rows = forecaster.window_size + extra_rows
    last_window = pd.DataFrame(
        {'1': 1000. + np.arange(n_rows), '2': 2000. + np.arange(n_rows)},
        index = pd.RangeIndex(start=50 - n_rows, stop=50, step=1)
    )
    last_window_copy = last_window.copy()
    last_window_stored_copy = {
        k: v.copy() for k, v in forecaster.last_window_.items()
    }

    forecaster._create_predict_inputs(steps=3, levels=levels, last_window=last_window)
    forecaster._create_predict_inputs(steps=3, levels=levels)

    pd.testing.assert_frame_equal(last_window, last_window_copy)
    for k, v in forecaster.last_window_.items():
        pd.testing.assert_series_equal(v, last_window_stored_copy[k])


def _exog_dict_create_predict_inputs(kind):
    """
    Exog of series '1' and '2' for the 3 steps after the training data, as
    `kind`: aligned with the predictions (`identical` index and columns) or in
    a form that `_create_predict_inputs` aligns (`reindex_like`).
    """
    idx = pd.date_range(start='2020-02-20', periods=6, freq='D')
    exog = {
        '1': pd.DataFrame(
                 {'exog_1': 100. + np.arange(6), 'exog_2': 200. + np.arange(6)},
                 index=idx
             ),
        '2': pd.DataFrame(
                 {'exog_1': 300. + np.arange(6), 'exog_2': 400. + np.arange(6)},
                 index=idx
             ),
    }
    if kind != 'extra_rows':
        exog = {k: v.iloc[:3] for k, v in exog.items()}
    if kind == 'named_index':
        exog = {k: v.rename_axis('date') for k, v in exog.items()}
    if kind == 'index_without_freq':
        exog = {k: v.set_axis(pd.DatetimeIndex(list(v.index))) for k, v in exog.items()}
    if kind == 'reordered_columns':
        exog = {k: v[['exog_2', 'exog_1']] for k, v in exog.items()}
    if kind == 'object_columns':
        exog = {k: v.set_axis(pd.Index(list(v.columns), dtype=object), axis=1)
                for k, v in exog.items()}

    return exog


@pytest.mark.parametrize(
    "kind",
    ['aligned', 'named_index', 'index_without_freq', 'extra_rows',
     'reordered_columns', 'object_columns'],
    ids=lambda kind: f'exog: {kind}'
)
def test_create_predict_inputs_output_when_exog_dict_aligned_or_not(kind):
    """
    Test _create_predict_inputs gives the same exog values when the exog of
    each series is already aligned with the predictions (its index and columns
    are identical to the ones of the predictions and the training, so it is not
    reindexed) and when it is aligned with `reindex_like` (named index, index
    without frequency, more rows than steps, columns in another order or of
    another dtype).
    """
    idx = pd.date_range(start='2020-01-01', periods=50, freq='D')
    series = {
        '1': pd.Series(np.arange(50, dtype=float), index=idx),
        '2': pd.Series(100. + np.arange(50, dtype=float), index=idx),
    }
    exog_train = {
        k: pd.DataFrame(
               {'exog_1': np.arange(50, dtype=float),
                'exog_2': np.arange(50, dtype=float) ** 2},
               index=idx
           )
        for k in series
    }
    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(), lags=3, transformer_series=None
    )
    forecaster.fit(series=series, exog=exog_train, suppress_warnings=True)
    exog = _exog_dict_create_predict_inputs(kind)
    results = forecaster._create_predict_inputs(steps=3, exog=exog)

    expected_exog = {
        1: np.array([[100., 200.], [300., 400.]]),
        2: np.array([[101., 201.], [301., 401.]]),
        3: np.array([[102., 202.], [302., 402.]]),
    }
    expected_index = pd.date_range(start='2020-02-20', periods=3, freq='D')

    assert results[1].keys() == expected_exog.keys()
    for k in expected_exog:
        np.testing.assert_array_equal(results[1][k], expected_exog[k])
    assert results[3] == ['1', '2']
    pd.testing.assert_index_equal(results[4], expected_index)


@pytest.mark.parametrize(
    "levels", [['1'], None], ids=lambda lv: f"levels: {lv}"
)
def test_create_predict_inputs_does_not_modify_exog_dict_when_aligned(levels):
    """
    Test _create_predict_inputs does not modify the exog of each series when it
    is already aligned with the predictions, so it is used without a copy, also
    when the categorical features are encoded without `transformer_exog` and
    with a single level (`pd.concat` of one DataFrame shares its data).
    """
    idx = pd.date_range(start='2020-01-01', periods=53, freq='D')
    series = {
        '1': pd.Series(np.arange(50, dtype=float), index=idx[:50]),
        '2': pd.Series(100. + np.arange(50, dtype=float), index=idx[:50]),
    }
    exog_full = pd.DataFrame(
        {'exog_1': np.arange(53, dtype=float),
         'exog_2': pd.Categorical(np.arange(53) % 3)},
        index=idx
    )
    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(), lags=3, transformer_series=None,
        transformer_exog=None, categorical_features='auto'
    )
    forecaster.fit(
        series=series, exog={k: exog_full.iloc[:50] for k in series},
        suppress_warnings=True
    )
    exog = {k: exog_full.iloc[50:].copy() for k in series}
    exog_copy = {k: v.copy() for k, v in exog.items()}

    results = forecaster._create_predict_inputs(steps=3, levels=levels, exog=exog)

    expected_exog_1 = np.array([[50., 2.], [50., 2.]])
    if levels == ['1']:
        expected_exog_1 = expected_exog_1[:1]
    np.testing.assert_array_equal(results[1][1], expected_exog_1)
    for k in exog:
        pd.testing.assert_frame_equal(exog[k], exog_copy[k])


@pytest.mark.parametrize(
    "kind",
    ['aligned', 'named_index', 'index_without_freq', 'extra_rows',
     'reordered_columns', 'object_columns', 'series'],
    ids=lambda kind: f'exog: {kind}'
)
def test_create_predict_inputs_output_when_exog_dict_shares_one_object(kind):
    """
    Test _create_predict_inputs gives the same output when several series share
    the same exog object as when each series has its own copy, whether the
    object is already aligned with the predictions or it is aligned with
    `reindex_like`. Series '1' and '3' share the object and series '2' has its
    own one, so the shared result is reused in between. The shared object is
    not modified.
    """
    idx = pd.date_range(start='2020-01-01', periods=50, freq='D')
    series = {
        '1': pd.Series(np.arange(50, dtype=float), index=idx),
        '2': pd.Series(100. + np.arange(50, dtype=float), index=idx),
        '3': pd.Series(200. + np.arange(50, dtype=float), index=idx),
    }
    exog_train = {
        k: pd.DataFrame(
               {'exog_1': np.arange(50, dtype=float),
                'exog_2': np.arange(50, dtype=float) ** 2},
               index=idx
           )
        for k in series
    }
    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(), lags=3, transformer_series=None
    )
    forecaster.fit(series=series, exog=exog_train, suppress_warnings=True)

    exog_kind = _exog_dict_create_predict_inputs(
        'aligned' if kind == 'series' else kind
    )
    exog_shared = exog_kind['1']
    if kind == 'series':
        exog_shared = exog_shared['exog_1']
    exog_shared_copy = exog_shared.copy()
    exog_2 = exog_kind['2']
    exog = {'1': exog_shared, '2': exog_2, '3': exog_shared}
    exog_own = {'1': exog_shared.copy(), '2': exog_2, '3': exog_shared.copy()}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        results = forecaster._create_predict_inputs(steps=3, exog=exog)
        expected = forecaster._create_predict_inputs(steps=3, exog=exog_own)

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert results[1].keys() == expected[1].keys()
    for k in expected[1]:
        np.testing.assert_array_equal(results[1][k], expected[1][k])
    assert results[3] == ['1', '2', '3']
    pd.testing.assert_index_equal(results[4], expected[4])
    if kind == 'series':
        pd.testing.assert_series_equal(exog_shared, exog_shared_copy)
    else:
        pd.testing.assert_frame_equal(exog_shared, exog_shared_copy)
