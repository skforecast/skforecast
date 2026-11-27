# Unit test fit ForecasterRecursiveMultiSeries
# ==============================================================================
import json
import os
import pickle
import re
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from skforecast.exceptions import DataTypeWarning
from skforecast.exceptions import MissingExogWarning
from skforecast.preprocessing import RollingFeatures, CalendarFeatures
from ....recursive import ForecasterRecursiveMultiSeries

# Fixtures
from .fixtures_forecaster_recursive_multiseries import (
    series_wide_range,
    series_dict_range,
    series_dict_dt,
    exog_wide_range,
    exog_wide_dt,
    exog_dict_range,
    series_dict_unordered,
    exog_dict_unordered
)

# Default dtype of a text column: object in pandas 2, str in pandas 3
STR_DTYPE = pd.Series(['a']).dtype


class LinearRegressionWritingColumn(LinearRegression):
    """
    LinearRegression that overwrites a column of the training matrix in place.
    """

    def fit(self, X, y, sample_weight=None):
        X[:, 0] = 0.
        return super().fit(X, y, sample_weight=sample_weight)


transformer_exog = ColumnTransformer(
                       [('scale', StandardScaler(), ['exog_1']),
                        ('onehot', OneHotEncoder(), ['exog_2'])],
                       remainder = 'passthrough',
                       verbose_feature_names_out = False
                   )


@pytest.mark.parametrize(
    "estimator",
    [LinearRegression(copy_X=False),
     make_pipeline(StandardScaler(copy=False), LinearRegression()),
     LinearRegressionWritingColumn()],
    ids=['LinearRegression(copy_X=False)', 'pipeline StandardScaler(copy=False)',
         'estimator writes a column']
)
@pytest.mark.parametrize(
    "encoding",
    ['ordinal', 'ordinal_category', 'onehot', None],
    ids=lambda encoding: f'encoding: {encoding}'
)
def test_fit_ValueError_when_estimator_modifies_X_train_in_place(estimator, encoding):
    """
    Test ValueError is raised when the estimator modifies the training matrix
    in place, because the in-sample residuals are calculated afterwards with
    the same matrix. Also when the matrix has columns of several dtypes
    (`encoding='ordinal_category'`), which are converted to a single array
    before training.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        estimator=estimator, lags=3, encoding=encoding
    )

    err_msg = re.escape(
        "The estimator has modified the training matrix in place during "
        "`fit`. The matrix is used again after training, to calculate the "
        "in-sample residuals or to fit the next candidates of a search "
        "with `OneStepAheadFold`, so the results would be wrong. This "
        "happens with estimators that do not copy their input, such as "
        "`LinearRegression(copy_X=False)` or a pipeline with "
        "`StandardScaler(copy=False)`. Use the default copy behavior of "
        "the estimator (`copy_X=True`, `copy=True`)."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.fit(series=series_wide_range)


def test_fit_TypeError_when_series_names_are_not_comparable():
    """
    Test TypeError is raised when the series names cannot be sorted, for
    example an integer and a string.
    """
    series = {
        1: series_dict_range['l1'].rename(1),
        '1': series_dict_range['l2'].rename('1')
    }
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)

    err_msg = re.escape(
        "The series names must be comparable with each other, for example "
        "all strings or all integers. Got: [1, '1']."
    )
    with pytest.raises(TypeError, match=err_msg):
        forecaster.fit(series=series)


@pytest.mark.parametrize(
    "encoding",
    ['ordinal', 'onehot', None],
    ids=lambda encoding: f'encoding: {encoding}'
)
def test_fit_when_estimator_modifies_X_train_in_place_and_probabilistic_mode_False(
    encoding
):
    """
    Test that an estimator that modifies the training matrix in place can be
    fitted when `_probabilistic_mode` is `False`: the in-sample residuals are
    not calculated, so the matrix is not used after training. Predictions are
    the same as with the default copy behavior of the estimator.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(copy_X=False), lags=3, encoding=encoding
    )
    forecaster._probabilistic_mode = False
    forecaster.fit(series=series_wide_range)

    forecaster_copy = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(), lags=3, encoding=encoding
    )
    forecaster_copy.fit(series=series_wide_range)

    assert forecaster.is_fitted
    pd.testing.assert_frame_equal(
        forecaster.predict(steps=3), forecaster_copy.predict(steps=3)
    )


@pytest.mark.parametrize(
    "encoding",
    ['ordinal', 'onehot', None],
    ids=lambda encoding: f'encoding: {encoding}'
)
def test_fit_when_pipeline_with_SimpleImputer_copy_False_fills_NaN_in_place(encoding):
    """
    Test that a pipeline with `SimpleImputer(copy=False)` can be fitted when
    `exog` has NaN. The imputer fills the NaN cells of the training matrix in
    place, which does not change the in-sample residuals because `predict`
    fills them again, so the check against in-place modifications ignores
    the cells that were NaN before training. Residuals and predictions are
    the same as with `SimpleImputer(copy=True)`.
    """
    series = series_wide_range.iloc[:12]
    exog = pd.DataFrame({'exog_1': np.arange(12, dtype=float)}, index=series.index)
    exog.loc[[4, 8], 'exog_1'] = np.nan
    exog_predict = pd.DataFrame(
        {'exog_1': np.arange(12, 14, dtype=float)}, index=pd.RangeIndex(12, 14)
    )

    forecaster = ForecasterRecursiveMultiSeries(
        estimator=make_pipeline(SimpleImputer(copy=False), LinearRegression()),
        lags=3, encoding=encoding
    )
    forecaster.fit(
        series=series, exog=exog, store_in_sample_residuals=True,
        suppress_warnings=True
    )
    forecaster_copy = ForecasterRecursiveMultiSeries(
        estimator=make_pipeline(SimpleImputer(copy=True), LinearRegression()),
        lags=3, encoding=encoding
    )
    forecaster_copy.fit(
        series=series, exog=exog, store_in_sample_residuals=True,
        suppress_warnings=True
    )

    assert forecaster.is_fitted
    assert (
        forecaster.in_sample_residuals_.keys()
        == forecaster_copy.in_sample_residuals_.keys()
    )
    for level, residuals in forecaster_copy.in_sample_residuals_.items():
        np.testing.assert_array_almost_equal(
            forecaster.in_sample_residuals_[level], residuals
        )
    pd.testing.assert_frame_equal(
        forecaster.predict(steps=2, exog=exog_predict),
        forecaster_copy.predict(steps=2, exog=exog_predict)
    )


@pytest.mark.parametrize(
    "forecaster_kwargs",
    [
        {"estimator": LinearRegression(), "lags": 3},
        {"estimator": LinearRegression(), "lags": 3,
         "window_features": RollingFeatures(stats=['mean'], window_sizes=4)},
        {"estimator": LinearRegression(), "lags": 3,
         "window_features": RollingFeatures(stats=['mean'], window_sizes=4),
         "transformer_series": StandardScaler(), "transformer_exog": StandardScaler()},
        {"estimator": LinearRegression(), "lags": 3,
         "window_features": RollingFeatures(stats=['mean'], window_sizes=4),
         "transformer_series": StandardScaler(), "transformer_exog": StandardScaler(),
         "differentiation": 1},
    ],
    ids=["base", "window_features", "transformers", "differentiation"]
)
def test_forecaster_fit_does_not_modify_series_exog(forecaster_kwargs):
    """
    Test forecaster.fit does not modify series and exog.
    """
    series_local = series_wide_range.copy()
    exog_local = exog_wide_range[['exog_1']].copy()
    series_copy = series_local.copy()
    exog_copy = exog_local.copy()

    forecaster = ForecasterRecursiveMultiSeries(**forecaster_kwargs)
    forecaster.fit(series=series_local, exog=exog_local)

    pd.testing.assert_frame_equal(series_local, series_copy)
    pd.testing.assert_frame_equal(exog_local, exog_copy)


@pytest.mark.parametrize(
    "forecaster_kwargs",
    [
        {"estimator": LinearRegression(), "lags": 3},
        {"estimator": LinearRegression(), "lags": 3,
         "window_features": RollingFeatures(stats=['mean'], window_sizes=4)},
        {"estimator": LinearRegression(), "lags": 3,
         "window_features": RollingFeatures(stats=['mean'], window_sizes=4),
         "transformer_series": StandardScaler(), "transformer_exog": StandardScaler()},
        {"estimator": LinearRegression(), "lags": 3,
         "window_features": RollingFeatures(stats=['mean'], window_sizes=4),
         "transformer_series": StandardScaler(), "transformer_exog": StandardScaler(),
         "differentiation": 1},
    ],
    ids=["base", "window_features", "transformers", "differentiation"]
)
def test_forecaster_fit_does_not_modify_series_exog_dict(forecaster_kwargs):
    """
    Test forecaster.fit does not modify series and exog when passed as dictionaries.
    """
    exog_dict_numeric = {
        'l1': exog_wide_range[['exog_1']].copy(),
        'l2': exog_wide_range['exog_1'].copy(),
    }
    series_local = {k: v.copy() for k, v in series_dict_range.items()}
    exog_local = {k: v.copy() for k, v in exog_dict_numeric.items()}
    series_copy = {k: v.copy() for k, v in series_local.items()}
    exog_copy = {k: v.copy() for k, v in exog_local.items()}

    forecaster = ForecasterRecursiveMultiSeries(**forecaster_kwargs)
    forecaster.fit(series=series_local, exog=exog_local)

    for k in series_local:
        pd.testing.assert_series_equal(series_local[k], series_copy[k])
    for k in exog_local:
        if isinstance(exog_local[k], pd.DataFrame):
            pd.testing.assert_frame_equal(exog_local[k], exog_copy[k])
        else:
            pd.testing.assert_series_equal(exog_local[k], exog_copy[k])


def test_forecaster_series_exog_features_stored():
    """
    Test forecaster stores series and exog features after fitting.
    """

    rolling = RollingFeatures(
        stats=['ratio_min_max', 'median'], window_sizes=4
    )
    calendar = CalendarFeatures(
        features=['day_of_week', 'weekend'], encoding="cyclical"
    )
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, window_features=rolling, 
        calendar_features=calendar, transformer_exog=transformer_exog
    )
    forecaster.fit(series=series_dict_dt, exog=exog_wide_dt)

    series_names_in_ = ['l1', 'l2']
    exog_in_ = True
    exog_type_in_ = type(exog_wide_range)
    exog_names_in_ = ['exog_1', 'exog_2']
    exog_dtypes_in_ = {
        'exog_1': np.dtype(float), 
        'exog_2': STR_DTYPE
    }
    # All floats
    exog_dtypes_out_ = {
        'exog_1': np.dtype(float), 
        'exog_2_a': np.dtype(float), 
        'exog_2_b': np.dtype(float)
    }
    categorical_features_names_in_ = []
    X_train_series_names_in_ = ['l1', 'l2']
    X_train_window_features_names_out_ = ['roll_ratio_min_max_4', 'roll_median_4']
    X_train_calendar_features_names_out_ = ['weekend', 'day_of_week_sin', 'day_of_week_cos']
    X_train_exog_names_out_ = ['exog_1', 'exog_2_a', 'exog_2_b']
    X_train_features_names_out_ = [
        'lag_1', 'lag_2', 'lag_3', 'roll_ratio_min_max_4', 'roll_median_4', 
        '_level_skforecast', 'exog_1', 'exog_2_a', 'exog_2_b', 
        'weekend', 'day_of_week_sin', 'day_of_week_cos'
    ]
    
    assert forecaster.series_names_in_ == series_names_in_
    assert forecaster.exog_in_ == exog_in_
    assert forecaster.exog_type_in_ == exog_type_in_
    assert forecaster.exog_names_in_ == exog_names_in_
    assert forecaster.exog_dtypes_in_ == exog_dtypes_in_
    assert forecaster.exog_dtypes_out_ == exog_dtypes_out_
    assert forecaster.categorical_features_names_in_ == categorical_features_names_in_
    assert forecaster.X_train_series_names_in_ == X_train_series_names_in_
    assert forecaster.X_train_window_features_names_out_ == X_train_window_features_names_out_
    assert forecaster.X_train_calendar_features_names_out_ == X_train_calendar_features_names_out_
    assert forecaster.X_train_exog_names_out_ == X_train_exog_names_out_
    assert forecaster.X_train_features_names_out_ == X_train_features_names_out_


def test_forecaster_series_not_matching_exog_features_stored():
    """
    Test forecaster stores series and exog features after fitting when exog keys
    do not match series keys.
    """
    exog_dict_no_match = {
        '1': exog_dict_range['l1'].copy(),
        '2': exog_dict_range['l2'].copy(),
    }
    rolling = RollingFeatures(
        stats=['ratio_min_max', 'median'], window_sizes=4
    )
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, window_features=rolling, transformer_exog=transformer_exog
    )

    warn_msg = re.escape(
        "No exogenous variables were found in `exog` that match the "
        "series IDs provided in `series`. As a result, no exogenous "
        "variables are included in the training matrices. Please "
        "review the series IDs in `exog` and ensure they match the "
        "following IDs: ['l1', 'l2']. The forecaster will be "
        "trained without exogenous variables."
    )
    with pytest.warns(MissingExogWarning, match = warn_msg):
        forecaster.fit(series=series_dict_range, exog=exog_dict_no_match)

    series_names_in_ = ['l1', 'l2']
    exog_in_ = False
    exog_type_in_ = None
    exog_names_in_ = None
    exog_dtypes_in_ = None
    exog_dtypes_out_ = None
    categorical_features_names_in_ = None
    X_train_series_names_in_ = ['l1', 'l2']
    X_train_window_features_names_out_ = ['roll_ratio_min_max_4', 'roll_median_4']
    X_train_exog_names_out_ = None
    X_train_features_names_out_ = [
        'lag_1', 'lag_2', 'lag_3', 'roll_ratio_min_max_4', 'roll_median_4', '_level_skforecast'
    ]
    
    assert forecaster.series_names_in_ == series_names_in_
    assert forecaster.exog_in_ == exog_in_
    assert forecaster.exog_type_in_ == exog_type_in_
    assert forecaster.exog_names_in_ == exog_names_in_
    assert forecaster.exog_dtypes_in_ == exog_dtypes_in_
    assert forecaster.exog_dtypes_out_ == exog_dtypes_out_
    assert forecaster.categorical_features_names_in_ == categorical_features_names_in_
    assert forecaster.X_train_series_names_in_ == X_train_series_names_in_
    assert forecaster.X_train_window_features_names_out_ == X_train_window_features_names_out_
    assert forecaster.X_train_exog_names_out_ == X_train_exog_names_out_
    assert forecaster.X_train_features_names_out_ == X_train_features_names_out_


def test_fit_correct_dict_create_series_weights_weight_func_transformer_series():
    """
    Test fit method creates correctly all the auxiliary dicts, series_weights_,
    weight_func_, transformer_series_.
    """
    series = pd.DataFrame(
        {'l1': np.arange(10), 'l2': np.arange(10), 'l3': np.arange(10)},
        index = pd.date_range('2022-01-04', periods=10, freq='D')
    ).to_dict(orient='series')

    def custom_weights(index):  # pragma: no cover
        """
        Return 0 if index is between '2022-01-08' and '2022-01-10', 1 otherwise.
        """
        weights = np.where(
                      (index >= '2022-01-08') & (index <= '2022-01-10'),
                      0,
                      1
                  )
        
        return weights

    transformer_series = {'l1': StandardScaler(), '_unknown_level': StandardScaler()}
    weight_func = {'l2': custom_weights}    
    series_weights = {'l1': 3., 'l3': 0.5, 'l4': 2.}

    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(), 
                     lags               = 3,
                     transformer_series = transformer_series,
                     weight_func        = weight_func,
                     series_weights     = series_weights
                 )
    
    forecaster.fit(series=series, store_in_sample_residuals=False)

    expected_transformer_series_ = {
        'l1': forecaster.transformer_series_['l1'], 
        'l2': None, 
        'l3': None,
        '_unknown_level': forecaster.transformer_series_['_unknown_level']
    }
    expected_weight_func_ = {
        'l1': forecaster._weight_func_all_1, 
        'l2': custom_weights, 
        'l3': forecaster._weight_func_all_1
    }
    expected_series_weights_ = {'l1': 3., 'l2': 1., 'l3': 0.5}

    assert forecaster.transformer_series_.keys() == expected_transformer_series_.keys()
    for k in expected_transformer_series_.keys():
        if expected_transformer_series_[k] is None:
            assert isinstance(forecaster.transformer_series_[k], type(None))
        else:
            assert isinstance(forecaster.transformer_series_[k], StandardScaler)
    assert forecaster.weight_func_.keys() == expected_weight_func_.keys()
    for key in forecaster.weight_func_.keys():
        assert forecaster.weight_func_[key].__code__.co_code == expected_weight_func_[key].__code__.co_code
    assert forecaster.series_weights_ == expected_series_weights_

    forecaster.fit(
        series = {k: v for k, v in series.items() if k in ['l1', 'l2']}, 
        store_in_sample_residuals = False
    )

    expected_transformer_series_ = {
        'l1': forecaster.transformer_series_['l1'], 
        'l2': None,
        '_unknown_level': forecaster.transformer_series_['_unknown_level']
    }
    expected_weight_func_ = {'l1': forecaster._weight_func_all_1, 'l2': custom_weights}
    expected_series_weights_ = {'l1': 3., 'l2': 1.}

    assert forecaster.transformer_series_.keys() == expected_transformer_series_.keys()
    for k in expected_transformer_series_.keys():
        if expected_transformer_series_[k] is None:
            assert isinstance(forecaster.transformer_series_[k], type(None))
        else:
            assert isinstance(forecaster.transformer_series_[k], StandardScaler)
    assert forecaster.weight_func_.keys() == expected_weight_func_.keys()
    for key in forecaster.weight_func_.keys():
        assert forecaster.weight_func_[key].__code__.co_code == expected_weight_func_[key].__code__.co_code
    assert forecaster.series_weights_ == expected_series_weights_


def test_forecaster_DatetimeIndex_index_freq_stored():
    """
    Test serie_with_DatetimeIndex.index.freq is stored in forecaster.index_freq_.
    """
    series = {
        '1': pd.Series(np.arange(5)), 
        '2': pd.Series(np.arange(5))
    }
    series['1'].index = pd.date_range(start='2022-01-01', periods=5, freq='D')
    series['2'].index = pd.date_range(start='2022-01-01', periods=5, freq='D')

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series)
    results = forecaster.index_freq_

    expected = series['1'].index.freq

    assert results == expected


def test_forecaster_index_step_stored():
    """
    Test serie without DatetimeIndex, step is stored in forecaster.index_freq_.
    """
    series = {
        '1': pd.Series(np.arange(5)), 
        '2': pd.Series(np.arange(5))
    }
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series)
    results = forecaster.index_freq_

    expected = series['1'].index.step

    assert results == expected


@pytest.mark.parametrize(
    "series_format",
    ['dict', 'wide'],
    ids=lambda series_format: f'series_format: {series_format}'
)
@pytest.mark.parametrize(
    "encoding",
    ['ordinal', 'ordinal_category', 'onehot', None],
    ids=lambda encoding: f'encoding: {encoding}'
)
def test_fit_when_series_names_are_integers(encoding, series_format):
    """
    Test that the forecaster is fitted when the series names are integers
    (keys of a dict or columns of a wide DataFrame), and that the names are
    kept as integers. Before, the encoding mapping was built with the names
    as strings and `fit` raised a `KeyError`.
    """
    series = {
        10: series_dict_range['l1'].rename(10),
        2: series_dict_range['l2'].rename(2),
        1: series_dict_range['l1'].rename(1) * 2
    }
    if series_format == 'wide':
        series = pd.DataFrame(series)

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=encoding
    )
    forecaster.fit(series=series, store_in_sample_residuals=True, suppress_warnings=True)

    assert forecaster.is_fitted
    assert forecaster.encoding_mapping_ == {1: 0, 2: 1, 10: 2}
    assert forecaster.series_names_in_ == [10, 2, 1]
    assert set(forecaster.X_train_series_names_in_) == {1, 2, 10}
    assert set(forecaster.last_window_.keys()) == {1, 2, 10}
    if encoding is None:
        assert list(forecaster.in_sample_residuals_.keys()) == ['_unknown_level']
    else:
        assert set(forecaster.in_sample_residuals_.keys()) == {1, 2, 10, '_unknown_level'}
    if encoding == 'onehot':
        assert forecaster.X_train_features_names_out_ == [
            'lag_1', 'lag_2', 'lag_3', '1', '2', '10'
        ]


@pytest.mark.parametrize("encoding", 
                         ['ordinal', 'ordinal_category', 'onehot'], 
                         ids = lambda encoding: f'encoding: {encoding}')
def test_fit_in_sample_residuals_stored(encoding):
    """
    Test that values of in_sample_residuals_ are stored after fitting
    when `store_in_sample_residuals=True`.
    """
    series = {
        '1': pd.Series(np.arange(5)), 
        '2': pd.Series(np.arange(5))
    }

    rolling = RollingFeatures(
        stats=['ratio_min_max', 'median'], window_sizes=4
    )
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=encoding, window_features=rolling
    )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results = forecaster.in_sample_residuals_

    expected = {
        '1': np.array([0.]),
        '2': np.array([0.]),
        '_unknown_level': np.array([0., 0.])
    }
    
    X_train_window_features_names_out_ = ['roll_ratio_min_max_4', 'roll_median_4']
    X_train_features_names_out_ = (
        ['lag_1', 'lag_2', 'lag_3', 'roll_ratio_min_max_4', 'roll_median_4', '_level_skforecast'] 
        if encoding != 'onehot' 
        else ['lag_1', 'lag_2', 'lag_3', 'roll_ratio_min_max_4', 'roll_median_4', '1', '2']
    )

    assert forecaster.series_names_in_ == ['1', '2']
    assert forecaster.X_train_series_names_in_ == ['1', '2']
    assert forecaster.X_train_window_features_names_out_ == X_train_window_features_names_out_
    assert forecaster.X_train_features_names_out_ == X_train_features_names_out_
    assert isinstance(results, dict)
    assert all(isinstance(x, np.ndarray) for x in results.values())
    assert results.keys() == expected.keys()
    for k in expected.keys():
        np.testing.assert_array_almost_equal(results[k], expected[k])


@pytest.mark.parametrize("encoding", 
                         ['ordinal', 'ordinal_category', 'onehot', None], 
                         ids = lambda encoding: f'encoding: {encoding}')
def test_fit_in_sample_residuals_by_bin_stored(encoding):
    """
    Test that values of in_sample_residuals_ are stored after fitting
    when `store_in_sample_residuals=True`.
    """
    rng = np.random.default_rng(1894)
    series = {
        "1": pd.Series(rng.normal(10, 5, 20)), 
        "2": pd.Series(rng.normal(10, 5, 20))
    }

    rolling = RollingFeatures(stats=["ratio_min_max", "median"], window_sizes=4)
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=encoding, window_features=rolling, binner_kwargs={"n_bins": 3}
    )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results_residuals = forecaster.in_sample_residuals_
    results_residuals_bin = forecaster.in_sample_residuals_by_bin_
    results_binner_intervals = forecaster.binner_intervals_

    if encoding is not None:
        expected_residuals = {
            '1': np.array([-1.31688412, -4.83850341, -6.89193651, -3.88484903,  0.93550062,
                    -5.05052676,  2.22146909,  8.10253854, -0.03935851,  2.44731007,
                    0.10143843,  0.99207366,  4.43782989, -6.31608079,  2.76608522,
                    6.3338936 ]),
            '2': np.array([ 1.35912405, -6.75955818, -1.96209665,  1.38750505,  3.0777547 ,
                    2.66627803,  7.45266459, -4.81736063,  2.97073109, -2.40028892,
                    0.69596738,  6.12813429, -2.54249734, -1.75531268, -2.58079634,
                    -2.92024843]),
            '_unknown_level': np.array([-1.31688412, -4.83850341, -6.89193651, -3.88484903,  0.93550062,
                    -5.05052676,  2.22146909,  8.10253854, -0.03935851,  2.44731007,
                    0.10143843,  0.99207366,  4.43782989, -6.31608079,  2.76608522,
                    6.3338936 ,  1.35912405, -6.75955818, -1.96209665,  1.38750505,
                    3.0777547 ,  2.66627803,  7.45266459, -4.81736063,  2.97073109,
                    -2.40028892,  0.69596738,  6.12813429, -2.54249734, -1.75531268,
                    -2.58079634, -2.92024843])
        }

        expected_residuals_bin = {
            '1': {
                    0: np.array([-1.31688412, -4.83850341,  0.99207366,  2.76608522,  6.3338936 ]),
                    1: np.array([-3.88484903,  0.93550062,  2.44731007,  0.10143843,  4.43782989, -6.31608079]),
                    2: np.array([-6.89193651, -5.05052676,  2.22146909,  8.10253854, -0.03935851])
                },
            '2': {
                    0: np.array([-6.75955818, -1.96209665,  3.0777547 ,  2.97073109, -2.40028892]),
                    1: np.array([1.38750505,  7.45266459, -4.81736063,  0.69596738, -2.92024843]),
                    2: np.array([1.35912405,  2.66627803,  6.12813429, -2.54249734, -1.75531268, -2.58079634])
                },
            '_unknown_level': {
                    0: np.array([-1.31688412, -4.83850341, -3.88484903,  0.99207366,  2.76608522,
                            6.3338936, -6.75955818, -1.96209665,  3.0777547,  2.97073109, -2.40028892]),
                    1: np.array([ 0.93550062, -0.03935851,  2.44731007,  0.10143843,  4.43782989,
                            -6.31608079,  7.45266459, -4.81736063,  0.69596738, -2.92024843]),
                    2: np.array([-6.89193651, -5.05052676,  2.22146909,  8.10253854,  1.35912405,
                            1.38750505,  2.66627803,  6.12813429, -2.54249734, -1.75531268, -2.58079634])
                }
            }

        expected_binner_intervals = {
            "1": {
                0: (7.268796921364613, 8.987998904684593),
                1: (8.987998904684593, 9.830020290458247),
                2: (9.830020290458247, 12.366369800747009),
            },
            "2": {
                0: (7.211558912361964, 9.22351564748152),
                1: (9.22351564748152, 11.350695147373349),
                2: (11.350695147373349, 11.844709741478395),
            },
            "_unknown_level": {
                0: (7.211558912361964, 9.016289033159751),
                1: (9.016289033159751, 10.730519759760567),
                2: (10.730519759760567, 12.366369800747009),
            },
        }

    else:
        expected_residuals = {
            '_unknown_level': np.array([-2.05820541, -5.47429818, -7.31524702, -4.29526858,  0.66810725,
                -5.37727919,  1.94669051,  7.67911837, -0.63572989,  1.79063442,
                -0.48881591,  0.44689235,  4.0759107 , -6.74143431,  2.16887587,
                5.7738583 ,  1.76518089, -6.32949235, -1.56689305,  1.98201019,
                3.56844034,  3.14094834,  7.85637887, -4.32241071,  3.27047182,
                -2.01737457,  1.26390611,  6.83575019, -2.02500966, -1.25537261,
                -2.00844951, -2.32189357])
        }

        expected_residuals_bin = {
            '_unknown_level': {
                0: np.array([-2.05820541, -5.47429818,  0.44689235,  2.16887587, -6.32949235,
                            -1.56689305,  3.56844034,  3.27047182, -2.01737457,  1.26390611, -2.32189357]),
                1: np.array([-4.29526858,  0.66810725,  1.79063442, -0.48881591,  4.0759107, -6.74143431,
                            5.7738583,  1.98201019,  7.85637887, -4.32241071]),
                2: np.array([-7.31524702, -5.37727919,  1.94669051,  7.67911837, -0.63572989,
                            1.76518089,  3.14094834,  6.83575019, -2.02500966, -1.25537261, -2.00844951])
            }
        }

        expected_binner_intervals = {
            "_unknown_level": {
                0: (6.81635531697439, 9.303028105789933),
                1: (9.303028105789933, 10.690051434777626),
                2: (10.690051434777626, 12.789680311772173),
            }
        }

    X_train_window_features_names_out_ = ["roll_ratio_min_max_4", "roll_median_4"]
    if encoding in ['ordinal', 'ordinal_category']:
        X_train_features_names_out_ = [
            "lag_1",
            "lag_2",
            "lag_3",
            "roll_ratio_min_max_4",
            "roll_median_4",
            "_level_skforecast",
        ]
    elif encoding == 'onehot':
        X_train_features_names_out_ = ["lag_1", "lag_2", "lag_3", "roll_ratio_min_max_4", "roll_median_4", "1", "2"]
    else:
        X_train_features_names_out_ = ["lag_1", "lag_2", "lag_3", "roll_ratio_min_max_4", "roll_median_4"]

    assert forecaster.series_names_in_ == ["1", "2"]
    assert forecaster.X_train_series_names_in_ == ["1", "2"]
    assert (
        forecaster.X_train_window_features_names_out_ == X_train_window_features_names_out_
    )
    assert forecaster.X_train_features_names_out_ == X_train_features_names_out_

    # In-sample residuals
    assert results_residuals.keys() == expected_residuals.keys()
    for k in results_residuals.keys():
        np.testing.assert_array_almost_equal(
            np.sort(results_residuals[k]), np.sort(expected_residuals[k])
        )

    # In-sample residuals by bin
    assert results_residuals_bin.keys() == expected_residuals_bin.keys()
    for level in results_residuals_bin.keys():
        assert results_residuals_bin[level].keys() == expected_residuals_bin[level].keys()
        for k in results_residuals_bin[level].keys():
            np.testing.assert_array_almost_equal(
                results_residuals_bin[level][k], expected_residuals_bin[level][k]
            )
    
    # Binner intervals
    assert results_binner_intervals.keys() == expected_binner_intervals.keys()
    for level in results_binner_intervals.keys():
        assert results_binner_intervals[level].keys() == expected_binner_intervals[level].keys()
        for k in results_binner_intervals[level].keys():
            assert results_binner_intervals[level][k][0] == approx(expected_binner_intervals[level][k][0])
            assert results_binner_intervals[level][k][1] == approx(expected_binner_intervals[level][k][1])


@pytest.mark.parametrize("encoding",
                         ['ordinal', 'ordinal_category', 'onehot'],
                         ids = lambda encoding: f'encoding: {encoding}')
def test_fit_in_sample_residuals_by_level_when_series_unordered_different_lengths_and_dropped(encoding):
    """
    Test the in-sample residuals, residuals by bin and binner intervals stored for
    each level when the series are not in alphabetical order, have different
    lengths and an interspersed NaN, and one series ('d') loses all its rows
    because it has no exog and `dropna_from_series=True`. The order of the keys
    is also checked: it follows the order of the series with every encoding.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=2, encoding=encoding, dropna_from_series=True,
        binner_kwargs={'n_bins': 2}
    )
    forecaster.fit(
        series=series_dict_unordered, exog=exog_dict_unordered,
        store_in_sample_residuals=True
    )

    if encoding == 'onehot':
        expected_residuals = {
            'c': np.array([2.35785357, -0.66170765, -1.19700919, -3.45838688,
                           -0.08898634, 3.04823649]),
            'a': np.array([0.51834364, 0.8441147, 1.91669612, 0.06325437,
                           -0.16432107, -0.53629588, -2.64179188]),
            'b': np.array([-1.17587909, 1.23566797, 0.65774457, -0.71753345]),
            '_unknown_level': np.array([
                2.35785357, -0.66170765, -1.19700919, -3.45838688, -0.08898634,
                3.04823649, 0.51834364, 0.8441147, 1.91669612, 0.06325437,
                -0.16432107, -0.53629588, -2.64179188, -1.17587909, 1.23566797,
                0.65774457, -0.71753345
            ])
        }
        expected_residuals_by_bin = {
            'c': {0: np.array([2.35785357, -0.66170765, -1.19700919]),
                  1: np.array([-3.45838688, -0.08898634, 3.04823649])},
            'a': {0: np.array([1.91669612, 0.06325437, -2.64179188]),
                  1: np.array([0.51834364, 0.8441147, -0.16432107, -0.53629588])},
            'b': {0: np.array([-1.17587909, 1.23566797]),
                  1: np.array([0.65774457, -0.71753345])},
            '_unknown_level': {
                0: np.array([2.35785357, -0.66170765, 1.91669612, 0.06325437,
                             -2.64179188, -1.17587909, 1.23566797, -0.71753345]),
                1: np.array([-1.19700919, -3.45838688, -0.08898634, 3.04823649,
                             0.51834364, 0.8441147, -0.16432107, -0.53629588,
                             0.65774457])
            }
        }
        expected_binner_intervals = {
            'c': {0: (6.822146430589493, 10.05299776610925),
                  1: (10.05299776610925, 10.971763505364349)},
            'a': {0: (7.146745630949951, 10.074321066273868),
                  1: (10.074321066273868, 12.626295879716096)},
            'b': {0: (3.62587909306756, 7.460932738187426),
                  1: (7.460932738187426, 9.192255430557589)},
            '_unknown_level': {0: (3.62587909306756, 9.192255430557589),
                               1: (9.192255430557589, 12.626295879716096)}
        }
    else:
        expected_residuals = {
            'a': np.array([2.49700869, 1.45587963, 1.20052686, 0.5560771,
                           1.4645523, 0.65310967, -2.95745736]),
            'b': np.array([-5.99486505, -0.25639827, -0.26807589, -3.22005454]),
            'c': np.array([2.24118559, -0.87331252, 0.10050154, -1.9052904,
                           1.57948265, 3.72713002]),
            '_unknown_level': np.array([
                2.24118559, -0.87331252, 0.10050154, -1.9052904, 1.57948265,
                3.72713002, 2.49700869, 1.45587963, 1.20052686, 0.5560771,
                1.4645523, 0.65310967, -2.95745736, -5.99486505, -0.25639827,
                -0.26807589, -3.22005454
            ])
        }
        expected_residuals_by_bin = {
            'a': {0: np.array([0.5560771, 1.4645523, -2.95745736]),
                  1: np.array([2.49700869, 1.45587963, 1.20052686, 0.65310967])},
            'b': {0: np.array([-5.99486505, -0.25639827]),
                  1: np.array([-0.26807589, -3.22005454])},
            'c': {0: np.array([2.24118559, -0.87331252, 0.10050154]),
                  1: np.array([-1.9052904, 1.57948265, 3.72713002])},
            '_unknown_level': {
                0: np.array([2.24118559, -0.87331252, 0.10050154, 1.57948265,
                             0.5560771, 1.4645523, -5.99486505, -0.25639827]),
                1: np.array([-1.9052904, 3.72713002, 2.49700869, 1.45587963,
                             1.20052686, 0.65310967, -2.95745736, -0.26807589,
                             -3.22005454])
            }
        }
        expected_binner_intervals = {
            'a': {0: (6.653922900508369, 8.972991306860978),
                  1: (8.972991306860978, 11.43689033146482)},
            'b': {0: (8.444865054761285, 9.377237082067435),
                  1: (9.377237082067435, 10.28005454280374)},
            'c': {0: (6.93881441329464, 8.570007908230785),
                  1: (8.570007908230785, 10.292869976253973)},
            '_unknown_level': {0: (6.653922900508369, 8.927457363589692),
                               1: (8.927457363589692, 11.43689033146482)}
        }

    expected_levels = ['c', 'a', 'b']
    expected_keys = expected_levels + ['_unknown_level']

    assert forecaster.series_names_in_ == ['c', 'a', 'd', 'b']
    assert forecaster.X_train_series_names_in_ == expected_levels
    assert list(forecaster.in_sample_residuals_) == expected_keys
    assert list(forecaster.in_sample_residuals_by_bin_) == expected_keys
    assert list(forecaster.binner_intervals_) == expected_keys
    for level in expected_keys:
        np.testing.assert_array_almost_equal(
            forecaster.in_sample_residuals_[level], expected_residuals[level]
        )
        results_bins = forecaster.in_sample_residuals_by_bin_[level]
        assert results_bins.keys() == expected_residuals_by_bin[level].keys()
        for k in results_bins.keys():
            np.testing.assert_array_almost_equal(
                results_bins[k], expected_residuals_by_bin[level][k]
            )
        results_intervals = forecaster.binner_intervals_[level]
        assert results_intervals.keys() == expected_binner_intervals[level].keys()
        for k in results_intervals.keys():
            np.testing.assert_array_almost_equal(
                results_intervals[k], expected_binner_intervals[level][k]
            )


@pytest.mark.parametrize("encoding",
                         ['ordinal', 'ordinal_category', 'onehot', None],
                         ids = lambda encoding: f'encoding: {encoding}')
def test_fit_same_residuals_when_residuals_greater_than_10_000(encoding):
    """
    Test fit return same residuals when residuals len is greater than 10_000.
    Testing with two different forecaster. Residuals shouldn't be more than
    1000 values.
    """
    rng = np.random.default_rng(12345)
    series = pd.DataFrame(
        {"1": rng.normal(10, 1, 15_000), "2": rng.normal(10, 1, 15_000)},
        index=pd.date_range(start="2000-01-01", periods=15_000, freq="h"),
    ).to_dict(orient='series')

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=encoding
    )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results_1 = forecaster.in_sample_residuals_

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=encoding
    )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results_2 = forecaster.in_sample_residuals_

    assert results_1.keys() == results_2.keys()
    assert np.all([len(v) == 10_000 for v in results_1.values()])
    assert np.all([len(v) == 10_000 for v in results_2.values()])
    for k in results_2.keys():
        np.testing.assert_array_equal(results_1[k], results_2[k])


@pytest.mark.parametrize("encoding", 
                         ['ordinal', 'ordinal_category', 'onehot', None], 
                         ids = lambda encoding: f'encoding: {encoding}')
def test_fit_same_residuals_by_bin_when_residuals_greater_than_10_000(encoding):
    """
    Test fit return same residuals by bien when residuals len is greater than 10_000.
    Testing with two different forecaster. Residuals shouldn't be more than
    1000 values.
    """
    rng = np.random.default_rng(12345)
    series = pd.DataFrame(
        {"1": rng.normal(10, 1, 15_000), "2": rng.normal(10, 1, 15_000)},
        index=pd.date_range(start="2000-01-01", periods=15_000, freq="h"),
    ).to_dict(orient='series')

    forecaster_1 = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=encoding, binner_kwargs={"n_bins": 3}
    )
    forecaster_1.fit(series=series, store_in_sample_residuals=True)
    results_1 = forecaster_1.in_sample_residuals_by_bin_

    forecaster_2 = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=encoding, binner_kwargs={"n_bins": 3}
    )
    forecaster_2.fit(series=series, store_in_sample_residuals=True)
    results_2 = forecaster_2.in_sample_residuals_by_bin_

    assert results_1.keys() == results_2.keys()
    for k in results_1.keys() & results_2.keys():
        for bin in results_1[k].keys():
            assert len(results_1[k][bin]) == len(results_2[k][bin])
            assert len(results_1[k][bin]) == 10_000 // 3
    
    assert forecaster_1.binner_intervals_.keys() == forecaster_2.binner_intervals_.keys()
    for level in forecaster_1.binner_intervals_.keys():
        for k in forecaster_1.binner_intervals_[level].keys():
            assert forecaster_1.binner_intervals_[level][k][0] == forecaster_2.binner_intervals_[level][k][0]
            assert forecaster_1.binner_intervals_[level][k][1] == forecaster_2.binner_intervals_[level][k][1]


@pytest.mark.parametrize("encoding", 
                         ['ordinal', 'ordinal_category', 'onehot', None], 
                         ids = lambda encoding: f'encoding: {encoding}')
def test_fit_in_sample_residuals_not_stored_probabilistic_mode_binned(encoding):
    """
    Test that values of in_sample_residuals_ are not stored after fitting
    when `store_in_sample_residuals=False`. Binner intervals are stored.
    """
    rng = np.random.default_rng(1894)
    series = {
        "1": pd.Series(rng.normal(10, 5, 20)), 
        "2": pd.Series(rng.normal(10, 5, 20))
    }

    rolling = RollingFeatures(stats=["ratio_min_max", "median"], window_sizes=4)
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=encoding, window_features=rolling, binner_kwargs={"n_bins": 3}
    )
    forecaster.fit(series=series, store_in_sample_residuals=False)
    results_residuals = forecaster.in_sample_residuals_
    results_residuals_bin = forecaster.in_sample_residuals_by_bin_
    results_binner_intervals = forecaster.binner_intervals_

    if encoding is not None:
        expected_residuals = {'1': None, '2': None, '_unknown_level': None}
        expected_residuals_by_bin = {'1': None, '2': None, '_unknown_level': None}
        expected_binner_intervals = {
            "1": {
                0: (7.268796921364613, 8.987998904684593),
                1: (8.987998904684593, 9.830020290458247),
                2: (9.830020290458247, 12.366369800747009),
            },
            "2": {
                0: (7.211558912361964, 9.22351564748152),
                1: (9.22351564748152, 11.350695147373349),
                2: (11.350695147373349, 11.844709741478395),
            },
            "_unknown_level": {
                0: (7.211558912361964, 9.016289033159751),
                1: (9.016289033159751, 10.730519759760567),
                2: (10.730519759760567, 12.366369800747009),
            },
        }

    else:
        expected_residuals = {'_unknown_level': None}
        expected_residuals_by_bin = {'_unknown_level': None}
        expected_binner_intervals = {
            "_unknown_level": {
                0: (6.81635531697439, 9.303028105789933),
                1: (9.303028105789933, 10.690051434777626),
                2: (10.690051434777626, 12.789680311772173),
            }
        }

    # In-sample residuals
    assert isinstance(results_residuals, dict)
    assert results_residuals.keys() == expected_residuals.keys()
    assert results_residuals == expected_residuals

    # In-sample residuals by bin
    assert isinstance(results_residuals_bin, dict)
    assert results_residuals_bin.keys() == expected_residuals_by_bin.keys()
    assert results_residuals_bin == expected_residuals_by_bin

    # Binner intervals
    assert results_binner_intervals.keys() == expected_binner_intervals.keys()
    for level in results_binner_intervals.keys():
        assert results_binner_intervals[level].keys() == expected_binner_intervals[level].keys()
        for k in results_binner_intervals[level].keys():
            assert results_binner_intervals[level][k][0] == approx(expected_binner_intervals[level][k][0])
            assert results_binner_intervals[level][k][1] == approx(expected_binner_intervals[level][k][1])


@pytest.mark.parametrize("encoding", 
                         ['ordinal', 'ordinal_category', 'onehot', None], 
                         ids = lambda encoding: f'encoding: {encoding}')
def test_fit_in_sample_residuals_not_stored_probabilistic_mode_False(encoding):
    """
    Test that values of in_sample_residuals_ are not stored after fitting
    when `store_in_sample_residuals=False` and _probabilistic_mode=False.
    """
    rng = np.random.default_rng(1894)
    series = {
        "1": pd.Series(rng.normal(10, 5, 20)), 
        "2": pd.Series(rng.normal(10, 5, 20))
    }

    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=encoding
    )
    forecaster._probabilistic_mode = False
    forecaster.fit(series=series, store_in_sample_residuals=False)
    results_residuals = forecaster.in_sample_residuals_
    results_residuals_bin = forecaster.in_sample_residuals_by_bin_
    results_binner_intervals = forecaster.binner_intervals_

    if encoding is not None:
        expected_residuals = {'1': None, '2': None, '_unknown_level': None}
        expected_residuals_by_bin = {'1': None, '2': None, '_unknown_level': None}
        expected_binner_intervals = {}

    else:
        expected_residuals = {'_unknown_level': None}
        expected_residuals_by_bin = {'_unknown_level': None}
        expected_binner_intervals = {}

    # In-sample residuals
    assert isinstance(results_residuals, dict)
    assert results_residuals.keys() == expected_residuals.keys()
    assert results_residuals == expected_residuals

    # In-sample residuals by bin
    assert isinstance(results_residuals_bin, dict)
    assert results_residuals_bin.keys() == expected_residuals_by_bin.keys()
    assert results_residuals_bin == expected_residuals_by_bin

    # Binner intervals
    assert results_binner_intervals == expected_binner_intervals


def test_fit_last_window_stored():
    """
    Test that values of last window are stored after fitting.
    """
    series = {
        '1': pd.Series(np.arange(5, dtype=float)), 
        '2': pd.Series(np.arange(5, dtype=float))
    }

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series)

    expected = {
        '1': pd.Series(
                 data  = np.array([2., 3., 4.]),
                 index = pd.RangeIndex(start=2, stop=5, step=1),
                 name  = '1',
                 dtype = float
             ),
        '2': pd.Series(
                 data  = np.array([2., 3., 4.]),
                 index = pd.RangeIndex(start=2, stop=5, step=1),
                 name  = '2',
                 dtype = float
             )
    }

    for k in forecaster.last_window_.keys():
        pd.testing.assert_series_equal(forecaster.last_window_[k], expected[k])


@pytest.mark.parametrize("encoding, encoding_mapping_", 
                         [('ordinal'         , {'1': 0, '2': 1}), 
                          ('ordinal_category', {'1': 0, '2': 1}),
                          ('onehot'          , {'1': 0, '2': 1}),
                          (None              , {'1': 0, '2': 1})], 
                         ids = lambda dt: f'encoding, mapping: {dt}')
def test_fit_encoding_mapping(encoding, encoding_mapping_):
    """
    Test the encoding mapping of _create_train_X_y.
    """
    series = {
        '1': pd.Series(np.arange(7, dtype=float)), 
        '2': pd.Series(np.arange(7, dtype=float))
    }
    
    forecaster = ForecasterRecursiveMultiSeries(
                     LinearRegression(),
                     lags     = 3,
                     encoding = encoding,
                 )
    forecaster.fit(series=series, suppress_warnings=True)
    
    assert forecaster.encoding_mapping_ == encoding_mapping_


def test_fit_resets_out_sample_residuals_on_refit():
    """
    Test that out_sample_residuals_ and out_sample_residuals_by_bin_ are reset
    to None when the forecaster is refitted.
    """
    series = {
        '1': pd.Series(np.arange(50, dtype=float)),
        '2': pd.Series(np.arange(50, dtype=float)),
    }
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding='ordinal'
    )
    forecaster.fit(series=series, suppress_warnings=True)
    forecaster.set_out_sample_residuals(
        y_true={'1': np.arange(1, 48, dtype=float), '2': np.arange(1, 48, dtype=float)},
        y_pred={'1': np.zeros(47), '2': np.zeros(47)},
    )

    assert forecaster.out_sample_residuals_ is not None
    assert forecaster.out_sample_residuals_by_bin_ is not None

    forecaster.fit(series=series, suppress_warnings=True)

    assert forecaster.out_sample_residuals_ is None
    assert forecaster.out_sample_residuals_by_bin_ is None


# ==============================================================================
# Tests: fit with categorical features and configure_estimator_categorical_features
# ==============================================================================
@pytest.mark.parametrize(
    "estimator, check_fn",
    [
        (
            CatBoostRegressor(
                iterations=10, random_seed=123, verbose=0,
                allow_writing_files=False
            ),
            None
        ),
        (
            LGBMRegressor(verbose=-1, random_state=123),
            None
        ),
        (
            XGBRegressor(random_state=123),
            lambda est, cat_idx, n_features: (
                est.get_params()['enable_categorical'] is True
                and est.get_params()['feature_types'] == [
                    'c' if i in cat_idx else 'q' for i in range(n_features)
                ]
            )
        ),
        (
            HistGradientBoostingRegressor(random_state=123),
            lambda est, cat_idx, n_features: (
                est.get_params()['categorical_features'] == cat_idx
            )
        ),
    ],
    ids=['CatBoostRegressor', 'LGBMRegressor', 'XGBRegressor', 'HistGradientBoostingRegressor']
)
def test_fit_configures_estimator_categorical_features(estimator, check_fn):
    """
    Test that fit correctly configures native categorical feature support
    for each supported estimator (CatBoostRegressor, LGBMRegressor,
    XGBRegressor, HistGradientBoostingRegressor).
    """
    series = {
        'l1': pd.Series(np.arange(20, dtype=float)),
        'l2': pd.Series(np.arange(20, dtype=float))
    }
    exog = pd.DataFrame({
        'exog_num': np.arange(100, 120, dtype=float),
        'exog_cat': pd.Categorical(range(20))
    })

    forecaster = ForecasterRecursiveMultiSeries(
        estimator=estimator, lags=3, encoding='ordinal_category',
        transformer_series=None, categorical_features='auto'
    )
    forecaster.fit(series=series, exog=exog, suppress_warnings=True)

    assert forecaster.is_fitted
    assert forecaster.categorical_features_names_in_ == ['exog_cat']
    assert 'exog_cat' in forecaster.X_train_features_names_out_

    if check_fn is not None:
        # With ordinal_category, _level_skforecast is also treated as categorical
        cat_idx = [
            forecaster.X_train_features_names_out_.index('exog_cat'),
            forecaster.X_train_features_names_out_.index('_level_skforecast')
        ]
        n_features = len(forecaster.X_train_features_names_out_)
        assert check_fn(forecaster.estimator, cat_idx, n_features)

    # fit_kwargs must not be mutated
    assert forecaster.fit_kwargs == {}


@pytest.mark.parametrize(
    "estimator, param_name, default_value",
    [
        (
            XGBRegressor(random_state=123),
            'feature_types',
            None
        ),
        (
            HistGradientBoostingRegressor(random_state=123),
            'categorical_features',
            'from_dtype'
        ),
    ],
    ids=['XGBRegressor', 'HistGradientBoostingRegressor']
)
def test_fit_resets_estimator_categorical_params_on_refit_without_categoricals(
    estimator, param_name, default_value
):
    """
    Test that fitting with categorical features and then refitting without
    categoricals resets the estimator's categorical parameters to their
    default values (XGBoost: feature_types=None,
    HistGradientBoosting: categorical_features='from_dtype').
    """
    series = {
        'l1': pd.Series(np.arange(20, dtype=float)),
        'l2': pd.Series(np.arange(20, dtype=float))
    }
    exog_with_cat = pd.DataFrame({
        'exog_num': np.arange(100, 120, dtype=float),
        'exog_cat': pd.Categorical(['a', 'b'] * 10)
    })
    exog_no_cat = pd.DataFrame({
        'exog_num': np.arange(100, 120, dtype=float)
    })

    forecaster = ForecasterRecursiveMultiSeries(
        estimator=estimator, lags=3, encoding='ordinal',
        transformer_series=None, categorical_features='auto'
    )

    # First fit — with categoricals
    forecaster.fit(series=series, exog=exog_with_cat, suppress_warnings=True)
    assert forecaster.categorical_features_names_in_ == ['exog_cat']

    # Second fit — without categoricals (auto detects no categories → [])
    forecaster.fit(series=series, exog=exog_no_cat, suppress_warnings=True)
    assert forecaster.categorical_features_names_in_ == []
    assert forecaster.estimator.get_params()[param_name] == default_value


@pytest.mark.parametrize(
    "estimator",
    [
        CatBoostRegressor(
            iterations=10, random_seed=123, verbose=0,
            allow_writing_files=False
        ),
        LGBMRegressor(verbose=-1, random_state=123),
        XGBRegressor(random_state=123),
        HistGradientBoostingRegressor(random_state=123),
    ],
    ids=['CatBoostRegressor', 'LGBMRegressor', 'XGBRegressor', 'HistGradientBoostingRegressor']
)
def test_fit_no_categoricals_with_supported_estimators(estimator):
    """
    Test that fit works correctly with supported estimators when
    categorical_features=None (no categorical encoding).
    """
    series = {
        'l1': pd.Series(np.arange(20, dtype=float)),
        'l2': pd.Series(np.arange(20, dtype=float))
    }
    exog = pd.DataFrame({
        'exog_num': np.arange(100, 120, dtype=float)
    })

    forecaster = ForecasterRecursiveMultiSeries(
        estimator=estimator, lags=3, encoding='ordinal',
        transformer_series=None, categorical_features=None
    )
    forecaster.fit(series=series, exog=exog, suppress_warnings=True)

    assert forecaster.is_fitted
    assert forecaster.categorical_features_names_in_ is None


@pytest.mark.parametrize(
    "estimator, check_fn",
    [
        (
            CatBoostRegressor(
                iterations=10, random_seed=123, verbose=0,
                allow_writing_files=False
            ),
            lambda model: model.get_cat_feature_indices() == []
        ),
        (
            LGBMRegressor(verbose=-1, random_state=123),
            lambda model: (
                model.booster_.params.get('categorical_column') is None
                and model.booster_.pandas_categorical is None
            )
        ),
        (
            XGBRegressor(random_state=123),
            lambda model: model.get_params()['feature_types'] is None
        ),
        (
            HistGradientBoostingRegressor(random_state=123),
            lambda model: (
                model.get_params()['categorical_features'] == 'from_dtype'
                and model.is_categorical_ is None
            )
        ),
    ],
    ids=['CatBoostRegressor', 'LGBMRegressor', 'XGBRegressor', 'HistGradientBoostingRegressor']
)
def test_fit_does_not_configure_categorical_features_when_estimator_is_Pipeline(
    estimator, check_fn
):
    """
    Test that fit does not configure native categorical features when the
    estimator is a Pipeline, whatever its last step is (CatBoostRegressor,
    LGBMRegressor, XGBRegressor, HistGradientBoostingRegressor): the previous
    steps may transform the columns, so the categorical exogenous variable and
    the series identifier of `encoding='ordinal_category'` reach the last step
    with their ordinal encoding, as numeric columns.
    """
    series = {
        'l1': pd.Series(np.arange(20, dtype=float)),
        'l2': pd.Series(np.arange(20, dtype=float))
    }
    exog = pd.DataFrame({
        'exog_num': np.arange(100, 120, dtype=float),
        'exog_cat': pd.Categorical(['a', 'b'] * 10)
    })

    forecaster = ForecasterRecursiveMultiSeries(
        estimator=make_pipeline(StandardScaler(), estimator),
        lags=3,
        encoding='ordinal_category',
        transformer_series=None,
        categorical_features='auto'
    )
    forecaster.fit(series=series, exog=exog, suppress_warnings=True)

    assert forecaster.is_fitted
    assert forecaster.categorical_features_names_in_ == ['exog_cat']
    assert 'exog_cat' in forecaster.X_train_features_names_out_
    assert '_level_skforecast' in forecaster.X_train_features_names_out_
    assert check_fn(forecaster.estimator[-1])

    # fit_kwargs must not be mutated
    assert forecaster.fit_kwargs == {}


@pytest.mark.parametrize(
    "exog_format",
    ['wide', 'dict'],
    ids=lambda exog_format: f'exog_format: {exog_format}'
)
def test_fit_and_predict_when_exog_index_is_descending(exog_format):
    """
    Test that an exog with the dates in descending order is used in fit and
    gives the same predictions as the exog sorted. Before, the exog was 
    empty after aligning and the forecaster was trained without it.
    """
    rng = np.random.default_rng(123)
    index = pd.date_range(start='2020-01-01', periods=30, freq='D')
    series = {
        'l1': pd.Series(rng.normal(size=30), index=index),
        'l2': pd.Series(rng.normal(size=30), index=index)
    }
    exog = pd.DataFrame(
        {'exog_1': rng.normal(size=35)},
        index=pd.date_range(start='2020-01-01', periods=35, freq='D')
    )
    exog_train = exog.iloc[:30]
    exog_pred = exog.iloc[30:]
    exog_train_descending = exog_train.iloc[::-1]
    exog_pred_descending = exog_pred.iloc[::-1]
    if exog_format == 'dict':
        exog_train = {'l1': exog_train, 'l2': exog_train}
        exog_pred = {'l1': exog_pred, 'l2': exog_pred}
        exog_train_descending = {
            'l1': exog_train_descending, 'l2': exog_train_descending
        }
        exog_pred_descending = {
            'l1': exog_pred_descending, 'l2': exog_pred_descending
        }

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, exog=exog_train, suppress_warnings=True)
    expected = forecaster.predict(steps=5, exog=exog_pred, suppress_warnings=True)

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, exog=exog_train_descending, suppress_warnings=True)
    results = forecaster.predict(
        steps=5, exog=exog_pred_descending, suppress_warnings=True
    )

    assert forecaster.exog_in_
    assert forecaster.exog_names_in_ == ['exog_1']
    pd.testing.assert_frame_equal(results, expected)


@pytest.mark.parametrize(
    "estimator",
    [
        CatBoostRegressor(iterations=2, verbose=0, allow_writing_files=False),
        make_pipeline(
            StandardScaler(),
            CatBoostRegressor(iterations=2, verbose=0, allow_writing_files=False)
        ),
    ],
    ids=["CatBoostRegressor", "Pipeline"]
)
def test_fit_replaces_fitted_CatBoost_estimator_with_unfitted_clone(estimator):
    """
    Test that the first fit keeps the estimator object and that a CatBoost
    estimator that is already fitted, also as the last step of a Pipeline or in
    a forecaster copied with pickle, is replaced with a clone before it is
    fitted again. No categorical features are used: CatBoost can block when a
    fitted model with categorical features is fitted again.
    """
    forecaster = ForecasterRecursiveMultiSeries(estimator=estimator, lags=3)
    estimator_new = forecaster.estimator
    forecaster.fit(series=series_wide_range)
    estimator_first_fit = forecaster.estimator
    forecaster_copy = pickle.loads(pickle.dumps(forecaster))
    estimator_copy = forecaster_copy.estimator
    forecaster.fit(series=series_wide_range)
    forecaster_copy.fit(series=series_wide_range)

    assert estimator_first_fit is estimator_new
    assert forecaster.estimator is not estimator_first_fit
    assert forecaster_copy.estimator is not estimator_copy
    assert forecaster.is_fitted
    assert forecaster_copy.is_fitted


@pytest.mark.parametrize(
    "estimator",
    [LinearRegression(), LGBMRegressor(n_estimators=2, verbose=-1)],
    ids=lambda estimator: f"estimator: {type(estimator).__name__}"
)
def test_fit_keeps_estimator_object_when_fitted_again_and_not_CatBoost(estimator):
    """
    Test that an estimator that is not a CatBoost model is fitted again without
    being replaced with a clone.
    """
    forecaster = ForecasterRecursiveMultiSeries(estimator=estimator, lags=3)
    estimator_new = forecaster.estimator
    forecaster.fit(series=series_wide_range)
    forecaster.fit(series=series_wide_range)

    assert forecaster.estimator is estimator_new


def test_fit_predict_output_when_CatBoost_with_categorical_features_fitted_again():
    """
    Test that a forecaster with a CatBoost estimator and categorical features
    that is fitted again, the same object or a copy made with pickle, predicts
    as a new forecaster fitted with the same data. With categorical features,
    CatBoost does not give the model of a new object when a fitted model is
    fitted again, and `predict` can block forever with a copy, so the fits run
    in a subprocess with a time limit.
    """
    code = textwrap.dedent("""
        import json
        import pickle
        import numpy as np
        import pandas as pd
        from catboost import CatBoostRegressor
        from skforecast.recursive import ForecasterRecursiveMultiSeries

        rng = np.random.RandomState(7)
        n = 260
        index = pd.date_range("2020-01-01", periods=n, freq="D")
        codes = rng.permutation(np.arange(n) % 6)
        effect = np.array([5.0, -3.0, 9.0, 0.0, -8.0, 2.0])
        l1 = effect[codes] + rng.normal(scale=0.3, size=n)
        exog = pd.DataFrame(
            {"c": pd.Categorical(codes), "x": rng.normal(size=n)}, index=index
        )
        l2 = 0.5 * effect[codes] + rng.normal(scale=0.3, size=n)
        series = pd.DataFrame({"l1": l1, "l2": l2}, index=index)

        def create_forecaster():
            return ForecasterRecursiveMultiSeries(
                estimator=CatBoostRegressor(
                    iterations=30, depth=3, verbose=0, thread_count=1,
                    random_seed=1, allow_writing_files=False
                ),
                lags=3,
                encoding="ordinal"
            )

        forecaster = create_forecaster()
        forecaster.fit(series=series.iloc[:200], exog=exog.iloc[:200])
        forecasters = {
            "new": create_forecaster(),
            "pickle": pickle.loads(pickle.dumps(forecaster)),
            "same": forecaster
        }
        predictions = {}
        for name, forecaster in forecasters.items():
            forecaster.fit(series=series.iloc[50:250], exog=exog.iloc[50:250])
            predictions[name] = forecaster.predict(
                steps=5, exog=exog.iloc[250:255]
            )["pred"].to_list()
        print(json.dumps(predictions))
    """)
    repo_root = Path(__file__).resolve().parents[4]

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output = True,
        encoding       = "utf-8",
        errors         = "replace",
        cwd            = repo_root,
        env            = {**os.environ, "PYTHONPATH": str(repo_root)},
        timeout        = 120
    )

    assert result.returncode == 0, result.stderr
    predictions = json.loads(result.stdout.splitlines()[-1])

    # NOTE: CatBoost results with categorical features change between platforms
    # (Linux, macOS), so the expected values are not hardcoded. They are the
    # predictions of a new forecaster fitted with the same data in the subprocess.
    expected = np.array(predictions["new"])

    assert expected.shape == (10,)
    np.testing.assert_array_almost_equal(predictions["same"], expected)
    np.testing.assert_array_almost_equal(predictions["pickle"], expected)


# ==============================================================================
# Tests: the estimator receives in fit the same values as in predict (numpy)
# ==============================================================================
exog_category = pd.DataFrame(
    {'exog_1': exog_wide_range['exog_1'].to_numpy(),
     'exog_cat': pd.Categorical([10, 20, 30, 40, 50] * 10),
     'exog_cat_2': pd.Categorical([3, 1, 2, 0, 4] * 10)}
)


def spy_fit(monkeypatch, estimator_class, error=None):
    """
    Record the training matrix, the keyword arguments and the params of the
    estimator in each call to the `fit` method of `estimator_class`. If
    `error` is given, it is raised instead of fitting. It must be called
    after creating the forecaster, which inspects the signature of `fit`.
    """
    calls = []
    original_fit = estimator_class.fit

    def fit(self, X, y, **kwargs):
        calls.append({'X': X, 'kwargs': kwargs, 'params': self.get_params()})
        if error is not None:
            raise error
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(estimator_class, 'fit', fit)

    return calls


@pytest.mark.parametrize(
    "encoding, n_features",
    [('ordinal', 7), ('ordinal_category', 7), ('onehot', 8), (None, 6)],
    ids=lambda value: f'{value}'
)
def test_fit_estimator_receives_float_numpy_array(monkeypatch, encoding, n_features):
    """
    Test that the estimator is fitted with a float64 numpy array, as in
    `predict`, where a `category` column not included in
    `categorical_features` has its values, not its codes.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator            = LinearRegression(),
                     lags                 = 3,
                     encoding             = encoding,
                     categorical_features = ['exog_cat_2']
                 )
    calls = spy_fit(monkeypatch, LinearRegression)
    forecaster.fit(
        series=series_wide_range, exog=exog_category, suppress_warnings=True
    )
    X = calls[0]['X']
    exog_cat_position = forecaster.X_train_features_names_out_.index('exog_cat')

    assert isinstance(X, np.ndarray)
    assert X.dtype == np.float64
    assert X.shape == (94, n_features)
    np.testing.assert_array_equal(
        X[:5, exog_cat_position], np.array([40., 50., 10., 20., 30.])
    )
    assert not hasattr(forecaster.estimator, 'feature_names_in_')


def test_fit_estimator_receives_NaN_and_0_1_when_nullable_and_bool_exog(monkeypatch):
    """
    Test that the missing values of a nullable column reach the estimator as
    NaN and the booleans as 0 and 1.
    """
    exog = pd.DataFrame(
        {'exog_1': exog_wide_range['exog_1'].to_numpy(),
         'exog_Int64': pd.array([1, None, 3, 4, 5] * 10, dtype='Int64'),
         'exog_bool': [True, False, True, True, False] * 10}
    )
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = HistGradientBoostingRegressor(max_iter=5),
                     lags      = 3,
                     encoding  = 'ordinal'
                 )
    calls = spy_fit(monkeypatch, HistGradientBoostingRegressor)
    forecaster.fit(series=series_wide_range, exog=exog, suppress_warnings=True)
    X = calls[0]['X']

    assert isinstance(X, np.ndarray)
    assert X.dtype == np.float64
    np.testing.assert_array_equal(
        X[:5, 5], np.array([4., 5., 1., np.nan, 3.])
    )
    np.testing.assert_array_equal(X[:5, 6], np.array([1., 0., 1., 0., 1.]))


def test_fit_CatBoost_receives_DataFrame_with_integer_categorical_columns(monkeypatch):
    """
    Test that a CatBoost model with categorical features is fitted with a
    DataFrame with the names of the features, where the categorical columns
    (the encoded exogenous variable and the series identifier) are integers.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = CatBoostRegressor(
                                     iterations=5, random_seed=123, verbose=0,
                                     allow_writing_files=False
                                 ),
                     lags      = 3,
                     encoding  = 'ordinal_category'
                 )
    calls = spy_fit(monkeypatch, CatBoostRegressor)
    forecaster.fit(
        series=series_wide_range, exog=exog_wide_range, suppress_warnings=True
    )
    X = calls[0]['X']

    assert isinstance(X, pd.DataFrame)
    assert X.columns.to_list() == [
        'lag_1', 'lag_2', 'lag_3', '_level_skforecast', 'exog_1', 'exog_2'
    ]
    assert calls[0]['kwargs']['cat_features'] == [5, 3]
    assert X.dtypes.astype(str).to_list() == [
        'float64', 'float64', 'float64', 'int64', 'float64', 'int64'
    ]
    assert X['_level_skforecast'].to_list() == [0] * 47 + [1] * 47
    assert forecaster.estimator.get_cat_feature_indices() == [3, 5]


@pytest.mark.parametrize(
    "categorical_features",
    [None, ['exog_cat_2']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_fit_DataTypeWarning_when_category_exog_not_in_categorical_features(
    categorical_features
):
    """
    Test DataTypeWarning is issued when an exogenous variable of dtype
    category is not included in `categorical_features`.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator            = LinearRegression(),
                     lags                 = 3,
                     categorical_features = categorical_features
                 )
    not_managed = ['exog_cat', 'exog_cat_2'] if categorical_features is None else ['exog_cat']

    warn_msg = re.escape(
        f"The exogenous variables {not_managed} have dtype `category` but "
        f"are not included in `categorical_features`, so the forecaster does "
        f"not encode them: the estimator receives their values, not their "
        f"category codes, as numeric columns. To treat them as categorical "
        f"features, include them in `categorical_features` or use "
        f"`categorical_features='auto'`."
    )
    with pytest.warns(DataTypeWarning, match=warn_msg):
        forecaster.fit(series=series_wide_range, exog=exog_category)


@pytest.mark.parametrize(
    "estimator, categorical_features, fit_kwargs",
    [(LinearRegression(), 'auto', None),
     (LGBMRegressor(verbose=-1, n_estimators=5), None,
      {'categorical_feature': ['exog_cat', 'exog_cat_2']}),
     (LGBMRegressor(verbose=-1, n_estimators=5), None,
      {'categorical_feature': [5, 6]}),
     (HistGradientBoostingRegressor(
         max_iter=5, categorical_features=['exog_cat', 'exog_cat_2']
      ), None, None),
     (XGBRegressor(
         n_estimators=5, enable_categorical=True,
         feature_types=['q', 'q', 'q', 'q', 'q', 'c', 'c']
      ), None, None)],
    ids=['auto', 'LGBMRegressor by name', 'LGBMRegressor by index',
         'HistGradientBoostingRegressor by name', 'XGBRegressor feature_types']
)
def test_fit_no_DataTypeWarning_when_category_exog_is_managed(
    estimator, categorical_features, fit_kwargs
):
    """
    Test that no DataTypeWarning is issued when the exogenous variables of
    dtype category are encoded by the forecaster, or the user has configured
    them as categorical features on the estimator.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator            = estimator,
                     lags                 = 3,
                     categorical_features = categorical_features,
                     fit_kwargs           = fit_kwargs
                 )
    exog = exog_category.assign(
        exog_cat=pd.Categorical([1, 2, 3, 4, 0] * 10)
    )

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        forecaster.fit(series=series_wide_range, exog=exog)

    assert not [w for w in record if issubclass(w.category, DataTypeWarning)]


@pytest.mark.parametrize(
    "column, dtype_repr",
    [(pd.Categorical(['a', 'b', 'c', 'd', 'e'] * 10), 'category'),
     (pd.Categorical([0.5, 1.5, 2.5, 3.5, 4.5] * 10), 'category'),
     (pd.Categorical(['1', '2', '3', '4', '5'] * 10), 'category')],
    ids=['text', 'decimals', 'numeric strings']
)
def test_fit_TypeError_when_category_exog_without_integers_not_in_categorical_features(
    column, dtype_repr
):
    """
    Test TypeError is raised in `fit`, with the name of the column, when an
    exogenous variable of dtype category whose categories are not integers is
    not included in the list of `categorical_features`. The DataTypeWarning of
    the columns passed as numeric is not issued for it before the error.
    """
    exog = exog_category.assign(exog_cat=column)
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator            = LinearRegression(),
                     lags                 = 3,
                     categorical_features = ['exog_cat_2']
                 )

    err_msg = re.escape(
        f"The following exogenous variables cannot be passed to the estimator "
        f"as numeric values: 'exog_cat' ({dtype_repr}). Include them in "
        f"`categorical_features`, or use `categorical_features='auto'`, so "
        f"that the forecaster encodes them. A column of dtype `category` that "
        f"is not included in `categorical_features` must have integer "
        f"categories."
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match=err_msg):
            forecaster.fit(series=series_wide_range, exog=exog)

    assert not [w for w in record if issubclass(w.category, DataTypeWarning)]


@pytest.mark.parametrize(
    "estimator, estimator_class, fit_kwargs, expected_fit, expected_after_fit",
    [(LGBMRegressor(verbose=-1, n_estimators=5), LGBMRegressor, None,
      lambda call, lags: call['kwargs']['categorical_feature'] == [lags],
      lambda est: True),
     (LGBMRegressor(verbose=-1, n_estimators=5), LGBMRegressor,
      {'categorical_feature': ['exog_cat']},
      lambda call, lags: call['kwargs']['categorical_feature'] == [lags + 2, lags],
      lambda est: True),
     (CatBoostRegressor(iterations=5, verbose=0, allow_writing_files=False),
      CatBoostRegressor, {'cat_features': ['exog_cat']},
      lambda call, lags: call['kwargs']['cat_features'] == [lags + 2, lags],
      lambda est: True),
     (XGBRegressor(n_estimators=5), XGBRegressor, None,
      lambda call, lags: (
          call['params']['feature_types']
          == ['q'] * lags + ['c'] + ['q'] * 2
          and call['params']['enable_categorical'] is True
      ),
      lambda est: est.get_params()['feature_types'] is None),
     (HistGradientBoostingRegressor(max_iter=5, categorical_features=['exog_cat']),
      HistGradientBoostingRegressor, None,
      lambda call, lags: call['params']['categorical_features'] == [lags, lags + 2],
      lambda est: est.get_params()['categorical_features'] == ['exog_cat'])],
    ids=['LGBMRegressor', 'LGBMRegressor with names', 'CatBoostRegressor with names',
         'XGBRegressor', 'HistGradientBoostingRegressor with names']
)
def test_fit_adds_level_to_user_categorical_config_when_ordinal_category_and_None(
    monkeypatch, estimator, estimator_class, fit_kwargs, expected_fit,
    expected_after_fit
):
    """
    Test that, with `encoding='ordinal_category'` and
    `categorical_features=None`, the series identifier is added to the
    categorical features the user has configured (given by name or by index),
    with the positions of the current matrix after `set_lags`. The params of
    the estimator are those of the user after each fit, and `fit_kwargs` of
    the forecaster is not modified.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator            = estimator,
                     lags                 = 3,
                     encoding             = 'ordinal_category',
                     categorical_features = None,
                     fit_kwargs           = fit_kwargs
                 )
    calls = spy_fit(monkeypatch, estimator_class)
    exog = exog_category.drop(columns='exog_cat_2')

    forecaster.fit(series=series_wide_range, exog=exog, suppress_warnings=True)
    assert expected_fit(calls[-1], 3)
    assert expected_after_fit(forecaster.estimator)

    forecaster.set_lags(5)
    forecaster.fit(series=series_wide_range, exog=exog, suppress_warnings=True)
    assert expected_fit(calls[-1], 5)
    assert expected_after_fit(forecaster.estimator)
    assert forecaster.fit_kwargs == (fit_kwargs if fit_kwargs is not None else {})


def test_fit_restores_estimator_categorical_params_when_fit_raises(monkeypatch):
    """
    Test that the categorical params of the estimator are those of the user
    when its `fit` method raises an error.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator            = HistGradientBoostingRegressor(
                                                categorical_features=['exog_cat']
                                            ),
                     lags                 = 3,
                     encoding             = 'ordinal_category',
                     categorical_features = None
                 )
    calls = spy_fit(
        monkeypatch, HistGradientBoostingRegressor, error=RuntimeError('fit failed')
    )

    with pytest.raises(RuntimeError, match='fit failed'):
        forecaster.fit(
            series=series_wide_range, exog=exog_category, suppress_warnings=True
        )

    assert calls[0]['params']['categorical_features'] == [3, 5]
    assert forecaster.estimator.get_params()['categorical_features'] == ['exog_cat']


@pytest.mark.parametrize(
    "cat_features",
    [(3, 5), ('_level_skforecast', 'exog_cat')],
    ids=lambda cf: f'cat_features: {cf}'
)
def test_fit_when_CatBoost_cat_features_in_constructor_include_level(cat_features):
    """
    Test that a CatBoost model with `cat_features` set in its constructor (a
    tuple) is fitted without changing them when they include the series
    identifier of `encoding='ordinal_category'`, and that its predictions are
    those of the same column passed as integer. The series depend on the
    category, so that the estimator uses the column.
    """
    rng = np.random.default_rng(123)
    cat_values = np.array([10, 20, 30, 40, 50] * 10)
    series = pd.DataFrame(
        {'l1': rng.normal(size=50) + (cat_values - 10) / 5,
         'l2': rng.normal(size=50) + 2 * (cat_values - 10) / 5}
    )
    exog_cat = pd.DataFrame(
        {'exog_1': rng.normal(size=50),
         'exog_cat': pd.Categorical(cat_values)}
    )
    exog_int = exog_cat.astype({'exog_cat': int})

    predictions = []
    for exog in (exog_cat, exog_int):
        forecaster = ForecasterRecursiveMultiSeries(
                         estimator            = CatBoostRegressor(
                                                    iterations=20, random_seed=123,
                                                    verbose=0,
                                                    allow_writing_files=False,
                                                    cat_features=cat_features
                                                ),
                         lags                 = 3,
                         encoding             = 'ordinal_category',
                         categorical_features = None
                     )
        forecaster.fit(series=series, exog=exog, suppress_warnings=True)
        predictions.append(
            forecaster.predict(
                steps             = 5,
                exog              = exog.set_axis(pd.RangeIndex(50, 100)),
                suppress_warnings = True
            )
        )

        assert forecaster.estimator.get_params()['cat_features'] == cat_features
        assert forecaster.estimator.get_cat_feature_indices() == [3, 5]
        assert forecaster.estimator.get_feature_importance()[5] > 0

    pd.testing.assert_frame_equal(predictions[0], predictions[1])


def test_fit_ValueError_when_CatBoost_cat_features_in_constructor_without_level():
    """
    Test ValueError is raised when `cat_features` is set in the constructor of
    CatBoost without the series identifier, which has to be added with
    `encoding='ordinal_category'` and `categorical_features=None`.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator            = CatBoostRegressor(
                                                iterations=5, verbose=0,
                                                allow_writing_files=False,
                                                cat_features=(5,)
                                            ),
                     lags                 = 3,
                     encoding             = 'ordinal_category',
                     categorical_features = None
                 )

    err_msg = re.escape(
        "With `encoding='ordinal_category'` and `categorical_features=None`, "
        "the series identifier (`_level_skforecast`) is added to the "
        "categorical features of the estimator, which is not possible when "
        "`cat_features` is set in the constructor of the CatBoost estimator. "
        "Pass them with `fit_kwargs={'cat_features': [...]}` instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.fit(
            series            = series_wide_range,
            exog              = exog_category.drop(columns='exog_cat_2'),
            suppress_warnings = True
        )


@pytest.mark.parametrize(
    "estimator",
    [LGBMRegressor(verbose=-1, n_estimators=5), XGBRegressor(n_estimators=5)],
    ids=['LGBMRegressor', 'XGBRegressor']
)
def test_fit_and_predict_when_names_have_characters_not_allowed_in_feature_names(
    estimator
):
    """
    Test that the forecaster is fitted when the names of the series (columns
    of the training matrix with `encoding='onehot'`) or of the exogenous
    variables have characters that LightGBM and XGBoost do not allow in
    feature names, since the estimator receives a numpy array.
    """
    series = series_wide_range.rename(columns={'1': 'store[1]', '2': 'store[2]'})
    exog = pd.DataFrame({'temp[C]': exog_wide_range['exog_1'].to_numpy()})
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = estimator,
                     lags      = 3,
                     encoding  = 'onehot'
                 )
    forecaster.fit(series=series, exog=exog, suppress_warnings=True)
    predictions = forecaster.predict(
        steps=3, exog=exog.set_axis(pd.RangeIndex(50, 100)), suppress_warnings=True
    )

    assert forecaster.X_train_features_names_out_ == [
        'lag_1', 'lag_2', 'lag_3', 'store[1]', 'store[2]', 'temp[C]'
    ]
    assert predictions['level'].to_list() == ['store[1]', 'store[2]'] * 3
    assert predictions['pred'].notna().all()


@pytest.mark.parametrize(
    "encoding, expected_categorical_feature",
    [('ordinal', [5]), ('ordinal_category', [5, 3])],
    ids=lambda value: f'{value}'
)
def test_fit_when_LGBMRegressor_categorical_feature_by_name_in_constructor(
    monkeypatch, encoding, expected_categorical_feature
):
    """
    Test that the categorical features set by name in the constructor of
    LightGBM are passed to `fit` by position when `categorical_features=None`,
    since the estimator is fitted with a numpy array, and that the series
    identifier of `encoding='ordinal_category'` is added to them. No
    DataTypeWarning is issued for the column, and the predictions are those of
    the same configuration given by position in `fit_kwargs`.
    """
    exog = exog_category.drop(columns='exog_cat_2')
    exog_predict = exog.set_axis(pd.RangeIndex(50, 100))
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator            = LGBMRegressor(
                                                verbose=-1, n_estimators=5,
                                                min_child_samples=5,
                                                categorical_feature=['exog_cat']
                                            ),
                     lags                 = 3,
                     encoding             = encoding,
                     categorical_features = None
                 )
    forecaster_2 = ForecasterRecursiveMultiSeries(
                       estimator            = LGBMRegressor(
                                                  verbose=-1, n_estimators=5,
                                                  min_child_samples=5
                                              ),
                       lags                 = 3,
                       encoding             = encoding,
                       categorical_features = None,
                       fit_kwargs           = {'categorical_feature': [5]}
                   )
    calls = spy_fit(monkeypatch, LGBMRegressor)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        forecaster.fit(series=series_wide_range, exog=exog)
    predictions = forecaster.predict(steps=3, exog=exog_predict)

    forecaster_2.fit(series=series_wide_range, exog=exog, suppress_warnings=True)
    predictions_2 = forecaster_2.predict(steps=3, exog=exog_predict)

    assert calls[0]['kwargs']['categorical_feature'] == expected_categorical_feature
    assert not [w for w in record if issubclass(w.category, DataTypeWarning)]
    assert forecaster.estimator.get_params()['categorical_feature'] == ['exog_cat']
    pd.testing.assert_frame_equal(predictions, predictions_2)
