# Unit test _evaluate_grid_hyperparameters_multiseries
# ==============================================================================
import re
import os
import warnings
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from skforecast.exceptions import (
    IgnoredArgumentWarning,
    MissingValuesWarning,
    OneStepAheadValidationWarning
)
from skforecast.metrics import mean_absolute_scaled_error
from skforecast.metrics import root_mean_squared_scaled_error
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.model_selection import backtesting_forecaster_multiseries
from skforecast.model_selection._search import _evaluate_grid_hyperparameters_multiseries
from skforecast.model_selection._split import TimeSeriesFold, OneStepAheadFold

# Fixtures
from ..fixtures_model_selection_multiseries import (
    series_wide_range,
    series_wide_dt,
    series_long_dt,
    series_dict_range,
    series_dict_dt,
    series_dict_nans,
    exog,
    exog_dict_nans
)

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # hide progress bar

pytestmark = [
    pytest.mark.filterwarnings("ignore::skforecast.exceptions.MissingExogWarning"),
    pytest.mark.filterwarnings("ignore::skforecast.exceptions.MissingValuesWarning"),
]

# NOTE: `short` has 20 values, 5 of them in the training set of
# `initial_train_size=45`, so the candidates with more than 4 lags do not
# predict it and evaluate fewer points than the others.
index_short = pd.date_range('2020-01-01', periods=60, freq='D')
series_short = {
    'l1': pd.Series(10 + np.sin(np.arange(60)), index=index_short),
    'l2': pd.Series(20 + 2 * np.cos(np.arange(60) / 2), index=index_short),
    'short': pd.Series(5 + np.arange(20) % 3, index=index_short[-20:], dtype=float),
}
# NOTE: `short` has 6 values, 5 of them in the training set of
# `initial_train_size=45` and 1 in the test set, so the candidates with more
# than 4 lags do not predict it, but evaluate more than 95% of the points of
# the others (30 of 31): the warning of the levels not predicted is shown, the
# warning of the points evaluated is not.
series_short_one_point = {
    'l1': series_short['l1'],
    'l2': series_short['l2'],
    'short': pd.Series(5 + np.arange(6) % 3, index=index_short[40:46], dtype=float),
}
N_EVALUATED_WARNING_MSG = "evaluated on fewer points than the most complete one"


def test_ValueError_evaluate_grid_hyperparameters_multiseries_when_return_best_and_len_series_exog_different():
    """
    Test ValueError is raised in _evaluate_grid_hyperparameters_multiseries when 
    `return_best = True` and length of `series` and `exog` do not match.
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator = Ridge(random_state=123),
                     level     = 'l1',
                     lags      = 2,
                     steps     = 3
                 )
    exog = series_wide_range['l1'].copy().iloc[:30].rename('exog').copy()

    cv = TimeSeriesFold(
            initial_train_size = 12,
            steps              = 4,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )

    err_msg = re.escape(
        f"`exog` must have same number of samples as `series`. "
        f"length `exog`: ({len(exog)}), length `series`: ({len(series_wide_range)})"
    )
    with pytest.raises(ValueError, match = err_msg):
        _evaluate_grid_hyperparameters_multiseries(
            forecaster         = forecaster,
            series             = series_wide_range,
            exog               = exog,
            cv                 = cv,
            param_grid         = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            metric             = 'mean_absolute_error',
            levels             = None,
            lags_grid          = [2, 4],
            return_best        = True,
            verbose            = False
        )


def test_TypeError_evaluate_grid_hyperparameters_multiseries_when_cv_not_valid():
    """
    Test TypeError is raised in _evaluate_grid_hyperparameters_multiseries when 
    cv is not a valid splitter.
    """
    class DummyCV:
        pass

    cv = DummyCV()
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = Ridge(random_state=123),
                     lags      = 3,
                     encoding  = 'onehot'
                 )

    err_msg = re.escape(
        f"`cv` must be an instance of `TimeSeriesFold` or `OneStepAheadFold`. "
        f"Got {type(cv)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        _evaluate_grid_hyperparameters_multiseries(
            forecaster         = forecaster,
            series             = series_dict_range,
            cv                 = cv,
            param_grid         = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            metric             = 'mean_absolute_error',
            levels             = None,
            lags_grid          = [2, 4],
            return_best        = True,
            verbose            = False
        )


def test_ValueError_evaluate_grid_hyperparameters_multiseries_when_not_allowed_aggregate_metric():
    """
    Test ValueError is raised in _evaluate_grid_hyperparameters_multiseries when 
    `aggregate_metric` has not a valid value.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = Ridge(random_state=123),
                     lags      = 3,
                     encoding  = 'onehot'
                 )
    cv = TimeSeriesFold(
            initial_train_size = 12,
            steps              = 4,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )

    err_msg = re.escape(
        "Allowed `aggregate_metric` are: ['average', 'weighted_average', 'pooling']. "
        "Got: ['not_valid']."
    )
    with pytest.raises(ValueError, match = err_msg):
        _evaluate_grid_hyperparameters_multiseries(
            forecaster         = forecaster,
            series             = series_dict_range,
            cv                 = cv,
            param_grid         = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            metric             = 'mean_absolute_error',
            aggregate_metric   = 'not_valid',
            levels             = None,
            lags_grid          = [2, 4],
            return_best        = True,
            verbose            = False
        )


def test_evaluate_grid_hyperparameters_multiseries_exception_when_metric_list_duplicate_names():
    """
    Test exception is raised in _evaluate_grid_hyperparameters when a `list` of 
    metrics is used with duplicate names.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = Ridge(random_state=123),
                     lags      = 3,
                     encoding  = 'onehot'
                 )
    cv = TimeSeriesFold(
            initial_train_size = 12,
            steps              = 4,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )
    
    err_msg = re.escape("When `metric` is a `list`, each metric name must be unique.")
    with pytest.raises(ValueError, match = err_msg):
        _evaluate_grid_hyperparameters_multiseries(
            forecaster         = forecaster,
            series             = series_dict_range,
            cv                 = cv,
            param_grid         = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            metric             = ['mean_absolute_error', mean_absolute_error],
            levels             = ['l1'],
            exog               = None,
            lags_grid          = [2, 4],
            return_best        = False,
            verbose            = False
        )


def test_evaluate_grid_hyperparameters_multiseries_ValueError_when_metric_named_n_evaluated():
    """
    Test ValueError is raised in _evaluate_grid_hyperparameters_multiseries,
    before evaluating any candidate, when a metric is named `n_evaluated`,
    the column of the backtesting metrics with the number of points evaluated.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = Ridge(random_state=123),
                     lags      = 3,
                     encoding  = 'onehot'
                 )
    cv = TimeSeriesFold(initial_train_size=12, steps=4, refit=False)

    def n_evaluated(y_true, y_pred):  # pragma: no cover
        return mean_absolute_error(y_true, y_pred)

    err_msg = re.escape(
        "The backtesting metrics include the column 'n_evaluated' with the "
        "number of points evaluated. Rename the metric 'n_evaluated' passed in "
        "`metric`."
    )
    with pytest.raises(ValueError, match = err_msg):
        _evaluate_grid_hyperparameters_multiseries(
            forecaster         = forecaster,
            series             = series_dict_range,
            cv                 = cv,
            param_grid         = [{'alpha': 0.01}, {'alpha': 0.1}],
            metric             = ['mean_absolute_error', n_evaluated],
            lags_grid          = [2, 4],
            return_best        = False,
            verbose            = False
        )


# ForecasterRecursiveMultiSeries
# ======================================================================================================================
def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterRecursiveMultiSeries_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterRecursiveMultiSeries 
    with mocked (mocked done in Skforecast v0.5.0).
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = Ridge(random_state=123),
                     lags               = 2, 
                     encoding           = 'onehot',
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_range['l1']) - 12,
            steps              = 3,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series_dict_range,
                  param_grid         = param_grid,
                  cv                 = cv,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  return_best        = False,
                  verbose            = False
              )

    expected_results = pd.DataFrame(
        {
            "levels": [["l1", "l2"]] * 6,
            "lags": [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2], [1, 2], [1, 2]],
            "lags_label": [
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2],
                [1, 2],
                [1, 2],
            ],
            "params": [
                {"alpha": 0.01},
                {"alpha": 0.1},
                {"alpha": 1},
                {"alpha": 1},
                {"alpha": 0.1},
                {"alpha": 0.01},
            ],
            "mean_absolute_error__weighted_average": np.array(
                [
                    0.20968100463227382,
                    0.20969259779858337,
                    0.20977945312386406,
                    0.21077344827205086,
                    0.21078653113227208,
                    0.21078779824759553,
                ]
            ),
            "n_evaluated": [24] * 6,
            "alpha": np.array([0.01, 0.1, 1.0, 1.0, 0.1, 0.01]),
        },
        index=pd.Index([0, 1, 2, 3, 4, 5], dtype="int64"),
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterRecursiveMultiSeries_with_window_features():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in 
    ForecasterRecursiveMultiSeries with window features 
    (mocked done in Skforecast v0.14.0).
    """
    window_features = RollingFeatures(
        stats=['mean', 'std', 'min', 'max', 'sum', 'median', 'ratio_min_max', 'coef_variation'],
        window_sizes=3,
    )
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = Ridge(random_state=123),
                     lags               = 2, 
                     window_features    = window_features,
                     encoding           = 'onehot',
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_range['l1']) - 12,
            steps              = 3,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series_dict_range,
                  param_grid         = param_grid,
                  cv                 = cv,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  return_best        = False,
                  verbose            = False
              )

    expected_results = pd.DataFrame(
        {'levels': [['l1', 'l2'],
                    ['l1', 'l2'],
                    ['l1', 'l2'],
                    ['l1', 'l2'],
                    ['l1', 'l2'],
                    ['l1', 'l2']],
         'lags': [np.array([1, 2, 3, 4]),
                    np.array([1, 2, 3, 4]),
                    np.array([1, 2]),
                    np.array([1, 2]),
                    np.array([1, 2, 3, 4]),
                    np.array([1, 2])],
         'lags_label': [np.array([1, 2, 3, 4]),
                    np.array([1, 2, 3, 4]),
                    np.array([1, 2]),
                    np.array([1, 2]),
                    np.array([1, 2, 3, 4]),
                    np.array([1, 2])],
         'params': [{'alpha': 1},
                    {'alpha': 0.1},
                    {'alpha': 1},
                    {'alpha': 0.1},
                    {'alpha': 0.01},
                    {'alpha': 0.01}],
         'mean_absolute_error__weighted_average': [0.2146029752649913,
                    0.2155171009179464,
                    0.21574815856300625,
                    0.21677673611895729,
                    0.22004258621311923,
                    0.22107598684517502],
         'n_evaluated': [24] * 6,
         'alpha': [1.0, 0.1, 1.0, 0.1, 0.01, 0.01]}
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterRecursiveMultiSeries_series_and_exog_dict_nans_with_window_features():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in 
    ForecasterRecursiveMultiSeries when series and exog are dicts with window features 
    (mocked done in Skforecast v0.14.0).
    """
    window_features = RollingFeatures(
        stats=['mean', 'std', 'min', 'max', 'sum', 'median', 'ratio_min_max', 'coef_variation'],
        window_sizes=3,
    )
    estimator = LGBMRegressor(
        n_estimators=10, random_state=123, verbose=-1, max_depth=3
    )
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = estimator,
                     lags               = 2, 
                     window_features    = window_features,
                     encoding           = 'onehot',
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = 38,
            steps              = 3,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )
    lags_grid = [2, 4]
    param_grid = [
        {'n_estimators': 15, 'max_depth': 5},
        {'n_estimators': 10, 'max_depth': 3}
    ]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series_dict_nans,
                  exog               = exog_dict_nans,
                  param_grid         = param_grid,
                  cv                 = cv,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  levels             = None,
                  lags_grid          = lags_grid,
                  return_best        = False,
                  verbose            = False
              )

    expected_results = pd.DataFrame({
        'levels': [['id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004'],
                   ['id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004'],
                   ['id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004'],
                   ['id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004']],
        'lags': [np.array([1, 2, 3, 4]),
                 np.array([1, 2]),
                 np.array([1, 2, 3, 4]),
                 np.array([1, 2])],
        'lags_label': [np.array([1, 2, 3, 4]),
                       np.array([1, 2]),
                       np.array([1, 2, 3, 4]),
                       np.array([1, 2])],
        'params': [{'n_estimators': 15, 'max_depth': 5},
                   {'n_estimators': 15, 'max_depth': 5},
                   {'n_estimators': 10, 'max_depth': 3},
                   {'n_estimators': 10, 'max_depth': 3}],
        'mean_absolute_error__weighted_average': [663.557949349482,
                                                  673.8431369197542,
                                                  761.8926543280367,
                                                  775.090407239743],
        'n_evaluated': [654] * 4,
        'n_estimators': [15, 15, 10, 10],
        'max_depth': [5, 5, 3, 3]
    })

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterRecursiveMultiSeries_lags_grid_dict_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterRecursiveMultiSeries 
    when `lags_grid` is a dict with mocked (mocked done in Skforecast v0.5.0).
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = Ridge(random_state=123),
                     lags               = 2,
                     encoding           = 'onehot', 
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_dt['l1']) - 12,
            steps              = 3,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )
    lags_grid = {'lags_1': 2, 'lags_2': 4}
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series_dict_dt,
                  cv                 = cv,
                  param_grid         = param_grid,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  return_best        = False,
                  verbose            = False
              )

    expected_results = pd.DataFrame(
        {
            "levels": [["l1", "l2"]] * 6,
            "lags": [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2], [1, 2], [1, 2]],
            "lags_label": ["lags_2", "lags_2", "lags_2", "lags_1", "lags_1", "lags_1"],
            "params": [
                {"alpha": 0.01},
                {"alpha": 0.1},
                {"alpha": 1},
                {"alpha": 1},
                {"alpha": 0.1},
                {"alpha": 0.01},
            ],
            "mean_absolute_error__weighted_average": np.array(
                [
                    0.20968100463227382,
                    0.20969259779858337,
                    0.20977945312386406,
                    0.21077344827205086,
                    0.21078653113227208,
                    0.21078779824759553,
                ]
            ),
            "n_evaluated": [24] * 6,
            "alpha": np.array([0.01, 0.1, 1.0, 1.0, 0.1, 0.01]),
        },
        index=pd.Index([0, 1, 2, 3, 4, 5], dtype="int64"),
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterRecursiveMultiSeries_lags_grid_is_None_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterRecursiveMultiSeries 
    when `lags_grid` is `None` with mocked (mocked done in Skforecast v0.5.0), 
    should use forecaster.lags as lags_grid.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = Ridge(random_state=123),
                     lags               = 2, 
                     encoding           = 'onehot',
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_range['l1']) - 12,
            steps              = 3,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )

    lags_grid = None
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series_long_dt,
                  param_grid         = param_grid,
                  cv                 = cv,
                  metric             = mean_absolute_error,
                  aggregate_metric   = 'weighted_average',
                  levels             = ['l1', 'l2'],
                  exog               = None,
                  lags_grid          = lags_grid,
                  return_best        = False,
                  verbose            = False
              )

    expected_results = pd.DataFrame(
        {
            "levels": [["l1", "l2"], ["l1", "l2"], ["l1", "l2"]],
            "lags": [[1, 2], [1, 2], [1, 2]],
            "lags_label": [[1, 2], [1, 2], [1, 2]],
            "params": [{"alpha": 1}, {"alpha": 0.1}, {"alpha": 0.01}],
            "mean_absolute_error__weighted_average": np.array(
                [0.21077344827205086, 0.21078653113227208, 0.21078779824759553]
            ),
            "n_evaluated": [24] * 3,
            "alpha": np.array([1.0, 0.1, 0.01]),
        },
        index=pd.Index([0, 1, 2], dtype="int64"),
    )

    pd.testing.assert_frame_equal(results, expected_results)


@pytest.mark.parametrize(
    "cv, expected_metric",
    [(TimeSeriesFold(steps=3, initial_train_size=len(series_dict_range['l1']) - 12, refit=False),
      np.array([0.21043602991610974, 0.21061383703517822, 0.21064405145051132])),
     (OneStepAheadFold(initial_train_size=len(series_dict_range['l1']) - 12),
      np.array([0.21088437764824652, 0.2113636838391947, 0.21144760769988624]))],
    ids=['TimeSeriesFold', 'OneStepAheadFold']
)
@pytest.mark.parametrize("lags_grid",
                         [None, [None]],
                         ids=lambda lg: f'lags_grid: {lg}')
def test_output_evaluate_grid_hyperparameters_multiseries_when_forecaster_lags_is_None(
    cv, expected_metric, lags_grid
):
    """
    Test output of _evaluate_grid_hyperparameters_multiseries when the forecaster
    has `lags=None` and only window features: `lags_grid=None` uses the lags of
    the forecaster and `lags_grid=[None]` evaluates the forecaster without lags.
    With `return_best=True`, the forecaster is refitted without lags.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = Ridge(random_state=123),
                     lags               = None,
                     window_features    = RollingFeatures(stats=['mean'], window_sizes=3),
                     encoding           = 'onehot',
                     transformer_series = None
                 )
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series_long_dt,
                  param_grid         = param_grid,
                  cv                 = cv,
                  metric             = mean_absolute_error,
                  aggregate_metric   = 'weighted_average',
                  levels             = ['l1', 'l2'],
                  exog               = None,
                  lags_grid          = lags_grid,
                  return_best        = True,
                  verbose            = False
              )

    expected_results = pd.DataFrame(
        {
            "levels": [["l1", "l2"], ["l1", "l2"], ["l1", "l2"]],
            "lags": [None, None, None],
            "lags_label": [None, None, None],
            "params": [{"alpha": 1}, {"alpha": 0.1}, {"alpha": 0.01}],
            "mean_absolute_error__weighted_average": expected_metric,
            "n_evaluated": [24] * 3,
            "alpha": np.array([1.0, 0.1, 0.01]),
        },
        index=pd.RangeIndex(start=0, stop=3, step=1),
    )

    pd.testing.assert_frame_equal(results, expected_results)
    assert forecaster.lags is None
    assert forecaster.estimator.alpha == 1
    assert forecaster.is_fitted


@pytest.mark.parametrize("levels", 
                         ['l1', ['l1']], 
                         ids = lambda value: f'levels: {value}')
def test_evaluate_grid_hyperparameters_multiseries_ForecasterRecursiveMultiSeries_levels_str_list(levels):
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterRecursiveMultiSeries 
    with mocked when `levels` is a `str` or a `list` (mocked done in Skforecast v0.5.0).
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = Ridge(random_state=123),
                     lags               = 2, 
                     encoding           = 'onehot',
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_range['l1']) - 12,
            steps              = 3,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series_long_dt,
                  param_grid         = param_grid,
                  cv                 = cv,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  levels             = levels,
                  exog               = None,
                  lags_grid          = lags_grid,
                  return_best        = False,
                  verbose            = False
              )

    expected_results = pd.DataFrame(
        {
            "levels": [["l1"]] * 6,
            "lags": [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2], [1, 2], [1, 2]],
            "lags_label": [
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2],
                [1, 2],
                [1, 2],
            ],
            "params": [
                {"alpha": 0.01},
                {"alpha": 0.1},
                {"alpha": 1},
                {"alpha": 0.01},
                {"alpha": 0.1},
                {"alpha": 1},
            ],
            "mean_absolute_error": np.array(
                [
                    0.20669393332187616,
                    0.20671040715338015,
                    0.20684013292264494,
                    0.2073988652614679,
                    0.20741562577568792,
                    0.2075484707375347,
                ]
            ),
            "n_evaluated": [12] * 6,
            "alpha": np.array([0.01, 0.1, 1.0, 0.01, 0.1, 1.0]),
        },
        index=pd.Index([0, 1, 2, 3, 4, 5], dtype="int64"),
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterRecursiveMultiSeries_multiple_metrics_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterRecursiveMultiSeries 
    with mocked when multiple metrics (mocked done in Skforecast v0.6.0).
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = Ridge(random_state=123),
                     lags               = 2, 
                     encoding           = 'onehot',
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_range['l1']) - 12,
            steps              = 3,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series_dict_range,
                  param_grid         = param_grid,
                  cv                 = cv,
                  metric             = [mean_squared_error, 'mean_absolute_error'],
                  aggregate_metric   = 'weighted_average',
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  return_best        = False,
                  verbose            = False
              )
    
    expected_results = pd.DataFrame(
        {
            "levels": [["l1", "l2"]] * 6,
            "lags": [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2], [1, 2], [1, 2]],
            "lags_label": [
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2],
                [1, 2],
                [1, 2],
            ],
            "params": [
                {"alpha": 0.01},
                {"alpha": 0.1},
                {"alpha": 1},
                {"alpha": 0.01},
                {"alpha": 0.1},
                {"alpha": 1},
            ],
            "mean_squared_error__weighted_average": np.array(
                [
                    0.06365397633008085,
                    0.06367614582294409,
                    0.06385378127252679,
                    0.06389613553855186,
                    0.06391570591810977,
                    0.06407787633532819,
                ]
            ),
            "mean_absolute_error__weighted_average": np.array(
                [
                    0.20968100463227382,
                    0.20969259779858337,
                    0.20977945312386406,
                    0.21078779824759553,
                    0.21078653113227208,
                    0.21077344827205086,
                ]
            ),
            "n_evaluated": [24] * 6,
            "alpha": np.array([0.01, 0.1, 1.0, 0.01, 0.1, 1.0]),
        },
        index=pd.Index([0, 1, 2, 3, 4, 5], dtype="int64"),
    )

    pd.testing.assert_frame_equal(results, expected_results)


@pytest.mark.parametrize("lags_grid", 
                         [[2, 4], {'lags_1': 2, 'lags_2': 4}], 
                         ids=lambda lg: f'lags_grid: {lg}')
def test_evaluate_grid_hyperparameters_multiseries_when_return_best_ForecasterRecursiveMultiSeries(lags_grid):
    """
    Test forecaster is refitted when `return_best = True` in 
    _evaluate_grid_hyperparameters_multiseries.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = Ridge(random_state=123),
                     lags               = 2, 
                     encoding           = 'onehot',
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_range['l1']) - 12,
            steps              = 3,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    _evaluate_grid_hyperparameters_multiseries(
        forecaster         = forecaster,
        series             = series_dict_range,
        param_grid         = param_grid,
        cv                 = cv,
        metric             = 'mean_absolute_error',
        aggregate_metric   = 'weighted_average',
        levels             = None,
        exog               = None,
        lags_grid          = lags_grid,
        return_best        = True,
        verbose            = False
    )

    expected_lags = np.array([1, 2, 3, 4])
    expected_alpha = 0.01
    expected_series_names_in_ = ['l1', 'l2']
    
    assert (expected_lags == forecaster.lags).all()
    assert expected_alpha == forecaster.estimator.alpha
    assert expected_series_names_in_ == forecaster.series_names_in_


def test_evaluate_grid_hyperparameters_multiseries_ForecasterRecursiveMultiSeries_output_file_single_level():
    """
    Test output file is created when output_file is passed to
    _evaluate_grid_hyperparameters_multiseries and single level.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        estimator=Ridge(random_state=123), lags=2
    )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_range['l1']) - 12,
            steps              = 3,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )
    lags_grid = {"lags_1": 2, "lags_2": 4}
    param_grid = [{"alpha": 0.01}, {"alpha": 0.1}, {"alpha": 1}]
    output_file = "test_evaluate_grid_hyperparameters_multiseries_output_file.txt"

    results = _evaluate_grid_hyperparameters_multiseries(
        forecaster=forecaster,
        series=series_dict_range,
        param_grid=param_grid,
        cv=cv,
        metric="mean_absolute_error",
        aggregate_metric="weighted_average",
        levels="l1",
        exog=None,
        lags_grid=lags_grid,
        return_best=False,
        verbose=False,
        show_progress=False,
        output_file=output_file,
    )
    results = results.astype({"levels": str, "lags": str, "params": str})

    assert os.path.isfile(output_file)
    output_file_content = pd.read_csv(output_file, sep="\t", low_memory=False)
    output_file_content = output_file_content.sort_values(
        by="mean_absolute_error"
    ).reset_index(drop=True)
    output_file_content = output_file_content.astype(
        {"levels": str, "lags": str, "params": str}
    )
    pd.testing.assert_frame_equal(results, output_file_content)
    os.remove(output_file)


def test_evaluate_grid_hyperparameters_multiseries_ForecasterRecursiveMultiSeries_output_file_multiple_metrics():
    """
    Test output file is created when output_file is passed to
    _evaluate_grid_hyperparameters_multiseries and list of metrics.
    """
    forecaster = ForecasterRecursiveMultiSeries(estimator=Ridge(random_state=123), lags=2)
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_range['l1']) - 12,
            steps              = 3,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )
    lags_grid = [2, 4]
    param_grid = [{"alpha": 0.01}, {"alpha": 0.1}, {"alpha": 1}]
    output_file = "test_evaluate_grid_hyperparameters_multiseries_output_file.txt"

    results = _evaluate_grid_hyperparameters_multiseries(
        forecaster=forecaster,
        series=series_dict_range,
        param_grid=param_grid,
        cv=cv,
        metric=[mean_squared_error, "mean_absolute_error"],
        aggregate_metric="weighted_average",
        levels=None,
        exog=None,
        lags_grid=lags_grid,
        return_best=False,
        verbose=False,
        show_progress=False,
        output_file=output_file,
    )
    results = results.astype(
        {"levels": str, "lags": str, "lags_label": str, "params": str}
    )

    assert os.path.isfile(output_file)
    output_file_content = pd.read_csv(output_file, sep="\t", low_memory=False)
    output_file_content = output_file_content.sort_values(
        by="mean_squared_error__weighted_average"
    ).reset_index(drop=True)
    output_file_content = output_file_content.astype(
        {"levels": str, "lags": str, "lags_label": str, "params": str}
    )
    pd.testing.assert_frame_equal(results, output_file_content)
    os.remove(output_file)


def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterRecursiveMultiSeries_multiple_metrics_aggregated_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterRecursiveMultiSeries 
    with mocked when multiple metrics (mocked done in Skforecast v0.6.0).
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = Ridge(random_state=123),
                     lags               = 2, 
                     encoding           = 'onehot',
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_range['l1']) - 12,
            steps              = 3,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series_dict_range,
                  param_grid         = param_grid,
                  cv                 = cv,
                  metric             = ['mean_absolute_error', 'mean_absolute_scaled_error'],
                  aggregate_metric   = ['weighted_average', 'average', 'pooling'],
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  return_best        = False,
                  verbose            = False,
                  show_progress      = False,
              )
    
    expected_results = pd.DataFrame({
        "levels": {
            0: ["l1", "l2"],
            1: ["l1", "l2"],
            2: ["l1", "l2"],
            3: ["l1", "l2"],
            4: ["l1", "l2"],
            5: ["l1", "l2"],
        },
        "lags": {
            0: np.array([1, 2, 3, 4]),
            1: np.array([1, 2, 3, 4]),
            2: np.array([1, 2, 3, 4]),
            3: np.array([1, 2]),
            4: np.array([1, 2]),
            5: np.array([1, 2]),
        },
        "lags_label": {
            0: np.array([1, 2, 3, 4]),
            1: np.array([1, 2, 3, 4]),
            2: np.array([1, 2, 3, 4]),
            3: np.array([1, 2]),
            4: np.array([1, 2]),
            5: np.array([1, 2]),
        },
        "params": {
            0: {"alpha": 0.01},
            1: {"alpha": 0.1},
            2: {"alpha": 1},
            3: {"alpha": 1},
            4: {"alpha": 0.1},
            5: {"alpha": 0.01},
        },
        "mean_absolute_error__weighted_average": {
            0: 0.20968100547390048,
            1: 0.20969259864077977,
            2: 0.20977945397058564,
            3: 0.21077344921320568,
            4: 0.21078653208835063,
            5: 0.21078779920557153,
        },
        "mean_absolute_error__average": {
            0: 0.20968100547390048,
            1: 0.20969259864077974,
            2: 0.20977945397058564,
            3: 0.21077344921320565,
            4: 0.21078653208835063,
            5: 0.21078779920557153,
        },
        "mean_absolute_error__pooling": {
            0: 0.20968100547390045,
            1: 0.2096925986407798,
            2: 0.20977945397058564,
            3: 0.21077344921320565,
            4: 0.21078653208835063,
            5: 0.21078779920557153,
        },
        'mean_absolute_scaled_error__weighted_average': {
            0: 0.798335248515498,
            1: 0.7983823516895249,
            2: 0.7987389489370742,
            3: 0.8120054679421529,
            4: 0.8119798820564519,
            5: 0.8119753734760063},
        'mean_absolute_scaled_error__average': {
            0: 0.7983352485154981,
            1: 0.7983823516895249,
            2: 0.7987389489370743,
            3: 0.812005467942153,
            4: 0.8119798820564519,
            5: 0.8119753734760063},
        'mean_absolute_scaled_error__pooling': {
            0: 0.7811335117239494,
            1: 0.7811767002385028,
            2: 0.7815002660695128,
            3: 0.7988421945907456,
            4: 0.7988917793592797,
            5: 0.7988965817986066
        },
        'n_evaluated': {0: 24, 1: 24, 2: 24, 3: 24, 4: 24, 5: 24},
        "alpha": {0: 0.01, 1: 0.1, 2: 1.0, 3: 1.0, 4: 0.1, 5: 0.01},
    })

    pd.testing.assert_frame_equal(results, expected_results)


# ForecasterDirectMultiVariate
# ======================================================================================================================
def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterDirectMultiVariate_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterDirectMultiVariate 
    with mocked (mocked done in Skforecast v0.6.0).
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3,
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_wide_range) - 12,
            steps              = 3,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series_wide_range,
                  param_grid         = param_grid,
                  cv                 = cv,
                  metric             = 'mean_absolute_error',
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  return_best        = False,
                  verbose            = True
              )
    
    expected_results = pd.DataFrame({
        'levels': [['l1']] * 6,
        'lags': [[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        'lags_label': [[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, 
                   {'alpha': 1}, {'alpha': 0.1}, {'alpha': 0.01}],
        'mean_absolute_error': np.array(
            [0.20115194, 0.20183032, 0.20566862,
             0.22224269, 0.22625017, 0.22644284]
        ),
        'n_evaluated': [12] * 6,
        'alpha': np.array([0.01, 0.1, 1., 1., 0.1, 0.01])
        },
        index = pd.Index([0, 1, 2, 3, 4, 5], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterDirectMultiVariate_lags_grid_dict_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterDirectMultiVariate 
    when `lags_grid` is a dict with mocked (mocked done in Skforecast v0.6.0)
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3,
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_wide_range) - 12,
            steps              = 3,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )
    lags_grid = {'lags_1': 2, 'lags_2': 4}
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series_wide_range,
                  param_grid         = param_grid,
                  cv                 = cv,
                  metric             = 'mean_absolute_error',
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  return_best        = False,
                  verbose            = True
              )
    
    expected_results = pd.DataFrame({
        'levels': [['l1']] * 6,
        'lags': [[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        'lags_label': ['lags_1', 'lags_1', 'lags_1', 'lags_2', 'lags_2', 'lags_2'],
        'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, 
                   {'alpha': 1}, {'alpha': 0.1}, {'alpha': 0.01}],
        'mean_absolute_error': np.array(
            [0.20115194, 0.20183032, 0.20566862,
             0.22224269, 0.22625017, 0.22644284]
            ),
        'n_evaluated': [12] * 6,
        'alpha': np.array([0.01, 0.1, 1., 1., 0.1, 0.01])
        },
        index = pd.Index([0, 1, 2, 3, 4, 5], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterDirectMultiVariate_lags_grid_is_None_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterDirectMultiVariate 
    when `lags_grid` is `None` with mocked (mocked done in Skforecast v0.6.0), 
    should use forecaster.lags as lags_grid.
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3,
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_wide_range) - 12,
            steps              = 3,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )
    lags_grid = None
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series_wide_range,
                  param_grid         = param_grid,
                  cv                 = cv,
                  metric             = mean_absolute_error,
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  return_best        = False,
                  verbose            = False
              )
    
    expected_results = pd.DataFrame({
        'levels': [['l1']] * 3,
        'lags': [[1, 2], [1, 2], [1, 2]],
        'lags_label': [[1, 2], [1, 2], [1, 2]],
        'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1.}],
        'mean_absolute_error': np.array([0.20115194, 0.20183032, 0.20566862]),
        'n_evaluated': [12] * 3,
        'alpha': np.array([0.01, 0.1, 1.])
    },
        index = pd.Index([0, 1, 2], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterDirectMultiVariate_lags_grid_is_list_of_dicts_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterDirectMultiVariate 
    when `lags_grid` is a list of dicts with mocked (mocked done in Skforecast v0.6.0).
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3,
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_wide_range) - 12,
            steps              = 3,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )
    lags_grid = [{'l1': 2, 'l2': 3}, {'l1': [1, 3], 'l2': 3}, {'l1': 2, 'l2': [1, 4]}]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series_wide_range,
                  param_grid         = param_grid,
                  cv                 = cv,
                  metric             = mean_absolute_error,
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  return_best        = False,
                  verbose            = False
              )
    
    expected_results = pd.DataFrame({
        'levels': [['l1']] * 9,
        'lags': [{'l1': np.array([1, 2]), 'l2': np.array([1, 2, 3])},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 2, 3])},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 2, 3])},
                   {'l1': np.array([1, 3]), 'l2': np.array([1, 2, 3])},
                   {'l1': np.array([1, 3]), 'l2': np.array([1, 2, 3])},
                   {'l1': np.array([1, 3]), 'l2': np.array([1, 2, 3])},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 4])},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 4])},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 4])}],
        'lags_label': [{'l1': np.array([1, 2]), 'l2': np.array([1, 2, 3])},
                        {'l1': np.array([1, 2]), 'l2': np.array([1, 2, 3])},
                        {'l1': np.array([1, 2]), 'l2': np.array([1, 2, 3])},
                        {'l1': np.array([1, 3]), 'l2': np.array([1, 2, 3])},
                        {'l1': np.array([1, 3]), 'l2': np.array([1, 2, 3])},
                        {'l1': np.array([1, 3]), 'l2': np.array([1, 2, 3])},
                        {'l1': np.array([1, 2]), 'l2': np.array([1, 4])},
                        {'l1': np.array([1, 2]), 'l2': np.array([1, 4])},
                        {'l1': np.array([1, 2]), 'l2': np.array([1, 4])}],
        'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1.}, 
                   {'alpha': 1.}, {'alpha': 0.1}, {'alpha': 0.01}, 
                   {'alpha': 1.}, {'alpha': 0.1}, {'alpha': 0.01}],
        'mean_absolute_error': np.array(
            [0.2053202, 0.20555199, 0.20677802, 
             0.21443621, 0.21801147, 0.21863968, 
             0.22401526, 0.22830217, 0.22878132]
            ),
        'n_evaluated': [12] * 9,
        'alpha': np.array([0.01, 0.1, 1., 1., 0.1, 0.01, 1., 0.1, 0.01])
        },
        index = pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterDirectMultiVariate_lags_grid_is_dict_of_dicts_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterDirectMultiVariate 
    when `lags_grid` is a dict of dicts with mocked (mocked done in Skforecast v0.6.0).
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3,
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_wide_range) - 12,
            steps              = 3,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )
    lags_grid = {
        'lags_1': {'l1': 2, 'l2': 3},
        'lags_2': {'l1': [1, 3], 'l2': 3},
        'lags_3': {'l1': 2, 'l2': [1, 4]},
        'lags_4': {'l1': 2, 'l2': None},
        'lags_5': {'l1': None, 'l2': 2},
        'lags_6': 3
    }
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series_wide_range,
                  param_grid         = param_grid,
                  cv                 = cv,
                  metric             = mean_absolute_error,
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  return_best        = False,
                  verbose            = False
              )
    
    expected_results = pd.DataFrame({
        'levels': [['l1']] * 18,
        'lags': [{'l1': np.array([1, 2]), 'l2': None},
                   {'l1': np.array([1, 2]), 'l2': None},
                   {'l1': np.array([1, 2]), 'l2': None},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 2, 3])},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 2, 3])},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 2, 3])},
                   {'l1': None, 'l2': np.array([1, 2])},
                   {'l1': None, 'l2': np.array([1, 2])},
                   {'l1': None, 'l2': np.array([1, 2])},
                   [1, 2, 3],
                   {'l1': np.array([1, 3]), 'l2': np.array([1, 2, 3])},
                   [1, 2, 3],
                   [1, 2, 3],
                   {'l1': np.array([1, 3]), 'l2': np.array([1, 2, 3])},
                   {'l1': np.array([1, 3]), 'l2': np.array([1, 2, 3])},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 4])},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 4])},
                   {'l1': np.array([1, 2]), 'l2': np.array([1, 4])}],
        'lags_label': ['lags_4', 'lags_4', 'lags_4', 
                       'lags_1', 'lags_1', 'lags_1', 
                       'lags_5', 'lags_5', 'lags_5',
                       'lags_6', 'lags_2', 'lags_6', 
                       'lags_6', 'lags_2', 'lags_2', 
                       'lags_3', 'lags_3', 'lags_3'],
        'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, 
                   {'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, 
                   {'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, 
                   {'alpha': 1}, {'alpha': 1}, {'alpha': 0.1}, 
                   {'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 0.01}, 
                   {'alpha': 1}, {'alpha': 0.1}, {'alpha': 0.01}],
        'mean_absolute_error': np.array(
            [0.20155258, 0.20208154, 0.20516149, 0.2053202, 0.20555199,
             0.20677802, 0.21005165, 0.21007475, 0.21071924, 0.21353688,
             0.21443621, 0.21622784, 0.2166998, 0.21801147, 0.21863968,
             0.22401526, 0.22830217, 0.22878132]),
        'n_evaluated': [12] * 18,
        'alpha': np.array([0.01, 0.1, 1., 0.01, 0.1, 1., 0.01, 0.1, 1., 1., 1.,
       0.1, 0.01, 0.1, 0.01, 1., 0.1, 0.01])
        },
        index = pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_multiseries_ForecasterDirectMultiVariate_multiple_metrics_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_multiseries in ForecasterDirectMultiVariate 
    with mocked when multiple metrics (mocked done in Skforecast v0.6.0).
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3,
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_wide_range) - 12,
            steps              = 3,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series_wide_range,
                  param_grid         = param_grid,
                  cv                 = cv,
                  metric             = [mean_squared_error, 'mean_absolute_error'],
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  return_best        = False,
                  verbose            = False
              )
    
    expected_results = pd.DataFrame({
        'levels': [['l1']] * 6,
        'lags': [[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        'lags_label': [[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, 
                   {'alpha': 1.}, {'alpha': 0.1}, {'alpha': 0.01}],
        'mean_squared_error': np.array([0.06260985, 0.06309219, 0.06627699, 
                                        0.08032378, 0.08400047, 0.08448937]),
        'mean_absolute_error': np.array(
            [0.20115194, 0.20183032, 0.20566862, 0.22224269, 0.22625017, 0.22644284]),
        'n_evaluated': [12] * 6,
        'alpha': np.array([0.01, 0.1, 1., 1., 0.1, 0.01])
        },
        index = pd.Index([0, 1, 2, 3, 4, 5], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results)


@pytest.mark.parametrize("lags_grid", 
                         [[2, 4], {'lags_1': 2, 'lags_2': 4}], 
                         ids=lambda lg: f'lags_grid: {lg}')
def test_evaluate_grid_hyperparameters_multiseries_when_return_best_ForecasterDirectMultiVariate(lags_grid):
    """
    Test forecaster is refitted when `return_best = True` in 
    _evaluate_grid_hyperparameters_multiseries.
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3,
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_wide_range) - 12,
            steps              = 3,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    _evaluate_grid_hyperparameters_multiseries(
        forecaster         = forecaster,
        series             = series_wide_range,
        param_grid         = param_grid,
        cv                 = cv,
        metric             = 'mean_absolute_error',
        aggregate_metric   = 'weighted_average',
        levels             = None,
        exog               = None,
        lags_grid          = lags_grid,
        return_best        = True,
        verbose            = True,
        show_progress      = False
    )

    expected_lags = np.array([1, 2])
    expected_alpha = 0.01
    expected_series_names_in_ = ['l1', 'l2']
    
    assert (expected_lags == forecaster.lags).all()
    for i in forecaster.steps:
        assert expected_alpha == forecaster.estimators_[i].alpha
    assert expected_series_names_in_ == forecaster.series_names_in_


def test_evaluate_grid_hyperparameters_multiseries_when_lags_grid_has_None_and_return_best():
    """
    Test that `None` in `lags_grid` evaluates the forecaster without lags (only
    window features) next to the other candidates, and that the forecaster is
    refitted with the best lags when `return_best=True`.
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = None,
                     steps              = 3,
                     window_features    = RollingFeatures(stats=['mean'], window_sizes=3),
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_wide_range) - 12,
            steps              = 3,
            refit              = False
         )
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series_wide_range,
                  param_grid         = param_grid,
                  cv                 = cv,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  levels             = None,
                  exog               = None,
                  lags_grid          = [None, 2],
                  return_best        = True,
                  verbose            = False,
                  show_progress      = False
              )

    expected_lags = [[1, 2], None, [1, 2], None, None, [1, 2]]
    expected_params = [
        {'alpha': 1}, {'alpha': 1}, {'alpha': 0.1},
        {'alpha': 0.1}, {'alpha': 0.01}, {'alpha': 0.01}
    ]
    expected_metric = np.array(
        [0.204584, 0.209270, 0.209546, 0.210975, 0.211978, 0.215558]
    )

    results_lags = [
        None if lags is None else lags.tolist() for lags in results['lags']
    ]
    results_lags_label = [
        None if lags is None else lags.tolist() for lags in results['lags_label']
    ]
    assert results_lags == expected_lags
    assert results_lags_label == expected_lags
    assert results['params'].to_list() == expected_params
    np.testing.assert_array_almost_equal(
        results['mean_absolute_error'].to_numpy(), expected_metric
    )
    np.testing.assert_array_equal(forecaster.lags, np.array([1, 2]))
    for i in forecaster.steps:
        assert forecaster.estimators_[i].alpha == 1
    assert forecaster.series_names_in_ == ['l1', 'l2']


def test_evaluate_grid_hyperparameters_multiseries_ForecasterDirectMultiVariate_output_file_single_level():
    """
    Test output file is created when output_file is passed to 
    _evaluate_grid_hyperparameters_multiseries and single level.
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator = Ridge(random_state=123),
                     level     = 'l1',
                     lags      = 2,
                     steps     = 3
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_wide_range) - 12,
            steps              = 3,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )
    lags_grid = {'lags_1': 2, 'lags_2': 4}
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]
    output_file = 'test_evaluate_grid_hyperparameters_multiseries_output_file.txt'

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series_wide_range,
                  param_grid         = param_grid,
                  cv                 = cv,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  levels             = 'l1',
                  exog               = None,
                  lags_grid          = lags_grid,
                  return_best        = False,
                  verbose            = False,
                  show_progress      = False,
                  output_file        = output_file
              )
    results  = results.astype({'levels': str, 'lags': str, 'lags_label': str, 'params': str})

    assert os.path.isfile(output_file)
    output_file_content = pd.read_csv(output_file, sep='\t', low_memory=False)
    output_file_content = output_file_content.sort_values(by='mean_absolute_error').reset_index(drop=True)
    output_file_content = output_file_content.astype({'levels': str, 'lags': str, 'lags_label': str, 'params': str})
    pd.testing.assert_frame_equal(results, output_file_content)
    os.remove(output_file)


def test_evaluate_grid_hyperparameters_multiseries_ForecasterDirectMultiVariate_output_file_multiple_metrics():
    """
    Test output file is created when output_file is passed to 
    _evaluate_grid_hyperparameters_multiseries and list of metrics.
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator = Ridge(random_state=123),
                     level     = 'l2',
                     lags      = 2,
                     steps     = 3
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_wide_range) - 12,
            steps              = 3,
            gap                = 0,
            refit              = False,
            fixed_train_size   = False,
         )
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]
    output_file = 'test_evaluate_grid_hyperparameters_multiseries_output_file.txt'

    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series_wide_range,
                  param_grid         = param_grid,
                  cv                 = cv,
                  metric             = [mean_squared_error, 'mean_absolute_error'],
                  levels             = None,
                  exog               = None,
                  lags_grid          = lags_grid,
                  return_best        = False,
                  verbose            = False,
                  show_progress      = False,
                  output_file        = output_file
              )
    results  = results.astype({'levels': str, 'lags': str, 'lags_label': str, 'params': str})

    assert os.path.isfile(output_file)
    output_file_content = pd.read_csv(output_file, sep='\t', low_memory=False)
    output_file_content = output_file_content.sort_values(
        by="mean_squared_error"
    ).reset_index(drop=True)
    output_file_content = output_file_content.astype({'levels': str, 'lags': str, 'lags_label': str, 'params': str})
    pd.testing.assert_frame_equal(results, output_file_content)
    os.remove(output_file)


@pytest.mark.parametrize(
        "forecaster",
        [
            ForecasterRecursiveMultiSeries(
                estimator=Ridge(random_state=678),
                lags=3,
                transformer_series=None,
                forecaster_id='Multiseries_no_transformer'
            ),
            ForecasterRecursiveMultiSeries(
                estimator=Ridge(random_state=678),
                lags=3,
                transformer_series=StandardScaler(),
                transformer_exog=StandardScaler(),
                forecaster_id='Multiseries_transformer'
            ),
            ForecasterDirectMultiVariate(
                estimator=Ridge(random_state=678),
                level='l1',
                lags=3,
                steps=1,
                transformer_series=StandardScaler(),
                transformer_exog=StandardScaler(),
                forecaster_id='Multivariate'
            )
        ], 
ids=lambda forecaster: f'forecaster: {forecaster.forecaster_id}')
def test_evaluate_grid_hyperparameters_equivalent_outputs_backtesting_and_one_step_ahead(
    forecaster,
):
    """
    Test that the output of _evaluate_grid_hyperparameters for backtesting and one-step-ahead
    is equivalent when steps=1 and refit=False.
    Results are not equivalent if differentiation is included.
    """
    
    if forecaster.forecaster_id == 'Multivariate':
        series = series_wide_dt
    else:
        series = series_dict_dt

    exog_datetime = exog.copy()
    exog_datetime.index = pd.date_range(
        start='2020-01-01', periods=len(exog), freq='D'
    )

    metrics = [
        "mean_absolute_error",
        "mean_squared_error",
        mean_absolute_percentage_error,
        mean_absolute_scaled_error,
        root_mean_squared_scaled_error,
    ]
    cv_backtesting = TimeSeriesFold(
            initial_train_size = 20,
            steps              = 1,
            refit              = False,
            fixed_train_size   = False,
         )
    param_grid = {
        "alpha": np.logspace(-1, 1, 3),
    }
    lags_grid = [3, 7]
    param_grid = list(ParameterGrid(param_grid))
    results_backtesting = _evaluate_grid_hyperparameters_multiseries(
        forecaster         = forecaster,
        series             = series,
        exog               = exog_datetime,
        cv                 = cv_backtesting,
        param_grid         = param_grid,
        lags_grid          = lags_grid,
        metric             = metrics,
        aggregate_metric   = ["average", "weighted_average", "pooling"],
        return_best        = False,
        n_jobs             = 'auto',
        verbose            = False,
        show_progress      = False
    )
    cv_one_step_ahead = OneStepAheadFold(initial_train_size = 20)
    results_one_step_ahead = _evaluate_grid_hyperparameters_multiseries(
        forecaster         = forecaster,
        series             = series,
        exog               = exog_datetime,
        cv                 = cv_one_step_ahead,
        param_grid         = param_grid,
        lags_grid          = lags_grid,
        metric             = metrics,
        aggregate_metric   = ["average", "weighted_average", "pooling"],
        return_best        = False,
        verbose            = False,
        show_progress      = False
    )

    pd.testing.assert_frame_equal(results_backtesting, results_one_step_ahead)


@pytest.mark.parametrize("initial_train_size", 
                         [213, '2016-07-31 00:00:00', pd.to_datetime('2016-07-31 00:00:00')], 
                         ids=lambda initial_train_size: f'initial_train_size: {initial_train_size}')
@pytest.mark.parametrize(
        "forecaster",
        [
            ForecasterRecursiveMultiSeries(
                estimator=LGBMRegressor(random_state=678, verbose=-1),
                lags=3,
                transformer_series=None,
                forecaster_id='Multiseries_no_transformer'
            ),
            ForecasterRecursiveMultiSeries(
                estimator=LGBMRegressor(random_state=678, verbose=-1),
                lags=3,
                transformer_series=StandardScaler(),
                transformer_exog=StandardScaler(),
                forecaster_id='Multiseries_transformer'
            )
        ],
ids=lambda forecaster: f'forecaster: {forecaster.forecaster_id}')
def test_evaluate_grid_hyperparameters_same_output_backtesting_and_one_step_ahead_when_series_is_dict(
    initial_train_size, forecaster,
):
    """
    Test that the output of evaluate_grid_hyperparameters for backtesting and one-step-ahead
    is equivalent when steps=1 and refit=False. Using series and exog as dictionaries.
    Results are not equivalent if differentiation is included.
    ForecasterMultiVariate is not included because it is not possible to use dictionaries
    as input.
    """

    metrics = [
        "mean_absolute_error",
        "mean_squared_error",
        mean_absolute_percentage_error,
        mean_absolute_scaled_error,
        root_mean_squared_scaled_error,
    ]
    param_grid = {
        "n_estimators": [5, 10, 15]
    }
    lags_grid = [3, 5]
    param_grid = list(ParameterGrid(param_grid))

    cv_backtesting = TimeSeriesFold(
        initial_train_size = initial_train_size,
        steps              = 1,
        refit              = False,
        fixed_train_size   = False,
    )
    cv_one_step_ahead = OneStepAheadFold(initial_train_size = initial_train_size)
    
    results_backtesting = _evaluate_grid_hyperparameters_multiseries(
        forecaster         = forecaster,
        series             = series_dict_nans,
        exog               = exog_dict_nans,
        cv                 = cv_backtesting,
        param_grid         = param_grid,
        lags_grid          = lags_grid,
        metric             = metrics,
        aggregate_metric   = ["average", "weighted_average", "pooling"],
        return_best        = False,
        n_jobs             = 'auto',
    )

    warn_msg = re.escape(
        "One-step-ahead predictions are used for faster model comparison, but they "
        "may not fully represent multi-step prediction performance. It is recommended "
        "to backtest the final model for a more accurate multi-step performance "
        "estimate."
    )
    with pytest.warns(OneStepAheadValidationWarning, match = warn_msg):
        results_one_step_ahead = _evaluate_grid_hyperparameters_multiseries(
            forecaster         = forecaster,
            series             = series_dict_nans,
            exog               = exog_dict_nans,
            cv                 = cv_one_step_ahead,
            param_grid         = param_grid,
            lags_grid          = lags_grid,
            metric             = metrics,
            aggregate_metric   = ["average", "weighted_average", "pooling"],
            return_best        = False,
            n_jobs             = 'auto',
        )

    pd.testing.assert_frame_equal(results_backtesting, results_one_step_ahead)


def test_evaluate_grid_hyperparameters_multiseries_warn_when_non_valid_params():
    """
    Test that a single warning, grouped by error, is raised when non valid
    params are included in param_grid, and the candidate is skipped.
    """

    param_grid = {
        "alpha": [0.1],
        "l1_ratio": [0.5, 10],  # 10 is not valid for ElasticNet
    }
    param_grid = list(ParameterGrid(param_grid))
    cv = TimeSeriesFold(steps=12, initial_train_size=30, refit=False)
    forecaster = ForecasterRecursiveMultiSeries(estimator=ElasticNet(), lags=5)

    msg = re.escape(
        "Parameters skipped: 1 of 2 candidates raises an error, so it is not in "
        "the results.\n"
        "\n"
        "InvalidParameterError: The 'l1_ratio' parameter of ElasticNet must be a "
        "float in the range [0.0, 1.0]. Got 10 instead.\n"
        "  - lags: [1, 2, 3, 4, 5], params: {'alpha': 0.1, 'l1_ratio': 10}"
    )
    with pytest.warns(RuntimeWarning, match=msg):
        results = _evaluate_grid_hyperparameters_multiseries(
            forecaster=forecaster,
            series=series_dict_dt,
            param_grid=param_grid,
            cv=cv,
            metric="mean_squared_error",
            return_best=True,
            n_jobs="auto",
            verbose=True,
            show_progress=False,
        )

    expected_results = pd.DataFrame(
        {
            "levels": {0: ["l1", "l2"]},
            "lags": {0: np.array([1, 2, 3, 4, 5])},
            "lags_label": {0: np.array([1, 2, 3, 4, 5])},
            "params": {0: {"alpha": 0.1, "l1_ratio": 0.5}},
            "mean_squared_error__weighted_average": {0: 0.06321861327736888},
            "mean_squared_error__average": {0: 0.06321861327736888},
            "mean_squared_error__pooling": {0: 0.06321861327736888},
            "n_evaluated": {0: 40},
            "alpha": {0: 0.1},
            "l1_ratio": {0: 0.5},
        }
    )
    pd.testing.assert_frame_equal(results, expected_results)


def test_evaluate_grid_hyperparameters_multiseries_warns_when_all_params_raise_exceptions():
    """
    Test that a RuntimeWarning is raised and an empty DataFrame is returned
    when all parameter combinations in param_grid are invalid and raise
    exceptions during evaluation (TimeSeriesFold).
    """
    # All l1_ratio values are invalid for ElasticNet (must be in [0.0, 1.0])
    param_grid = [
        {"alpha": 0.1, "l1_ratio": 10},
        {"alpha": 0.5, "l1_ratio": 20},
    ]
    cv = TimeSeriesFold(steps=12, initial_train_size=30, refit=False)
    forecaster = ForecasterRecursiveMultiSeries(estimator=ElasticNet(), lags=5)

    warn_msg = re.escape(
        "Parameters skipped: 2 of 2 candidates raise an error, so the results "
        "are empty."
    )
    with pytest.warns(RuntimeWarning, match=warn_msg):
        results = _evaluate_grid_hyperparameters_multiseries(
            forecaster    = forecaster,
            series        = series_dict_dt,
            param_grid    = param_grid,
            cv            = cv,
            metric        = "mean_squared_error",
            return_best   = False,
            n_jobs        = "auto",
            verbose       = False,
            show_progress = False,
        )

    assert results.empty
    assert isinstance(results, pd.DataFrame)
    expected_columns = [
        'levels', 'lags', 'lags_label', 'params',
        'mean_squared_error__weighted_average',
        'mean_squared_error__average',
        'mean_squared_error__pooling',
        'n_evaluated',
    ]
    assert list(results.columns) == expected_columns


def test_evaluate_grid_hyperparameters_multiseries_warns_when_all_params_raise_exceptions_OneStepAheadFold():
    """
    Test that a RuntimeWarning is raised and an empty DataFrame is returned
    when all parameter combinations raise exceptions with OneStepAheadFold
    (multiseries).
    """
    param_grid = [
        {"alpha": 0.1, "l1_ratio": 10},
        {"alpha": 0.5, "l1_ratio": 20},
    ]
    cv = OneStepAheadFold(initial_train_size=30)
    forecaster = ForecasterRecursiveMultiSeries(estimator=ElasticNet(), lags=5)

    warn_msg = re.escape(
        "Parameters skipped: 2 of 2 candidates raise an error, so the results "
        "are empty."
    )
    with pytest.warns(RuntimeWarning, match=warn_msg):
        results = _evaluate_grid_hyperparameters_multiseries(
            forecaster    = forecaster,
            series        = series_dict_dt,
            param_grid    = param_grid,
            cv            = cv,
            metric        = "mean_squared_error",
            return_best   = False,
            verbose       = False,
            show_progress = False,
        )

    assert results.empty
    assert isinstance(results, pd.DataFrame)
    expected_columns = [
        'levels', 'lags', 'lags_label', 'params',
        'mean_squared_error__weighted_average',
        'mean_squared_error__average',
        'mean_squared_error__pooling',
        'n_evaluated',
    ]
    assert list(results.columns) == expected_columns


@pytest.mark.parametrize(
    "cv",
    [
        TimeSeriesFold(steps=10, initial_train_size=213, refit=False),
        OneStepAheadFold(initial_train_size=213),
    ],
    ids=lambda cv: f"cv: {type(cv).__name__}",
)
@pytest.mark.parametrize(
    "aggregate_metric",
    [
        ["weighted_average", "average", "pooling"],
        ["pooling", "average"],
    ],
    ids=lambda agg: f"aggregate_metric: {agg}",
)
def test_evaluate_grid_hyperparameters_multiseries_aggregated_metrics_match_their_column(
    cv, aggregate_metric
):
    """
    Test that each column `<metric>__<aggregation>` of the results contains
    the metric of its aggregation, whatever the order of `aggregate_metric`.
    The series are evaluated on different numbers of points, so the weighted
    average differs from the average. The expected values are those of a
    search with each aggregation alone.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LGBMRegressor(n_estimators=5, random_state=123, verbose=-1),
        lags=3,
    )
    kwargs = {
        "forecaster": forecaster,
        "series": series_dict_nans,
        "param_grid": [{"n_estimators": 5}, {"n_estimators": 10}],
        "cv": cv,
        "metric": ["mean_absolute_error", "mean_squared_error"],
        "return_best": False,
        "verbose": False,
        "show_progress": False,
    }

    def by_params(results, column):
        return results.set_index(results["params"].astype(str))[column].sort_index()

    results = _evaluate_grid_hyperparameters_multiseries(
        aggregate_metric=aggregate_metric, **kwargs
    )

    for aggregation in aggregate_metric:
        expected = _evaluate_grid_hyperparameters_multiseries(
            aggregate_metric=aggregation, **kwargs
        )
        for metric in ["mean_absolute_error", "mean_squared_error"]:
            column = f"{metric}__{aggregation}"
            pd.testing.assert_series_equal(
                by_params(results, column), by_params(expected, column)
            )
    weighted_average = _evaluate_grid_hyperparameters_multiseries(
        aggregate_metric="weighted_average", **kwargs
    )
    assert not np.allclose(
        by_params(results, "mean_absolute_error__average"),
        by_params(weighted_average, "mean_absolute_error__weighted_average"),
    )


@pytest.mark.parametrize(
    "cv",
    [
        TimeSeriesFold(initial_train_size=45, steps=5, refit=False),
        OneStepAheadFold(initial_train_size=45),
    ],
    ids=lambda cv: f"cv: {type(cv).__name__}",
)
@pytest.mark.parametrize(
    "levels, expected_n_evaluated",
    [
        (None, [30, 45]),
        (['l1', 'l2'], [30]),
        ('l1', [15]),
    ],
    ids=lambda x: f"{x}",
)
def test_evaluate_grid_hyperparameters_multiseries_n_evaluated_equal_to_backtesting(
    cv, levels, expected_n_evaluated
):
    """
    Test that the column `n_evaluated` of each candidate is the `n_evaluated`
    of the backtesting of the same configuration: the row of the aggregation
    (sum over the levels), or the level if only one is evaluated. With
    `OneStepAheadFold`, the backtesting with `steps=1` and `refit=False`
    evaluates the same points, also when `levels` is a subset of the series,
    and so do the metrics (`pooling` included).
    """
    forecaster = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=2)
    results = _evaluate_grid_hyperparameters_multiseries(
                  forecaster         = forecaster,
                  series             = series_short,
                  cv                 = cv,
                  param_grid         = [{'alpha': 0.1}, {'alpha': 1}],
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = ['pooling', 'average'],
                  levels             = levels,
                  lags_grid          = [2, 8],
                  return_best        = False,
                  verbose            = False,
                  show_progress      = False,
                  suppress_warnings  = True
              )

    if isinstance(cv, OneStepAheadFold):
        cv_backtesting = TimeSeriesFold(initial_train_size=45, steps=1, refit=False)
    else:
        cv_backtesting = cv
    metric_column = (
        'mean_absolute_error' if levels == 'l1' else 'mean_absolute_error__pooling'
    )
    row = 'l1' if levels == 'l1' else 'pooling'
    for lags, params, metric_value, n_evaluated in zip(
        results['lags'], results['params'], results[metric_column], results['n_evaluated']
    ):
        metrics, _ = backtesting_forecaster_multiseries(
            forecaster        = ForecasterRecursiveMultiSeries(
                                    estimator=Ridge(**params), lags=lags
                                ),
            series            = series_short,
            cv                = cv_backtesting,
            metric            = 'mean_absolute_error',
            levels            = levels,
            show_progress     = False,
            suppress_warnings = True
        )
        metrics = metrics.set_index('levels')
        assert n_evaluated == metrics.loc[row, 'n_evaluated']
        assert metric_value == pytest.approx(metrics.loc[row, 'mean_absolute_error'])

    assert results['n_evaluated'].dtype == 'int64'
    assert sorted(results['n_evaluated'].unique()) == expected_n_evaluated


@pytest.mark.parametrize(
    "cv",
    [
        TimeSeriesFold(initial_train_size=45, steps=5, refit=False),
        OneStepAheadFold(initial_train_size=45),
    ],
    ids=lambda cv: f"cv: {type(cv).__name__}",
)
@pytest.mark.parametrize(
    "lags_grid",
    [[2, 8], [8, 2]],
    ids=lambda lags_grid: f"lags_grid: {lags_grid}",
)
def test_evaluate_grid_hyperparameters_multiseries_MissingValuesWarning_once_when_candidates_evaluate_different_points(
    cv, lags_grid
):
    """
    Test that a single MissingValuesWarning is raised at the end of the search
    when some candidates evaluate less than 95% of the points of the most
    complete one, whatever the order of the grid. It lists the candidates affected,
    says that the best candidate is one of them and gives the series that
    every candidate evaluates on the same points. It covers the warning of the
    levels not predicted of the candidates, which is not shown. The warning
    filters are the same after the search.
    """
    forecaster = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=2)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        filters_before = list(warnings.filters)
        results = _evaluate_grid_hyperparameters_multiseries(
                      forecaster         = forecaster,
                      series             = series_short,
                      cv                 = cv,
                      param_grid         = [{'alpha': 0.1}, {'alpha': 1}],
                      metric             = 'mean_absolute_error',
                      lags_grid          = lags_grid,
                      return_best        = False,
                      verbose            = False,
                      show_progress      = False
                  )
        filters_after = list(warnings.filters)

    assert filters_after == filters_before
    messages = [
        str(w.message) for w in record
        if w.category is MissingValuesWarning and N_EVALUATED_WARNING_MSG in str(w.message)
    ]
    assert len(messages) == 1
    assert not any(
        str(w.message).startswith("Some levels are not predicted") for w in record
    )
    assert (
        "2 of 4 candidates are evaluated on fewer points than the most complete "
        "one, so their metrics are not comparable. The best candidate of the "
        "ranking is one of them, so the ranking may favor it. To compare the "
        "candidates on the same points, pass the series listed below in `levels` "
        "or remove the candidates with larger windows. See the column "
        "`n_evaluated` of the results.\n"
        "\n"
        "Candidates with less than 95% of the 45 points of the most complete one:\n"
        "  - lags: [1, 2, ..., 8], 2 candidates (rows 0, 1): 30 (66.7%)\n"
        "\n"
        "Series that every candidate evaluates on the same points:\n"
        "  levels=['l1', 'l2']"
    ) in messages[0]
    assert results['n_evaluated'].to_list() == [30, 30, 45, 45]


@pytest.mark.parametrize(
    "lags_grid",
    [[2, 5], [5, 2]],
    ids=lambda lags_grid: f"lags_grid: {lags_grid}",
)
def test_evaluate_grid_hyperparameters_multiseries_warnings_of_candidates_once_with_count(
    lags_grid
):
    """
    Test that the warnings of the candidates and of the refit are shown once at
    the end of the search, whatever the order of the grid: the warning of the
    levels not predicted with the number of candidates in which it is raised,
    and the warning of the bins of the residuals, raised only in the refit, as
    it is. The candidates evaluate more than 95% of the points of the most
    complete one, so there is no warning of the points evaluated.
    """
    forecaster = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=2)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _evaluate_grid_hyperparameters_multiseries(
            forecaster         = forecaster,
            series             = series_short_one_point,
            cv                 = TimeSeriesFold(initial_train_size=45, steps=5, refit=False),
            param_grid         = [{'alpha': 0.1}, {'alpha': 1}],
            metric             = 'mean_absolute_error',
            lags_grid          = lags_grid,
            return_best        = True,
            verbose            = False,
            show_progress      = False
        )

    assert [w.category for w in record] == [
        MissingValuesWarning, IgnoredArgumentWarning
    ]
    assert record[1].message.message.startswith(
        "The number of bins has been reduced from 10 to"
    )
    messages = [
        w.message.message for w in record if w.category is MissingValuesWarning
    ]
    assert len(messages) == 1
    assert messages[0].startswith(
        "Some levels are not predicted in folds where they have observed values, "
        "so their metrics, and the aggregated metrics, do not include those "
        "folds (in 2 of 4 candidates). Levels never predicted have NaN metrics "
        "and are left out of the aggregated metrics.\n\nLevels not predicted:\n"
        "  - Too short for the `window_size` (5) in the training set: "
        "'short' (never predicted)."
    )


def test_evaluate_grid_hyperparameters_multiseries_suppress_warnings_keeps_parameters_skipped():
    """
    Test that, with `suppress_warnings=True`, the skforecast warnings of the
    candidates are not shown, but the candidates skipped are.
    """
    forecaster = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=2)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results = _evaluate_grid_hyperparameters_multiseries(
                      forecaster         = forecaster,
                      series             = series_short,
                      cv                 = TimeSeriesFold(initial_train_size=45, steps=5, refit=False),
                      param_grid         = [{'alpha': 0.1}, {'alpha': 1}],
                      metric             = 'mean_absolute_error',
                      lags_grid          = [2, 8, 50],
                      return_best        = True,
                      verbose            = False,
                      show_progress      = False,
                      suppress_warnings  = True
                  )

    assert len(results) == 4
    assert [w.category for w in record] == [RuntimeWarning]
    message = str(record[0].message)
    assert message.startswith(
        "Parameters skipped: 2 of 6 candidates raise an error, so they are not "
        "in the results.\n\nValueError: "
    )
    assert message.endswith("  - lags: [1, 2, ..., 50] (2 candidates)")


@pytest.mark.parametrize(
    "levels, suppress_warnings",
    [(['l1', 'l2'], False), (None, True)],
    ids=lambda x: f"{x}",
)
def test_evaluate_grid_hyperparameters_multiseries_no_MissingValuesWarning_n_evaluated(
    levels, suppress_warnings
):
    """
    Test that the warning about the points evaluated is not raised when every
    candidate evaluates the same points (`levels` without the short series),
    or when `suppress_warnings=True`.
    """
    forecaster = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=2)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _evaluate_grid_hyperparameters_multiseries(
            forecaster         = forecaster,
            series             = series_short,
            cv                 = TimeSeriesFold(initial_train_size=45, steps=5, refit=False),
            param_grid         = [{'alpha': 0.1}, {'alpha': 1}],
            metric             = 'mean_absolute_error',
            levels             = levels,
            lags_grid          = [2, 8],
            return_best        = True,
            verbose            = False,
            show_progress      = False,
            suppress_warnings  = suppress_warnings
        )

    assert not any(N_EVALUATED_WARNING_MSG in str(w.message) for w in record)


def test_evaluate_grid_hyperparameters_multiseries_OneStepAheadFold_skips_candidates_without_data():
    """
    Test that, with a `OneStepAheadFold`, the candidates whose lags leave no
    data to train with, or no level of `levels` to evaluate in the test set,
    are skipped with the warning of the candidates skipped, grouped by error,
    instead of aborting the search or returning rows without points evaluated.
    `short` has 5 values in the training set, so it is not trained with 8 lags.
    """
    forecaster = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=2)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results = _evaluate_grid_hyperparameters_multiseries(
                      forecaster         = forecaster,
                      series             = series_short,
                      cv                 = OneStepAheadFold(initial_train_size=45),
                      param_grid         = [{'alpha': 0.1}, {'alpha': 1}],
                      metric             = 'mean_absolute_error',
                      levels             = ['short'],
                      lags_grid          = [2, 8, 50],
                      return_best        = False,
                      verbose            = False,
                      show_progress      = False
                  )

    assert results['lags_label'].map(len).tolist() == [2, 2]
    assert results['n_evaluated'].tolist() == [15, 15]
    messages = [str(w.message) for w in record if w.category is RuntimeWarning]
    assert messages == [
        "Parameters skipped: 4 of 6 candidates raise an error, so they are not in "
        "the results.\n\nValueError: No level of `levels` can be evaluated in the "
        "test set: none of them has more than `window_size` (8) observations in "
        "the training set, can be predicted after it and has observed values in "
        "the test set. Review `levels`, `initial_train_size` of `cv` or the span "
        "of the series.\n  - lags: [1, 2, ..., 8] (2 candidates)\n\nValueError: "
        "`initial_train_size` must be greater than the `window_size` of the "
        "forecaster (50), so that the first training set has data to train with. "
        "Got 45.\n  - lags: [1, 2, ..., 50] (2 candidates)"
    ]
