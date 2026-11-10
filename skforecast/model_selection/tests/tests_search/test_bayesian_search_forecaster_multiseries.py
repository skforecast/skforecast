# Unit test bayesian_search_forecaster_multiseries
# ==============================================================================
import os
import re
import platform
import pytest
from unittest.mock import patch
from joblib import Parallel, cpu_count
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from pytest import approx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.model_selection import backtesting_forecaster_multiseries
from skforecast.model_selection import bayesian_search_forecaster_multiseries
from skforecast.model_selection._split import TimeSeriesFold, OneStepAheadFold
from skforecast.preprocessing import RollingFeatures
from skforecast.exceptions import MissingValuesWarning, OneStepAheadValidationWarning
from skforecast.metrics import mean_absolute_scaled_error
from sklearn.metrics import mean_absolute_percentage_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm
from functools import partialmethod
import warnings

# Fixtures
from ..fixtures_model_selection_multiseries import (
    series_wide_range,
    series_wide_dt,
    series_long_dt,
    series_dict_range,
    series_dict_dt,
    series_wide_dt_item_sales,
    series_long_dt_item_sales,
    series_dict_dt_item_sales,
    exog_wide_dt_item_sales,
    exog_long_dt_item_sales,
    series_dict_nans,
    exog_dict_nans,
    series_dict_nans_train,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # hide progress bar

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


def test_TypeError_bayesian_search_forecaster_multiseries_when_cv_not_valid():
    """
    Test TypeError is raised in bayesian_search_forecaster_multiseries when cv is not
    a valid splitter.
    """
    class DummyCV:
        pass

    cv = DummyCV()
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot'
                 )
    
    def search_space(trial):  # pragma: no cover
        search_space  = {
            'alpha': trial.suggest_float('not_alpha', 1e-2, 1.0),
            'lags': trial.suggest_categorical('lags', [2, 4])
        }

        return search_space
    
    err_msg = re.escape(
        f"`cv` must be an instance of `TimeSeriesFold` or `OneStepAheadFold`. "
        f"Got {type(cv)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series_dict_range,
            search_space       = search_space,
            cv                 = cv,
            metric             = 'mean_absolute_error',
            aggregate_metric   = 'not_valid',
            n_trials           = 10,
            random_state       = 123,
            return_best        = False,
            verbose            = False,
        )


def test_ValueError_bayesian_search_forecaster_multiseries_when_not_allowed_aggregate_metric():
    """
    Test ValueError is raised in bayesian_search_forecaster_multiseries when 
    `aggregate_metric` has not a valid value.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot'
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_range['l1']) - 12,
            steps              = 3,
            refit              = True,
            fixed_train_size   = True
        )
    
    def search_space(trial):  # pragma: no cover
        search_space  = {'alpha': trial.suggest_float('not_alpha', 1e-2, 1.0)}

        return search_space

    err_msg = re.escape(
        "Allowed `aggregate_metric` are: ['average', 'weighted_average', 'pooling']. "
        "Got: ['not_valid']."
    )
    with pytest.raises(ValueError, match = err_msg):
        bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series_dict_range,
            search_space       = search_space,
            cv                 = cv,
            metric             = 'mean_absolute_error',
            aggregate_metric   = 'not_valid',
            n_trials           = 10,
            random_state       = 123,
            return_best        = False,
            verbose            = False,
        )


def test_ValueError_bayesian_search_forecaster_multiseries_metric_list_duplicate_names():
    """
    Test ValueError is raised in bayesian_search_forecaster_multiseries when a `list` 
    of metrics is used with duplicate names.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot'
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_range['l1']) - 12,
            steps              = 3,
            refit              = True,
            fixed_train_size   = True
        )
    
    def search_space(trial):  # pragma: no cover
        search_space  = {'alpha': trial.suggest_float('not_alpha', 1e-2, 1.0)}

        return search_space

    err_msg = re.escape("When `metric` is a `list`, each metric name must be unique.")
    with pytest.raises(ValueError, match = err_msg):
        bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series_dict_range,
            search_space       = search_space,
            cv                 = cv,
            metric             = ['mean_absolute_error', mean_absolute_error],
            aggregate_metric   = 'weighted_average',
            n_trials           = 10,
            random_state       = 123,
            return_best        = False,
            verbose            = False,
        )


def test_ValueError_bayesian_search_forecaster_multiseries_when_metric_named_n_evaluated():
    """
    Test ValueError is raised in bayesian_search_forecaster_multiseries, before
    the first trial, when a metric is named `n_evaluated`, the column of the
    backtesting metrics with the number of points evaluated.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot'
                 )
    cv = TimeSeriesFold(initial_train_size=len(series_dict_range['l1']) - 12, steps=3)

    def search_space(trial):  # pragma: no cover
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}

        return search_space

    def n_evaluated(y_true, y_pred):  # pragma: no cover
        return mean_absolute_error(y_true, y_pred)

    err_msg = re.escape(
        "The backtesting metrics include the column 'n_evaluated' with the "
        "number of points evaluated. Rename the metric 'n_evaluated' passed in "
        "`metric`."
    )
    with pytest.raises(ValueError, match = err_msg):
        bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series_dict_range,
            search_space       = search_space,
            cv                 = cv,
            metric             = n_evaluated,
            n_trials           = 10,
            return_best        = False,
            verbose            = False,
        )


def test_ValueError_bayesian_search_forecaster_multiseries_when_search_space_names_do_not_match():
    """
    Test ValueError is raised when search_space key name do not match the trial 
    object name from optuna.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot'
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_range['l1']) - 12,
            steps              = 3,
            refit              = True,
            fixed_train_size   = True
        )

    def search_space(trial):
        search_space  = {'alpha': trial.suggest_float('not_alpha', 1e-2, 1.0)}

        return search_space
    
    err_msg = re.escape(
        "`search_space` dict keys must match the names passed to "
        "`trial.suggest_*()`.\n"
        "  Dict keys    : ['alpha']\n"
        "  Suggest names: ['not_alpha']"
    )
    with pytest.raises(ValueError, match = err_msg):
        bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series_dict_range,
            cv                 = cv,
            search_space       = search_space,
            metric             = 'mean_absolute_error',
            aggregate_metric   = 'weighted_average',
            n_trials           = 10,
            random_state       = 123,
            return_best        = False,
            verbose            = False,
        )


def test_ValueError_bayesian_search_forecaster_multiseries_when_search_space_names_do_not_match_OneStepAheadFold():
    """
    Test ValueError is raised when search_space key name do not match the trial 
    object name from optuna using OneStepAheadFold.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot'
                 )
    cv = OneStepAheadFold(
            initial_train_size = len(series_dict_range['l1']) - 12,
        )

    def search_space(trial):
        search_space = {'alpha': trial.suggest_float('not_alpha', 1e-2, 1.0)}

        return search_space

    err_msg = re.escape(
        "`search_space` dict keys must match the names passed to "
        "`trial.suggest_*()`.\n"
        "  Dict keys    : ['alpha']\n"
        "  Suggest names: ['not_alpha']"
    )
    with pytest.raises(ValueError, match=err_msg):
        bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series_dict_range,
            cv                 = cv,
            search_space       = search_space,
            metric             = 'mean_absolute_error',
            aggregate_metric   = 'weighted_average',
            n_trials           = 10,
            random_state       = 123,
            return_best        = False,
            verbose            = False,
        )


# This mark allows to only run test with "slow" label or all except this, "not slow".
# The mark should be included in the pytest.ini file
# pytest -m slow --verbose
# pytest -m "not slow" --verbose
@pytest.mark.slow
def test_results_output_bayesian_search_forecaster_multiseries_with_mocked_when_lags_grid_dict():
    """
    Test output of bayesian_search_forecaster_multiseries in ForecasterRecursiveMultiSeries 
    when `lags_grid` is a dict with mocked (mocked done in skforecast v0.12.0).
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = RandomForestRegressor(random_state=123, n_jobs=1),
                     lags      = 2,
                     encoding  = 'onehot',
                     transformer_series = StandardScaler()
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_range['l1']) - 12,
            steps              = 3,
            refit              = True,
            fixed_train_size   = True
        )

    def search_space(trial):
        search_space  = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 20),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 1., log=True),
            'max_features': trial.suggest_categorical('max_features', ['log2', 'sqrt']),
            'lags': trial.suggest_categorical('lags', [2, 4])
        }
        
        return search_space

    results = bayesian_search_forecaster_multiseries(
                  forecaster         = forecaster,
                  series             = series_dict_range,
                  cv                 = cv,
                  search_space       = search_space,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False
              )[0]

    if platform.system() == 'Darwin':
        expected_results = pd.DataFrame(
            np.array([
                [['l1', 'l2'],
                    np.array([1, 2]),
                    {'n_estimators': 13, 'min_samples_leaf': 0.20142843988664705, 'max_features': 'sqrt'},
                    0.21000466923864858,
                    13,
                    0.20142843988664705,
                    'sqrt'],
                [['l1', 'l2'],
                    np.array([1, 2]),
                    {'n_estimators': 11, 'min_samples_leaf': 0.2714570796881702, 'max_features': 'sqrt'},
                    0.2108843883798094,
                    11,
                    0.2714570796881702,
                    'sqrt'],
                [['l1', 'l2'],
                    np.array([1, 2]),
                    {'n_estimators': 17, 'min_samples_leaf': 0.19325882509735576, 'max_features': 'sqrt'},
                    0.21223533093219435,
                    17,
                    0.19325882509735576,
                    'sqrt'],
                [['l1', 'l2'],
                    np.array([1, 2]),
                    {'n_estimators': 14, 'min_samples_leaf': 0.1147302385573586, 'max_features': 'sqrt'},
                    0.21315705728046805,
                    14,
                    0.1147302385573586,
                    'sqrt'],
                [['l1', 'l2'],
                    np.array([1, 2, 3, 4]),
                    {'n_estimators': 13, 'min_samples_leaf': 0.2599119286878713, 'max_features': 'log2'},
                    0.21559787742691414,
                    13,
                    0.2599119286878713,
                    'log2'],
                [['l1', 'l2'],
                    np.array([1, 2, 3, 4]),
                    {'n_estimators': 20, 'min_samples_leaf': 0.48398258917593734, 'max_features': 'log2'},
                    0.21615288060051888,
                    20,
                    0.48398258917593734,
                    'log2'],
                [['l1', 'l2'],
                    np.array([1, 2]),
                    {'n_estimators': 15, 'min_samples_leaf': 0.41010449151752726, 'max_features': 'sqrt'},
                    0.2163760081019307,
                    15,
                    0.41010449151752726,
                    'sqrt'],
                [['l1', 'l2'],
                    np.array([1, 2]),
                    {'n_estimators': 14, 'min_samples_leaf': 0.7823285204656389, 'max_features': 'log2'},
                    0.21653840486608775,
                    14,
                    0.7823285204656389,
                    'log2'],
                [['l1', 'l2'],
                    np.array([1, 2]),
                    {'n_estimators': 15, 'min_samples_leaf': 0.34027307604369605, 'max_features': 'sqrt'},
                    0.2166204088831145,
                    15,
                    0.34027307604369605,
                    'sqrt'],
                [['l1', 'l2'],
                    np.array([1, 2, 3, 4]),
                    {'n_estimators': 17, 'min_samples_leaf': 0.21035794225904134, 'max_features': 'log2'},
                    0.21732156972764352,
                    17,
                    0.21035794225904134,
                    'log2']], dtype=object),
            columns=['levels', 'lags', 'params', 'mean_absolute_error__weighted_average', 'n_estimators', 'min_samples_leaf', 'max_features'],
            index=pd.RangeIndex(start=0, stop=10, step=1)
        ).astype({'mean_absolute_error__weighted_average': float, 'n_estimators': int, 'min_samples_leaf': float})
    else:
        expected_results = pd.DataFrame(
            np.array([
                [['l1', 'l2'],
                    np.array([1, 2]),
                    {'n_estimators': 13, 'min_samples_leaf': 0.20142843988664705, 'max_features': 'sqrt'},
                    0.21000466923864858,
                    13,
                    0.20142843988664705,
                    'sqrt'],
                [['l1', 'l2'],
                    np.array([1, 2]),
                    {'n_estimators': 11, 'min_samples_leaf': 0.2714570796881701, 'max_features': 'sqrt'},
                    0.2108843883798094,
                    11,
                    0.2714570796881701,
                    'sqrt'],
                [['l1', 'l2'],
                    np.array([1, 2]),
                    {'n_estimators': 17, 'min_samples_leaf': 0.19325882509735576, 'max_features': 'sqrt'},
                    0.21223533093219435,
                    17,
                    0.19325882509735576,
                    'sqrt'],
                [['l1', 'l2'],
                    np.array([1, 2]),
                    {'n_estimators': 14, 'min_samples_leaf': 0.1147302385573586, 'max_features': 'sqrt'},
                    0.21340636192242327,
                    14,
                    0.1147302385573586,
                    'sqrt'],
                [['l1', 'l2'],
                    np.array([1, 2, 3, 4]),
                    {'n_estimators': 13, 'min_samples_leaf': 0.2599119286878713, 'max_features': 'log2'},
                    0.21559787742691414,
                    13,
                    0.2599119286878713,
                    'log2'],
                [['l1', 'l2'],
                    np.array([1, 2, 3, 4]),
                    {'n_estimators': 20, 'min_samples_leaf': 0.4839825891759374, 'max_features': 'log2'},
                    0.21615288060051888,
                    20,
                    0.4839825891759374,
                    'log2'],
                [['l1', 'l2'],
                    np.array([1, 2]),
                    {'n_estimators': 15, 'min_samples_leaf': 0.41010449151752726, 'max_features': 'sqrt'},
                    0.2163760081019307,
                    15,
                    0.41010449151752726,
                    'sqrt'],
                [['l1', 'l2'],
                    np.array([1, 2]),
                    {'n_estimators': 14, 'min_samples_leaf': 0.782328520465639, 'max_features': 'log2'},
                    0.21653840486608775,
                    14,
                    0.782328520465639,
                    'log2'],
                [['l1', 'l2'],
                    np.array([1, 2]),
                    {'n_estimators': 15, 'min_samples_leaf': 0.34027307604369605, 'max_features': 'sqrt'},
                    0.2166204088831145,
                    15,
                    0.34027307604369605,
                    'sqrt'],
                [['l1', 'l2'],
                    np.array([1, 2, 3, 4]),
                    {'n_estimators': 17, 'min_samples_leaf': 0.21035794225904136, 'max_features': 'log2'},
                    0.21732156972764352,
                    17,
                    0.21035794225904136,
                    'log2']], dtype=object),
            columns=['levels', 'lags', 'params', 'mean_absolute_error__weighted_average', 'n_estimators', 'min_samples_leaf', 'max_features'],
            index=pd.RangeIndex(start=0, stop=10, step=1)
        ).astype({'mean_absolute_error__weighted_average': float, 'n_estimators': int, 'min_samples_leaf': float})
    expected_results.insert(4, 'n_evaluated', [24] * 10)

    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results, check_dtype=False)


def test_results_output_bayesian_search_forecaster_multiseries_ForecasterRecursiveMultiSeries_with_levels():
    """
    Test output of bayesian_search_forecaster_multiseries in 
    ForecasterRecursiveMultiSeries for level 'l1' with mocked 
    (mocked done in skforecast v0.12.0).
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot',
                     transformer_series = StandardScaler()
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_range['l1']) - 12,
            steps              = 3,
            refit              = True,
            fixed_train_size   = True
        )
    levels = ['l1']

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1.0),
            'lags': trial.suggest_categorical('lags', [2, 4])
        }
        
        return search_space

    results = bayesian_search_forecaster_multiseries(
                  forecaster         = forecaster,
                  series             = series_dict_range,
                  cv                 = cv,
                  levels             = levels,
                  search_space       = search_space,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([
            [['l1'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.23598059857016607},
                0.2158500990715522,
                0.23598059857016607],
            [['l1'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.398196343012209},
                0.2158706018070041,
                0.398196343012209],
            [['l1'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.4441865222328282},
                0.21587636323102785,
                0.4441865222328282],
            [['l1'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.53623586010342},
                0.21588782726158717,
                0.53623586010342],
            [['l1'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.7252189487445193},
                0.21591108476752116,
                0.7252189487445193],
            [['l1'],
                np.array([1, 2]),
                {'alpha': 0.5558016213920624},
                0.21631472411547312,
                0.5558016213920624],
            [['l1'],
                np.array([1, 2]),
                {'alpha': 0.6995044937418831},
                0.21631966674575429,
                0.6995044937418831],
            [['l1'],
                np.array([1, 2]),
                {'alpha': 0.7406154516747153},
                0.2163210721487403,
                0.7406154516747153],
            [['l1'],
                np.array([1, 2]),
                {'alpha': 0.8509374761370117},
                0.21632482482968562,
                0.8509374761370117],
            [['l1'],
                np.array([1, 2]),
                {'alpha': 0.9809565564007693},
                0.2163292127503296,
                0.9809565564007693]], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error', 'alpha'],
        index=pd.RangeIndex(start=0, stop=10, step=1)
    ).astype({'mean_absolute_error': float, 'alpha': float})
    expected_results.insert(4, 'n_evaluated', [12] * 10)

    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results, check_dtype=False)


def test_results_output_bayesian_search_forecaster_multiseries_ForecasterRecursiveMultiSeries_with_multiple_metrics_aggregated():
    """
    Test output of bayesian_search_forecaster_multiseries in ForecasterRecursiveMultiSeries
    with multiple metrics and aggregated metrics (mocked done in skforecast v0.12.0).
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot',
                     transformer_series = StandardScaler()
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_range['l1']) - 12,
            steps              = 3,
            refit              = True,
            fixed_train_size   = True
        )

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1.0),
            'lags': trial.suggest_categorical('lags', [2, 4])
        }

        return search_space

    results = bayesian_search_forecaster_multiseries(
                  forecaster         = forecaster,
                  series             = series_dict_range,
                  search_space       = search_space,
                  cv                 = cv,
                  metric             = ['mean_absolute_error', 'mean_absolute_scaled_error'],
                  aggregate_metric   = ['weighted_average', 'average', 'pooling'],
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False
              )[0]

    expected_results = pd.DataFrame({
        'levels': [
            ['l1', 'l2'],
            ['l1', 'l2'],
            ['l1', 'l2'],
            ['l1', 'l2'],
            ['l1', 'l2'],
            ['l1', 'l2'],
            ['l1', 'l2'],
            ['l1', 'l2'],
            ['l1', 'l2'],
            ['l1', 'l2'],
        ],
        'lags': [
            np.array([1, 2]),
            np.array([1, 2]),
            np.array([1, 2]),
            np.array([1, 2]),
            np.array([1, 2]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
        ],
        'params': [
            {'alpha': 0.5558016213920624},
            {'alpha': 0.6995044937418831},
            {'alpha': 0.7406154516747153},
            {'alpha': 0.8509374761370117},
            {'alpha': 0.9809565564007693},
            {'alpha': 0.23598059857016607},
            {'alpha': 0.398196343012209},
            {'alpha': 0.4441865222328282},
            {'alpha': 0.53623586010342},
            {'alpha': 0.7252189487445193},
        ],
        'mean_absolute_error__weighted_average': [
            0.21324663796176382,
            0.2132571094660072,
            0.21326009091608622,
            0.21326806055662118,
            0.2132773952926551,
            0.21476196207156512,
            0.21477679099211167,
            0.21478095843883202,
            0.2147892513261171,
            0.21480607764821474,
        ],
        'mean_absolute_error__average': [
            0.21324663796176382,
            0.21325710946600718,
            0.21326009091608622,
            0.21326806055662115,
            0.2132773952926551,
            0.21476196207156514,
            0.21477679099211167,
            0.21478095843883202,
            0.21478925132611706,
            0.21480607764821472,
        ],
        'mean_absolute_error__pooling': [
            0.21324663796176382,
            0.21325710946600726,
            0.21326009091608622,
            0.21326806055662118,
            0.21327739529265513,
            0.21476196207156514,
            0.21477679099211167,
            0.21478095843883202,
            0.21478925132611706,
            0.21480607764821472,
        ],
        'mean_absolute_scaled_error__weighted_average': [
            0.7923109431516052,
            0.7923475860368283,
            0.7923580182679315,
            0.7923859027861416,
            0.7924185605513081,
            0.7880617091025902,
            0.7881186058178458,
            0.7881345955402543,
            0.788166413485017,
            0.788230970932795,
        ],
        'mean_absolute_scaled_error__average': [
            0.7923109431516052,
            0.7923475860368283,
            0.7923580182679316,
            0.7923859027861416,
            0.7924185605513081,
            0.7880617091025902,
            0.7881186058178459,
            0.7881345955402543,
            0.7881664134850169,
            0.788230970932795,
        ],
        'mean_absolute_scaled_error__pooling': [
            0.7819568303260771,
            0.7819952284192204,
            0.7820061611367338,
            0.7820353851137734,
            0.7820696147769971,
            0.7760652225280101,
            0.776118808411715,
            0.7761338679242804,
            0.7761638351556942,
            0.7762246388626313,
        ],
        'n_evaluated': [24] * 10,
        'alpha': [
            0.5558016213920624,
            0.6995044937418831,
            0.7406154516747153,
            0.8509374761370117,
            0.9809565564007693,
            0.23598059857016607,
            0.398196343012209,
            0.4441865222328282,
            0.53623586010342,
            0.7252189487445193,
        ],
    })

    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results, check_dtype=False)


def test_results_output_bayesian_search_forecaster_multiseries_ForecasterRecursiveMultiSeries_window_features_multiple_metrics_aggregated():
    """
    Test output of bayesian_search_forecaster_multiseries in ForecasterRecursiveMultiSeries
    with window features and multiple metrics and aggregated metrics
    (mocked done in skforecast v0.12.0).
    """
    window_features = RollingFeatures(
        stats=['mean', 'std', 'min', 'max', 'sum', 'median', 'ratio_min_max', 'coef_variation'],
        window_sizes=3,
    )
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = Ridge(random_state=123),
                     lags      = 2,
                     window_features = window_features,
                     encoding  = 'onehot',
                     transformer_series = StandardScaler()
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_long_dt.loc['l1']) - 12,
            steps              = 3,
            refit              = True,
            fixed_train_size   = True
        )

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1.0),
            'lags': trial.suggest_categorical('lags', [2, 4])
        }

        return search_space

    results = bayesian_search_forecaster_multiseries(
                  forecaster         = forecaster,
                  series             = series_long_dt,
                  search_space       = search_space,
                  cv                 = cv,
                  metric             = ['mean_absolute_error', 'mean_absolute_scaled_error'],
                  aggregate_metric   = ['weighted_average', 'average', 'pooling'],
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False
              )[0]

    expected_results = pd.DataFrame({
        'levels': [
            ['l1', 'l2'],
            ['l1', 'l2'],
            ['l1', 'l2'],
            ['l1', 'l2'],
            ['l1', 'l2'],
            ['l1', 'l2'],
            ['l1', 'l2'],
            ['l1', 'l2'],
            ['l1', 'l2'],
            ['l1', 'l2'],
        ],
        'lags': [
            np.array([1, 2]),
            np.array([1, 2]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2]),
            np.array([1, 2]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
        ],
        'params': [
            {'alpha': 0.9809565564007693},
            {'alpha': 0.8509374761370117},
            {'alpha': 0.7252189487445193},
            {'alpha': 0.7406154516747153},
            {'alpha': 0.6995044937418831},
            {'alpha': 0.53623586010342},
            {'alpha': 0.5558016213920624},
            {'alpha': 0.4441865222328282},
            {'alpha': 0.398196343012209},
            {'alpha': 0.23598059857016607},
        ],
        'mean_absolute_error__weighted_average': [
            0.266209774594638,
            0.2663778037208765,
            0.2664010145777917,
            0.2665530523449524,
            0.2666288782988022,
            0.26683839212167015,
            0.2669606099393226,
            0.26716247569103446,
            0.26737008166011794,
            0.2685748853275215,
        ],
        'mean_absolute_error__average': [
            0.26620977459463796,
            0.2663778037208765,
            0.2664010145777917,
            0.2665530523449524,
            0.2666288782988022,
            0.26683839212167015,
            0.2669606099393225,
            0.26716247569103446,
            0.26737008166011794,
            0.26857488532752155,
        ],
        'mean_absolute_error__pooling': [
            0.26620977459463796,
            0.26637780372087644,
            0.2664010145777917,
            0.2665530523449524,
            0.2666288782988022,
            0.26683839212167015,
            0.2669606099393225,
            0.26716247569103446,
            0.26737008166011794,
            0.2685748853275215,
        ],
        'mean_absolute_scaled_error__weighted_average': [
            0.9930667497722298,
            0.9937439823757044,
            0.9923479377236472,
            0.9944435611595945,
            0.994744372168292,
            0.9941069879981377,
            0.9960491623317846,
            0.9953844945941341,
            0.996195904784012,
            1.0008499618646218,
        ],
        'mean_absolute_scaled_error__average': [
            0.9930667497722299,
            0.9937439823757045,
            0.9923479377236472,
            0.9944435611595946,
            0.994744372168292,
            0.9941069879981376,
            0.9960491623317846,
            0.9953844945941341,
            0.996195904784012,
            1.000849961864622,
        ],
        'mean_absolute_scaled_error__pooling': [
            0.969431956811412,
            0.9700438532186673,
            0.9626684384225735,
            0.970682040215464,
            0.970958168704471,
            0.9642489487589695,
            0.9721662056891969,
            0.9654200592524217,
            0.9661702655321596,
            0.9705239518983445,
        ],
        'n_evaluated': [24] * 10,
        'alpha': [
            0.9809565564007693,
            0.8509374761370117,
            0.7252189487445193,
            0.7406154516747153,
            0.6995044937418831,
            0.53623586010342,
            0.5558016213920624,
            0.4441865222328282,
            0.398196343012209,
            0.23598059857016607,
        ],
    })

    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results, check_dtype=False)


def test_results_output_bayesian_search_forecaster_multiseries_with_kwargs_create_study():
    """
    Test output of bayesian_search_forecaster_multiseries in ForecasterRecursiveMultiSeries 
    when `kwargs_create_study` with mocked (mocked done in skforecast v0.12.0).
    All trials are startup trials of the sampler (`n_startup_trials=10`), so the
    results only depend on the seed and not on the internals of TPE, which
    change between optuna versions.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot',
                     transformer_series = StandardScaler()
                 )

    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_dt['l1']) - 12,
            steps              = 3,
            refit              = False,
            fixed_train_size   = True
        )

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 2e-2, 2.0),
            'lags': trial.suggest_categorical('lags', [2, 4])
        }
        
        return search_space

    kwargs_create_study = {
        "sampler": TPESampler(seed=123, n_startup_trials=10, n_ei_candidates=18)
    }
    results = bayesian_search_forecaster_multiseries(
                  forecaster          = forecaster,
                  series              = series_dict_dt,
                  cv                  = cv,
                  search_space        = search_space,
                  metric              = 'mean_absolute_error',
                  aggregate_metric    = 'weighted_average',
                  n_trials            = 10,
                  random_state        = 123,
                  return_best         = False,
                  verbose             = False,
                  kwargs_create_study = kwargs_create_study
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.47196119714033213},
                0.20881449475639347,
                0.47196119714033213],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.796392686024418},
                0.20883044337865528,
                0.796392686024418],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.8883730444656563},
                0.20883491982666835,
                0.8883730444656563],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 1.07247172020684},
                0.2088438203726152,
                1.07247172020684],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 1.4504378974890386},
                0.20886185085049824,
                1.4504378974890386],
            [['l1', 'l2'],
                np.array([1, 2]),
                {'alpha': 1.1116032427841247},
                0.20914976196539467,
                1.1116032427841247],
            [['l1', 'l2'],
                np.array([1, 2]),
                {'alpha': 1.3990089874837661},
                0.20915713939422753,
                1.3990089874837661],
            [['l1', 'l2'],
                np.array([1, 2]),
                {'alpha': 1.4812309033494306},
                0.2091592392813534,
                1.4812309033494306],
            [['l1', 'l2'],
                np.array([1, 2]),
                {'alpha': 1.7018749522740233},
                0.20916485106998076,
                1.7018749522740233],
            [['l1', 'l2'],
                np.array([1, 2]),
                {'alpha': 1.9619131128015386},
                0.20917142150909918,
                1.9619131128015386]], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error__weighted_average', 'alpha'],
        index=pd.RangeIndex(start=0, stop=10, step=1)
    ).astype({'mean_absolute_error__weighted_average': float, 'alpha': float})
    expected_results.insert(4, 'n_evaluated', [24] * 10)

    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results)


def test_results_output_bayesian_search_forecaster_multiseries_with_kwargs_study_optimize():
    """
    Test output of bayesian_search_forecaster_multiseries in ForecasterRecursiveMultiSeries 
    when `kwargs_study_optimize` with mocked (mocked done in skforecast v0.12.0).
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot',
                     transformer_series = StandardScaler()
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_dt['l1']) - 12,
            steps              = 3,
            refit              = False,
            fixed_train_size   = True
        )

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1.0),
            'lags': trial.suggest_categorical('lags', [2, 4])
        }
         
        return search_space

    kwargs_study_optimize = {'gc_after_trial': True}
    results = bayesian_search_forecaster_multiseries(
                  forecaster            = forecaster,
                  series                = series_dict_dt,
                  cv                    = cv,
                  search_space          = search_space,
                  metric                = 'mean_absolute_error',
                  aggregate_metric      = 'weighted_average',
                  n_trials              = 10,
                  random_state          = 123,
                  return_best           = False,
                  verbose               = False,
                  kwargs_study_optimize = kwargs_study_optimize
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.23598059857016607},
                0.20880273566554797,
                0.23598059857016607],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.398196343012209},
                0.20881083349292262,
                0.398196343012209],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.4441865222328282},
                0.2088131177203144,
                0.4441865222328282],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.53623586010342},
                0.20881767431515372,
                0.53623586010342],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.7252189487445193},
                0.20882696592349723,
                0.7252189487445193],
            [['l1', 'l2'],
                np.array([1, 2]),
                {'alpha': 0.5558016213920624},
                0.20913532870732662,
                0.5558016213920624],
            [['l1', 'l2'],
                np.array([1, 2]),
                {'alpha': 0.6995044937418831},
                0.20913908163271674,
                0.6995044937418831],
            [['l1', 'l2'],
                np.array([1, 2]),
                {'alpha': 0.7406154516747153},
                0.209140152549569,
                0.7406154516747153],
            [['l1', 'l2'],
                np.array([1, 2]),
                {'alpha': 0.8509374761370117},
                0.20914302038975388,
                0.8509374761370117],
            [['l1', 'l2'],
                np.array([1, 2]),
                {'alpha': 0.9809565564007693},
                0.20914638910039662,
                0.9809565564007693]], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error__weighted_average', 'alpha'],
        index=pd.RangeIndex(start=0, stop=10, step=1)
    ).astype({'mean_absolute_error__weighted_average': float, 'alpha': float})
    expected_results.insert(4, 'n_evaluated', [24] * 10)

    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results)


def test_results_output_bayesian_search_forecaster_multiseries_when_lags_is_not_provided():
    """
    Test output of bayesian_search_forecaster_multiseries in ForecasterRecursiveMultiSeries 
    when `lags` is not provided (mocked done in skforecast v0.12.0), 
    should use forecaster.lags.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = Ridge(random_state=123),
                     lags      = 4,
                     encoding  = 'onehot',
                     transformer_series = StandardScaler()
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_dt['l1']) - 12,
            steps              = 3,
            refit              = True,
            fixed_train_size   = True
        )

    def search_space(trial):
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}
        
        return search_space

    results = bayesian_search_forecaster_multiseries(
                  forecaster         = forecaster,
                  series             = series_dict_dt,
                  cv                 = cv,
                  search_space       = search_space,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.2345829390285611},
                0.21476183342122757,
                0.2345829390285611],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.29327794160087567},
                0.21476722307839205,
                0.29327794160087567],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.398196343012209},
                0.21477679099211167,
                0.398196343012209],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.42887539552321635},
                0.21477957279597848,
                0.42887539552321635],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.48612258246951734},
                0.21478474449018078,
                0.48612258246951734],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.5558016213920624},
                0.21479100579021138,
                0.5558016213920624],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.6879814411990146},
                0.21480278321679924,
                0.6879814411990146],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.6995044937418831},
                0.2148038037681593,
                0.6995044937418831],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.7222742800877074},
                0.2148058175049471,
                0.7222742800877074],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.9809565564007693},
                0.21482842793875442,
                0.9809565564007693]], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error__weighted_average', 'alpha'],
        index=pd.RangeIndex(start=0, stop=10, step=1)
    ).astype({'mean_absolute_error__weighted_average': float, 'alpha': float})
    expected_results.insert(4, 'n_evaluated', [24] * 10)

    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results)


def test_results_output_bayesian_search_forecaster_multiseries_ForecasterDirectMultiVariate():
    """
    Test output of bayesian_search_forecaster_multiseries in 
    ForecasterDirectMultiVariate with mocked (mocked done in skforecast v0.12.0).
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
            refit              = True,
            fixed_train_size   = True
    )

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1.0),
            'lags': trial.suggest_categorical('lags', [2, 4])
        }
        
        return search_space

    results = bayesian_search_forecaster_multiseries(
                  forecaster         = forecaster,
                  series             = series_wide_range,
                  cv                 = cv,
                  search_space       = search_space,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([
            [['l1'],
                np.array([1, 2]),
                {'alpha': 0.5558016213920624},
                0.20494430799685603,
                0.5558016213920624],
            [['l1'],
                np.array([1, 2]),
                {'alpha': 0.6995044937418831},
                0.2054241757124218,
                0.6995044937418831],
            [['l1'],
                np.array([1, 2]),
                {'alpha': 0.7406154516747153},
                0.2055505658009149,
                0.7406154516747153],
            [['l1'],
                np.array([1, 2]),
                {'alpha': 0.8509374761370117},
                0.20586868696104654,
                0.8509374761370117],
            [['l1'],
                np.array([1, 2]),
                {'alpha': 0.9809565564007693},
                0.20620855083897363,
                0.9809565564007693],
            [['l1'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.7252189487445193},
                0.2170248206892771,
                0.7252189487445193],
            [['l1'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.53623586010342},
                0.21761843345398177,
                0.53623586010342],
            [['l1'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.4441865222328282},
                0.21794402393310322,
                0.4441865222328282],
            [['l1'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.398196343012209},
                0.21811674876142448,
                0.398196343012209],
            [['l1'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.23598059857016607},
                0.21912194726679404,
                0.23598059857016607]], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error', 'alpha'],
        index=pd.RangeIndex(start=0, stop=10, step=1)
    ).astype({'mean_absolute_error': float, 'alpha': float})
    expected_results.insert(4, 'n_evaluated', [12] * 10)

    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results)


def test_results_output_bayesian_search_forecaster_multiseries_ForecasterDirectMultiVariate_lags_dict():
    """
    Test output of bayesian_search_forecaster_multiseries in 
    ForecasterDirectMultiVariate when lags is a dict 
    with mocked (mocked done in skforecast v0.12.0).
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
            refit              = True,
            fixed_train_size   = True
    )

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1.0),
            'lags': trial.suggest_categorical('lags', [{'l1': 2, 'l2': [1, 3]}, 
                                                       {'l1': None, 'l2': [1, 3]}, 
                                                       {'l1': [1, 3], 'l2': None}])
        }
        
        return search_space

    results = bayesian_search_forecaster_multiseries(
                  forecaster         = forecaster,
                  series             = series_wide_range,
                  cv                 = cv,
                  search_space       = search_space,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([
            [['l1'],
                {'l1': np.array([1, 2]), 'l2': np.array([1, 3])},
                {'alpha': 0.30077690592494105},
                0.20844762947854312,
                0.30077690592494105],
            [['l1'],
                {'l1': np.array([1, 2]), 'l2': np.array([1, 3])},
                {'alpha': 0.4365541356963474},
                0.20880336411565956,
                0.4365541356963474],
            [['l1'],
                {'l1': np.array([1, 2]), 'l2': np.array([1, 3])},
                {'alpha': 0.6380569489658079},
                0.2092371153650312,
                0.6380569489658079],
            [['l1'],
                {'l1': None, 'l2': np.array([1, 3])},
                {'alpha': 0.7252189487445193},
                0.21685083725475654,
                0.7252189487445193],
            [['l1'],
                {'l1': None, 'l2': np.array([1, 3])},
                {'alpha': 0.7222742800877074},
                0.2168551702095223,
                0.7222742800877074],
            [['l1'],
                {'l1': None, 'l2': np.array([1, 3])},
                {'alpha': 0.43208779389318014},
                0.21733651515831423,
                0.43208779389318014],
            [['l1'],
                {'l1': np.array([1, 3]), 'l2': None},
                {'alpha': 0.6995044937418831},
                0.22066810286127028,
                0.6995044937418831],
            [['l1'],
                {'l1': np.array([1, 3]), 'l2': None},
                {'alpha': 0.48612258246951734},
                0.22159811332626014,
                0.48612258246951734],
            [['l1'],
                {'l1': np.array([1, 3]), 'l2': None},
                {'alpha': 0.4441865222328282},
                0.22180308084369335,
                0.4441865222328282],
            [['l1'],
                {'l1': np.array([1, 3]), 'l2': None},
                {'alpha': 0.190666813148965},
                0.22324045507529866,
                0.190666813148965]], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error', 'alpha'],
        index=pd.RangeIndex(start=0, stop=10, step=1)
    ).astype({'mean_absolute_error': float, 'alpha': float})
    expected_results.insert(4, 'n_evaluated', [12] * 10)

    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results)


def test_evaluate_bayesian_search_forecaster_multiseries_when_return_best_ForecasterRecursiveMultiSeries():
    """
    Test forecaster is refitted when return_best=True in 
    bayesian_search_forecaster_multiseries with ForecasterRecursiveMultiSeries 
    (mocked done in skforecast v0.12.0).
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot',
                     transformer_series = StandardScaler()
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_range['l1']) - 12,
            steps              = 3,
            refit              = True,
            fixed_train_size   = True
    )
    
    def search_space(trial):
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}

        return search_space

    bayesian_search_forecaster_multiseries(
        forecaster         = forecaster,
        series             = series_dict_range,
        cv                 = cv,
        search_space       = search_space,
        metric             = 'mean_absolute_error',
        aggregate_metric   = 'weighted_average',
        n_trials           = 10,
        random_state       = 123,
        return_best        = True,
        verbose            = True
    )
    expected_lags = np.array([1, 2])
    expected_alpha = 0.2345829390285611
    
    np.testing.assert_array_almost_equal(forecaster.lags, expected_lags)
    assert expected_alpha == forecaster.estimator.alpha


def test_results_opt_best_output_bayesian_search_forecaster_multiseries_with_output_study_best_trial_optuna():
    """
    Test results_opt_best output of bayesian_search_forecaster_multiseries with output 
    study.best_trial optuna (mocked done in skforecast v0.12.0).
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot'
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_range['l1']) - 12,
            steps              = 3,
            refit              = True,
            fixed_train_size   = True
         )

    metric        = 'mean_absolute_error'
    verbose       = False
    show_progress = False
    n_trials      = 10
    random_state  = 123

    def objective(
        trial,
        forecaster    = forecaster,
        series        = series_dict_range,
        cv            = cv,
        metric        = metric,
        verbose       = verbose,
        show_progress = show_progress
    ) -> float:
        
        alpha = trial.suggest_float('alpha', 1e-2, 1.0)
        forecaster = ForecasterRecursiveMultiSeries(
                         estimator = Ridge(random_state=random_state, 
                                           alpha=alpha),
                         lags      = 2,
                         encoding  = 'onehot'
                     )
        metrics_levels, _ = backtesting_forecaster_multiseries(
                                forecaster         = forecaster,
                                series             = series,
                                cv                 = cv,
                                metric             = metric,
                                verbose            = verbose,
                                show_progress      = show_progress     
                            )

        return abs(metrics_levels.iloc[:, 1].mean())

    study = optuna.create_study(
        direction="minimize", sampler=TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials)
    best_trial = study.best_trial

    def search_space(trial):
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}

        return search_space

    return_best  = False
    results_opt_best = bayesian_search_forecaster_multiseries(
                           forecaster    = forecaster,
                           series        = series_dict_range,
                           cv            = cv,
                           search_space  = search_space,
                           metric        = metric,
                           n_trials      = n_trials,
                           return_best   = return_best,
                           verbose       = verbose,
                           show_progress = show_progress
                       )[1]

    assert best_trial.number == results_opt_best.best_trial.number
    assert best_trial.values == approx(results_opt_best.best_trial.values)
    assert best_trial.params == results_opt_best.best_trial.params


def test_bayesian_search_forecaster_multiseries_ForecasterRecursiveMultiSeries_output_file():
    """
    Test output file of bayesian_search_forecaster_multiseries in 
    ForecasterRecursiveMultiSeries.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = RandomForestRegressor(random_state=123),
                     lags      = 2 
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_range['l1']) - 12,
            steps              = 3,
            refit              = False,
            fixed_train_size   = False
         )

    def search_space(trial):
        search_space  = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 20),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 1., log=True),
            'max_features': trial.suggest_categorical('max_features', ['log2', 'sqrt'])
        }
        
        return search_space

    output_file = 'test_bayesian_search_forecaster_multiseries_output_file.txt'
    _ = bayesian_search_forecaster_multiseries(
            forecaster    = forecaster,
            series        = series_dict_range,
            cv            = cv,
            search_space  = search_space,
            metric        = 'mean_absolute_error',
            n_trials      = 10,
            random_state  = 123,
            return_best   = False,
            verbose       = False,
            output_file   = output_file
        )[0]

    assert os.path.isfile(output_file)
    os.remove(output_file)


def test_bayesian_search_forecaster_multiseries_ForecasterDirectMultiVariate_output_file():
    """
    Test output file of bayesian_search_forecaster_multiseries in 
    ForecasterDirectMultiVariate.
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator = RandomForestRegressor(random_state=123),
                     level     = 'l1',
                     lags      = 2,
                     steps     = 3
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_wide_range) - 12,
            steps              = 3,
            refit              = False,
            fixed_train_size   = False
         )

    def search_space(trial):
        search_space  = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 20),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 1., log=True),
            'max_features': trial.suggest_categorical('max_features', ['log2', 'sqrt'])
        }
        
        return search_space

    output_file = 'test_bayesian_search_forecaster_multiseries_output_file_3.txt'
    _ = bayesian_search_forecaster_multiseries(
            forecaster   = forecaster,
            series       = series_wide_range,
            cv           = cv,
            search_space = search_space,
            metric       = 'mean_absolute_error',
            n_trials     = 10,
            random_state = 123,
            return_best  = False,
            verbose      = False,
            output_file  = output_file
        )[0]

    assert os.path.isfile(output_file)
    os.remove(output_file)


def test_ValueError_bayesian_search_forecaster_multiseries_when_return_best_and_len_series_exog_different():
    """
    Test ValueError is raised in bayesian_search_forecaster_multiseries when 
    return_best and length of `series` and `exog` do not match.
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator = Ridge(random_state=123),
                     level     = 'l1',
                     steps     = 3,
                     lags      = 2
                 )
    
    cv = TimeSeriesFold(
            initial_train_size = len(series_wide_range) - 12,
            steps              = 3,
            refit              = True,
            fixed_train_size   = True
        )
    exog = series_wide_range.iloc[:30].copy()

    def search_space(trial):  # pragma: no cover
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}

        return search_space

    err_msg = re.escape(
        f"`exog` must have same number of samples as `series`. "
        f"length `exog`: ({len(exog)}), length `series`: ({len(series_wide_range)})"
    )
    with pytest.raises(ValueError, match = err_msg):
        bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series_wide_range,
            exog               = exog,
            cv                 = cv,
            search_space       = search_space,
            metric             = 'mean_absolute_error',
            aggregate_metric   = 'weighted_average',
            n_trials           = 10,
            random_state       = 123,
            return_best        = True,
            verbose            = False,
        )


@pytest.mark.parametrize("initial_train_size",
                         [1000, '2014-09-26 00:00:00', pd.to_datetime('2014-09-26 00:00:00')],
                         ids=lambda initial_train_size: f'initial_train_size: {initial_train_size}')
def test_bayesian_search_forecaster_multiseries_ForecasterDirectMultiVariate_one_step_ahead(initial_train_size):
    """
    Test output of bayesian_search_forecaster_multiseries when forecaster is ForecasterRecursiveMultiSeries
    and method is one_step_ahead.
    """
    metrics = ['mean_absolute_error', mean_absolute_percentage_error, mean_absolute_scaled_error]
    forecaster = ForecasterDirectMultiVariate(
                    estimator          = Ridge(random_state=123),
                    lags               = 10,
                    steps              = 10,
                    level              = 'item_1',
                    transformer_series = StandardScaler(),
                    transformer_exog   = StandardScaler(),
                )
    cv = OneStepAheadFold(initial_train_size = initial_train_size)

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-3, 1e3, log=True),
            'lags': trial.suggest_categorical('lags', [3, 5]),
        }

        return search_space
    
    warn_msg = re.escape(
        "One-step-ahead predictions are used for faster model comparison, but they "
        "may not fully represent multi-step prediction performance. It is recommended "
        "to backtest the final model for a more accurate multi-step performance "
        "estimate."
    )
    with pytest.warns(OneStepAheadValidationWarning, match = warn_msg):
        results, _ = bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series_wide_dt_item_sales,
            exog               = exog_wide_dt_item_sales,
            cv                 = cv,
            search_space       = search_space,
            n_trials           = 5,
            metric             = metrics,
            aggregate_metric   = ["average", "weighted_average", "pooling"],
            return_best        = False,
            n_jobs             = 'auto',
            verbose            = False,
            show_progress      = False
        )

    expected_results = pd.DataFrame({
        "levels": [["item_1"], ["item_1"], ["item_1"], ["item_1"], ["item_1"]],
        "lags": [
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
        ],
        "params": [
            {"alpha": 0.2252709077935534},
            {"alpha": 0.4279898484823994},
            {"alpha": 2.031835829826598},
            {"alpha": 15.094374246471325},
            {"alpha": 766.6289057556013},
        ],
        "mean_absolute_error": [
            0.7984524540765968,
            0.7986282481567063,
            0.8345353589834665,
            0.8372706471178882,
            1.1243829076961176,
        ],
        "mean_absolute_percentage_error": [
            0.03815078895052657,
            0.03816123329315277,
            0.039892049528787554,
            0.04013598113903565,
            0.05626606940150185,
        ],
        "mean_absolute_scaled_error": [
            0.5253268062267301,
            0.5254424666423076,
            0.548427455162373,
            0.5502249908743697,
            0.7389051285340059,
        ],
        "n_evaluated": [88] * 5,
        "alpha": [
            0.2252709077935534,
            0.4279898484823994,
            2.031835829826598,
            15.094374246471325,
            766.6289057556013,
        ],
    },
        index=pd.RangeIndex(start=0, stop=5, step=1)
    )

    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results)


@pytest.mark.parametrize("series",
                         [series_wide_range, series_dict_range],
                         ids = lambda series: f'series type: {type(series)}')
def test_bayesian_search_forecaster_multiseries_ForecasterRecursiveMultiSeries(series):
    """
    Test output of bayesian_search_forecaster_multiseries in 
    ForecasterRecursiveMultiSeries with mocked (mocked done in Skforecast v0.12.0).
    """
    if isinstance(series, pd.DataFrame):
        series = series.rename(columns={'1': 'l1', '2': 'l2'})
    
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot',
                     transformer_series = StandardScaler()
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series['l1']) - 12,
            steps              = 3,
            refit              = False,
            fixed_train_size   = True
        )

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1.0),
            'lags': trial.suggest_categorical('lags', [2, 4])
        }

        return search_space

    results = bayesian_search_forecaster_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  cv                 = cv,
                  search_space       = search_space,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False,
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([[list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.23598059857016607}, 0.20880273566554797,
            0.23598059857016607],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.398196343012209}, 0.2088108334929226,
            0.398196343012209],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.4441865222328282}, 0.2088131177203144,
            0.4441865222328282],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.53623586010342}, 0.2088176743151537,
            0.53623586010342],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.7252189487445193}, 0.2088269659234972,
            0.7252189487445193],
        [list(['l1', 'l2']), np.array([1, 2]), {'alpha': 0.5558016213920624},
            0.20913532870732662, 0.5558016213920624],
        [list(['l1', 'l2']), np.array([1, 2]), {'alpha': 0.6995044937418831},
            0.20913908163271674, 0.6995044937418831],
        [list(['l1', 'l2']), np.array([1, 2]), {'alpha': 0.7406154516747153},
            0.20914015254956903, 0.7406154516747153],
        [list(['l1', 'l2']), np.array([1, 2]), {'alpha': 0.8509374761370117},
            0.20914302038975385, 0.8509374761370117],
        [list(['l1', 'l2']), np.array([1, 2]), {'alpha': 0.9809565564007693},
            0.20914638910039665, 0.9809565564007693]], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error__weighted_average', 'alpha'],
        index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64')
    ).astype({
        'mean_absolute_error__weighted_average': float, 
        'alpha': float
    })
    expected_results.insert(4, 'n_evaluated', [24] * 10)

    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results)


@pytest.mark.parametrize("initial_train_size",
                         [1000, '2014-09-26 00:00:00', pd.to_datetime('2014-09-26 00:00:00')],
                         ids=lambda initial_train_size: f'initial_train_size: {initial_train_size}')
def test_bayesian_search_forecaster_multiseries_ForecasterRecursiveMultiSeries_one_step_ahead(initial_train_size):
    """
    Test output of bayesian_search_forecaster_multiseries when forecaster is ForecasterRecursiveMultiSeries
    and method is one_step_ahead.
    """
    metrics = ['mean_absolute_error', mean_absolute_percentage_error, mean_absolute_scaled_error]
    forecaster = ForecasterRecursiveMultiSeries(
            estimator          = Ridge(random_state=123),
            lags               = 3,
            encoding           = 'ordinal',
            transformer_series = StandardScaler(),
            transformer_exog   = StandardScaler(),
            weight_func        = None,
            series_weights     = None,
            differentiation    = None,
            dropna_from_series = False,
        )

    cv = OneStepAheadFold(initial_train_size = initial_train_size)
    levels = ["item_1", "item_2", "item_3"]

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-3, 1e3, log=True),
            'lags': trial.suggest_categorical('lags', [3, 5]),
        }

        return search_space
    
    warn_msg = re.escape(
        "One-step-ahead predictions are used for faster model comparison, but they "
        "may not fully represent multi-step prediction performance. It is recommended "
        "to backtest the final model for a more accurate multi-step performance "
        "estimate."
    )
    with pytest.warns(OneStepAheadValidationWarning, match = warn_msg):
        results, _ = bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series_dict_dt_item_sales,
            exog               = exog_wide_dt_item_sales,
            search_space       = search_space,
            cv                 = cv,
            n_trials           = 5,
            metric             = metrics,
            levels             = levels,
            aggregate_metric   = ["average", "weighted_average", "pooling"],
            return_best        = False,
            n_jobs             = 'auto',
            verbose            = False,
            show_progress      = False
        )

    expected_results = pd.DataFrame(
        {
            "levels": [
                ["item_1", "item_2", "item_3"],
                ["item_1", "item_2", "item_3"],
                ["item_1", "item_2", "item_3"],
                ["item_1", "item_2", "item_3"],
                ["item_1", "item_2", "item_3"],
            ],
            "lags": [
                np.array([1, 2, 3]),
                np.array([1, 2, 3, 4, 5]),
                np.array([1, 2, 3, 4, 5]),
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
            ],
            "params": [
                {"alpha": 766.6289057556013},
                {"alpha": 0.4279898484823994},
                {"alpha": 0.2252709077935534},
                {"alpha": 15.094374246471325},
                {"alpha": 2.031835829826598},
            ],
            "mean_absolute_error__average": [
                2.487682902584588,
                2.5778104346448623,
                2.5779063732830942,
                2.599406545255343,
                2.605166414875337,
            ],
            "mean_absolute_error__weighted_average": [
                2.487682902584588,
                2.5778104346448623,
                2.5779063732830947,
                2.599406545255343,
                2.6051664148753373,
            ],
            "mean_absolute_error__pooling": [
                2.4876829025845875,
                2.5778104346448623,
                2.5779063732830947,
                2.599406545255343,
                2.6051664148753373,
            ],
            "mean_absolute_percentage_error__average": [
                0.13537962851907534,
                0.14171211559040905,
                0.1417181361315976,
                0.1425099620058197,
                0.14287737387127633,
            ],
            "mean_absolute_percentage_error__weighted_average": [
                0.13537962851907534,
                0.14171211559040905,
                0.14171813613159756,
                0.14250996200581972,
                0.14287737387127633,
            ],
            "mean_absolute_percentage_error__pooling": [
                0.13537962851907534,
                0.14171211559040905,
                0.14171813613159756,
                0.14250996200581975,
                0.14287737387127633,
            ],
            "mean_absolute_scaled_error__average": [
                0.9807885628114613,
                0.9887040054273549,
                0.9887272793513265,
                0.9982897158009892,
                0.999726922014544,
            ],
            "mean_absolute_scaled_error__weighted_average": [
                0.9807885628114613,
                0.9887040054273548,
                0.9887272793513265,
                0.9982897158009892,
                0.9997269220145439,
            ],
            "mean_absolute_scaled_error__pooling": [
                0.994386675351447,
                1.0309527259247835,
                1.0309910949992809,
                1.039045301850066,
                1.0413476602398484,
            ],
            "n_evaluated": [291] * 5,
            "alpha": [
                766.6289057556013,
                0.4279898484823994,
                0.2252709077935534,
                15.094374246471325,
                2.031835829826598,
            ],
        },
        index=pd.RangeIndex(start=0, stop=5, step=1),
    )

    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results)


@pytest.mark.parametrize("initial_train_size",
                         [1000, '2014-09-26 00:00:00', pd.to_datetime('2014-09-26 00:00:00')],
                         ids=lambda initial_train_size: f'initial_train_size: {initial_train_size}')
def test_bayesian_search_forecaster_multiseries_ForecasterRecursiveMultiSeries_one_step_ahead_long_format(initial_train_size):
    """
    Test output of bayesian_search_forecaster_multiseries when forecaster is ForecasterRecursiveMultiSeries
    and method is one_step_ahead.
    """
    metrics = ['mean_absolute_error', mean_absolute_percentage_error, mean_absolute_scaled_error]
    forecaster = ForecasterRecursiveMultiSeries(
            estimator          = Ridge(random_state=123),
            lags               = 3,
            encoding           = 'ordinal',
            transformer_series = StandardScaler(),
            transformer_exog   = StandardScaler(),
            weight_func        = None,
            series_weights     = None,
            differentiation    = None,
            dropna_from_series = False,
        )

    cv = OneStepAheadFold(initial_train_size = initial_train_size)
    levels = ["item_1", "item_2", "item_3"]

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-3, 1e3, log=True),
            'lags': trial.suggest_categorical('lags', [3, 5]),
        }

        return search_space
    
    warn_msg = re.escape(
        "One-step-ahead predictions are used for faster model comparison, but they "
        "may not fully represent multi-step prediction performance. It is recommended "
        "to backtest the final model for a more accurate multi-step performance "
        "estimate."
    )
    with pytest.warns(OneStepAheadValidationWarning, match = warn_msg):
        results, _ = bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series_long_dt_item_sales,
            exog               = exog_long_dt_item_sales,
            search_space       = search_space,
            cv                 = cv,
            n_trials           = 5,
            metric             = metrics,
            levels             = levels,
            aggregate_metric   = ["average", "weighted_average", "pooling"],
            return_best        = False,
            n_jobs             = 'auto',
            verbose            = False,
            show_progress      = False
        )

    expected_results = pd.DataFrame(
        {
            "levels": [
                ["item_1", "item_2", "item_3"],
                ["item_1", "item_2", "item_3"],
                ["item_1", "item_2", "item_3"],
                ["item_1", "item_2", "item_3"],
                ["item_1", "item_2", "item_3"],
            ],
            "lags": [
                np.array([1, 2, 3]),
                np.array([1, 2, 3, 4, 5]),
                np.array([1, 2, 3, 4, 5]),
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
            ],
            "params": [
                {"alpha": 766.6289057556013},
                {"alpha": 0.4279898484823994},
                {"alpha": 0.2252709077935534},
                {"alpha": 15.094374246471325},
                {"alpha": 2.031835829826598},
            ],
            "mean_absolute_error__average": [
                2.487682902584588,
                2.5778104346448623,
                2.5779063732830942,
                2.599406545255343,
                2.605166414875337,
            ],
            "mean_absolute_error__weighted_average": [
                2.487682902584588,
                2.5778104346448623,
                2.5779063732830947,
                2.599406545255343,
                2.6051664148753373,
            ],
            "mean_absolute_error__pooling": [
                2.4876829025845875,
                2.5778104346448623,
                2.5779063732830947,
                2.599406545255343,
                2.6051664148753373,
            ],
            "mean_absolute_percentage_error__average": [
                0.13537962851907534,
                0.14171211559040905,
                0.1417181361315976,
                0.1425099620058197,
                0.14287737387127633,
            ],
            "mean_absolute_percentage_error__weighted_average": [
                0.13537962851907534,
                0.14171211559040905,
                0.14171813613159756,
                0.14250996200581972,
                0.14287737387127633,
            ],
            "mean_absolute_percentage_error__pooling": [
                0.13537962851907534,
                0.14171211559040905,
                0.14171813613159756,
                0.14250996200581975,
                0.14287737387127633,
            ],
            "mean_absolute_scaled_error__average": [
                0.9807885628114613,
                0.9887040054273549,
                0.9887272793513265,
                0.9982897158009892,
                0.999726922014544,
            ],
            "mean_absolute_scaled_error__weighted_average": [
                0.9807885628114613,
                0.9887040054273548,
                0.9887272793513265,
                0.9982897158009892,
                0.9997269220145439,
            ],
            "mean_absolute_scaled_error__pooling": [
                0.994386675351447,
                1.0309527259247835,
                1.0309910949992809,
                1.039045301850066,
                1.0413476602398484,
            ],
            "n_evaluated": [291] * 5,
            "alpha": [
                766.6289057556013,
                0.4279898484823994,
                0.2252709077935534,
                15.094374246471325,
                2.031835829826598,
            ],
        },
        index=pd.RangeIndex(start=0, stop=5, step=1),
    )

    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results)


def test_output_bayesian_search_forecaster_multiseries_series_and_exog_dict_multiple_metrics_aggregated_with_mocked():
    """
    Test output of bayesian_search_forecaster_multiseries in ForecasterRecursiveMultiSeries
    when series and exog are dictionaries and multiple aggregated metrics
    (mocked done in Skforecast v0.12.0).
    """

    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags=14,
        encoding="ordinal",
        dropna_from_series=False,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )

    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_nans_train["id_1000"]),
            steps              = 24,
            refit              = False,
    )
    lags_grid = [[5], [1, 7, 14]]

    def search_space(trial):
        search_space = {
            "n_estimators": trial.suggest_int("n_estimators", 2, 5),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "lags": trial.suggest_categorical("lags", lags_grid),
        }

        return search_space

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="optuna")

        results_search, _ = bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series_dict_nans,
            cv                 = cv,
            exog               = exog_dict_nans,
            search_space       = search_space,
            metric             = ['mean_absolute_error', 'mean_absolute_scaled_error'],
            aggregate_metric   = ['weighted_average', 'average', 'pooling'],
            n_trials           = 3,
            return_best        = False,
            show_progress      = False,
            verbose            = False,
            suppress_warnings  = True,
        )

    expected = pd.DataFrame(
        {
            "levels": [
                ["id_1000", "id_1001", "id_1002", "id_1003", "id_1004"],
                ["id_1000", "id_1001", "id_1002", "id_1003", "id_1004"],
                ["id_1000", "id_1001", "id_1002", "id_1003", "id_1004"],
            ],
            "lags": [np.array([1, 7, 14]), np.array([1, 7, 14]), np.array([5])],
            "params": [
                {"n_estimators": 4, "max_depth": 3},
                {"n_estimators": 3, "max_depth": 3},
                {"n_estimators": 4, "max_depth": 3},
            ],
            "mean_absolute_error__weighted_average": [
                709.8836514262415,
                721.1848222120482,
                754.3537196425694,
            ],
            "mean_absolute_error__average": [
                749.8761502029433,
                760.659082077477,
                777.6874712018467,
            ],
            "mean_absolute_error__pooling": [
                709.8836514262414,
                721.1848222120483,
                754.3537196425694,
            ],
            "mean_absolute_scaled_error__weighted_average": [
                2.0713511786893473,
                2.104173580194738,
                2.06373433155371,
            ],
            "mean_absolute_scaled_error__average": [
                1.720281630023928,
                1.7468333254211739,
                1.7217882951816943,
            ],
            "mean_absolute_scaled_error__pooling": [
                1.7240004031507636,
                1.7514460598462835,
                1.7678687830707494,
            ],
            "n_evaluated": [417] * 3,
            "n_estimators": [4, 3, 4],
            "max_depth": [3, 3, 3],
        }
    )

    pd.testing.assert_frame_equal(expected, results_search.drop(columns=["trial_number"]))


def test_output_bayesian_search_forecaster_multiseries_series_and_exog_dict_window_features_with_mocked():
    """
    Test output of bayesian_search_forecaster_multiseries in ForecasterRecursiveMultiSeries
    when series and exog are dictionaries and window features are included 
    (mocked done in Skforecast v0.14.0).
    """
    window_features = RollingFeatures(
        stats = ['mean', 'std', 'min', 'max', 'sum', 'median', 'ratio_min_max', 'coef_variation'],
        window_sizes = 3,
    )
    estimator = LGBMRegressor(
        n_estimators=2, random_state=123, verbose=-1, max_depth=2
    )
    forecaster = ForecasterRecursiveMultiSeries(
        estimator          = estimator,
        lags               = 14,
        window_features    = window_features,
        encoding           = "ordinal",
        dropna_from_series = False,
        transformer_series = StandardScaler(),
        transformer_exog   = StandardScaler(),
    )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_nans_train["id_1000"]),
            steps              = 24,
            refit              = False,
            fixed_train_size   = True
    )
    lags_grid = [[5], [1, 7, 14]]

    def search_space(trial):
        search_space = {
            "n_estimators": trial.suggest_int("n_estimators", 2, 5),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "lags": trial.suggest_categorical("lags", lags_grid),
        }

        return search_space

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="optuna")

        results_search, _ = bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series_dict_nans,
            exog               = exog_dict_nans,
            cv                 = cv,
            search_space       = search_space,
            metric             = "mean_absolute_error",
            aggregate_metric   = "weighted_average",
            n_trials           = 3,
            return_best        = False,
            show_progress      = False,
            verbose            = False,
            suppress_warnings  = True,
        )

    expected = pd.DataFrame({
                'levels': [
                    ['id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004'],
                    ['id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004'],
                    ['id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004'],
                ],
                'lags': [np.array([1, 7, 14]), np.array([1, 7, 14]), np.array([5])],
                'params': [
                    {'n_estimators': 4, 'max_depth': 3},
                    {'n_estimators': 3, 'max_depth': 3},
                    {'n_estimators': 4, 'max_depth': 3},
                ],
                'mean_absolute_error__weighted_average': [
                    705.2775299463871,
                    721.2904821438083,
                    744.3600945094412,
                ],
                'n_evaluated': [417] * 3,
                'n_estimators': [4, 3, 4],
                'max_depth': [3, 3, 3],
            })

    pd.testing.assert_frame_equal(expected, results_search.drop(columns=["trial_number"]))


def test_output_bayesian_search_forecaster_multiseries_series_and_exog_dict_with_mocked():
    """
    Test output of bayesian_search_forecaster_multiseries in ForecasterRecursiveMultiSeries
    when series and exog are dictionaries (mocked done in Skforecast v0.12.0).
    """

    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags=14,
        encoding="ordinal",
        dropna_from_series=False,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_nans_train["id_1000"]),
            steps              = 24,
            refit              = False,
            fixed_train_size   = True
    )
    lags_grid = [[5], [1, 7, 14]]

    def search_space(trial):
        search_space = {
            "n_estimators": trial.suggest_int("n_estimators", 2, 5),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "lags": trial.suggest_categorical("lags", lags_grid),
        }

        return search_space

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="optuna")

        results_search, _ = bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series_dict_nans,
            exog               = exog_dict_nans,
            cv                 = cv,
            search_space       = search_space,
            metric             = "mean_absolute_error",
            aggregate_metric   = "weighted_average",
            n_trials           = 3,
            return_best        = False,
            show_progress      = False,
            verbose            = False,
            suppress_warnings  = True,
        )

    expected = pd.DataFrame(
        np.array(
            [
                [
                    list(["id_1000", "id_1001", "id_1002", "id_1003", "id_1004"]),
                    np.array([1, 7, 14]),
                    {"n_estimators": 4, "max_depth": 3},
                    709.8836514262415,
                    4,
                    3,
                ],
                [
                    list(["id_1000", "id_1001", "id_1002", "id_1003", "id_1004"]),
                    np.array([1, 7, 14]),
                    {"n_estimators": 3, "max_depth": 3},
                    721.1848222120482,
                    3,
                    3,
                ],
                [
                    list(["id_1000", "id_1001", "id_1002", "id_1003", "id_1004"]),
                    np.array([5]),
                    {"n_estimators": 4, "max_depth": 3},
                    754.3537196425694,
                    4,
                    3,
                ],
            ],
            dtype=object,
        ),
        columns=[
            "levels",
            "lags",
            "params",
            "mean_absolute_error__weighted_average",
            "n_estimators",
            "max_depth",
        ],
        index=pd.Index([0, 1, 2], dtype="int64"),
    ).astype(
        {
            "mean_absolute_error__weighted_average": float,
            "n_estimators": int,
            "max_depth": int,
        }
    )
    expected.insert(4, 'n_evaluated', [417] * 3)

    results_search = results_search.astype(
        {
            "mean_absolute_error__weighted_average": float,
            "n_estimators": int,
            "max_depth": int,
        }
    )

    pd.testing.assert_frame_equal(expected, results_search.drop(columns=["trial_number"]))


def test_output_bayesian_search_forecaster_multiseries_series_and_exog_dict_with_mocked_skip_folds():
    """
    Test output of bayesian_search_forecaster_multiseries in ForecasterRecursiveMultiSeries 
    when series and exog are dictionaries (mocked done in Skforecast v0.12.0) and skip_folds.
    """

    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags=14,
        encoding='ordinal',
        dropna_from_series=False,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_nans_train["id_1000"]),
            steps              = 24,
            refit              = False,
            skip_folds         = 2,
    )
    lags_grid = [[5], [1, 7, 14]]

    def search_space(trial):
        search_space = {
            "n_estimators": trial.suggest_int("n_estimators", 2, 5),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "lags": trial.suggest_categorical("lags", lags_grid),
        }

        return search_space

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="optuna")
        
        results_search, _ = bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series_dict_nans,
            exog               = exog_dict_nans,
            cv                 = cv,
            search_space       = search_space,
            metric             = 'mean_absolute_error',          
            n_trials           = 3,
            return_best        = False,
            show_progress      = False,
            verbose            = False,
            suppress_warnings  = True
        )
    
    expected = pd.DataFrame(
        np.array([[list(['id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004']),
        np.array([1,  7, 14]), {'n_estimators': 4, 'max_depth': 3}, 
            694.843611295513, 718.447039241195, 694.843611295513, 4, 3],
        [list(['id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004']),
        np.array([1,  7, 14]), {'n_estimators': 3, 'max_depth': 3},
            704.8898538858386, 726.7646039380159, 704.8898538858386, 3, 3],
        [list(['id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004']),
         np.array([5]), {'n_estimators': 4, 'max_depth': 3},
            730.1888301097707, 735.3584948212444, 730.1888301097707, 4, 3]], dtype=object),
        columns=['levels', 'lags', 'params', 
                 'mean_absolute_error__weighted_average', 
                 'mean_absolute_error__average', 
                 'mean_absolute_error__pooling',
                 'n_estimators', 'max_depth'],
        index=pd.Index([0, 1, 2], dtype='int64')
    ).astype({
        'mean_absolute_error__weighted_average': float,
        'mean_absolute_error__average': float,
        'mean_absolute_error__pooling': float,
        'n_estimators': int,
        'max_depth': int
    })
    expected.insert(6, 'n_evaluated', [225] * 3)

    results_search = results_search.astype({
        'mean_absolute_error__weighted_average': float,
        'mean_absolute_error__average': float,
        'mean_absolute_error__pooling': float,
        'n_estimators': int,
        'max_depth': int
    })

    pd.testing.assert_frame_equal(expected, results_search.drop(columns=["trial_number"]))


def test_results_output_bayesian_search_forecaster_multiseries_ForecasterRecursiveMultiSeries_with_window_features():
    """
    Test output of bayesian_search_forecaster_multiseries in 
    ForecasterRecursiveMultiSeries with window features 
    (mocked done in Skforecast v0.14.0).
    """
    window_features = RollingFeatures(
        stats=['mean', 'std', 'min', 'max', 'sum', 'median', 'ratio_min_max', 'coef_variation'],
        window_sizes=3,
    )

    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = Ridge(random_state=123),
                     lags      = 2,
                     window_features = window_features,
                     encoding  = 'onehot',
                     transformer_series = StandardScaler()
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_range['l1']) - 12,
            steps              = 3,
            refit              = False,
            fixed_train_size   = True
        )

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1.0),
            'lags': trial.suggest_categorical('lags', [2, 4])
        }

        return search_space

    results = bayesian_search_forecaster_multiseries(
                  forecaster         = forecaster,
                  series             = series_dict_range,
                  cv                 = cv,
                  search_space       = search_space,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False,
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([
            [['l1', 'l2'],
                np.array([1, 2]),
                {'alpha': 0.5558016213920624},
                0.26169284425167877,
                0.5558016213920624],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.23598059857016607},
                0.2643237951473078,
                0.23598059857016607],
            [['l1', 'l2'],
                np.array([1, 2]),
                {'alpha': 0.6995044937418831},
                0.2686147300996643,
                0.6995044937418831],
            [['l1', 'l2'],
                np.array([1, 2]),
                {'alpha': 0.7406154516747153},
                0.27087715575521176,
                0.7406154516747153],
            [['l1', 'l2'],
                np.array([1, 2]),
                {'alpha': 0.8509374761370117},
                0.27770070699157484,
                0.8509374761370117],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.7252189487445193},
                0.27967030912996743,
                0.7252189487445193],
            [['l1', 'l2'],
                np.array([1, 2]),
                {'alpha': 0.9809565564007693},
                0.287498421578392,
                0.9809565564007693],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.53623586010342},
                0.29887291134275246,
                0.53623586010342],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.4441865222328282},
                0.345749311164625,
                0.4441865222328282],
            [['l1', 'l2'],
                np.array([1, 2, 3, 4]),
                {'alpha': 0.398196343012209},
                0.5104999292169362,
                0.398196343012209]], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error__weighted_average', 'alpha'],
        index=pd.RangeIndex(start=0, stop=10, step=1)
    ).astype({'mean_absolute_error__weighted_average': float, 'alpha': float})
    expected_results.insert(4, 'n_evaluated', [24] * 10)

    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results)


def test_results_output_bayesian_search_forecaster_multivariate_ForecasterDirectMultiVariate():
    """
    Test output of bayesian_search_forecaster_multivariate in 
    ForecasterDirectMultiVariate with mocked (mocked done in Skforecast v0.12.0).
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator = Ridge(random_state=123),
                     level     = 'l1',
                     lags      = 2,
                     steps     = 3
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_wide_dt) - 12,
            steps              = 3,
            refit              = False,
    )

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1.0),
            'lags': trial.suggest_categorical('lags', [2, {'l1': 4, 'l2': [2, 3]}])
        }

        return search_space

    results = bayesian_search_forecaster_multiseries(
                  forecaster         = forecaster,
                  series             = series_wide_dt,
                  cv                 = cv,
                  search_space       = search_space,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False,
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([
            [['l1'],
                {'l1': np.array([1, 2, 3, 4]), 'l2': np.array([2, 3])},
                {'alpha': 0.23598059857016607},
                0.19308110319514993,
                0.23598059857016607],
            [['l1'],
                {'l1': np.array([1, 2, 3, 4]), 'l2': np.array([2, 3])},
                {'alpha': 0.398196343012209},
                0.1931744420708601,
                0.398196343012209],
            [['l1'],
                {'l1': np.array([1, 2, 3, 4]), 'l2': np.array([2, 3])},
                {'alpha': 0.4441865222328282},
                0.19320049540447037,
                0.4441865222328282],
            [['l1'],
                {'l1': np.array([1, 2, 3, 4]), 'l2': np.array([2, 3])},
                {'alpha': 0.53623586010342},
                0.19325210858832276,
                0.53623586010342],
            [['l1'],
                {'l1': np.array([1, 2, 3, 4]), 'l2': np.array([2, 3])},
                {'alpha': 0.7252189487445193},
                0.19335589494249983,
                0.7252189487445193],
            [['l1'],
                np.array([1, 2]),
                {'alpha': 0.5558016213920624},
                0.20131081099888368,
                0.5558016213920624],
            [['l1'],
                np.array([1, 2]),
                {'alpha': 0.6995044937418831},
                0.2013710017368262,
                0.6995044937418831],
            [['l1'],
                np.array([1, 2]),
                {'alpha': 0.7406154516747153},
                0.2013880862681147,
                0.7406154516747153],
            [['l1'],
                np.array([1, 2]),
                {'alpha': 0.8509374761370117},
                0.20143363961627603,
                0.8509374761370117],
            [['l1'],
                np.array([1, 2]),
                {'alpha': 0.9809565564007693},
                0.20148678375852938,
                0.9809565564007693]], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error', 'alpha'],
        index=pd.RangeIndex(start=0, stop=10, step=1)
    ).astype({'mean_absolute_error': float, 'alpha': float})
    expected_results.insert(4, 'n_evaluated', [12] * 10)

    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results)


def test_results_output_bayesian_search_forecaster_multivariate_ForecasterDirectMultiVariate_window_features():
    """
    Test output of bayesian_search_forecaster_multivariate in 
    ForecasterDirectMultiVariate with mocked (mocked done in Skforecast v0.12.0).
    """
    window_features = RollingFeatures(
        stats = ['mean', 'std', 'min', 'max', 'sum', 'median', 'ratio_min_max', 'coef_variation'],
        window_sizes = 3,
    )
    forecaster = ForecasterDirectMultiVariate(
                     estimator       = Ridge(random_state=123),
                     level           = 'l1',
                     steps           = 3,
                     lags            = 2,
                     window_features = window_features
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series_wide_range) - 12,
            steps              = 3,
            refit              = False,
    )

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1.0),
            'lags': trial.suggest_categorical('lags', [2, {'l1': 4, 'l2': [2, 3]}])
        }

        return search_space

    results = bayesian_search_forecaster_multiseries(
                  forecaster         = forecaster,
                  series             = series_wide_range,
                  cv                 = cv,
                  search_space       = search_space,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False,
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([
            [['l1'],
                {'l1': np.array([1, 2, 3, 4]), 'l2': np.array([2, 3])},
                {'alpha': 0.7252189487445193},
                0.2336486272843502,
                0.7252189487445193],
            [['l1'],
                {'l1': np.array([1, 2, 3, 4]), 'l2': np.array([2, 3])},
                {'alpha': 0.23598059857016607},
                0.23399963314273878,
                0.23598059857016607],
            [['l1'],
                {'l1': np.array([1, 2, 3, 4]), 'l2': np.array([2, 3])},
                {'alpha': 0.53623586010342},
                0.23401124553469277,
                0.53623586010342],
            [['l1'],
                {'l1': np.array([1, 2, 3, 4]), 'l2': np.array([2, 3])},
                {'alpha': 0.4441865222328282},
                0.234115147686308,
                0.4441865222328282],
            [['l1'],
                {'l1': np.array([1, 2, 3, 4]), 'l2': np.array([2, 3])},
                {'alpha': 0.398196343012209},
                0.23414113077335508,
                0.398196343012209],
            [['l1'],
                np.array([1, 2]),
                {'alpha': 0.9809565564007693},
                0.250967375463984,
                0.9809565564007693],
            [['l1'],
                np.array([1, 2]),
                {'alpha': 0.8509374761370117},
                0.25166609972144904,
                0.8509374761370117],
            [['l1'],
                np.array([1, 2]),
                {'alpha': 0.7406154516747153},
                0.2522467306377772,
                0.7406154516747153],
            [['l1'],
                np.array([1, 2]),
                {'alpha': 0.6995044937418831},
                0.2524589067210421,
                0.6995044937418831],
            [['l1'],
                np.array([1, 2]),
                {'alpha': 0.5558016213920624},
                0.2531740553380533,
                0.5558016213920624]], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error', 'alpha'],
        index=pd.RangeIndex(start=0, stop=10, step=1)
    ).astype({'mean_absolute_error': float, 'alpha': float})
    expected_results.insert(4, 'n_evaluated', [12] * 10)

    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results)


def test_bayesian_search_forecaster_multiseries_xgboost_categorical_no_ValueError_on_cache_hit():
    """
    Test that bayesian_search_forecaster_multiseries does not raise a ValueError
    when XGBRegressor with categorical features is used and the search space
    includes multiple lag combinations with OneStepAheadFold (cache hit scenario).

    Root cause: `configure_estimator_categorical_features` sets `feature_types` on
    the estimator via `set_params` (persistent mutation). When a cached split is
    reused the estimator's `feature_types` must be restored to match the cached
    X_train column count, otherwise XGBoost raises:
      ValueError: feature types must have the same length as the number of
                  data columns, expected <N>, got <M>
    """
    # np.random.seed(123); s1 = np.random.rand(50); s2 = np.random.rand(50)
    s1 = np.array([0.69646919, 0.28613933, 0.22685145, 0.55131477, 0.71946897,
                   0.42310646, 0.9807642 , 0.68482974, 0.4809319 , 0.39211752,
                   0.34317802, 0.72904971, 0.43857224, 0.0596779 , 0.39804426,
                   0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759,
                   0.63440096, 0.84943179, 0.72445532, 0.61102351, 0.72244338,
                   0.32295891, 0.36178866, 0.22826323, 0.29371405, 0.63097612,
                   0.09210494, 0.43370117, 0.43086276, 0.4936851 , 0.42583029,
                   0.31226122, 0.42635131, 0.89338916, 0.94416002, 0.50183668,
                   0.62395295, 0.1156184 , 0.31728548, 0.41482621, 0.86630916,
                   0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453])
    s2 = np.array([0.12062867, 0.8263408 , 0.60306013, 0.54506801, 0.34276383,
                   0.30412079, 0.41702221, 0.68130077, 0.87545684, 0.51042234,
                   0.66931378, 0.58593655, 0.6249035 , 0.67468905, 0.84234244,
                   0.08319499, 0.76368284, 0.24366637, 0.19422296, 0.57245696,
                   0.09571252, 0.88532683, 0.62724897, 0.72341636, 0.01612921,
                   0.59443188, 0.55678519, 0.15895964, 0.15307052, 0.69552953,
                   0.31876643, 0.6919703 , 0.55438325, 0.38895057, 0.92513249,
                   0.84167   , 0.35739757, 0.04359146, 0.30476807, 0.39818568,
                   0.70495883, 0.99535848, 0.35591487, 0.76254781, 0.59317692,
                   0.6917018 , 0.15112745, 0.39887629, 0.2408559 , 0.34345601])
    series_local = pd.DataFrame({'s1': s1, 's2': s2})

    # Hardcoded categorical exog with three levels ('f', 'g', 'h')
    exog_local = pd.DataFrame({
        'cat_feat': pd.Categorical(
            ['f', 'f', 'g', 'h', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'h', 'g', 'g',
             'g', 'f', 'f', 'g', 'g', 'g', 'h', 'f', 'g', 'g', 'f', 'h', 'h', 'h',
             'g', 'f', 'f', 'h', 'g', 'h', 'g', 'h', 'g', 'h', 'g', 'h', 'g', 'h',
             'f', 'g', 'h', 'h', 'g', 'f', 'f', 'g']
        )
    })

    forecaster = ForecasterRecursiveMultiSeries(
        estimator=XGBRegressor(n_estimators=10, random_state=123, verbosity=0),
        lags=3
    )
    forecaster.categorical_features = ['cat_feat']

    cv = OneStepAheadFold(initial_train_size=35)

    def search_space(trial):
        return {
            'lags': trial.suggest_categorical('lags', [[1, 2, 3], [1, 2, 3, 4, 5]]),
            'n_estimators': trial.suggest_int('n_estimators', 5, 10),
        }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        results, _ = bayesian_search_forecaster_multiseries(
            forecaster   = forecaster,
            series       = series_local,
            exog         = exog_local,
            cv           = cv,
            search_space = search_space,
            metric       = 'mean_absolute_error',
            n_trials     = 6,
            random_state = 123,
            return_best  = False,
            verbose      = False,
            show_progress = False
        )

    expected_results = pd.DataFrame({
        'levels': [['s1', 's2']] * 6,
        'lags': [
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
        ],
        'params': [
            {'n_estimators': 7},
            {'n_estimators': 7},
            {'n_estimators': 7},
            {'n_estimators': 6},
            {'n_estimators': 6},
            {'n_estimators': 9},
        ],
        'mean_absolute_error__weighted_average': [
            0.232613, 0.243916, 0.243916, 0.244897, 0.244897, 0.246239
        ],
        'mean_absolute_error__average': [
            0.232613, 0.243916, 0.243916, 0.244897, 0.244897, 0.246239
        ],
        'mean_absolute_error__pooling': [
            0.232613, 0.243916, 0.243916, 0.244897, 0.244897, 0.246239
        ],
        'n_evaluated': [30] * 6,
        'n_estimators': [7, 7, 7, 6, 6, 9],
    }, index=pd.RangeIndex(start=0, stop=6, step=1))

    pd.testing.assert_frame_equal(
        results.drop(columns=['trial_number']), expected_results, atol=1e-4
    )


@pytest.mark.parametrize(
    "kwargs_create_study, expected_direction",
    [(None, "maximize"), ({"direction": "minimize"}, "minimize")],
    ids=lambda dt: f"{dt}",
)
def test_bayesian_search_forecaster_multiseries_direction_when_metric_greater_is_better(
    kwargs_create_study, expected_direction
):
    """
    Test that, with a metric in which greater values are better (`r2_score`),
    the study is maximized by default, and that the results (sorted by the
    aggregated metric) and `return_best` follow the direction of the study,
    also when it is set in `kwargs_create_study` (with a warning when it
    contradicts the metric).
    """
    forecaster = ForecasterRecursiveMultiSeries(
        estimator=Ridge(random_state=123), lags=2
    )
    cv = TimeSeriesFold(
        steps=3, initial_train_size=len(series_wide_range) - 12, refit=False
    )

    def search_space(trial):
        return {"alpha": trial.suggest_float("alpha", 1e-2, 1e2, log=True)}

    kwargs = dict(
        forecaster=forecaster,
        series=series_wide_range,
        cv=cv,
        search_space=search_space,
        metric=r2_score,
        aggregate_metric="average",
        n_trials=8,
        random_state=123,
        return_best=True,
        kwargs_create_study=kwargs_create_study,
    )
    if expected_direction == "minimize":
        with pytest.warns(UserWarning, match="The optuna study direction is 'minimize'"):
            results, study = bayesian_search_forecaster_multiseries(**kwargs)
    else:
        results, study = bayesian_search_forecaster_multiseries(**kwargs)

    assert study.direction.name.lower() == expected_direction
    assert results["r2_score__average"].nunique() > 1
    if expected_direction == "maximize":
        assert results["r2_score__average"].is_monotonic_decreasing
    else:
        assert results["r2_score__average"].is_monotonic_increasing
    assert results.loc[0, "trial_number"] == study.best_trial.number
    assert forecaster.estimator.alpha == study.best_params["alpha"]


@pytest.mark.parametrize(
    "cv",
    [
        TimeSeriesFold(steps=10, initial_train_size=213, refit=False),
        OneStepAheadFold(initial_train_size=213),
    ],
    ids=lambda cv: f"cv: {type(cv).__name__}",
)
def test_bayesian_search_forecaster_multiseries_aggregated_metrics_match_their_column(cv):
    """
    Test that each column `<metric>__<aggregation>` of the results, and the
    value optimized by optuna (the first one), contain the metric of their
    aggregation with the default `aggregate_metric` (weighted average first).
    The series are evaluated on different numbers of points, so the weighted
    average differs from the average. The expected values are those of a
    search with each aggregation alone (the first trials are random, so all
    the searches evaluate the same parameters).
    """
    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LGBMRegressor(n_estimators=5, random_state=123, verbose=-1),
        lags=3,
    )

    def search_space(trial):
        return {"n_estimators": trial.suggest_int("n_estimators", 2, 20)}

    def search(aggregate_metric):
        results, _ = bayesian_search_forecaster_multiseries(
            forecaster       = forecaster,
            series           = series_dict_nans,
            cv               = cv,
            search_space     = search_space,
            metric           = "mean_absolute_error",
            aggregate_metric = aggregate_metric,
            n_trials         = 3,
            random_state     = 123,
            return_best      = False,
            show_progress    = False,
        )
        return results.set_index(results["params"].astype(str)).sort_index()

    results = search(["weighted_average", "average", "pooling"])
    for aggregation in ["weighted_average", "average", "pooling"]:
        column = f"mean_absolute_error__{aggregation}"
        pd.testing.assert_series_equal(results[column], search(aggregation)[column])
    assert not np.allclose(
        results["mean_absolute_error__weighted_average"],
        results["mean_absolute_error__average"],
    )


@pytest.mark.parametrize(
    "cv",
    [
        TimeSeriesFold(initial_train_size=45, steps=5, refit=False),
        OneStepAheadFold(initial_train_size=45),
    ],
    ids=lambda cv: f"cv: {type(cv).__name__}",
)
def test_bayesian_search_forecaster_multiseries_n_evaluated_user_attr_and_MissingValuesWarning_once(cv):
    """
    Test that the number of points evaluated of each trial is stored as the
    `user_attr` `n_evaluated` and in the column `n_evaluated` of the results,
    without changing the value optimized by optuna, and that a single
    MissingValuesWarning is raised at the end of the search when some trials
    evaluate less than 95% of the points of the most complete one.
    """
    forecaster = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=2)

    def search_space(trial):
        search_space = {
            'alpha': trial.suggest_float('alpha', 0.1, 1.0),
            'lags': trial.suggest_categorical('lags', [2, 8]),
        }

        return search_space

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results, study = bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series_short,
            search_space       = search_space,
            cv                 = cv,
            metric             = 'mean_absolute_error',
            aggregate_metric   = 'weighted_average',
            n_trials           = 6,
            random_state       = 123,
            return_best        = False,
            verbose            = False,
            show_progress      = False
        )

    for trial in study.trials:
        n_evaluated = 30 if trial.params['lags'] == 8 else 45
        assert trial.user_attrs['n_evaluated'] == n_evaluated
        assert trial.value == trial.user_attrs['mean_absolute_error__weighted_average']
    pd.testing.assert_series_equal(
        results['n_evaluated'],
        pd.Series(
            [study.trials[n].user_attrs['n_evaluated'] for n in results['trial_number']],
            name='n_evaluated'
        )
    )

    messages = [
        str(w.message) for w in record
        if w.category is MissingValuesWarning
        and "evaluated on fewer points than the most complete one" in str(w.message)
    ]
    n_affected = (results['n_evaluated'] == 30).sum()
    assert 0 < n_affected < len(results)
    assert len(messages) == 1
    assert (
        f"{n_affected} of 6 trials are evaluated on fewer points than the "
        f"most complete one"
    ) in messages[0]
    trials_affected = results.loc[results['n_evaluated'] == 30, 'trial_number'].tolist()
    if n_affected == 1:
        expected_trials = f"trial {trials_affected[0]}"
    else:
        expected_trials = (
            f"{n_affected} trials ({', '.join(map(str, trials_affected))})"
        )
    assert (
        f"Trials with less than 95% of the 45 points of the most complete one:\n"
        f"  - lags: [1, 2, ..., 8], {expected_trials}: 30 (66.7%)"
    ) in messages[0]
    assert (
        "Series that every candidate evaluates on the same points:\n"
        "  levels=['l1', 'l2']"
    ) in messages[0]


@pytest.mark.parametrize(
    "n_jobs",
    [1, 2],
    ids=lambda n_jobs: f"n_jobs: {n_jobs}",
)
def test_bayesian_search_forecaster_multiseries_warnings_of_trials_once_with_count(n_jobs):
    """
    Test that the warning of the levels not predicted, raised in the trials
    with 8 lags, is shown once at the end of the search with the number of
    trials in which it is raised, also when optuna runs the trials in threads
    (`n_jobs` of `study.optimize`). The trials evaluate more than 95% of the
    points of the most complete one, so there is no warning of the points
    evaluated.
    """
    forecaster = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=2)

    def search_space(trial):
        search_space = {
            'alpha': trial.suggest_float('alpha', 0.1, 1.0),
            'lags': trial.suggest_categorical('lags', [2, 8]),
        }

        return search_space

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _, study = bayesian_search_forecaster_multiseries(
            forecaster            = forecaster,
            series                = series_short_one_point,
            search_space          = search_space,
            cv                    = TimeSeriesFold(initial_train_size=45, steps=5, refit=False),
            metric                = 'mean_absolute_error',
            n_trials              = 8,
            random_state          = 123,
            return_best           = False,
            show_progress         = False,
            kwargs_study_optimize = {'n_jobs': n_jobs}
        )

    n_lags_8 = sum(trial.params['lags'] == 8 for trial in study.trials)
    assert 0 < n_lags_8 < 8
    messages = [
        w.message.message for w in record if w.category is MissingValuesWarning
    ]
    assert len(messages) == 1
    assert messages[0].startswith(
        f"Some levels are not predicted in folds where they have observed "
        f"values, so their metrics, and the aggregated metrics, do not include "
        f"those folds (in {n_lags_8} of 8 trials)."
    )


def test_bayesian_search_forecaster_multiseries_when_study_is_loaded_from_storage():
    """
    Test that a study loaded from storage with trials of a previous call
    (`load_if_exists=True`) can be resumed: the results include every trial
    with its `n_evaluated`, and the warning about the points evaluated only
    uses the trials of this call (4 of the 8 trials of the study).
    """
    storage = optuna.storages.InMemoryStorage()

    def search_space(trial):
        search_space = {
            'alpha': trial.suggest_float('alpha', 0.1, 1.0),
            'lags': trial.suggest_categorical('lags', [2, 8]),
        }

        return search_space

    kwargs = {
        'series': series_short,
        'search_space': search_space,
        'cv': TimeSeriesFold(initial_train_size=45, steps=5, refit=False),
        'metric': 'mean_absolute_error',
        'n_trials': 4,
        'return_best': False,
        'show_progress': False,
        'kwargs_create_study': {
            'storage': storage, 'study_name': 'resumed', 'load_if_exists': True
        },
    }
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results_first, _ = bayesian_search_forecaster_multiseries(
            forecaster   = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=2),
            random_state = 123,
            **kwargs
        )
        results, study = bayesian_search_forecaster_multiseries(
            forecaster   = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=2),
            random_state = 456,
            **kwargs
        )

    assert len(results) == 8
    assert len(study.trials) == 8
    assert results['n_evaluated'].isin([30, 45]).all()
    messages = [
        str(w.message) for w in record
        if w.category is MissingValuesWarning
        and "evaluated on fewer points than the most complete one" in str(w.message)
    ]
    assert len(messages) == 2
    assert results.loc[0, 'trial_number'] == 4
    assert messages[-1].startswith(
        "1 of 4 trials is evaluated on fewer points than the most complete "
        "one, so its metrics are not comparable. It is the best candidate of the "
        "ranking"
    )



@pytest.mark.parametrize(
    "cv",
    [TimeSeriesFold(steps=5, initial_train_size=45, refit=False),
     OneStepAheadFold(initial_train_size=45)],
    ids=lambda cv: f"cv: {type(cv).__name__}"
)
def test_bayesian_search_forecaster_multiseries_skips_trials_whose_evaluation_fails(cv):
    """
    Test that the trials whose evaluation fails (50 lags, not smaller than
    `initial_train_size`) are marked as failed and reported in the warning of
    the candidates skipped, identified by their number, instead of stopping
    the search. With a `OneStepAheadFold`, the split of the lags that fail is
    cached with its error.
    """
    forecaster = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=2)

    def search_space(trial):
        return {'lags': trial.suggest_categorical('lags', [2, 50])}

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results, study = bayesian_search_forecaster_multiseries(
                             forecaster    = forecaster,
                             series        = series_short,
                             cv            = cv,
                             search_space  = search_space,
                             metric        = 'mean_absolute_error',
                             n_trials      = 6,
                             return_best   = True,
                             show_progress = False
                         )

    failed = [t.number for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    assert failed == [t.number for t in study.trials if t.params['lags'] == 50]
    assert 0 < len(failed) < 6
    assert len(results) == 6 - len(failed)
    assert forecaster.max_lag == 2
    messages = [str(w.message) for w in record if w.category is RuntimeWarning]
    assert len(messages) == 1
    for trial_number in failed:
        assert f"  - trial {trial_number}, params: {{'lags': 50}}" in messages[0]


def test_bayesian_search_forecaster_multiseries_every_trial_fails_returns_empty_results():
    """
    Test that, when the evaluation of every trial fails, the results are
    empty, with the columns of the metrics, and the forecaster is not
    refitted.
    """
    forecaster = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=2)

    def search_space(trial):
        return {'lags': trial.suggest_categorical('lags', [50, 55])}

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results, _ = bayesian_search_forecaster_multiseries(
                         forecaster    = forecaster,
                         series        = series_short,
                         cv            = TimeSeriesFold(steps=5, initial_train_size=45),
                         search_space  = search_space,
                         metric        = 'mean_absolute_error',
                         n_trials      = 3,
                         return_best   = True,
                         show_progress = False
                     )

    # NOTE: The column `params` is duplicated when the results are empty.
    assert results.empty
    assert list(dict.fromkeys(results.columns)) == [
        'trial_number', 'levels', 'lags', 'params',
        'mean_absolute_error__weighted_average', 'mean_absolute_error__average',
        'mean_absolute_error__pooling', 'n_evaluated'
    ]
    assert not forecaster.is_fitted
    assert any(
        str(w.message).startswith("Parameters skipped: 3 of 3 trials raise an error")
        for w in record
    )


def test_bayesian_search_forecaster_multiseries_refit_without_series_too_short():
    """
    Test that the refit of the best trial leaves out the series with
    `window_size` observations or fewer (`short` has 6 values and the
    forecaster 8 lags) with a RuntimeWarning, instead of failing at the end of
    the search.
    """
    forecaster = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=8)

    def search_space(trial):
        return {'alpha': trial.suggest_float('alpha', 0.1, 1)}

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results, _ = bayesian_search_forecaster_multiseries(
                         forecaster    = forecaster,
                         series        = series_short_one_point,
                         cv            = TimeSeriesFold(initial_train_size=45, steps=5),
                         search_space  = search_space,
                         metric        = 'mean_absolute_error',
                         n_trials      = 3,
                         return_best   = True,
                         show_progress = False
                     )

    assert len(results) == 3
    assert forecaster.series_names_in_ == ['l1', 'l2']
    assert any(
        w.category is RuntimeWarning
        and str(w.message).endswith("Series left out: 'short' (6 observations).")
        for w in record
    )


@pytest.mark.parametrize(
    "estimator, refit, n_jobs, expected_n_jobs",
    [(Ridge(), True, 'auto', 1),
     (HistGradientBoostingRegressor(max_iter=5), True, 'auto', max(1, cpu_count() - 1)),
     (HistGradientBoostingRegressor(max_iter=5), False, 'auto', max(1, cpu_count() - 1)),
     (DecisionTreeRegressor(max_depth=2), False, 'auto', 1),
     (HistGradientBoostingRegressor(max_iter=5), True, 2, 2)],
    ids=[
        'auto, Ridge', 'auto, HistGradientBoostingRegressor',
        'auto, HistGradientBoostingRegressor, no refit',
        'auto, DecisionTreeRegressor, no refit', 'int'
    ]
)
def test_bayesian_search_forecaster_multiseries_n_jobs_auto_resolved_with_the_rules_of_each_candidate(
    estimator, refit, n_jobs, expected_n_jobs
):
    """
    Test that bayesian_search_forecaster_multiseries resolves `n_jobs='auto'` with
    `select_n_jobs_backtesting` for each candidate and passes the integer to
    `backtesting_forecaster_multiseries`, so the work threshold of a single backtesting does not
    reduce it. Without refit, the rules give one job, except with
    `HistGradientBoostingRegressor`. `Parallel` is mocked with a
    sequential one, so no workers are started.
    """
    forecaster = ForecasterRecursiveMultiSeries(estimator=estimator, lags=3)
    cv = TimeSeriesFold(
             steps              = 10,
             initial_train_size = 10,
             refit              = refit,
             fixed_train_size   = False
         )

    with (
        patch(
            "skforecast.model_selection._search.backtesting_forecaster_multiseries",
            wraps=backtesting_forecaster_multiseries
        ) as mock_backtesting,
        patch(
            "skforecast.model_selection._validation.Parallel",
            side_effect=lambda **kwargs: Parallel(n_jobs=1)
        ) as mock_parallel
    ):
        bayesian_search_forecaster_multiseries(
            forecaster    = forecaster,
            series        = series_wide_range,
            cv            = cv,
            search_space  = lambda trial: {'random_state': trial.suggest_int('random_state', 1, 2)},
            metric        = 'mean_absolute_error',
            n_trials      = 2,
            return_best   = False,
            n_jobs        = n_jobs,
            show_progress = False
        )

    assert mock_backtesting.call_count == 2
    for call in mock_backtesting.call_args_list:
        assert call.kwargs['n_jobs'] == expected_n_jobs
    for call in mock_parallel.call_args_list:
        assert call.kwargs == {'n_jobs': expected_n_jobs}
