# Unit test _bayesian_search_optuna_multiseries
# ==============================================================================
import os
import re
import sys
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.model_selection import backtesting_forecaster_multiseries
from skforecast.model_selection._search import _bayesian_search_optuna_multiseries
from skforecast.model_selection._split import TimeSeriesFold
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm
from functools import partialmethod

# Fixtures
from ..fixtures_model_selection_multiseries import series

optuna.logging.set_verbosity(optuna.logging.WARNING)
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # hide progress bar


def test_ValueError_bayesian_search_optuna_multiseries_when_not_allowed_aggregate_metric():
    """
    Test ValueError is raised in _bayesian_search_optuna_multiseries when 
    `aggregate_metric` has not a valid value.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot'
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series[:-12]),
            steps              = 3,
            refit              = True,
            fixed_train_size   = True
    )
    
    def search_space(trial):  # pragma: no cover
        search_space  = {'alpha': trial.suggest_float('not_alpha', 1e-2, 1.0)}

        return search_space

    err_msg = re.escape(
        ("Allowed `aggregate_metric` are: ['average', 'weighted_average', 'pooling']. "
         "Got: ['not_valid'].")
    )
    with pytest.raises(ValueError, match = err_msg):
        _bayesian_search_optuna_multiseries(
            forecaster         = forecaster,
            series             = series,
            search_space       = search_space,
            cv                 = cv,
            metric             = 'mean_absolute_error',
            aggregate_metric   = 'not_valid',
            n_trials           = 10,
            random_state       = 123,
            return_best        = False,
            verbose            = False,
        )


def test_ValueError_bayesian_search_optuna_multiseries_metric_list_duplicate_names():
    """
    Test ValueError is raised in _bayesian_search_optuna_multiseries when a `list` 
    of metrics is used with duplicate names.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot'
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series[:-12]),
            steps              = 3,
            refit              = True,
            fixed_train_size   = True
    )
    
    def search_space(trial):  # pragma: no cover
        search_space  = {'alpha': trial.suggest_float('not_alpha', 1e-2, 1.0)}

        return search_space

    err_msg = re.escape("When `metric` is a `list`, each metric name must be unique.")
    with pytest.raises(ValueError, match = err_msg):
        _bayesian_search_optuna_multiseries(
            forecaster         = forecaster,
            series             = series,
            search_space       = search_space,
            cv                 = cv,
            metric             = ['mean_absolute_error', mean_absolute_error],
            aggregate_metric   = 'weighted_average',
            n_trials           = 10,
            random_state       = 123,
            return_best        = False,
            verbose            = False,
        )


def test_ValueError_bayesian_search_optuna_multiseries_when_search_space_names_do_not_match():
    """
    Test ValueError is raised when search_space key name do not match the trial 
    object name from optuna.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot'
                 )
    cv = TimeSeriesFold(
                initial_train_size = len(series[:-12]),
                steps              = 3,
                refit              = True,
                fixed_train_size   = True
        )

    def search_space(trial):
        search_space  = {'alpha': trial.suggest_float('not_alpha', 1e-2, 1.0)}

        return search_space
    
    err_msg = re.escape(
        ("Some of the key values do not match the search_space key names.\n"
         "  Search Space keys  : ['alpha']\n"
         "  Trial objects keys : ['not_alpha']")
    )
    with pytest.raises(ValueError, match = err_msg):
        _bayesian_search_optuna_multiseries(
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
        )


# This mark allows to only run test with "slow" label or all except this, "not slow".
# The mark should be included in the pytest.ini file
# pytest -m slow --verbose
# pytest -m "not slow" --verbose
@pytest.mark.slow
@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires python3.8 or higher")
def test_results_output_bayesian_search_optuna_multiseries_with_mocked_when_lags_grid_dict():
    """
    Test output of _bayesian_search_optuna_multiseries in ForecasterRecursiveMultiSeries 
    when `lags_grid` is a dict with mocked (mocked done in skforecast v0.12.0).
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = RandomForestRegressor(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot',
                     transformer_series = StandardScaler()
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series[:-12]),
            steps              = 3,
            refit              = True,
            fixed_train_size   = True
    )

    def search_space(trial):
        search_space  = {
            'n_estimators'    : trial.suggest_int('n_estimators', 10, 20),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 1., log=True),
            'max_features'    : trial.suggest_categorical('max_features', ['log2', 'sqrt']),
            'lags'            : trial.suggest_categorical('lags', [2, 4])
        }
        
        return search_space

    results = _bayesian_search_optuna_multiseries(
                  forecaster         = forecaster,
                  series             = series,
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
        [list(['l1', 'l2']), np.array([1, 2]),
            {'n_estimators': 13, 'min_samples_leaf': 0.20142843988664705, 'max_features': 'sqrt'},
            0.21000466923864858, 13, 0.20142843988664705, 'sqrt'],
        [list(['l1', 'l2']), np.array([1, 2]),
            {'n_estimators': 11, 'min_samples_leaf': 0.2714570796881701, 'max_features': 'sqrt'},
            0.21088438837980938, 11, 0.2714570796881701, 'sqrt'],
        [list(['l1', 'l2']), np.array([1, 2]),
            {'n_estimators': 17, 'min_samples_leaf': 0.19325882509735576, 'max_features': 'sqrt'},
            0.21223533093219432, 17, 0.19325882509735576, 'sqrt'],
        [list(['l1', 'l2']), np.array([1, 2]),
            {'n_estimators': 14, 'min_samples_leaf': 0.1147302385573586, 'max_features': 'sqrt'},
            0.2134063619224233, 14, 0.1147302385573586, 'sqrt'],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'n_estimators': 13, 'min_samples_leaf': 0.2599119286878713, 'max_features': 'log2'},
            0.2155978774269141, 13, 0.2599119286878713, 'log2'],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'n_estimators': 20, 'min_samples_leaf': 0.4839825891759374, 'max_features': 'log2'},
            0.2161528806005189, 20, 0.4839825891759374, 'log2'],
        [list(['l1', 'l2']), np.array([1, 2]),
            {'n_estimators': 15, 'min_samples_leaf': 0.41010449151752726, 'max_features': 'sqrt'},
            0.2163760081019307, 15, 0.41010449151752726, 'sqrt'],
        [list(['l1', 'l2']), np.array([1, 2]),
            {'n_estimators': 14, 'min_samples_leaf': 0.782328520465639, 'max_features': 'log2'},
            0.21653840486608772, 14, 0.782328520465639, 'log2'],
        [list(['l1', 'l2']), np.array([1, 2]),
            {'n_estimators': 15, 'min_samples_leaf': 0.34027307604369605, 'max_features': 'sqrt'},
            0.2166204088831145, 15, 0.34027307604369605, 'sqrt'],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'n_estimators': 17, 'min_samples_leaf': 0.21035794225904136, 'max_features': 'log2'},
            0.21732156972764355, 17, 0.21035794225904136, 'log2']],
        dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error__weighted_average', 'n_estimators', 'min_samples_leaf', 'max_features'],
        index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64')
    ).astype({
        'mean_absolute_error__weighted_average': float, 
        'n_estimators': int, 
        'min_samples_leaf': float
    })

    pd.testing.assert_frame_equal(results, expected_results, check_dtype=False)


def test_results_output_bayesian_search_optuna_multiseries_ForecasterAutoregMultiSeries_with_levels():
    """
    Test output of _bayesian_search_optuna_multiseries in 
    ForecasterRecursiveMultiSeries for level 'l1' with mocked 
    (mocked done in skforecast v0.12.0).
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot',
                     transformer_series = StandardScaler()
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series[:-12]),
            steps              = 3,
            refit              = True,
            fixed_train_size   = True
    )
    levels = ['l1']

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1.0),
            'lags' : trial.suggest_categorical('lags', [2, 4])
        }
        
        return search_space

    results = _bayesian_search_optuna_multiseries(
                  forecaster         = forecaster,
                  series             = series,
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
        np.array([[list(['l1']), np.array([1, 2, 3, 4]),
            {'alpha': 0.23598059857016607}, 0.2158500990715522,
            0.23598059857016607],
        [list(['l1']), np.array([1, 2, 3, 4]), {'alpha': 0.398196343012209},
            0.2158706018070041, 0.398196343012209],
        [list(['l1']), np.array([1, 2, 3, 4]), {'alpha': 0.4441865222328282},
            0.21587636323102785, 0.4441865222328282],
        [list(['l1']), np.array([1, 2, 3, 4]), {'alpha': 0.53623586010342},
            0.21588782726158717, 0.53623586010342],
        [list(['l1']), np.array([1, 2, 3, 4]), {'alpha': 0.7252189487445193},
            0.21591108476752116, 0.7252189487445193],
        [list(['l1']), np.array([1, 2]), {'alpha': 0.5558016213920624},
            0.21631472411547312, 0.5558016213920624],
        [list(['l1']), np.array([1, 2]), {'alpha': 0.6995044937418831},
            0.21631966674575429, 0.6995044937418831],
        [list(['l1']), np.array([1, 2]), {'alpha': 0.7406154516747153},
            0.2163210721487403, 0.7406154516747153],
        [list(['l1']), np.array([1, 2]), {'alpha': 0.8509374761370117},
            0.21632482482968562, 0.8509374761370117],
        [list(['l1']), np.array([1, 2]), {'alpha': 0.9809565564007693},
            0.2163292127503296, 0.9809565564007693]], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error', 'alpha'],
        index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64')
    ).astype({
        'mean_absolute_error': float, 
        'alpha': float
    })

    pd.testing.assert_frame_equal(results, expected_results, check_dtype=False)


def test_results_output_bayesian_search_optuna_multiseries_ForecasterAutoregMultiSeries_with_multiple_metrics_aggregated():
    """
    Test output of _bayesian_search_optuna_multiseries in ForecasterRecursiveMultiSeries
    with multiple metrics and aggregated metrics (mocked done in skforecast v0.12.0).
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot',
                     transformer_series = StandardScaler()
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series[:-12]),
            steps              = 3,
            refit              = True,
            fixed_train_size   = True
    )

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1.0),
            'lags' : trial.suggest_categorical('lags', [2, 4])
        }

        return search_space

    results = _bayesian_search_optuna_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  search_space       = search_space,
                  cv                 = cv,
                  metric             = ['mean_absolute_error', 'mean_absolute_scaled_error'],
                  aggregate_metric   = ['weighted_average', 'average', 'pooling'],
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False
              )[0]

    expected_results = pd.DataFrame(
        {
            "levels": [
                ["l1", "l2"],
                ["l1", "l2"],
                ["l1", "l2"],
                ["l1", "l2"],
                ["l1", "l2"],
                ["l1", "l2"],
                ["l1", "l2"],
                ["l1", "l2"],
                ["l1", "l2"],
                ["l1", "l2"],
            ],
            "lags": [
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
            "params": [
                {"alpha": 0.5558016213920624},
                {"alpha": 0.6995044937418831},
                {"alpha": 0.7406154516747153},
                {"alpha": 0.8509374761370117},
                {"alpha": 0.9809565564007693},
                {"alpha": 0.23598059857016607},
                {"alpha": 0.398196343012209},
                {"alpha": 0.4441865222328282},
                {"alpha": 0.53623586010342},
                {"alpha": 0.7252189487445193},
            ],
            "mean_absolute_error__weighted_average": [
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
            "mean_absolute_error__average": [
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
            "mean_absolute_error__pooling": [
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
            "mean_absolute_scaled_error__weighted_average": [
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
            "mean_absolute_scaled_error__average": [
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
            "mean_absolute_scaled_error__pooling": [
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
            "alpha": [
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
        }
    )

    pd.testing.assert_frame_equal(results, expected_results, check_dtype=False)


def test_results_output_bayesian_search_optuna_multiseries_with_kwargs_create_study():
    """
    Test output of _bayesian_search_optuna_multiseries in ForecasterRecursiveMultiSeries 
    when `kwargs_create_study` with mocked (mocked done in skforecast v0.12.0).
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot',
                     transformer_series = StandardScaler()
                 )

    cv = TimeSeriesFold(
            initial_train_size = len(series[:-12]),
            steps              = 3,
            refit              = False,
            fixed_train_size   = True
    )

    def search_space(trial):
        search_space  = {
            'alpha' : trial.suggest_float('alpha', 2e-2, 2.0),
            'lags'  : trial.suggest_categorical('lags', [2, 4])
        }
        
        return search_space

    kwargs_create_study = {
        "sampler": TPESampler(seed=123, prior_weight=2.0, consider_magic_clip=False)
    }
    results = _bayesian_search_optuna_multiseries(
                  forecaster          = forecaster,
                  series              = series,
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
        np.array([[list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.47196119714033213}, 0.20881449475639347,
            0.47196119714033213],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.796392686024418}, 0.2088304433786553,
            0.796392686024418],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.8883730444656563}, 0.20883491982666835,
            0.8883730444656563],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 1.07247172020684}, 0.20884382037261517,
            1.07247172020684],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 1.4504378974890386}, 0.20886185085049824,
            1.4504378974890386],
        [list(['l1', 'l2']), np.array([1, 2]), {'alpha': 1.1116032427841247},
            0.20914976196539467, 1.1116032427841247],
        [list(['l1', 'l2']), np.array([1, 2]), {'alpha': 1.3990089874837661},
            0.20915713939422753, 1.3990089874837661],
        [list(['l1', 'l2']), np.array([1, 2]), {'alpha': 1.4812309033494306},
            0.20915923928135344, 1.4812309033494306],
        [list(['l1', 'l2']), np.array([1, 2]), {'alpha': 1.7018749522740233},
            0.20916485106998073, 1.7018749522740233],
        [list(['l1', 'l2']), np.array([1, 2]), {'alpha': 1.9619131128015386},
            0.2091714215090992, 1.9619131128015386]], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error__weighted_average', 'alpha'],
        index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64')
    ).astype({
        'mean_absolute_error__weighted_average': float,
        'alpha': float
    })

    pd.testing.assert_frame_equal(results, expected_results)


def test_results_output_bayesian_search_optuna_multiseries_with_kwargs_study_optimize():
    """
    Test output of _bayesian_search_optuna_multiseries in ForecasterRecursiveMultiSeries 
    when `kwargs_study_optimize` with mocked (mocked done in skforecast v0.12.0).
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = RandomForestRegressor(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot',
                     transformer_series = StandardScaler()
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series[:-12]),
            steps              = 3,
            refit              = False,
            fixed_train_size   = True
    )

    def search_space(trial):
        search_space  = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 200),
            'max_depth'   : trial.suggest_int('max_depth', 20, 35, log=True),
            'max_features': trial.suggest_categorical('max_features', ['log2', 'sqrt']),
            'lags': trial.suggest_categorical('lags', [2, 4])
        }
         
        return search_space

    kwargs_study_optimize = {'timeout': 3.0}
    results = _bayesian_search_optuna_multiseries(
                  forecaster            = forecaster,
                  series                = series,
                  cv                    = cv,
                  search_space          = search_space,
                  metric                = 'mean_absolute_error',
                  aggregate_metric      = 'weighted_average',
                  n_trials              = 10,
                  random_state          = 123,
                  n_jobs                = 1,
                  return_best           = False,
                  verbose               = False,
                  kwargs_study_optimize = kwargs_study_optimize
              )[0].reset_index(drop=True)
    
    expected_results = pd.DataFrame(
        np.array([[list(['l1', 'l2']), np.array([1, 2]),
            {'n_estimators': 144, 'max_depth': 20, 'max_features': 'sqrt'},
            0.18642719373739342, 144, 20, 'sqrt'],
        [list(['l1', 'l2']), np.array([1, 2]),
            {'n_estimators': 143, 'max_depth': 33, 'max_features': 'log2'},
            0.1898478778067653, 143, 33, 'log2'],
        [list(['l1', 'l2']), np.array([1, 2]),
            {'n_estimators': 153, 'max_depth': 27, 'max_features': 'sqrt'},
            0.19105592191594672, 153, 27, 'sqrt'],
        [list(['l1', 'l2']), np.array([1, 2]),
            {'n_estimators': 170, 'max_depth': 23, 'max_features': 'sqrt'},
            0.19258452474386697, 170, 23, 'sqrt'],
        [list(['l1', 'l2']), np.array([1, 2]),
            {'n_estimators': 109, 'max_depth': 25, 'max_features': 'sqrt'},
            0.19273502480138094, 109, 25, 'sqrt'],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'n_estimators': 199, 'max_depth': 29, 'max_features': 'log2'},
            0.2220112988108661, 199, 29, 'log2'],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'n_estimators': 172, 'max_depth': 24, 'max_features': 'log2'},
            0.22300765864212552, 172, 24, 'log2']], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error__weighted_average', 'n_estimators', 'max_depth', 'max_features'],
        index=pd.RangeIndex(start=0, stop=7, step=1)
    ).astype({
        'mean_absolute_error__weighted_average': float,
        'n_estimators': int,
        'max_depth': int
    })

    pd.testing.assert_frame_equal(results.head(1), expected_results.head(1), check_dtype=False)


def test_results_output_bayesian_search_optuna_multiseries_when_lags_is_not_provided():
    """
    Test output of _bayesian_search_optuna_multiseries in ForecasterRecursiveMultiSeries 
    when `lags` is not provided (mocked done in skforecast v0.12.0), 
    should use forecaster.lags.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 4,
                     encoding  = 'onehot',
                     transformer_series = StandardScaler()
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series[:-12]),
            steps              = 3,
            refit              = True,
            fixed_train_size   = True
    )

    def search_space(trial):
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}
        
        return search_space

    results = _bayesian_search_optuna_multiseries(
                  forecaster         = forecaster,
                  series             = series,
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
        np.array([[list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.2345829390285611}, 0.21476183342122757,
            0.2345829390285611],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.29327794160087567}, 0.21476722307839208,
            0.29327794160087567],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.398196343012209}, 0.21477679099211167,
            0.398196343012209],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.42887539552321635}, 0.2147795727959785,
            0.42887539552321635],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.48612258246951734}, 0.21478474449018078,
            0.48612258246951734],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.5558016213920624}, 0.2147910057902114,
            0.5558016213920624],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.6879814411990146}, 0.21480278321679924,
            0.6879814411990146],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.6995044937418831}, 0.2148038037681593,
            0.6995044937418831],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.7222742800877074}, 0.2148058175049471,
            0.7222742800877074],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.9809565564007693}, 0.2148284279387544,
            0.9809565564007693]], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error__weighted_average', 'alpha'],
        index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64')
    ).astype({
        'mean_absolute_error__weighted_average': float,
        'alpha': float
    })

    pd.testing.assert_frame_equal(results, expected_results)


def test_results_output_bayesian_search_optuna_multiseries_ForecasterAutoregMultiVariate():
    """
    Test output of _bayesian_search_optuna_multiseries in 
    ForecasterDirectMultiVariate with mocked (mocked done in skforecast v0.12.0).
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3,
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series[:-12]),
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

    results = _bayesian_search_optuna_multiseries(
                  forecaster         = forecaster,
                  series             = series,
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
        np.array([[list(['l1']), np.array([1, 2]), {'alpha': 0.5558016213920624},
            0.20494430799685603, 0.5558016213920624],
        [list(['l1']), np.array([1, 2]), {'alpha': 0.6995044937418831},
            0.2054241757124218, 0.6995044937418831],
        [list(['l1']), np.array([1, 2]), {'alpha': 0.7406154516747153},
            0.205550565800915, 0.7406154516747153],
        [list(['l1']), np.array([1, 2]), {'alpha': 0.8509374761370117},
            0.20586868696104646, 0.8509374761370117],
        [list(['l1']), np.array([1, 2]), {'alpha': 0.9809565564007693},
            0.20620855083897363, 0.9809565564007693],
        [list(['l1']), np.array([1, 2, 3, 4]), {'alpha': 0.7252189487445193},
            0.21702482068927706, 0.7252189487445193],
        [list(['l1']), np.array([1, 2, 3, 4]), {'alpha': 0.53623586010342},
            0.2176184334539817, 0.53623586010342],
        [list(['l1']), np.array([1, 2, 3, 4]), {'alpha': 0.4441865222328282},
            0.21794402393310316, 0.4441865222328282],
        [list(['l1']), np.array([1, 2, 3, 4]), {'alpha': 0.398196343012209},
            0.21811674876142453, 0.398196343012209],
        [list(['l1']), np.array([1, 2, 3, 4]),
            {'alpha': 0.23598059857016607}, 0.21912194726679404,
            0.23598059857016607]], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error', 'alpha'],
        index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64')
    ).astype({
        'mean_absolute_error': float,
        'alpha': float
    })

    pd.testing.assert_frame_equal(results, expected_results)


def test_results_output_bayesian_search_optuna_multiseries_ForecasterAutoregMultiVariate_lags_dict():
    """
    Test output of _bayesian_search_optuna_multiseries in 
    ForecasterDirectMultiVariate when lags is a dict 
    with mocked (mocked done in skforecast v0.12.0).
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3,
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series[:-12]),
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

    results = _bayesian_search_optuna_multiseries(
                  forecaster         = forecaster,
                  series             = series,
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
        [list(['l1']), {'l1': np.array([1, 2]), 'l2': np.array([1, 3])},
            {'alpha': 0.30077690592494105}, 0.20844762947854312,
            0.30077690592494105],
        [list(['l1']), {'l1': np.array([1, 2]), 'l2': np.array([1, 3])},
            {'alpha': 0.4365541356963474}, 0.20880336411565956,
            0.4365541356963474],
        [list(['l1']), {'l1': np.array([1, 2]), 'l2': np.array([1, 3])},
            {'alpha': 0.6380569489658079}, 0.2092371153650312,
            0.6380569489658079],
        [list(['l1']), {'l1': None, 'l2': np.array([1, 3])},
            {'alpha': 0.7252189487445193}, 0.21685083725475654,
            0.7252189487445193],
        [list(['l1']), {'l1': None, 'l2': np.array([1, 3])},
            {'alpha': 0.7222742800877074}, 0.2168551702095223,
            0.7222742800877074],
        [list(['l1']), {'l1': None, 'l2': np.array([1, 3])},
            {'alpha': 0.43208779389318014}, 0.21733651515831423,
            0.43208779389318014],
        [list(['l1']), {'l1': np.array([1, 3]), 'l2': None},
            {'alpha': 0.6995044937418831}, 0.22066810286127028,
            0.6995044937418831],
        [list(['l1']), {'l1': np.array([1, 3]), 'l2': None},
            {'alpha': 0.48612258246951734}, 0.22159811332626014,
            0.48612258246951734],
        [list(['l1']), {'l1': np.array([1, 3]), 'l2': None},
            {'alpha': 0.4441865222328282}, 0.2218030808436934,
            0.4441865222328282],
        [list(['l1']), {'l1': np.array([1, 3]), 'l2': None},
            {'alpha': 0.190666813148965}, 0.22324045507529866,
            0.190666813148965]], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error', 'alpha'],
        index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64')
    ).astype({
        'mean_absolute_error': float,
        'alpha': float
    })

    pd.testing.assert_frame_equal(results, expected_results)


def test_evaluate_bayesian_search_optuna_multiseries_when_return_best_ForecasterAutoregMultiSeries():
    """
    Test forecaster is refitted when return_best=True in 
    _bayesian_search_optuna_multiseries with ForecasterRecursiveMultiSeries 
    (mocked done in skforecast v0.12.0).
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot',
                     transformer_series = StandardScaler()
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series[:-12]),
            steps              = 3,
            refit              = True,
            fixed_train_size   = True
    )
    
    def search_space(trial):
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}

        return search_space

    _bayesian_search_optuna_multiseries(
        forecaster         = forecaster,
        series             = series,
        cv                 = cv,
        search_space       = search_space,
        metric             = 'mean_absolute_error',
        aggregate_metric   = 'weighted_average',
        n_trials           = 10,
        random_state       = 123,
        return_best        = True,
        verbose            = False
    )
    expected_lags = np.array([1, 2])
    expected_alpha = 0.2345829390285611
    
    np.testing.assert_array_equal(forecaster.lags, expected_lags)
    assert expected_alpha == forecaster.regressor.alpha


def test_results_opt_best_output__bayesian_search_optuna_multiseries_with_output_study_best_trial_optuna():
    """
    Test results_opt_best output of _bayesian_search_optuna_multiseries with output 
    study.best_trial optuna (mocked done in skforecast v0.12.0).
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot'
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series[:-12]),
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
        series        = series,
        cv            = cv,
        metric        = metric,
        verbose       = verbose,
        show_progress = show_progress
    ) -> float:
        
        alpha = trial.suggest_float('alpha', 1e-2, 1.0)
        forecaster = ForecasterRecursiveMultiSeries(
                         regressor = Ridge(random_state=random_state, 
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

    study = optuna.create_study(direction="minimize", 
                                sampler=TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)
    best_trial = study.best_trial

    def search_space(trial):
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}

        return search_space

    return_best  = False
    results_opt_best = _bayesian_search_optuna_multiseries(
                           forecaster    = forecaster,
                           series        = series,
                           cv            = cv,
                           search_space  = search_space,
                           metric        = metric,
                           n_trials      = n_trials,
                           return_best   = return_best,
                           verbose       = verbose,
                           show_progress = show_progress
                       )[1]

    assert best_trial.number == results_opt_best.number
    assert best_trial.values == approx(results_opt_best.values)
    assert best_trial.params == results_opt_best.params


def test_bayesian_search_optuna_multiseries_ForecasterAutoregMultiSeries_output_file():
    """
    Test output file of _bayesian_search_optuna_multiseries in 
    ForecasterRecursiveMultiSeries.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = RandomForestRegressor(random_state=123),
                     lags      = 2 
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series[:-12]),
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

    output_file = 'test_bayesian_search_optuna_multiseries_output_file.txt'
    _ = _bayesian_search_optuna_multiseries(
            forecaster    = forecaster,
            series        = series,
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


def test_bayesian_search_optuna_multiseries_ForecasterAutoregMultiVariate_output_file():
    """
    Test output file of _bayesian_search_optuna_multiseries in 
    ForecasterDirectMultiVariate.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor = RandomForestRegressor(random_state=123),
                     level     = 'l1',
                     lags      = 2,
                     steps     = 3
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series[:-12]),
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

    output_file = 'test_bayesian_search_optuna_multiseries_output_file_3.txt'
    _ = _bayesian_search_optuna_multiseries(
            forecaster   = forecaster,
            series       = series,
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
