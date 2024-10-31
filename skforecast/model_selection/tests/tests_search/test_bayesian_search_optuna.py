# Unit test _bayesian_search_optuna
# ==============================================================================
import re
import pytest
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from skforecast.metrics import mean_absolute_scaled_error, root_mean_squared_scaled_error
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection._search import _bayesian_search_optuna
from skforecast.model_selection._split import TimeSeriesFold, OneStepAheadFold
from skforecast.preprocessing import RollingFeatures
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm
from functools import partialmethod

# Fixtures
from ..fixtures_model_selection import y
from ..fixtures_model_selection import y_feature_selection
from ..fixtures_model_selection import exog_feature_selection

optuna.logging.set_verbosity(optuna.logging.WARNING)
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # hide progress bar


def test_TypeError_bayesian_search_optuna_when_cv_not_valid():
    """
    Test TypeError is raised in _bayesian_search_optuna when cv is not
    a valid splitter.
    """
    class DummyCV:
        pass

    cv = DummyCV()
    forecaster = ForecasterRecursive(
                     regressor = Ridge(random_state=123),
                     lags      = 2
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
        _bayesian_search_optuna(
            forecaster         = forecaster,
            y                  = y,
            cv                 = cv,
            search_space       = search_space,
            metric             = ['mean_absolute_error', mean_absolute_error],
            n_trials           = 10,
            random_state       = 123,
            return_best        = False,
            verbose            = False,
        )


def test_TypeError_bayesian_search_optuna_when_forecaster_not_OneStepAhead():
    """
    Test TypeError is raised in _bayesian_search_optuna when forecaster is not
    allowed to use OneStepAheadFold.
    """

    cv = OneStepAheadFold(
             initial_train_size    = 100,
             return_all_indexes    = False,
         )
    
    class DummyForecaster:
        pass
    forecaster = DummyForecaster()
    
    def search_space(trial):  # pragma: no cover
        search_space  = {
            'alpha': trial.suggest_float('not_alpha', 1e-2, 1.0),
            'lags': trial.suggest_categorical('lags', [2, 4])
        }

        return search_space
    
    err_msg = re.escape(
        f"Only forecasters of type ['ForecasterRecursive', 'ForecasterDirect'] are allowed "
        f"when using `cv` of type `OneStepAheadFold`. Got {type(forecaster).__name__}."
    )
    with pytest.raises(TypeError, match = err_msg):
        _bayesian_search_optuna(
            forecaster         = forecaster,
            y                  = y,
            cv                 = cv,
            search_space       = search_space,
            metric             = ['mean_absolute_error', mean_absolute_error],
            n_trials           = 10,
            random_state       = 123,
            return_best        = False,
            verbose            = False,
        )


def test_ValueError_bayesian_search_optuna_metric_list_duplicate_names():
    """
    Test ValueError is raised in _bayesian_search_optuna when a `list` of 
    metrics is used with duplicate names.
    """
    forecaster = ForecasterRecursive(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    
    def search_space(trial):  # pragma: no cover
        search_space  = {
            'alpha': trial.suggest_float('not_alpha', 1e-2, 1.0),
            'lags': trial.suggest_categorical('lags', [2, 4])
        }

        return search_space

    err_msg = re.escape("When `metric` is a `list`, each metric name must be unique.")
    with pytest.raises(ValueError, match = err_msg):
        _bayesian_search_optuna(
            forecaster         = forecaster,
            y                  = y,
            cv                 = cv,
            search_space       = search_space,
            metric             = ['mean_absolute_error', mean_absolute_error],
            n_trials           = 10,
            random_state       = 123,
            return_best        = False,
            verbose            = False,
        )


def test_ValueError_bayesian_search_optuna_when_search_space_names_do_not_match():
    """
    Test ValueError is raised when search_space key name do not match the trial 
    object name from optuna.
    """
    forecaster = ForecasterRecursive(
                     regressor = Ridge(random_state=123),
                     lags      = 2 
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('not_alpha', 1e-2, 1.0),
            'lags': trial.suggest_categorical('lags', [2, 4])
        }

        return search_space
    
    err_msg = re.escape(
        ("Some of the key values do not match the search_space key names.\n"
         "  Search Space keys  : ['alpha', 'lags']\n"
         "  Trial objects keys : ['not_alpha', 'lags'].")
    )
    with pytest.raises(ValueError, match = err_msg):
        _bayesian_search_optuna(
            forecaster         = forecaster,
            y                  = y,
            cv                 = cv, 
            search_space       = search_space,
            metric             = 'mean_absolute_error',
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
def test_results_output_bayesian_search_optuna_ForecasterRecursive():
    """
    Test output of _bayesian_search_optuna in ForecasterRecursive with mocked
    (mocked done in Skforecast v0.4.3).
    """
    forecaster = ForecasterRecursive(
                     regressor = RandomForestRegressor(random_state=123),
                     lags      = 2
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )

    def search_space(trial):
        search_space  = {
            'n_estimators'    : trial.suggest_int('n_estimators', 10, 20),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 1., log=True),
            'max_features'    : trial.suggest_categorical('max_features', ['log2', 'sqrt']),
            'lags'            : trial.suggest_categorical('lags', [2, 4])
        } 
        
        return search_space

    results = _bayesian_search_optuna(
                  forecaster         = forecaster,
                  y                  = y,
                  cv                 = cv,
                  search_space       = search_space,
                  metric             = 'mean_absolute_error',
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([[np.array([1, 2, 3, 4]),
                {'n_estimators': 20, 'min_samples_leaf': 0.4839825891759374, 'max_features': 'log2'},
                0.21252324019730395, 20, 0.4839825891759374, 'log2'],
            [np.array([1, 2]),
                {'n_estimators': 15, 'min_samples_leaf': 0.34027307604369605, 'max_features': 'sqrt'},
                0.21479600790277778, 15, 0.34027307604369605, 'sqrt'],
            [np.array([1, 2]),
                {'n_estimators': 15, 'min_samples_leaf': 0.41010449151752726, 'max_features': 'sqrt'},
                0.21479600790277778, 15, 0.41010449151752726, 'sqrt'],
            [np.array([1, 2]),
                {'n_estimators': 14, 'min_samples_leaf': 0.782328520465639, 'max_features': 'log2'},
                0.21661388810185186, 14, 0.782328520465639, 'log2'],
            [np.array([1, 2]),
                {'n_estimators': 13, 'min_samples_leaf': 0.20142843988664705, 'max_features': 'sqrt'},
                0.22084665733716438, 13, 0.20142843988664705, 'sqrt'],
            [np.array([1, 2, 3, 4]),
                {'n_estimators': 17, 'min_samples_leaf': 0.21035794225904136, 'max_features': 'log2'},
                0.2229839747692736, 17, 0.21035794225904136, 'log2'],
            [np.array([1, 2]),
                {'n_estimators': 17, 'min_samples_leaf': 0.19325882509735576, 'max_features': 'sqrt'},
                0.2235827591941962, 17, 0.19325882509735576, 'sqrt'],
            [np.array([1, 2]),
                {'n_estimators': 11, 'min_samples_leaf': 0.2714570796881701, 'max_features': 'sqrt'},
                0.22384439522399655, 11, 0.2714570796881701, 'sqrt'],
            [np.array([1, 2, 3, 4]),
                {'n_estimators': 13, 'min_samples_leaf': 0.2599119286878713, 'max_features': 'log2'},
                0.22522461935181756, 13, 0.2599119286878713, 'log2'],
            [np.array([1, 2]),
                {'n_estimators': 14, 'min_samples_leaf': 0.1147302385573586, 'max_features': 'sqrt'},
                0.22764885273610677, 14, 0.1147302385573586, 'sqrt']],
            dtype=object),
        columns=['lags', 'params', 'mean_absolute_error', 'n_estimators', 'min_samples_leaf', 'max_features'],
        index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results, check_dtype=False)


def test_results_output_bayesian_search_optuna_window_features_ForecasterRecursive():
    """
    Test output of _bayesian_search_optuna in ForecasterRecursive including
    window_features with mocked (mocked done in Skforecast v0.4.3).
    """
    window_features = RollingFeatures(
        stats = ['mean', 'std', 'min', 'max', 'sum', 'median', 'ratio_min_max', 'coef_variation'],
        window_sizes = 3,
    )
    forecaster = ForecasterRecursive(
                     regressor = RandomForestRegressor(random_state=123),
                     lags      = 2,
                     window_features = window_features,
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )

    def search_space(trial):
        search_space  = {
            'n_estimators'    : trial.suggest_int('n_estimators', 10, 20),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 1., log=True),
            'max_features'    : trial.suggest_categorical('max_features', ['log2', 'sqrt']),
            'lags'            : trial.suggest_categorical('lags', [2, 4])
        } 
        
        return search_space

    results = _bayesian_search_optuna(
                  forecaster         = forecaster,
                  y                  = y,
                  cv                 = cv,
                  search_space       = search_space,
                  metric             = 'mean_absolute_error',
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False
              )[0]
    
    expected_results = pd.DataFrame({
        'lags': [
            np.array([1, 2, 3, 4]),
            np.array([1, 2]),
            np.array([1, 2]),
            np.array([1, 2]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2]),
            np.array([1, 2]),
            np.array([1, 2]),
            np.array([1, 2]),
        ],
        'params': [
            {
                'n_estimators': 20,
                'min_samples_leaf': 0.4839825891759374,
                'max_features': 'log2',
            },
            {
                'n_estimators': 15,
                'min_samples_leaf': 0.41010449151752726,
                'max_features': 'sqrt',
            },
            {
                'n_estimators': 14,
                'min_samples_leaf': 0.782328520465639,
                'max_features': 'log2',
            },
            {
                'n_estimators': 15,
                'min_samples_leaf': 0.34027307604369605,
                'max_features': 'sqrt',
            },
            {
                'n_estimators': 13,
                'min_samples_leaf': 0.2599119286878713,
                'max_features': 'log2',
            },
            {
                'n_estimators': 17,
                'min_samples_leaf': 0.21035794225904136,
                'max_features': 'log2',
            },
            {
                'n_estimators': 11,
                'min_samples_leaf': 0.2714570796881701,
                'max_features': 'sqrt',
            },
            {
                'n_estimators': 13,
                'min_samples_leaf': 0.20142843988664705,
                'max_features': 'sqrt',
            },
            {
                'n_estimators': 17,
                'min_samples_leaf': 0.19325882509735576,
                'max_features': 'sqrt',
            },
            {
                'n_estimators': 14,
                'min_samples_leaf': 0.1147302385573586,
                'max_features': 'sqrt',
            },
        ],
        'mean_absolute_error': [
            0.21252324019730395,
            0.21851156682698414,
            0.22027505037414963,
            0.22530355563144466,
            0.2378358005415178,
            0.2384269598653063,
            0.25241649504238356,
            0.2584381014601676,
            0.26067982740527423,
            0.2697438465540914,
        ],
        'n_estimators': [20, 15, 14, 15, 13, 17, 11, 13, 17, 14],
        'min_samples_leaf': [
            0.4839825891759374,
            0.41010449151752726,
            0.782328520465639,
            0.34027307604369605,
            0.2599119286878713,
            0.21035794225904136,
            0.2714570796881701,
            0.20142843988664705,
            0.19325882509735576,
            0.1147302385573586,
        ],
        'max_features': [
            'log2',
            'sqrt',
            'log2',
            'sqrt',
            'log2',
            'log2',
            'sqrt',
            'sqrt',
            'sqrt',
            'sqrt',
        ],
    })

    pd.testing.assert_frame_equal(results, expected_results, check_dtype=False)
    

def test_results_output_bayesian_search_optuna_ForecasterRecursive_with_kwargs_create_study():
    """
    Test output of _bayesian_search_optuna in ForecasterRecursive with 
    kwargs_create_study with mocked (mocked done in Skforecast v0.4.3).
    """
    forecaster = ForecasterRecursive(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1.0),
            'lags': trial.suggest_categorical('lags', [4, 2])
        }
        
        return search_space

    kwargs_create_study = {
        'sampler': TPESampler(seed=123, prior_weight=2.0, consider_magic_clip=False)
    }
    results = _bayesian_search_optuna(
                  forecaster          = forecaster,
                  y                   = y,
                  cv                  = cv, 
                  search_space        = search_space,
                  metric              = 'mean_absolute_error',
                  n_trials            = 10,
                  random_state        = 123,
                  return_best         = False,
                  verbose             = False,
                  kwargs_create_study = kwargs_create_study
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([[np.array([1, 2]), {'alpha': 0.23598059857016607},
            0.21239141697571848, 0.23598059857016607],
        [np.array([1, 2]), {'alpha': 0.398196343012209}, 0.21271021033387602,
            0.398196343012209],
        [np.array([1, 2]), {'alpha': 0.4441865222328282}, 0.2127897499229874,
            0.4441865222328282],
        [np.array([1, 2]), {'alpha': 0.53623586010342}, 0.21293692257888705,
            0.53623586010342],
        [np.array([1, 2]), {'alpha': 0.7252189487445193},
            0.21319693043832985, 0.7252189487445193],
        [np.array([1, 2, 3, 4]), {'alpha': 0.9809565564007693},
            0.21539791166603497, 0.9809565564007693],
        [np.array([1, 2, 3, 4]), {'alpha': 0.8509374761370117},
            0.21557690844753197, 0.8509374761370117],
        [np.array([1, 2, 3, 4]), {'alpha': 0.7406154516747153},
            0.2157346392837304, 0.7406154516747153],
        [np.array([1, 2, 3, 4]), {'alpha': 0.6995044937418831},
            0.21579460210585208, 0.6995044937418831],
        [np.array([1, 2, 3, 4]), {'alpha': 0.5558016213920624},
            0.21600778429729228, 0.5558016213920624]], dtype=object),
        columns=['lags', 'params', 'mean_absolute_error', 'alpha'],
        index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64')
    ).astype({
        'mean_absolute_error': float,
        'alpha': float
    })

    pd.testing.assert_frame_equal(results, expected_results)


def test_results_output_bayesian_search_optuna_ForecasterRecursive_with_kwargs_study_optimize():
    """
    Test output of _bayesian_search_optuna in ForecasterRecursive when 
    kwargs_study_optimize with mocked (mocked done in Skforecast v0.4.3).
    """
    forecaster = ForecasterRecursive(
                     regressor = RandomForestRegressor(random_state=123),
                     lags      = 2 
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )

    def search_space(trial):
        search_space  = {
            'n_estimators': trial.suggest_int('n_estimators', 2, 10),
            'max_depth'   : trial.suggest_int('max_depth', 2, 10, log=True),
            'max_features': trial.suggest_categorical('max_features', ['log2', 'sqrt']),
            'lags'        : trial.suggest_categorical('lags', [2, 4])
        } 
        
        return search_space

    kwargs_study_optimize = {'timeout': 10}
    results = _bayesian_search_optuna(
                    forecaster            = forecaster,
                    y                     = y,
                    cv                    = cv,
                    search_space          = search_space,
                    metric                = 'mean_absolute_error',
                    n_trials              = 5,
                    random_state          = 123,
                    n_jobs                = 1,
                    return_best           = False,
                    verbose               = False,
                    kwargs_study_optimize = kwargs_study_optimize
                )[0].reset_index(drop=True)

    expected_results = pd.DataFrame(
        np.array([[np.array([1, 2, 3, 4]),
            {'n_estimators': 8, 'max_depth': 3, 'max_features': 'log2'},
            0.2176619102322017, 8, 3, 'log2'],
        [np.array([1, 2]),
            {'n_estimators': 8, 'max_depth': 3, 'max_features': 'sqrt'},
            0.21923614756760298, 8, 3, 'sqrt'],
        [np.array([1, 2]),
            {'n_estimators': 5, 'max_depth': 2, 'max_features': 'sqrt'},
            0.22116013675443522, 5, 2, 'sqrt'],
        [np.array([1, 2, 3, 4]),
            {'n_estimators': 10, 'max_depth': 6, 'max_features': 'log2'},
            0.22221487679563792, 10, 6, 'log2'],
        [np.array([1, 2]),
            {'n_estimators': 6, 'max_depth': 4, 'max_features': 'sqrt'},
            0.22883925084220677, 6, 4, 'sqrt']], dtype=object),
        columns=['lags', 'params', 'mean_absolute_error', 'n_estimators', 'max_depth', 'max_features'],
        index=pd.RangeIndex(start=0, stop=5, step=1)
    ).astype({
        'mean_absolute_error': float,
        'n_estimators': int,
        'max_depth': int
    })

    pd.testing.assert_frame_equal(results, expected_results, check_dtype=False)


def test_results_output_bayesian_search_optuna_ForecasterRecursive_when_lags_not_in_search_space():
    """
    Test output of _bayesian_search_optuna in ForecasterRecursive when lag is not 
    in search_space with mocked (mocked done in Skforecast v0.4.3).
    """
    forecaster = ForecasterRecursive(
                     regressor = Ridge(random_state=123),
                     lags      = 4
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    
    def search_space(trial):
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}

        return search_space

    results = _bayesian_search_optuna(
                  forecaster         = forecaster,
                  y                  = y,
                  search_space       = search_space,
                  cv                 = cv,
                  metric             = 'mean_absolute_error',
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([[np.array([1, 2, 3, 4]), {'alpha': 0.9809565564007693},
            0.21539791166603497, 0.9809565564007693],
        [np.array([1, 2, 3, 4]), {'alpha': 0.7222742800877074},
            0.21576131952657338, 0.7222742800877074],
        [np.array([1, 2, 3, 4]), {'alpha': 0.6995044937418831},
            0.21579460210585208, 0.6995044937418831],
        [np.array([1, 2, 3, 4]), {'alpha': 0.6879814411990146},
            0.21581150916013203, 0.6879814411990146],
        [np.array([1, 2, 3, 4]), {'alpha': 0.5558016213920624},
            0.21600778429729228, 0.5558016213920624],
        [np.array([1, 2, 3, 4]), {'alpha': 0.48612258246951734},
            0.21611205459571634, 0.48612258246951734],
        [np.array([1, 2, 3, 4]), {'alpha': 0.42887539552321635},
            0.2161973389956996, 0.42887539552321635],
        [np.array([1, 2, 3, 4]), {'alpha': 0.398196343012209},
            0.21624265320052985, 0.398196343012209],
        [np.array([1, 2, 3, 4]), {'alpha': 0.29327794160087567},
            0.2163933942116072, 0.29327794160087567],
        [np.array([1, 2, 3, 4]), {'alpha': 0.2345829390285611},
            0.21647289061896782, 0.2345829390285611]], dtype=object),
        columns=['lags', 'params', 'mean_absolute_error', 'alpha'],
        index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64')
    ).astype({
        'mean_absolute_error': float,
        'alpha': float
    })
    
    pd.testing.assert_frame_equal(results, expected_results)


def test_evaluate_bayesian_search_optuna_when_return_best_ForecasterRecursive():
    """
    Test forecaster is refitted when return_best=True in _bayesian_search_optuna
    with a ForecasterRecursive.
    """
    forecaster = ForecasterRecursive(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    
    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1.0),
            'lags': trial.suggest_categorical('lags', [2, 4])
        }
        
        return search_space

    _bayesian_search_optuna(
        forecaster         = forecaster,
        y                  = y,
        cv                 = cv,
        search_space       = search_space,
        metric             = 'mean_absolute_error',
        n_trials           = 10,
        return_best        = True,
        verbose            = False
    )
    
    expected_lags = np.array([1, 2])
    expected_alpha = 0.5558016213920624

    np.testing.assert_array_equal(forecaster.lags, expected_lags)
    assert expected_alpha == forecaster.regressor.alpha


def test_results_opt_best_output_bayesian_search_optuna_with_output_study_best_trial_optuna():
    """
    Test results_opt_best output of _bayesian_search_optuna with output 
    study.best_trial optuna.
    """
    forecaster = ForecasterRecursive(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )

    n_validation = 12
    y_train = y[:-n_validation]
    metric = 'mean_absolute_error'
    verbose = False
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    n_trials = 10
    random_state = 123

    def objective(
        trial,
        forecaster         = forecaster,
        y                  = y,
        cv                 = cv,
        metric             = metric,
        verbose            = verbose,
    ) -> float:
        
        alpha = trial.suggest_float('alpha', 1e-2, 1.0)
        lags  = trial.suggest_categorical('lags', [4, 2])
        
        forecaster = ForecasterRecursive(
                        regressor = Ridge(random_state=random_state, 
                                          alpha=alpha),
                        lags      = lags
                     )

        metric, _ = backtesting_forecaster(
                        forecaster         = forecaster,
                        y                  = y,
                        cv                 = cv,
                        metric             = metric,
                        verbose            = verbose       
                    )
        metric = metric.iat[0, 0]
        return metric
  
    study = optuna.create_study(direction="minimize", 
                                sampler=TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1.0),
            'lags': trial.suggest_categorical('lags', [4, 2])
        }
        return search_space
    return_best  = False

    results_opt_best = _bayesian_search_optuna(
                           forecaster         = forecaster,
                           y                  = y,
                           cv                 = cv,
                           search_space       = search_space,
                           metric             = metric,
                           n_trials           = n_trials,
                           return_best        = return_best,
                           verbose            = verbose
                       )[1]

    assert best_trial.number == results_opt_best.number
    assert best_trial.values == results_opt_best.values
    assert best_trial.params == results_opt_best.params


def test_results_output_bayesian_search_optuna_ForecasterDirect():
    """
    Test output of _bayesian_search_optuna in ForecasterDirect with mocked
    (mocked done in Skforecast v0.4.3).
    """    
    forecaster = ForecasterDirect(
                     regressor = RandomForestRegressor(random_state=123),
                     steps     = 3,
                     lags      = 4
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    
    def search_space(trial):
        search_space  = {
            'n_estimators'    : trial.suggest_int('n_estimators', 10, 20),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 1., log=True),
            'max_features'    : trial.suggest_categorical('max_features', ['log2', 'sqrt']),
            'lags'            : trial.suggest_categorical('lags', [2, 4])
        } 
        
        return search_space

    results = _bayesian_search_optuna(
                  forecaster   = forecaster,
                  y            = y,
                  cv           = cv,
                  search_space = search_space,
                  metric       = 'mean_absolute_error',
                  n_trials     = 10,
                  random_state = 123,
                  return_best  = False,
                  verbose      = False
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([[np.array([1, 2, 3, 4]),
            {'n_estimators': 17, 'min_samples_leaf': 0.21035794225904136, 'max_features': 'log2'},
            0.20462849703549102, 17, 0.21035794225904136, 'log2'],
        [np.array([1, 2, 3, 4]),
            {'n_estimators': 13, 'min_samples_leaf': 0.2599119286878713, 'max_features': 'log2'},
            0.20911589421562574, 13, 0.2599119286878713, 'log2'],
        [np.array([1, 2, 3, 4]),
            {'n_estimators': 20, 'min_samples_leaf': 0.4839825891759374, 'max_features': 'log2'},
            0.2130714159765625, 20, 0.4839825891759374, 'log2'],
        [np.array([1, 2]),
            {'n_estimators': 15, 'min_samples_leaf': 0.34027307604369605, 'max_features': 'sqrt'},
            0.2133466144141166, 15, 0.34027307604369605, 'sqrt'],
        [np.array([1, 2]),
            {'n_estimators': 15, 'min_samples_leaf': 0.41010449151752726, 'max_features': 'sqrt'},
            0.21383764123529414, 15, 0.41010449151752726, 'sqrt'],
        [np.array([1, 2]),
            {'n_estimators': 14, 'min_samples_leaf': 0.782328520465639, 'max_features': 'log2'},
            0.21488066970063024, 14, 0.782328520465639, 'log2'],
        [np.array([1, 2]),
            {'n_estimators': 11, 'min_samples_leaf': 0.2714570796881701, 'max_features': 'sqrt'},
            0.21935169972870014, 11, 0.2714570796881701, 'sqrt'],
        [np.array([1, 2]),
            {'n_estimators': 13, 'min_samples_leaf': 0.20142843988664705, 'max_features': 'sqrt'},
            0.22713854310135304, 13, 0.20142843988664705, 'sqrt'],
        [np.array([1, 2]),
            {'n_estimators': 14, 'min_samples_leaf': 0.1147302385573586, 'max_features': 'sqrt'},
            0.22734290011048722, 14, 0.1147302385573586, 'sqrt'],
        [np.array([1, 2]),
            {'n_estimators': 17, 'min_samples_leaf': 0.19325882509735576, 'max_features': 'sqrt'},
            0.2279790315504743, 17, 0.19325882509735576, 'sqrt']], dtype=object),
        columns=['lags', 'params', 'mean_absolute_error', 'n_estimators', 'min_samples_leaf', 'max_features'],
        index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64')
    ).astype({
        'mean_absolute_error': float, 
        'n_estimators': int, 
        'min_samples_leaf': float
    })

    results = results.astype({
        'mean_absolute_error': float, 
        'n_estimators': int, 
        'min_samples_leaf': float
    })

    pd.testing.assert_frame_equal(results, expected_results)


def test_results_output_bayesian_search_optuna_window_features_ForecasterDirect():
    """
    Test output of _bayesian_search_optuna in ForecasterDirect including
    window_features with mocked (mocked done in Skforecast v0.4.3).
    """
    window_features = RollingFeatures(
        stats = ['mean', 'std', 'min', 'max', 'sum', 'median', 'ratio_min_max', 'coef_variation'],
        window_sizes = 3,
    )
    forecaster = ForecasterDirect(
                     regressor = RandomForestRegressor(random_state=123),
                     steps     = 3,
                     lags      = 4,
                     window_features = window_features,
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    
    def search_space(trial):
        search_space  = {
            'n_estimators'    : trial.suggest_int('n_estimators', 10, 20),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 1., log=True),
            'max_features'    : trial.suggest_categorical('max_features', ['log2', 'sqrt']),
            'lags'            : trial.suggest_categorical('lags', [2, 4])
        } 
        
        return search_space

    results = _bayesian_search_optuna(
                  forecaster   = forecaster,
                  y            = y,
                  cv           = cv,
                  search_space = search_space,
                  metric       = 'mean_absolute_error',
                  n_trials     = 10,
                  random_state = 123,
                  return_best  = False,
                  verbose      = False
              )[0]
    
    expected_results = pd.DataFrame(
        {
            'lags': [
                np.array([1, 2, 3, 4]),
                np.array([1, 2]),
                np.array([1, 2]),
                np.array([1, 2, 3, 4]),
                np.array([1, 2]),
                np.array([1, 2, 3, 4]),
                np.array([1, 2]),
                np.array([1, 2]),
                np.array([1, 2]),
                np.array([1, 2]),
            ],
            'params': [
                {
                    'n_estimators': 20,
                    'min_samples_leaf': 0.4839825891759374,
                    'max_features': 'log2',
                },
                {
                    'n_estimators': 15,
                    'min_samples_leaf': 0.41010449151752726,
                    'max_features': 'sqrt',
                },
                {
                    'n_estimators': 14,
                    'min_samples_leaf': 0.782328520465639,
                    'max_features': 'log2',
                },
                {
                    'n_estimators': 17,
                    'min_samples_leaf': 0.21035794225904136,
                    'max_features': 'log2',
                },
                {
                    'n_estimators': 15,
                    'min_samples_leaf': 0.34027307604369605,
                    'max_features': 'sqrt',
                },
                {
                    'n_estimators': 13,
                    'min_samples_leaf': 0.2599119286878713,
                    'max_features': 'log2',
                },
                {
                    'n_estimators': 13,
                    'min_samples_leaf': 0.20142843988664705,
                    'max_features': 'sqrt',
                },
                {
                    'n_estimators': 14,
                    'min_samples_leaf': 0.1147302385573586,
                    'max_features': 'sqrt',
                },
                {
                    'n_estimators': 17,
                    'min_samples_leaf': 0.19325882509735576,
                    'max_features': 'sqrt',
                },
                {
                    'n_estimators': 11,
                    'min_samples_leaf': 0.2714570796881701,
                    'max_features': 'sqrt',
                },
            ],
            'mean_absolute_error': [
                0.2130714159765625,
                0.21370598876262625,
                0.21466582599386722,
                0.21724682621084832,
                0.21740193466522548,
                0.22083884690087485,
                0.22198405983119776,
                0.2246121914126593,
                0.22842351518266302,
                0.22906678546095408,
            ],
            'n_estimators': [20, 15, 14, 17, 15, 13, 13, 14, 17, 11],
            'min_samples_leaf': [
                0.4839825891759374,
                0.41010449151752726,
                0.782328520465639,
                0.21035794225904136,
                0.34027307604369605,
                0.2599119286878713,
                0.20142843988664705,
                0.1147302385573586,
                0.19325882509735576,
                0.2714570796881701,
            ],
            'max_features': [
                'log2',
                'sqrt',
                'log2',
                'log2',
                'sqrt',
                'log2',
                'sqrt',
                'sqrt',
                'sqrt',
                'sqrt',
            ],
        }
    )

    pd.testing.assert_frame_equal(results, expected_results)

    
def test_bayesian_search_optuna_output_file():
    """ 
    Test output file of _bayesian_search_optuna.
    """

    forecaster = ForecasterRecursive(
                     regressor = RandomForestRegressor(random_state=123),
                     lags      = 2 
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )

    def search_space(trial):
        search_space  = {
            'n_estimators'    : trial.suggest_int('n_estimators', 10, 20),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 1., log=True),
            'max_features'    : trial.suggest_categorical('max_features', ['log2', 'sqrt']),
            'lags'            : trial.suggest_categorical('lags', [2, 4])
        } 
        
        return search_space

    output_file = 'test_bayesian_search_optuna_output_file.txt'
    _ = _bayesian_search_optuna(
            forecaster   = forecaster,
            y            = y,
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


@pytest.mark.parametrize(
        "forecaster",
        [
            ForecasterRecursive(
                regressor=Ridge(random_state=678),
                lags=3,
                transformer_y=None,
                forecaster_id='Recursive_no_transformer'
            ),
            ForecasterDirect(
                regressor=Ridge(random_state=678),
                steps=1,
                lags=3,
                transformer_y=None,
                forecaster_id='Direct_no_transformer'
            ),
            ForecasterRecursive(
                regressor=Ridge(random_state=678),
                lags=3,
                transformer_y=StandardScaler(),
                transformer_exog=StandardScaler(),
                forecaster_id='Recursive_transformers'
            ),
            ForecasterDirect(
                regressor=Ridge(random_state=678),
                steps=1,
                lags=3,
                transformer_y=StandardScaler(),
                transformer_exog=StandardScaler(),
                forecaster_id='Direct_transformer'
            )
        ],
ids=lambda forecaster: f'forecaster: {forecaster.forecaster_id}')
def test_bayesian_search_optuna_outputs_backtesting_one_step_ahead(
    forecaster,
):
    """
    Test that the outputs of _bayesian_search_optuna are equivalent when
    using backtesting and one-step-ahead.
    """
    metrics = [
        "mean_absolute_error",
        "mean_squared_error",
        mean_absolute_percentage_error,
        mean_absolute_scaled_error,
        root_mean_squared_scaled_error,
    ]

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1.0),
            'lags': trial.suggest_categorical('lags', [2, 4])
        }
        
        return search_space
    
    cv_backtesnting = TimeSeriesFold(
            steps                 = 1,
            initial_train_size    = 100,
            window_size           = None,
            differentiation       = None,
            refit                 = False,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    cv_one_step_ahead = OneStepAheadFold(
            initial_train_size    = 100,
            return_all_indexes    = False,
        )
    
    results_backtesting = _bayesian_search_optuna(
        forecaster   = forecaster,
        y            = y_feature_selection,
        exog         = exog_feature_selection,
        cv           = cv_backtesnting,
        search_space = search_space,
        metric       = metrics,
        n_trials     = 10,
        random_state = 123,
        return_best  = False,
        verbose      = False
    )[0]

    warn_msg = re.escape(
        "One-step-ahead predictions are used for faster model comparison, but they "
        "may not fully represent multi-step prediction performance. It is recommended "
        "to backtest the final model for a more accurate multi-step performance "
        "estimate."
    )
    with pytest.warns(UserWarning, match = warn_msg):
        results_one_step_ahead = _bayesian_search_optuna(
            forecaster   = forecaster,
            y            = y_feature_selection,
            exog         = exog_feature_selection,
            cv           = cv_one_step_ahead,
            search_space = search_space,
            metric       = metrics,
            n_trials     = 10,
            random_state = 123,
            return_best  = False,
            verbose      = False
        )[0]

    pd.testing.assert_frame_equal(results_backtesting, results_one_step_ahead)
