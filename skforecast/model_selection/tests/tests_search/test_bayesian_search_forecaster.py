# Unit test bayesian_search_forecaster
# ==============================================================================
import re
import pytest
from unittest.mock import patch
from joblib import Parallel, cpu_count
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error, r2_score, zero_one_loss
)
from xgboost import XGBRegressor
from skforecast.exceptions import (
    LongTrainingWarning, MissingValuesWarning, OneStepAheadValidationWarning
)
from skforecast.metrics import mean_absolute_scaled_error, root_mean_squared_scaled_error
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveClassifier
from skforecast.direct import ForecasterDirect
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import bayesian_search_forecaster
from skforecast.model_selection._split import TimeSeriesFold, OneStepAheadFold
from skforecast.preprocessing import RollingFeatures
import warnings
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm
from functools import partialmethod

# Fixtures
from ..fixtures_model_selection import y, y_clf
from ..fixtures_model_selection import y_feature_selection
from ..fixtures_model_selection import exog_feature_selection

optuna.logging.set_verbosity(optuna.logging.WARNING)
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # hide progress bar


def test_TypeError_bayesian_search_forecaster_when_cv_not_valid():
    """
    Test TypeError is raised in bayesian_search_forecaster when cv is not
    a valid splitter.
    """
    class DummyCV:
        pass

    cv = DummyCV()
    forecaster = ForecasterRecursive(
                     estimator = Ridge(random_state=123),
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
        bayesian_search_forecaster(
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


def test_ValueError_bayesian_search_forecaster_metric_list_duplicate_names():
    """
    Test ValueError is raised in bayesian_search_forecaster when a `list` of 
    metrics is used with duplicate names.
    """
    forecaster = ForecasterRecursive(
                     estimator = Ridge(random_state=123),
                     lags      = 2
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
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
        bayesian_search_forecaster(
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


def test_ValueError_bayesian_search_forecaster_when_metric_named_n_evaluated():
    """
    Test ValueError is raised in bayesian_search_forecaster, before the first
    trial, when a metric is named `n_evaluated`, the column of the backtesting
    metrics with the number of points evaluated.
    """
    forecaster = ForecasterRecursive(estimator=Ridge(random_state=123), lags=2)
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y[:-12]), refit=False)

    def search_space(trial):  # pragma: no cover
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}

        return search_space

    def n_evaluated(y_true, y_pred):  # pragma: no cover
        return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

    err_msg = re.escape(
        "The backtesting metrics include the column 'n_evaluated' with the "
        "number of points evaluated. Rename the metric 'n_evaluated' passed in "
        "`metric`."
    )
    with pytest.raises(ValueError, match = err_msg):
        bayesian_search_forecaster(
            forecaster         = forecaster,
            y                  = y,
            cv                 = cv,
            search_space       = search_space,
            metric             = n_evaluated,
            n_trials           = 10,
            return_best        = False,
            verbose            = False,
        )


@pytest.mark.parametrize(
    'cv',
    [TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=True, fixed_train_size=True),
     OneStepAheadFold(initial_train_size=len(y) - 12)],
    ids=lambda cv: f'cv: {type(cv).__name__}'
)
def test_ValueError_bayesian_search_forecaster_when_search_space_names_do_not_match(cv):
    """
    Test ValueError is raised when search_space key name do not match the trial 
    object name from optuna.
    """
    forecaster = ForecasterRecursive(
                     estimator = Ridge(random_state=123),
                     lags      = 2
                 )

    def search_space(trial):
        search_space = {
            'alpha': trial.suggest_float('not_alpha', 1e-2, 1.0),
            'lags': trial.suggest_categorical('lags', [2, 4])
        }

        return search_space

    err_msg = re.escape(
        "`search_space` dict keys must match the names passed to "
        "`trial.suggest_*()`.\n"
        "  Dict keys    : ['alpha', 'lags']\n"
        "  Suggest names: ['not_alpha', 'lags']"
    )
    with pytest.raises(ValueError, match=err_msg):
        bayesian_search_forecaster(
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
def test_results_output_bayesian_search_forecaster_ForecasterRecursive():
    """
    Test output of bayesian_search_forecaster in ForecasterRecursive with mocked
    (mocked done in Skforecast v0.4.3).
    """
    forecaster = ForecasterRecursive(
                     estimator = RandomForestRegressor(random_state=123),
                     lags      = 2
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )

    def search_space(trial):
        search_space  = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 20),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 1., log=True),
            'max_features': trial.suggest_categorical('max_features', ['log2', 'sqrt']),
            'lags': trial.suggest_categorical('lags', [2, 4])
        } 
        
        return search_space

    results = bayesian_search_forecaster(
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
        np.array([
            [np.array([1, 2, 3, 4]),
                {'n_estimators': 20, 'min_samples_leaf': 0.4839825891759374, 'max_features': 'log2'},
                0.21252324019730398,
                20,
                0.4839825891759374,
                'log2'],
            [np.array([1, 2]),
                {'n_estimators': 15, 'min_samples_leaf': 0.34027307604369605, 'max_features': 'sqrt'},
                0.21479600790277778,
                15,
                0.34027307604369605,
                'sqrt'],
            [np.array([1, 2]),
                {'n_estimators': 15, 'min_samples_leaf': 0.41010449151752726, 'max_features': 'sqrt'},
                0.21479600790277778,
                15,
                0.41010449151752726,
                'sqrt'],
            [np.array([1, 2]),
                {'n_estimators': 14, 'min_samples_leaf': 0.782328520465639, 'max_features': 'log2'},
                0.21661388810185186,
                14,
                0.782328520465639,
                'log2'],
            [np.array([1, 2]),
                {'n_estimators': 13, 'min_samples_leaf': 0.20142843988664705, 'max_features': 'sqrt'},
                0.22084665733716444,
                13,
                0.20142843988664705,
                'sqrt'],
            [np.array([1, 2, 3, 4]),
                {'n_estimators': 17, 'min_samples_leaf': 0.21035794225904136, 'max_features': 'log2'},
                0.2229839747692736,
                17,
                0.21035794225904136,
                'log2'],
            [np.array([1, 2]),
                {'n_estimators': 17, 'min_samples_leaf': 0.19325882509735576, 'max_features': 'sqrt'},
                0.22358275919419623,
                17,
                0.19325882509735576,
                'sqrt'],
            [np.array([1, 2]),
                {'n_estimators': 11, 'min_samples_leaf': 0.2714570796881701, 'max_features': 'sqrt'},
                0.22384439522399655,
                11,
                0.2714570796881701,
                'sqrt'],
            [np.array([1, 2, 3, 4]),
                {'n_estimators': 13, 'min_samples_leaf': 0.2599119286878713, 'max_features': 'log2'},
                0.2252246193518175,
                13,
                0.2599119286878713,
                'log2'],
            [np.array([1, 2]),
                {'n_estimators': 14, 'min_samples_leaf': 0.1147302385573586, 'max_features': 'sqrt'},
                0.22764885273610677,
                14,
                0.1147302385573586,
                'sqrt']], dtype=object),
        columns=['lags', 'params', 'mean_absolute_error', 'n_estimators', 'min_samples_leaf', 'max_features'],
        index=pd.RangeIndex(start=0, stop=10, step=1)
    )

    expected_results.insert(3, 'n_evaluated', [12] * 10)
    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results, check_dtype=False)


def test_results_output_bayesian_search_forecaster_window_features_ForecasterRecursive():
    """
    Test output of bayesian_search_forecaster in ForecasterRecursive including
    window_features with mocked (mocked done in Skforecast v0.4.3).
    """
    window_features = RollingFeatures(
        stats = ['mean', 'std', 'min', 'max', 'sum', 'median', 'ratio_min_max', 'coef_variation'],
        window_sizes = 3,
    )
    forecaster = ForecasterRecursive(
                     estimator = RandomForestRegressor(random_state=123),
                     lags      = 2,
                     window_features = window_features,
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )

    def search_space(trial):
        search_space  = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 20),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 1., log=True),
            'max_features': trial.suggest_categorical('max_features', ['log2', 'sqrt']),
            'lags': trial.suggest_categorical('lags', [2, 4])
        } 
        
        return search_space

    results = bayesian_search_forecaster(
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
            {'n_estimators': 20, 'min_samples_leaf': 0.4839825891759374, 'max_features': 'log2'},
            {'n_estimators': 15, 'min_samples_leaf': 0.41010449151752726, 'max_features': 'sqrt'},
            {'n_estimators': 14, 'min_samples_leaf': 0.782328520465639, 'max_features': 'log2'},
            {'n_estimators': 15, 'min_samples_leaf': 0.34027307604369605, 'max_features': 'sqrt'},
            {'n_estimators': 13, 'min_samples_leaf': 0.2599119286878713, 'max_features': 'log2'},
            {'n_estimators': 17, 'min_samples_leaf': 0.21035794225904136, 'max_features': 'log2'},
            {'n_estimators': 11, 'min_samples_leaf': 0.2714570796881701, 'max_features': 'sqrt'},
            {'n_estimators': 13, 'min_samples_leaf': 0.20142843988664705, 'max_features': 'sqrt'},
            {'n_estimators': 17, 'min_samples_leaf': 0.19325882509735576, 'max_features': 'sqrt'},
            {'n_estimators': 14, 'min_samples_leaf': 0.1147302385573586, 'max_features': 'sqrt'},
        ],
        'mean_absolute_error': [
            0.21252324019730398,
            0.21851156682698417,
            0.22027505037414966,
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

    expected_results.insert(3, 'n_evaluated', [12] * 10)
    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results, check_dtype=False)
    

def test_results_output_bayesian_search_forecaster_ForecasterRecursive_with_kwargs_create_study():
    """
    Test output of bayesian_search_forecaster in ForecasterRecursive with 
    kwargs_create_study with mocked (mocked done in Skforecast v0.4.3).
    All trials are startup trials of the sampler (`n_startup_trials=10`), so the
    results only depend on the seed and not on the internals of TPE, which
    change between optuna versions.
    """
    forecaster = ForecasterRecursive(
                     estimator = Ridge(random_state=123),
                     lags      = 2
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
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
        'sampler': TPESampler(seed=123, n_startup_trials=10, n_ei_candidates=18)
    }
    results = bayesian_search_forecaster(
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
        np.array([
            [np.array([1, 2]),
                {'alpha': 0.23598059857016607},
                0.21239141697571848,
                0.23598059857016607],
            [np.array([1, 2]),
                {'alpha': 0.398196343012209},
                0.21271021033387602,
                0.398196343012209],
            [np.array([1, 2]),
                {'alpha': 0.4441865222328282},
                0.2127897499229874,
                0.4441865222328282],
            [np.array([1, 2]),
                {'alpha': 0.53623586010342},
                0.21293692257888708,
                0.53623586010342],
            [np.array([1, 2]),
                {'alpha': 0.7252189487445193},
                0.21319693043832985,
                0.7252189487445193],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.9809565564007693},
                0.21539791166603497,
                0.9809565564007693],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.8509374761370117},
                0.21557690844753197,
                0.8509374761370117],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.7406154516747153},
                0.2157346392837304,
                0.7406154516747153],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.6995044937418831},
                0.2157946021058521,
                0.6995044937418831],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.5558016213920624},
                0.21600778429729228,
                0.5558016213920624]], dtype=object),
        columns=['lags', 'params', 'mean_absolute_error', 'alpha'],
        index=pd.RangeIndex(start=0, stop=10, step=1)
    ).astype({'mean_absolute_error': float, 'alpha': float})

    expected_results.insert(3, 'n_evaluated', [12] * 10)
    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results)


def test_results_output_bayesian_search_forecaster_ForecasterRecursive_with_kwargs_study_optimize():
    """
    Test output of bayesian_search_forecaster in ForecasterRecursive when 
    kwargs_study_optimize with mocked (mocked done in Skforecast v0.4.3).
    """
    forecaster = ForecasterRecursive(
                     estimator = RandomForestRegressor(random_state=123),
                     lags      = 2 
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
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
            'max_depth': trial.suggest_int('max_depth', 2, 10, log=True),
            'max_features': trial.suggest_categorical('max_features', ['log2', 'sqrt']),
            'lags': trial.suggest_categorical('lags', [2, 4])
        } 
        
        return search_space

    kwargs_study_optimize = {'timeout': 10}
    results = bayesian_search_forecaster(
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
        np.array([
            [np.array([1, 2, 3, 4]),
                {'n_estimators': 8, 'max_depth': 3, 'max_features': 'log2'},
                0.2176619102322017,
                8,
                3,
                'log2'],
            [np.array([1, 2]),
                {'n_estimators': 8, 'max_depth': 3, 'max_features': 'sqrt'},
                0.21923614756760298,
                8,
                3,
                'sqrt'],
            [np.array([1, 2]),
                {'n_estimators': 5, 'max_depth': 2, 'max_features': 'sqrt'},
                0.22116013675443522,
                5,
                2,
                'sqrt'],
            [np.array([1, 2, 3, 4]),
                {'n_estimators': 10, 'max_depth': 6, 'max_features': 'log2'},
                0.2222148767956379,
                10,
                6,
                'log2'],
            [np.array([1, 2]),
                {'n_estimators': 6, 'max_depth': 4, 'max_features': 'sqrt'},
                0.22883925084220677,
                6,
                4,
                'sqrt']], dtype=object),
        columns=['lags', 'params', 'mean_absolute_error', 'n_estimators', 'max_depth', 'max_features'],
        index=pd.RangeIndex(start=0, stop=5, step=1)
    ).astype({'mean_absolute_error': float, 'n_estimators': int, 'max_depth': int})

    expected_results.insert(3, 'n_evaluated', [12] * 5)
    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results, check_dtype=False)


def test_results_output_bayesian_search_forecaster_ForecasterRecursive_when_lags_not_in_search_space():
    """
    Test output of bayesian_search_forecaster in ForecasterRecursive when lag is not 
    in search_space with mocked (mocked done in Skforecast v0.4.3).
    """
    forecaster = ForecasterRecursive(
                     estimator = Ridge(random_state=123),
                     lags      = 4
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
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

    results = bayesian_search_forecaster(
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
        np.array([
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.9809565564007693},
                0.21539791166603497,
                0.9809565564007693],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.7222742800877074},
                0.21576131952657338,
                0.7222742800877074],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.6995044937418831},
                0.21579460210585208,
                0.6995044937418831],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.6879814411990146},
                0.21581150916013206,
                0.6879814411990146],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.5558016213920624},
                0.21600778429729228,
                0.5558016213920624],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.48612258246951734},
                0.21611205459571634,
                0.48612258246951734],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.42887539552321635},
                0.2161973389956996,
                0.42887539552321635],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.398196343012209},
                0.2162426532005299,
                0.398196343012209],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.29327794160087567},
                0.2163933942116072,
                0.29327794160087567],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.2345829390285611},
                0.21647289061896782,
                0.2345829390285611]], dtype=object),
        columns=['lags', 'params', 'mean_absolute_error', 'alpha'],
        index=pd.RangeIndex(start=0, stop=10, step=1)
    ).astype({'mean_absolute_error': float, 'alpha': float})
    
    expected_results.insert(3, 'n_evaluated', [12] * 10)
    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results)


def test_evaluate_bayesian_search_forecaster_when_return_best_ForecasterRecursive():
    """
    Test forecaster is refitted when return_best=True in bayesian_search_forecaster
    with a ForecasterRecursive.
    """
    forecaster = ForecasterRecursive(
                     estimator = Ridge(random_state=123),
                     lags      = 2
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
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

    bayesian_search_forecaster(
        forecaster         = forecaster,
        y                  = y,
        cv                 = cv,
        search_space       = search_space,
        metric             = 'mean_absolute_error',
        n_trials           = 10,
        return_best        = True,
        verbose            = True
    )
    
    expected_lags = np.array([1, 2])
    expected_alpha = 0.5558016213920624

    np.testing.assert_array_almost_equal(forecaster.lags, expected_lags)
    assert expected_alpha == forecaster.estimator.alpha


def test_results_opt_best_output_bayesian_search_forecaster_with_output_study_best_trial_optuna():
    """
    Test results_opt_best output of bayesian_search_forecaster with output 
    study.best_trial optuna.
    """
    forecaster = ForecasterRecursive(
                     estimator = Ridge(random_state=123),
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
                        estimator = Ridge(random_state=random_state, 
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

    results_opt_best = bayesian_search_forecaster(
                           forecaster         = forecaster,
                           y                  = y,
                           cv                 = cv,
                           search_space       = search_space,
                           metric             = metric,
                           n_trials           = n_trials,
                           return_best        = return_best,
                           verbose            = verbose
                       )[1]

    assert best_trial.number == results_opt_best.best_trial.number
    assert best_trial.values == results_opt_best.best_trial.values
    assert best_trial.params == results_opt_best.best_trial.params


def test_results_output_bayesian_search_forecaster_ForecasterDirect():
    """
    Test output of bayesian_search_forecaster in ForecasterDirect with mocked
    (mocked done in Skforecast v0.4.3).
    """    
    forecaster = ForecasterDirect(
                     estimator = RandomForestRegressor(random_state=123),
                     steps     = 3,
                     lags      = 4
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    
    def search_space(trial):
        search_space  = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 20),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 1., log=True),
            'max_features': trial.suggest_categorical('max_features', ['log2', 'sqrt']),
            'lags': trial.suggest_categorical('lags', [2, 4])
        } 
        
        return search_space

    results = bayesian_search_forecaster(
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
        np.array([
            [np.array([1, 2, 3, 4]),
                {'n_estimators': 17, 'min_samples_leaf': 0.21035794225904136, 'max_features': 'log2'},
                0.20462849703549102,
                17,
                0.21035794225904136,
                'log2'],
            [np.array([1, 2, 3, 4]),
                {'n_estimators': 13, 'min_samples_leaf': 0.2599119286878713, 'max_features': 'log2'},
                0.20911589421562574,
                13,
                0.2599119286878713,
                'log2'],
            [np.array([1, 2, 3, 4]),
                {'n_estimators': 20, 'min_samples_leaf': 0.4839825891759374, 'max_features': 'log2'},
                0.2130714159765625,
                20,
                0.4839825891759374,
                'log2'],
            [np.array([1, 2]),
                {'n_estimators': 15, 'min_samples_leaf': 0.34027307604369605, 'max_features': 'sqrt'},
                0.21334661441411654,
                15,
                0.34027307604369605,
                'sqrt'],
            [np.array([1, 2]),
                {'n_estimators': 15, 'min_samples_leaf': 0.41010449151752726, 'max_features': 'sqrt'},
                0.21383764123529414,
                15,
                0.41010449151752726,
                'sqrt'],
            [np.array([1, 2]),
                {'n_estimators': 14, 'min_samples_leaf': 0.782328520465639, 'max_features': 'log2'},
                0.21488066970063024,
                14,
                0.782328520465639,
                'log2'],
            [np.array([1, 2]),
                {'n_estimators': 11, 'min_samples_leaf': 0.2714570796881701, 'max_features': 'sqrt'},
                0.21935169972870014,
                11,
                0.2714570796881701,
                'sqrt'],
            [np.array([1, 2]),
                {'n_estimators': 13, 'min_samples_leaf': 0.20142843988664705, 'max_features': 'sqrt'},
                0.22713854310135292,
                13,
                0.20142843988664705,
                'sqrt'],
            [np.array([1, 2]),
                {'n_estimators': 14, 'min_samples_leaf': 0.1147302385573586, 'max_features': 'sqrt'},
                0.22734290011048722,
                14,
                0.1147302385573586,
                'sqrt'],
            [np.array([1, 2]),
                {'n_estimators': 17, 'min_samples_leaf': 0.19325882509735576, 'max_features': 'sqrt'},
                0.22797903155047428,
                17,
                0.19325882509735576,
                'sqrt']], dtype=object),
        columns=['lags', 'params', 'mean_absolute_error', 'n_estimators', 'min_samples_leaf', 'max_features'],
        index=pd.RangeIndex(start=0, stop=10, step=1)
    ).astype({'mean_absolute_error': float, 'n_estimators': int, 'min_samples_leaf': float})

    results = results.astype({
        'mean_absolute_error': float, 
        'n_estimators': int, 
        'min_samples_leaf': float
    })

    expected_results.insert(3, 'n_evaluated', [12] * 10)
    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results)


def test_results_output_bayesian_search_forecaster_window_features_ForecasterDirect():
    """
    Test output of bayesian_search_forecaster in ForecasterDirect including
    window_features with mocked (mocked done in Skforecast v0.4.3).
    """
    window_features = RollingFeatures(
        stats = ['mean', 'std', 'min', 'max', 'sum', 'median', 'ratio_min_max', 'coef_variation'],
        window_sizes = 3,
    )
    forecaster = ForecasterDirect(
                     estimator = RandomForestRegressor(random_state=123),
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
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    
    def search_space(trial):
        search_space  = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 20),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 1., log=True),
            'max_features': trial.suggest_categorical('max_features', ['log2', 'sqrt']),
            'lags': trial.suggest_categorical('lags', [2, 4])
        } 
        
        return search_space

    results = bayesian_search_forecaster(
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
    
    expected_results = pd.DataFrame({
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
            {'n_estimators': 20, 'min_samples_leaf': 0.4839825891759374, 'max_features': 'log2'},
            {'n_estimators': 15, 'min_samples_leaf': 0.41010449151752726, 'max_features': 'sqrt'},
            {'n_estimators': 14, 'min_samples_leaf': 0.782328520465639, 'max_features': 'log2'},
            {'n_estimators': 17, 'min_samples_leaf': 0.21035794225904136, 'max_features': 'log2'},
            {'n_estimators': 15, 'min_samples_leaf': 0.34027307604369605, 'max_features': 'sqrt'},
            {'n_estimators': 13, 'min_samples_leaf': 0.2599119286878713, 'max_features': 'log2'},
            {'n_estimators': 13, 'min_samples_leaf': 0.20142843988664705, 'max_features': 'sqrt'},
            {'n_estimators': 14, 'min_samples_leaf': 0.1147302385573586, 'max_features': 'sqrt'},
            {'n_estimators': 17, 'min_samples_leaf': 0.19325882509735576, 'max_features': 'sqrt'},
            {'n_estimators': 11, 'min_samples_leaf': 0.2714570796881701, 'max_features': 'sqrt'},
        ],
        'mean_absolute_error': [
            0.2130714159765625,
            0.21370598876262625,
            0.21466582599386722,
            0.21724682621084823,
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
    })

    expected_results.insert(3, 'n_evaluated', [12] * 10)
    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results)

    
def test_bayesian_search_forecaster_output_file():
    """ 
    Test output file of bayesian_search_forecaster.
    """

    forecaster = ForecasterRecursive(
                     estimator = RandomForestRegressor(random_state=123),
                     lags      = 2 
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )

    def search_space(trial):
        search_space  = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 20),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 1., log=True),
            'max_features': trial.suggest_categorical('max_features', ['log2', 'sqrt']),
            'lags': trial.suggest_categorical('lags', [2, 4])
        } 
        
        return search_space

    output_file = 'test_bayesian_search_forecaster_output_file.txt'
    _ = bayesian_search_forecaster(
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


@pytest.mark.parametrize("initial_train_size", 
                         [450, '2020-01-19 17:00:00', pd.to_datetime('2020-01-19 17:00:00')], 
                         ids=lambda initial_train_size: f'initial_train_size: {initial_train_size}')
@pytest.mark.parametrize(
        "forecaster",
        [
            ForecasterRecursive(
                estimator=Ridge(random_state=678),
                lags=3,
                transformer_y=None,
                forecaster_id='Recursive_no_transformer'
            ),
            ForecasterDirect(
                estimator=Ridge(random_state=678),
                steps=1,
                lags=3,
                transformer_y=None,
                forecaster_id='Direct_no_transformer'
            ),
            ForecasterRecursive(
                estimator=Ridge(random_state=678),
                lags=3,
                transformer_y=StandardScaler(),
                transformer_exog=StandardScaler(),
                forecaster_id='Recursive_transformers'
            ),
            ForecasterDirect(
                estimator=Ridge(random_state=678),
                steps=1,
                lags=3,
                transformer_y=StandardScaler(),
                transformer_exog=StandardScaler(),
                forecaster_id='Direct_transformer'
            )
        ],
ids=lambda forecaster: f'forecaster: {forecaster.forecaster_id}')
def test_bayesian_search_forecaster_outputs_backtesting_one_step_ahead(
    forecaster, initial_train_size
):
    """
    Test that the outputs of bayesian_search_forecaster are equivalent when
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
            initial_train_size    = initial_train_size,
            window_size           = None,
            refit                 = False,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    cv_one_step_ahead = OneStepAheadFold(
        initial_train_size    = initial_train_size,
        return_all_indexes    = False,
    )
    
    results_backtesting = bayesian_search_forecaster(
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
    with pytest.warns(OneStepAheadValidationWarning, match = warn_msg):
        results_one_step_ahead = bayesian_search_forecaster(
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

    pd.testing.assert_frame_equal(results_backtesting.drop(columns=["trial_number"]), results_one_step_ahead.drop(columns=["trial_number"]))


def test_ValueError_bayesian_search_forecaster_when_return_best_and_len_y_exog_different():
    """
    Test ValueError is raised in bayesian_search_forecaster when return_best 
    and length of `y` and `exog` do not match.
    """
    forecaster = ForecasterRecursive(
                    estimator = Ridge(random_state=123),
                    lags      = 2
                 )
    exog = y[:30]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y[:-12]),
            window_size           = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    
    def search_space(trial):  # pragma: no cover
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}
        return search_space

    err_msg = re.escape(
        f"`exog` must have same number of samples as `y`. "
        f"length `exog`: ({len(exog)}), length `y`: ({len(y)})"
    )
    with pytest.raises(ValueError, match = err_msg):
        bayesian_search_forecaster(
            forecaster         = forecaster,
            y                  = y,
            exog               = exog,
            cv                 = cv,
            search_space       = search_space,
            metric             = 'mean_absolute_error',
            n_trials           = 10,
            random_state       = 123,
            return_best        = True,
            verbose            = False,
        )


def test_results_output_bayesian_search_forecaster_optuna_ForecasterRecursive_window_features_with_mocked():
    """
    Test output of bayesian_search_forecaster in ForecasterRecursive with window features 
    using mocked using optuna (mocked done in Skforecast v0.4.3).
    """
    window_features = RollingFeatures(
        stats        = ['mean', 'std', 'min', 'max', 'sum', 'median', 'ratio_min_max', 'coef_variation'],
        window_sizes = 3,
    )
    forecaster = ForecasterRecursive(
                     estimator       = Ridge(random_state=123),
                     lags            = 4,
                     window_features = window_features,
                 )

    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y[:-12]),
            window_size           = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    
    def search_space(trial):  # pragma: no cover
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}
        return search_space

    results = bayesian_search_forecaster(
                  forecaster         = forecaster,
                  y                  = y,
                  cv                 = cv,
                  search_space       = search_space,
                  metric             = 'mean_absolute_error',
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False,
              )[0]
    
    expected_results = pd.DataFrame({
        'lags': [
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 3, 4]),
        ],
        'params': [
            {'alpha': 0.9809565564007693},
            {'alpha': 0.7222742800877074},
            {'alpha': 0.6995044937418831},
            {'alpha': 0.6879814411990146},
            {'alpha': 0.5558016213920624},
            {'alpha': 0.48612258246951734},
            {'alpha': 0.42887539552321635},
            {'alpha': 0.398196343012209},
            {'alpha': 0.29327794160087567},
            {'alpha': 0.2345829390285611},
        ],
        'mean_absolute_error': [
            0.23783372219201282,
            0.24038797813509474,
            0.2406573143006138,
            0.2407972445231639,
            0.24261647067658373,
            0.24378706429210695,
            0.24490876891227567,
            0.24558757754181923,
            0.24851969600385518,
            0.2508328027649238,
        ],
        'alpha': [
            0.9809565564007693,
            0.7222742800877074,
            0.6995044937418831,
            0.6879814411990146,
            0.5558016213920624,
            0.48612258246951734,
            0.42887539552321635,
            0.398196343012209,
            0.29327794160087567,
            0.2345829390285611,
        ],
    })

    expected_results.insert(3, 'n_evaluated', [12] * 10)
    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results)


def test_results_output_bayesian_search_forecaster_optuna_ForecasterRecursive_with_mocked():
    """
    Test output of bayesian_search_forecaster in ForecasterRecursive with 
    mocked using optuna (mocked done in Skforecast v0.4.3).
    """    
    forecaster = ForecasterRecursive(
                     estimator = Ridge(random_state=123),
                     lags      = 4
                 )

    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y[:-12]),
            window_size           = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    
    def search_space(trial):  # pragma: no cover
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}
        return search_space

    results = bayesian_search_forecaster(
                  forecaster         = forecaster,
                  y                  = y,
                  cv                 = cv,
                  search_space       = search_space,
                  metric             = 'mean_absolute_error',
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False,
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.9809565564007693},
                0.21539791166603497,
                0.9809565564007693],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.7222742800877074},
                0.21576131952657338,
                0.7222742800877074],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.6995044937418831},
                0.21579460210585208,
                0.6995044937418831],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.6879814411990146},
                0.21581150916013206,
                0.6879814411990146],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.5558016213920624},
                0.21600778429729228,
                0.5558016213920624],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.48612258246951734},
                0.21611205459571634,
                0.48612258246951734],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.42887539552321635},
                0.2161973389956996,
                0.42887539552321635],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.398196343012209},
                0.2162426532005299,
                0.398196343012209],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.29327794160087567},
                0.2163933942116072,
                0.29327794160087567],
            [np.array([1, 2, 3, 4]),
                {'alpha': 0.2345829390285611},
                0.21647289061896782,
                0.2345829390285611]], dtype=object),
        columns=['lags', 'params', 'mean_absolute_error', 'alpha'],
        index=pd.RangeIndex(start=0, stop=10, step=1)
    ).astype({'mean_absolute_error': float, 'alpha': float})

    expected_results.insert(3, 'n_evaluated', [12] * 10)
    pd.testing.assert_frame_equal(results.drop(columns=["trial_number"]), expected_results)


def test_bayesian_search_forecaster_xgboost_categorical_no_ValueError_on_cache_hit():
    """
    Test that bayesian_search_forecaster does not raise a ValueError when
    XGBRegressor with categorical features is used and the search space includes
    multiple lag combinations with OneStepAheadFold (cache hit scenario).

    Root cause: `configure_estimator_categorical_features` sets `feature_types` on
    the estimator via `set_params` (persistent mutation). When a cached split is
    reused the estimator's `feature_types` must be restored to match the cached
    X_train column count, otherwise XGBoost raises:
      ValueError: feature types must have the same length as the number of
                  data columns, expected <N>, got <M>
    """
    # np.random.seed(123); y = np.random.rand(50)
    y_local = pd.Series(
        np.array([0.69646919, 0.28613933, 0.22685145, 0.55131477, 0.71946897,
                  0.42310646, 0.9807642 , 0.68482974, 0.4809319 , 0.39211752,
                  0.34317802, 0.72904971, 0.43857224, 0.0596779 , 0.39804426,
                  0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759,
                  0.63440096, 0.84943179, 0.72445532, 0.61102351, 0.72244338,
                  0.32295891, 0.36178866, 0.22826323, 0.29371405, 0.63097612,
                  0.09210494, 0.43370117, 0.43086276, 0.4936851 , 0.42583029,
                  0.31226122, 0.42635131, 0.89338916, 0.94416002, 0.50183668,
                  0.62395295, 0.1156184 , 0.31728548, 0.41482621, 0.86630916,
                  0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453])
    )
    # Hardcoded categorical exog with three levels ('f', 'g', 'h')
    exog_local = pd.DataFrame({
        'cat_feat': pd.Categorical(
            ['f', 'f', 'g', 'h', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'h', 'g', 'g',
             'g', 'f', 'f', 'g', 'g', 'g', 'h', 'f', 'g', 'g', 'f', 'h', 'h', 'h',
             'g', 'f', 'f', 'h', 'g', 'h', 'g', 'h', 'g', 'h', 'g', 'h', 'g', 'h',
             'f', 'g', 'h', 'h', 'g', 'f', 'f', 'g']
        )
    })

    forecaster = ForecasterRecursive(
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
        results, _ = bayesian_search_forecaster(
            forecaster   = forecaster,
            y            = y_local,
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

    expected_results = pd.DataFrame(
        data=np.array(
            [[np.array([1, 2, 3, 4, 5]), {'n_estimators': 7}, 0.267985, 7],
             [np.array([1, 2, 3]),        {'n_estimators': 6}, 0.275470, 6],
             [np.array([1, 2, 3]),        {'n_estimators': 6}, 0.275470, 6],
             [np.array([1, 2, 3]),        {'n_estimators': 7}, 0.281943, 7],
             [np.array([1, 2, 3]),        {'n_estimators': 7}, 0.281943, 7],
             [np.array([1, 2, 3]),        {'n_estimators': 9}, 0.285263, 9]],
            dtype=object
        ),
        columns=['lags', 'params', 'mean_absolute_error', 'n_estimators'],
        index=pd.RangeIndex(start=0, stop=6, step=1)
    ).astype({'mean_absolute_error': float, 'n_estimators': int})

    expected_results.insert(3, 'n_evaluated', [15] * 6)
    pd.testing.assert_frame_equal(
        results.drop(columns=['trial_number']), expected_results, atol=1e-4
    )


@pytest.mark.parametrize(
    "cv",
    [
        TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=False),
        OneStepAheadFold(initial_train_size=len(y) - 12),
    ],
    ids=lambda cv: f"cv: {type(cv).__name__}",
)
@pytest.mark.parametrize(
    "kwargs_create_study, expected_direction",
    [
        (None, "maximize"),
        ({"direction": "maximize"}, "maximize"),
        ({"direction": "minimize"}, "minimize"),
    ],
    ids=lambda dt: f"{dt}",
)
def test_bayesian_search_forecaster_direction_when_metric_greater_is_better(
    cv, kwargs_create_study, expected_direction
):
    """
    Test that, with a metric in which greater values are better (`r2_score`),
    the study is maximized by default, and that the results and `return_best`
    follow the direction of the study, also when it is set in
    `kwargs_create_study`. A direction that contradicts the metric raises a
    warning, which points to the code of the user (here, the test). Before the
    fix, the default direction was 'minimize' and the results were always
    sorted in ascending order.
    """
    forecaster = ForecasterRecursive(estimator=Ridge(random_state=123), lags=2)

    def search_space(trial):
        return {
            "alpha": trial.suggest_float("alpha", 1e-2, 1e2, log=True),
            "lags": trial.suggest_categorical("lags", [2, 4]),
        }

    kwargs = dict(
        forecaster=forecaster,
        y=y,
        cv=cv,
        search_space=search_space,
        metric=r2_score,
        n_trials=10,
        random_state=123,
        return_best=True,
        kwargs_create_study=kwargs_create_study,
    )
    warn_msg = "The optuna study direction is 'minimize'"
    if expected_direction == "minimize":
        with pytest.warns(UserWarning, match=warn_msg) as record:
            results, study = bayesian_search_forecaster(**kwargs)
        record = [w for w in record if warn_msg in str(w.message)]
        assert record[0].filename == __file__
    else:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            results, study = bayesian_search_forecaster(**kwargs)
        assert not any(warn_msg in str(w.message) for w in record)

    assert study.direction.name.lower() == expected_direction
    assert results["r2_score"].nunique() > 1
    if expected_direction == "maximize":
        assert results["r2_score"].is_monotonic_decreasing
    else:
        assert results["r2_score"].is_monotonic_increasing
    assert results.loc[0, "trial_number"] == study.best_trial.number
    assert results.loc[0, "r2_score"] == study.best_value
    assert forecaster.estimator.alpha == study.best_params["alpha"]
    np.testing.assert_array_equal(forecaster.lags, results.loc[0, "lags"])


def test_bayesian_search_forecaster_ForecasterRecursiveClassifier_minimizes_loss():
    """
    Test that a classification forecaster minimizes a loss (`zero_one_loss`)
    by default and that `return_best` refits it with the best trial. Before
    the fix, every metric was maximized in classification tasks.
    """
    forecaster = ForecasterRecursiveClassifier(
        estimator=LogisticRegression(max_iter=1000), lags=2
    )
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y_clf) - 12, refit=False)

    def search_space(trial):
        return {
            "C": trial.suggest_float("C", 1e-4, 10, log=True),
            "lags": trial.suggest_categorical("lags", [2, 4]),
        }

    results, study = bayesian_search_forecaster(
        forecaster=forecaster,
        y=y_clf,
        cv=cv,
        search_space=search_space,
        metric=zero_one_loss,
        n_trials=10,
        random_state=123,
        return_best=True,
    )

    assert study.direction == optuna.study.StudyDirection.MINIMIZE
    assert results["zero_one_loss"].is_monotonic_increasing
    assert results.loc[0, "zero_one_loss"] == study.best_value
    assert forecaster.estimator.C == results.loc[0, "C"]


def test_bayesian_search_forecaster_LongTrainingWarning_once_with_fits_of_the_search():
    """
    Test that LongTrainingWarning is raised once, with the number of fits of
    each trial and of the whole search, instead of once per trial.
    """
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=2)
    cv = TimeSeriesFold(steps=1, initial_train_size=440, refit=True)

    def search_space(trial):
        search_space = {'alpha': trial.suggest_float('alpha', 0.1, 1.0)}

        return search_space

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        bayesian_search_forecaster(
            forecaster    = forecaster,
            y             = y_feature_selection,
            cv            = cv,
            search_space  = search_space,
            metric        = 'mean_absolute_error',
            n_trials      = 4,
            random_state  = 123,
            return_best   = False,
            verbose       = False,
            show_progress = False
        )

    messages = [
        w.message.message for w in record if w.category is LongTrainingWarning
    ]
    assert messages == [
        "Each trial fits the forecaster about 60 times, so the search (4 trials) "
        "fits it about 240 times. This can take substantial amounts of time. If "
        "not feasible, try with `refit = False`."
    ]


@pytest.mark.parametrize(
    "cv",
    [TimeSeriesFold(steps=3, initial_train_size=30, refit=False),
     OneStepAheadFold(initial_train_size=30)],
    ids=lambda cv: f"cv: {type(cv).__name__}"
)
def test_bayesian_search_forecaster_skips_trials_whose_evaluation_fails(cv, capfd):
    """
    Test that the trials whose evaluation fails (40 lags, not smaller than
    `initial_train_size`) are marked as failed and reported in the warning of
    the candidates skipped, identified by their number, instead of stopping
    the search. Optuna does not log them.
    """
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=2)

    def search_space(trial):
        return {'lags': trial.suggest_categorical('lags', [2, 40])}

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results, study = bayesian_search_forecaster(
                             forecaster    = forecaster,
                             y             = y,
                             cv            = cv,
                             search_space  = search_space,
                             metric        = 'mean_absolute_error',
                             n_trials      = 6,
                             return_best   = True,
                             show_progress = False
                         )

    failed = [t.number for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    assert failed == [t.number for t in study.trials if t.params['lags'] == 40]
    assert 0 < len(failed) < 6
    assert len(results) == 6 - len(failed)
    assert results['lags'].map(len).unique().tolist() == [2]
    assert forecaster.max_lag == 2
    messages = [str(w.message) for w in record if w.category is RuntimeWarning]
    assert len(messages) == 1
    verb = "raises" if len(failed) == 1 else "raise"
    assert messages[0].startswith(
        f"Parameters skipped: {len(failed)} of 6 trials {verb} an error"
    )
    assert "`initial_train_size` must be greater than the `window_size` of the " \
           "forecaster (40)" in messages[0]
    for trial_number in failed:
        assert f"  - trial {trial_number}, params: {{'lags': 40}}" in messages[0]
    assert "Trial" not in capfd.readouterr().err


def test_bayesian_search_forecaster_OneStepAheadFold_window_size_of_forecaster_not_checked():
    """
    Test that, with a `OneStepAheadFold`, the `window_size` of the forecaster
    passed (70 lags) is not checked against `initial_train_size` (60) when
    the `search_space` does not use it: the trials (3 lags) are evaluated
    without errors, as with a `TimeSeriesFold`.
    """
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=70)

    def search_space(trial):
        return {
            'lags': trial.suggest_categorical('lags', [3]),
            'alpha': trial.suggest_float('alpha', 0.1, 1)
        }

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results, study = bayesian_search_forecaster(
                             forecaster    = forecaster,
                             y             = y_feature_selection.iloc[:80],
                             cv            = OneStepAheadFold(initial_train_size=60),
                             search_space  = search_space,
                             metric        = 'mean_absolute_error',
                             n_trials      = 3,
                             random_state  = 123,
                             return_best   = True,
                             show_progress = False
                         )

    assert len(results) == 3
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert results['lags'].map(len).tolist() == [3, 3, 3]
    assert forecaster.max_lag == 3
    assert [w for w in record if w.category is RuntimeWarning] == []


def test_bayesian_search_forecaster_every_trial_fails_returns_empty_results():
    """
    Test that, when the evaluation of every trial fails, the results are
    empty, the forecaster is not refitted and the warning says so.
    """
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=2)

    def search_space(trial):
        return {'lags': trial.suggest_categorical('lags', [40, 45])}

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results, study = bayesian_search_forecaster(
                             forecaster    = forecaster,
                             y             = y,
                             cv            = TimeSeriesFold(steps=3, initial_train_size=30),
                             search_space  = search_space,
                             metric        = 'mean_absolute_error',
                             n_trials      = 3,
                             return_best   = True,
                             show_progress = False
                         )

    assert results.empty
    assert not forecaster.is_fitted
    assert len(study.trials) == 3
    messages = [str(w.message) for w in record if w.category is RuntimeWarning]
    assert messages[0].startswith(
        "Parameters skipped: 3 of 3 trials raise an error, so the results are empty."
    )


def test_bayesian_search_forecaster_ValueError_search_space_stops_the_search(capfd):
    """
    Test that an error of the `search_space` (not of the evaluation of the
    trial) stops the search, also with a `catch` of the user in
    `kwargs_study_optimize`. Optuna does not log the failed trial, so the
    error is shown once, when it is raised.
    """
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=2)

    def search_space(trial):
        return {'alpha': trial.suggest_float('alpha', 0.1, 1) / 0}

    with pytest.raises(ZeroDivisionError):
        bayesian_search_forecaster(
            forecaster            = forecaster,
            y                     = y,
            cv                    = TimeSeriesFold(steps=3, initial_train_size=30),
            search_space          = search_space,
            metric                = 'mean_absolute_error',
            n_trials              = 3,
            return_best           = False,
            show_progress         = False,
            kwargs_study_optimize = {'catch': (KeyError,)}
        )

    assert "Trial 0 failed" not in capfd.readouterr().err


def test_bayesian_search_forecaster_TimeSeriesFold_NaN_in_test_set_single_series():
    """
    Test that, with a `TimeSeriesFold` and NaN in `y` inside the test set,
    every trial is evaluated (the dates without a true value are left out of
    the metrics) on the same points, so no trial is skipped and the
    MissingValuesWarning of `n_evaluated` is not raised.
    """
    y_nan = y.copy()
    y_nan.iloc[[35, 40, 45]] = np.nan
    forecaster = ForecasterRecursive(
        estimator=HistGradientBoostingRegressor(max_iter=5, random_state=123), lags=2
    )

    def search_space(trial):
        return {'lags': trial.suggest_categorical('lags', [2, 10])}

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results, _ = bayesian_search_forecaster(
                         forecaster    = forecaster,
                         y             = y_nan,
                         cv            = TimeSeriesFold(steps=5, initial_train_size=30),
                         search_space  = search_space,
                         metric        = 'mean_absolute_error',
                         n_trials      = 4,
                         return_best   = False,
                         show_progress = False
                     )

    assert len(results) == 4
    assert results['mean_absolute_error'].notna().all()
    assert results['n_evaluated'].tolist() == [17] * 4
    assert [w for w in record if w.category is RuntimeWarning] == []
    assert [
        w for w in record
        if w.category is MissingValuesWarning and "comparable" in str(w.message)
    ] == []


def test_bayesian_search_forecaster_MissingValuesWarning_n_evaluated_single_series():
    """
    Test that, with a `OneStepAheadFold`, `dropna_from_series=True` and NaN in
    `y`, the trials with larger windows are evaluated on fewer points and a
    MissingValuesWarning lists them by trial number, once at the end of the
    search.
    """
    y_nan = y.copy()
    y_nan.iloc[[35, 40, 45]] = np.nan
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=2, dropna_from_series=True)

    def search_space(trial):
        return {'lags': trial.suggest_categorical('lags', [2, 10])}

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results, study = bayesian_search_forecaster(
                             forecaster    = forecaster,
                             y             = y_nan,
                             cv            = OneStepAheadFold(initial_train_size=30),
                             search_space  = search_space,
                             metric        = 'mean_absolute_error',
                             n_trials      = 4,
                             return_best   = False,
                             show_progress = False
                         )

    trials_10 = sorted(t.number for t in study.trials if t.params['lags'] == 10)
    assert 0 < len(trials_10) < 4
    messages = [
        w.message.message for w in record
        if w.category is MissingValuesWarning and "comparable" in str(w.message)
    ]
    assert len(messages) == 1
    assert messages[0].startswith(
        f"{len(trials_10)} of 4 trials {'is' if len(trials_10) == 1 else 'are'} "
        f"evaluated on fewer points than the most complete one"
    )
    assert "fill the gaps of `y`" in messages[0]
    assert "  - lags: [1, 2, ..., 10]" in messages[0]


@pytest.mark.parametrize(
    "estimator, n_jobs, expected_n_jobs",
    [(Ridge(), 'auto', 1),
     (HistGradientBoostingRegressor(max_iter=5), 'auto', max(1, cpu_count() - 1)),
     (HistGradientBoostingRegressor(max_iter=5), 2, 2)],
    ids=['auto, Ridge', 'auto, HistGradientBoostingRegressor', 'int']
)
def test_bayesian_search_forecaster_n_jobs_auto_resolved_with_the_rules_of_each_candidate(
    estimator, n_jobs, expected_n_jobs
):
    """
    Test that bayesian_search_forecaster resolves `n_jobs='auto'` with
    `select_n_jobs_backtesting` for each candidate and passes the integer to
    `backtesting_forecaster`, so the work threshold of a single backtesting does not
    reduce it. `Parallel` is mocked with a sequential one, so no workers are
    started.
    """
    forecaster = ForecasterRecursive(estimator=estimator, lags=3)
    cv = TimeSeriesFold(
             steps              = 10,
             initial_train_size = 10,
             refit              = True,
             fixed_train_size   = False
         )

    with (
        patch(
            "skforecast.model_selection._search.backtesting_forecaster",
            wraps=backtesting_forecaster
        ) as mock_backtesting,
        patch(
            "skforecast.model_selection._validation.Parallel",
            side_effect=lambda **kwargs: Parallel(n_jobs=1)
        ) as mock_parallel
    ):
        bayesian_search_forecaster(
            forecaster    = forecaster,
            y             = y,
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


class RidgeWarnsInPredict(Ridge):
    """
    Ridge that warns in `predict` when `alpha` is greater than 0.5. It is not
    a class of scikit-learn, so the forecaster calls its `predict`.
    """

    def predict(self, X):
        if self.alpha > 0.5:
            warnings.warn("Large alpha in predict. Details of the fold.", UserWarning)
        return super().predict(X)


def test_bayesian_search_forecaster_warnings_of_folds_in_processes_with_trials_in_threads():
    """
    Test that the warnings raised in the folds that run in worker processes
    (`n_jobs=2`), shown again in the thread of each trial (`n_jobs` of
    `study.optimize`), are collected with their trial: the warning raised in
    the trials with `alpha` greater than 0.5 is shown once with the number of
    those trials.
    """
    def search_space(trial):
        return {'alpha': trial.suggest_float('alpha', 0.01, 1.0)}

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _, study = bayesian_search_forecaster(
            forecaster            = ForecasterRecursive(RidgeWarnsInPredict(), lags=3),
            y                     = y,
            cv                    = TimeSeriesFold(steps=2, initial_train_size=40, refit=False),
            search_space          = search_space,
            metric                = 'mean_absolute_error',
            n_trials              = 8,
            random_state          = 123,
            return_best           = False,
            n_jobs                = 2,
            show_progress         = False,
            kwargs_study_optimize = {'n_jobs': 2}
        )

    n_large_alpha = sum(trial.params['alpha'] > 0.5 for trial in study.trials)
    assert 0 < n_large_alpha < 8
    messages = [str(w.message) for w in record if w.category is UserWarning]
    assert messages == [
        f"Large alpha in predict (in {n_large_alpha} of 8 trials). Details of the fold."
    ]
