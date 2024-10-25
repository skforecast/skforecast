# Unit test bayesian_search_forecaster
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
# from skopt.space import Real
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection._search import bayesian_search_forecaster
from skforecast.model_selection._split import TimeSeriesFold

# Fixtures
from ..fixtures_model_selection import y


def test_ValueError_bayesian_search_forecaster_when_return_best_and_len_y_exog_different():
    """
    Test ValueError is raised in bayesian_search_forecaster when return_best 
    and length of `y` and `exog` do not match.
    """
    forecaster = ForecasterRecursive(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )
    exog = y[:30]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y[:-12]),
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
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}
        return search_space

    err_msg = re.escape(
        (f"`exog` must have same number of samples as `y`. "
         f"length `exog`: ({len(exog)}), length `y`: ({len(y)})")
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


def test_results_output_bayesian_search_forecaster_optuna_ForecasterAutoreg_with_mocked():
    """
    Test output of bayesian_search_forecaster in ForecasterRecursive with 
    mocked using optuna (mocked done in Skforecast v0.4.3).
    """    
    forecaster = ForecasterRecursive(
                     regressor = Ridge(random_state=123),
                     lags      = 4
                 )

    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y[:-12]),
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
        np.array([[np.array([1, 2, 3, 4]), {'alpha': 0.9809565564007693},
            0.21539791166603497, 0.9809565564007693],
        [np.array([1, 2, 3, 4]), {'alpha': 0.7222742800877074},
            0.21576131952657338, 0.7222742800877074],
        [np.array([1, 2, 3, 4]), {'alpha': 0.6995044937418831},
            0.21579460210585208, 0.6995044937418831],
        [np.array([1, 2, 3, 4]), {'alpha': 0.6879814411990146},
            0.21581150916013206, 0.6879814411990146],
        [np.array([1, 2, 3, 4]), {'alpha': 0.5558016213920624},
            0.21600778429729228, 0.5558016213920624],
        [np.array([1, 2, 3, 4]), {'alpha': 0.48612258246951734},
            0.21611205459571634, 0.48612258246951734],
        [np.array([1, 2, 3, 4]), {'alpha': 0.42887539552321635},
            0.2161973389956996, 0.42887539552321635],
        [np.array([1, 2, 3, 4]), {'alpha': 0.398196343012209},
            0.2162426532005299, 0.398196343012209],
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
