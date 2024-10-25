# Unit test grid_search_forecaster
# ==============================================================================
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_percentage_error
from skforecast.metrics import mean_absolute_scaled_error
from skforecast.metrics import root_mean_squared_scaled_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from skforecast.model_selection._search import grid_search_forecaster
from skforecast.model_selection._split import TimeSeriesFold
from skforecast.model_selection._split import OneStepAheadFold

from tqdm import tqdm
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # hide progress bar

# Fixtures
from ..fixtures_model_selection import y

 
def test_output_grid_search_forecaster_ForecasterAutoreg_with_mocked():
    """
    Test output of grid_search_forecaster in ForecasterRecursive with mocked
    (mocked done in Skforecast v0.4.3)
    """
    forecaster = ForecasterRecursive(
        regressor=Ridge(random_state=123),
        lags=2,  # Placeholder, the value will be overwritten
    )
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    param_grid = {"alpha": [0.01, 0.1, 1]}
    idx = len(lags_grid) * len(param_grid["alpha"])
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = False,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False
         )

    results = grid_search_forecaster(
        forecaster=forecaster,
        y=y,
        cv=cv,
        lags_grid=lags_grid,
        param_grid=param_grid,
        metric="mean_squared_error",
        return_best=False,
        verbose=False,
    )

    expected_results = pd.DataFrame(
        {
            "lags": [[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            "lags_label": [
                [1, 2],
                [1, 2],
                [1, 2],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
            ],
            "params": [
                {"alpha": 0.01},
                {"alpha": 0.1},
                {"alpha": 1},
                {"alpha": 0.01},
                {"alpha": 0.1},
                {"alpha": 1},
            ],
            "mean_squared_error": np.array(
                [0.06464646, 0.06502362, 0.06745534, 0.06779272, 0.06802481, 0.06948609]
            ),
            "alpha": np.array([0.01, 0.1, 1.0, 0.01, 0.1, 1.0]),
        },
        index=pd.RangeIndex(start=0, stop=idx, step=1),
    )

    pd.testing.assert_frame_equal(results, expected_results)


forecasters = [
    ForecasterRecursive(regressor=Ridge(random_state=678), lags=3),
    ForecasterDirect(regressor=Ridge(random_state=678), lags=3, steps=1),
    ForecasterRecursive(
        regressor=Ridge(random_state=678),
        lags=3,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
    ),
    ForecasterDirect(
        regressor=Ridge(random_state=678),
        lags=3,
        steps=1,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
    ),
]
@pytest.mark.parametrize("forecaster", forecasters)
def test_output_grid_search_forecaster_equivalence_backtesting_one_step_ahead(
    forecaster,
):
    """
    Test that output of grid_search_forecaster is equivalent with backtesting and
    one step ahead one step=1, fixed_train_size=True retran=False
    """

    metrics = [
        "mean_absolute_error",
        "mean_squared_error",
        mean_absolute_percentage_error,
        mean_absolute_scaled_error,
        root_mean_squared_scaled_error,
    ]
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    param_grid = {"alpha": [0.01, 0.1, 1]}
    cv_backtesting = TimeSeriesFold(
            steps                 = 1,
            initial_train_size    = len(y_train),
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
            initial_train_size    = len(y_train),
            return_all_indexes    = False,
        )

    results_backtesting = grid_search_forecaster(
        forecaster=forecaster,
        y=y,
        cv=cv_backtesting,
        lags_grid=lags_grid,
        param_grid=param_grid,
        metric=metrics,
        return_best=False,
        verbose=False,
    )

    results_one_step_ahead = grid_search_forecaster(
        forecaster=forecaster,
        y=y,
        cv=cv_one_step_ahead,
        lags_grid=lags_grid,
        param_grid=param_grid,
        metric=metrics,
        return_best=False,
        verbose=False,
    )

    pd.testing.assert_frame_equal(results_backtesting, results_one_step_ahead)
