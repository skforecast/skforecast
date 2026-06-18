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

 
def test_output_grid_search_forecaster_ForecasterRecursive_with_mocked():
    """
    Test output of grid_search_forecaster in ForecasterRecursive with mocked
    (mocked done in Skforecast v0.4.3)
    """
    forecaster = ForecasterRecursive(
        estimator=Ridge(random_state=123),
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
    ForecasterRecursive(estimator=Ridge(random_state=678), lags=3),
    ForecasterDirect(estimator=Ridge(random_state=678), lags=3, steps=1),
    ForecasterRecursive(
        estimator=Ridge(random_state=678),
        lags=3,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
    ),
    ForecasterDirect(
        estimator=Ridge(random_state=678),
        lags=3,
        steps=1,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
    ),
]
@pytest.mark.parametrize("forecaster", forecasters)
def test_grid_search_forecaster_equivalence_backtesting_one_step_ahead(
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


@pytest.mark.parametrize("forecaster", forecasters)
def test_grid_search_forecaster_one_step_ahead_date_initial_train_size_equivalence(
    forecaster,
):
    """
    Test that grid_search_forecaster with OneStepAheadFold returns identical results
    whether `initial_train_size` is given as an integer or as the equivalent date.
    """
    metrics = ["mean_absolute_error", "mean_squared_error"]
    n_validation = 12
    initial_train_size_int = len(y) - n_validation  # 38
    lags_grid = [2, 4]
    param_grid = {"alpha": [0.01, 0.1, 1]}

    # Datetime-indexed copy so a date `initial_train_size` is valid. Position 37
    # (the 38th observation) is 2020-02-07; `date_to_index_position` resolves it to 38.
    y_datetime = y.copy()
    y_datetime.index = pd.date_range(start='2020-01-01', periods=len(y), freq='D')

    cv_int = OneStepAheadFold(initial_train_size=initial_train_size_int)
    cv_date = OneStepAheadFold(initial_train_size="2020-02-07")

    results_int = grid_search_forecaster(
        forecaster=forecaster,
        y=y_datetime,
        cv=cv_int,
        lags_grid=lags_grid,
        param_grid=param_grid,
        metric=metrics,
        return_best=False,
        verbose=False,
    )
    results_date = grid_search_forecaster(
        forecaster=forecaster,
        y=y_datetime,
        cv=cv_date,
        lags_grid=lags_grid,
        param_grid=param_grid,
        metric=metrics,
        return_best=False,
        verbose=False,
    )

    pd.testing.assert_frame_equal(results_int, results_date)
    # `initial_train_size` must remain the raw user input on the (deep-copied) cv objects.
    assert cv_int.initial_train_size == initial_train_size_int
    assert cv_date.initial_train_size == "2020-02-07"
