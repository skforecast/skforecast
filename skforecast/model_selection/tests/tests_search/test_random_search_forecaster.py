# Unit test random_search_forecaster
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection._search import random_search_forecaster
from skforecast.model_selection._split import TimeSeriesFold

# Fixtures
from ..fixtures_model_selection import y

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # hide progress bar


def test_output_random_search_forecaster_ForecasterAutoreg_with_mocked():
    """
    Test output of random_search_forecaster in ForecasterRecursive with mocked
    (mocked done in Skforecast v0.4.3)
    """
    forecaster = ForecasterRecursive(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    param_distributions = {'alpha': np.logspace(-5, 3, 10)}
    n_iter = 3
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

    results = random_search_forecaster(
                  forecaster           = forecaster,
                  y                    = y,
                  cv                   = cv,
                  lags_grid            = lags_grid,
                  param_distributions  = param_distributions,
                  metric               = 'mean_squared_error',
                  n_iter               = n_iter,
                  random_state         = 123,
                  return_best          = False,
                  verbose              = False
              )

    expected_results = pd.DataFrame(
        {
            "lags": [[1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2]],
            "lags_label": [
                [1, 2],
                [1, 2],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2],
            ],
            "params": [
                {"alpha": 1e-05},
                {"alpha": 0.03593813663804626},
                {"alpha": 1e-05},
                {"alpha": 0.03593813663804626},
                {"alpha": 16.681005372000556},
                {"alpha": 16.681005372000556},
            ],
            "mean_squared_error": np.array(
                [0.06460234, 0.06475887, 0.06776596, 0.06786132, 0.0713478, 0.07161]
            ),
            "alpha": np.array(
                [
                    1.00000000e-05,
                    3.59381366e-02,
                    1.00000000e-05,
                    3.59381366e-02,
                    1.66810054e01,
                    1.66810054e01,
                ]
            ),
        },
        index=pd.Index([0, 1, 2, 3, 4, 5], dtype="int64"),
    )

    pd.testing.assert_frame_equal(results, expected_results)
