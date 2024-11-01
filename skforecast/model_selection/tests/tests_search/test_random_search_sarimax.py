# Unit test random_search_sarimax
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.sarimax import Sarimax
from skforecast.recursive import ForecasterSarimax
from skforecast.model_selection._split import TimeSeriesFold
from skforecast.model_selection._search import random_search_sarimax

# Fixtures
from ....recursive.tests.tests_forecaster_sarimax.fixtures_forecaster_sarimax import y_datetime

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # hide progress bar


def test_output_random_search_sarimax_sarimax_with_mocked():
    """
    Test output of random_search_sarimax in ForecasterSarimax with mocked
    (mocked done in Skforecast v0.7.0).
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )

    param_distributions = {'order': [(2, 2, 0), (3, 2, 0)]}

    results = random_search_sarimax(
                  forecaster          = forecaster,
                  y                   = y_datetime,
                  cv                  = cv,
                  param_distributions = param_distributions,
                  metric              = 'mean_absolute_error',
                  n_iter              = 2,
                  random_state        = 123,
                  return_best         = False,
                  verbose             = False
              )
    
    expected_results = pd.DataFrame(
        data  = {'params': np.array([{'order': (3, 2, 0)}, {'order': (2, 2, 0)}], dtype=object),
                'mean_absolute_error': np.array([0.15357204, 0.19853423]),
                'order'              : [(3, 2, 0), (2, 2, 0)]},
        index = pd.Index(np.array([0, 1]), dtype="int64")
    )

    pd.testing.assert_frame_equal(results, expected_results, atol=0.001)
