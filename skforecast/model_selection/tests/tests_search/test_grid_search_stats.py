# Unit test grid_search_stats
# ==============================================================================
import pytest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import skforecast.stats._arima as arima_module
from skforecast.stats import Arima, Sarimax
from skforecast.recursive import ForecasterStats
from skforecast.model_selection._split import TimeSeriesFold
from skforecast.model_selection._search import grid_search_stats

# Fixtures
from ....recursive.tests.tests_forecaster_stats.fixtures_forecaster_stats import y_datetime

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # hide progress bar


def test_output_grid_search_stats_sarimax_with_mocked():
    """
    Test output of grid_search_stats in ForecasterStats with mocked
    (mocked done in Skforecast v0.7.0).
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    param_grid = [{'order': [(2, 2, 0), (3, 2, 0)],
                   'trend': [None, 'c']}]

    results = grid_search_stats(
                  forecaster  = forecaster,
                  y           = y_datetime,
                  cv          = cv,
                  param_grid  = param_grid,
                  metric      = 'mean_absolute_error',
                  return_best = False,
                  verbose     = False
              )
    
    expected_results = pd.DataFrame(
        data  = {'params'             : [{'order': (3, 2, 0), 'trend': None},
                                         {'order': (3, 2, 0), 'trend': 'c'}, 
                                         {'order': (2, 2, 0), 'trend': 'c'}, 
                                         {'order': (2, 2, 0), 'trend': None}],
                 'mean_absolute_error': np.array([0.15357204, 0.1548934 , 0.19852912, 0.19853423]),
                 'order'              : [(3, 2, 0), (3, 2, 0), (2, 2, 0), (2, 2, 0)],
                 'trend'              : [None, 'c', 'c', None]},
        index = pd.Index(np.array([0, 1, 2, 3]), dtype="int64")
    )

    pd.testing.assert_frame_equal(results, expected_results, atol=0.0001)


def test_grid_search_stats_automatic_search_runs_once_per_candidate():
    """
    Test that grid_search_stats with an automatic Arima runs the automatic
    selection once per candidate (in the first fit of its backtesting) and
    reuses the selected model in the other folds (`freeze_params=True`).
    Before the fix, it was repeated in every fold of every candidate.
    """
    forecaster = ForecasterStats(estimator=Arima(order=None, seasonal_order=None))
    cv = TimeSeriesFold(
             steps              = 3,
             initial_train_size = len(y_datetime) - 12,
             refit              = True,
             verbose            = False
         )

    with patch(
        'skforecast.stats._arima.auto_arima', wraps=arima_module.auto_arima
    ) as mock_auto_arima:
        results = grid_search_stats(
                      forecaster        = forecaster,
                      y                 = y_datetime,
                      cv                = cv,
                      param_grid        = {'max_p': [2, 3]},
                      metric            = 'mean_absolute_error',
                      return_best       = False,
                      n_jobs            = 1,
                      suppress_warnings = True,
                      verbose           = False
                  )

    assert len(results) == 2
    assert mock_auto_arima.call_count == 2


def test_grid_search_stats_maximizes_metric_when_greater_is_better():
    """
    Test that, with a metric in which greater values are better (`r2_score`),
    the results are sorted in descending order and `return_best` refits the
    forecaster with the best candidate. Before the fix, the results were
    always sorted in ascending order.
    """
    forecaster = ForecasterStats(estimator=Arima(order=(1, 0, 0)))
    cv = TimeSeriesFold(
             steps              = 3,
             initial_train_size = len(y_datetime) - 12,
             refit              = False
         )
    results = grid_search_stats(
                  forecaster  = forecaster,
                  y           = y_datetime,
                  cv          = cv,
                  param_grid  = {'order': [(1, 0, 0), (2, 0, 0), (1, 1, 0)]},
                  metric      = r2_score,
                  return_best = True
              )

    assert results['r2_score'].nunique() > 1
    assert results['r2_score'].is_monotonic_decreasing
    assert results.loc[0, 'order'] == (1, 1, 0)
    assert forecaster.estimators[0].get_params()['order'] == (1, 1, 0)


def test_grid_search_stats_ValueError_when_metric_name_not_valid():
    """
    Test ValueError is raised before evaluating any candidate when a metric
    name is not valid. Before, every candidate was skipped with a warning and
    an empty DataFrame was returned.
    """
    forecaster = ForecasterStats(estimator=Arima(order=(1, 0, 0)))
    cv = TimeSeriesFold(
             steps              = 3,
             initial_train_size = len(y_datetime) - 12,
             refit              = False
         )

    with pytest.raises(ValueError, match="Allowed metrics are"):
        grid_search_stats(
            forecaster  = forecaster,
            y           = y_datetime,
            cv          = cv,
            param_grid  = {'order': [(1, 0, 0), (2, 0, 0)]},
            metric      = 'not_a_metric',
            return_best = False
        )
