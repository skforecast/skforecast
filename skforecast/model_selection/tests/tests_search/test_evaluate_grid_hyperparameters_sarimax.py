# Unit test _evaluate_grid_hyperparameters_sarimax
# ==============================================================================
import os
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from skforecast.sarimax import Sarimax
from skforecast.recursive import ForecasterSarimax
from skforecast.model_selection._split import TimeSeriesFold
from skforecast.model_selection._search import _evaluate_grid_hyperparameters_sarimax

# Fixtures
# from skforecast.recursive.tests.tests_forecaster_sarimax.fixtures_forecaster_sarimax import y_datetime
from ....recursive.tests.tests_forecaster_sarimax.fixtures_forecaster_sarimax import y_datetime
from ....recursive.tests.tests_forecaster_sarimax.fixtures_forecaster_sarimax import exog_datetime

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # hide progress bar


def test_ValueError_evaluate_grid_hyperparameters_sarimax_when_return_best_and_len_y_exog_different():
    """
    Test ValueError is raised in _evaluate_grid_hyperparameters_sarimax when return_best 
    and length of `y` and `exog` do not match.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    exog_test = exog_datetime[:30].copy()
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )

    err_msg = re.escape(
        (f'`exog` must have same number of samples as `y`. '
         f'length `exog`: ({len(exog_test)}), length `y`: ({len(y_datetime)})')
    )
    with pytest.raises(ValueError, match = err_msg):
        _evaluate_grid_hyperparameters_sarimax(
            forecaster  = forecaster,
            y           = y_datetime,
            cv          = cv,
            exog        = exog_test,
            param_grid  = [{'order': (1, 1, 1)}, {'order': (1, 2, 2)}, {'order': (1, 2, 3)}],
            metric      = 'mean_absolute_error',
            return_best = True,
            verbose     = False
        )


def test_exception_evaluate_grid_hyperparameters_sarimax_metric_list_duplicate_names():
    """
    Test exception is raised in _evaluate_grid_hyperparameters_sarimax when a `list` of 
    metrics is used with duplicate names.
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
    
    err_msg = re.escape('When `metric` is a `list`, each metric name must be unique.')
    with pytest.raises(ValueError, match = err_msg):
        _evaluate_grid_hyperparameters_sarimax(
            forecaster  = forecaster,
            y           = y_datetime,
            cv          = cv,
            exog        = exog_datetime,
            param_grid  = [{'order': (1, 1, 1)}, {'order': (1, 2, 2)}, {'order': (1, 2, 3)}],
            metric      = ['mean_absolute_error', mean_absolute_error],
            return_best = True,
            verbose     = False
        )


def test_output_evaluate_grid_hyperparameters_sarimax_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_sarimax in ForecasterSarimax with mocked
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
    
    param_grid = [{'order': (3, 2, 0), 'trend': None}, 
                  {'order': (3, 2, 0), 'trend': 'c'}]

    results = _evaluate_grid_hyperparameters_sarimax(
                  forecaster  = forecaster,
                  y           = y_datetime,
                  cv          = cv,
                  param_grid  = param_grid,
                  metric      = 'mean_squared_error',
                  return_best = False,
                  verbose     = False
              )
    
    expected_results = pd.DataFrame(
        data  = {'params'            : [{'order': (3, 2, 0), 'trend': None}, 
                                        {'order': (3, 2, 0), 'trend': 'c'}],
                 'mean_squared_error': np.array([0.03683793, 0.03740798]),
                 'order'             : [(3, 2, 0), (3, 2, 0)],
                 'trend'             : [None, 'c']},
        index = pd.Index(np.array([0, 1]), dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results, atol=0.0001)


def test_output_evaluate_grid_hyperparameters_sarimax_exog_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_sarimax in ForecasterSarimax 
    with exog with mocked (mocked done in Skforecast v0.7.0).
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
    
    param_grid = [{'order': (3, 2, 0), 'trend': None}, 
                  {'order': (3, 2, 0), 'trend': 'c'}]

    results = _evaluate_grid_hyperparameters_sarimax(
                  forecaster  = forecaster,
                  y           = y_datetime,
                  cv          = cv,
                  exog        = exog_datetime,
                  param_grid  = param_grid,
                  metric      = 'mean_squared_error',
                  return_best = False,
                  verbose     = False
              )
    
    expected_results = pd.DataFrame(
        data  = {'params'            : [{'order': (3, 2, 0), 'trend': None}, 
                                        {'order': (3, 2, 0), 'trend': 'c'}],
                 'mean_squared_error': np.array([0.18551857, 0.19151678]),
                 'order'             : [(3, 2, 0), (3, 2, 0)],
                 'trend'             : [None, 'c']},
        index = pd.Index(np.array([0, 1]), dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results, atol=0.0001)


def test_output_evaluate_grid_hyperparameters_sarimax_metric_list_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters_sarimax in ForecasterSarimax 
    with multiple metrics with mocked (mocked done in Skforecast v0.7.0).
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    param_grid = [{'order': (3, 2, 0), 'trend': None}, 
                  {'order': (3, 2, 0), 'trend': 'c'}]

    results = _evaluate_grid_hyperparameters_sarimax(
                  forecaster  = forecaster,
                  y           = y_datetime,
                  cv          = cv,
                  param_grid  = param_grid,
                  metric      = [mean_absolute_error, 'mean_squared_error'],
                  return_best = False,
                  verbose     = False
              )
    
    expected_results = pd.DataFrame(
        data  = {'params'             : [{'order': (3, 2, 0), 'trend': None}, 
                                         {'order': (3, 2, 0), 'trend': 'c'}],
                 'mean_absolute_error': np.array([0.15724498, 0.16638452]),
                 'mean_squared_error' : np.array([0.0387042 , 0.04325543]),
                 'order'              : [(3, 2, 0), (3, 2, 0)],
                 'trend'              : [None, 'c']},
        index = pd.Index(np.array([0, 1]), dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results, atol=0.0001)
    

def test_evaluate_grid_hyperparameters_sarimax_when_return_best():
    """
    Test forecaster is refitted when return_best=True in 
    _evaluate_grid_hyperparameters_sarimax.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = True,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    param_grid = [{'order': (3, 2, 0), 'trend': None}, 
                  {'order': (3, 2, 0), 'trend': 'c'}]

    _evaluate_grid_hyperparameters_sarimax(
        forecaster            = forecaster,
        y                     = y_datetime,
        cv                    = cv,
        param_grid            = param_grid,
        metric                = mean_absolute_error,
        return_best           = True,
        suppress_warnings_fit = False,
        verbose               = False
    )
    
    expected_params = {
        'concentrate_scale': False,
        'dates': None,
        'disp': False,
        'enforce_invertibility': True,
        'enforce_stationarity': True,
        'freq': None,
        'hamilton_representation': False,
        'maxiter': 1000,
        'measurement_error': False,
        'method': 'cg',
        'missing': 'none',
        'mle_regression': True,
        'order': (3, 2, 0),
        'seasonal_order': (0, 0, 0, 0),
        'simple_differencing': False,
        'sm_fit_kwargs': {},
        'sm_init_kwargs': {},
        'sm_predict_kwargs': {},
        'start_params': None,
        'time_varying_regression': False,
        'trend': None,
        'trend_offset': 1,
        'use_exact_diffuse': False,
        'validate_specification': True
    }
    
    assert expected_params == forecaster.params


def test_evaluate_grid_hyperparameters_sarimax_output_file_when_single_metric():
    """
    Test output file is created when output_file is passed to
    _evaluate_grid_hyperparameters_sarimax and single metric.
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
    
    param_grid = [{'order': (3, 2, 0), 'trend': None}, 
                  {'order': (1, 1, 0), 'trend': 'c'}]
    output_file = 'test_evaluate_grid_hyperparameters_sarimax_output_file.txt'

    results = _evaluate_grid_hyperparameters_sarimax(
                  forecaster  = forecaster,
                  y           = y_datetime,
                  cv          = cv,
                  param_grid  = param_grid,
                  metric      = 'mean_squared_error',
                  return_best = False,
                  verbose     = False,
                  output_file = output_file
              )
    results  = results.astype({'params': str, 'order': str})

    def convert_none(val):  # pragma: no cover
        if val == 'None':
            return None
        return val

    assert os.path.isfile(output_file)
    output_file_content = pd.read_csv(output_file, sep='\t', low_memory=False, converters={'trend': convert_none})
    output_file_content = output_file_content.sort_values(by='mean_squared_error').reset_index(drop=True)
    output_file_content = output_file_content.astype({'params': str, 'order': str})
    pd.testing.assert_frame_equal(results, output_file_content)
    os.remove(output_file)


def test_evaluate_grid_hyperparameters_sarimax_output_file_when_metric_list():
    """
    Test output file is created when output_file is passed to
    _evaluate_grid_hyperparameters_sarimax and metric as list.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    param_grid = [{'order': (3, 2, 0), 'trend': None}, 
                  {'order': (1, 1, 0), 'trend': 'c'}]
    output_file = 'test_evaluate_grid_hyperparameters_sarimax_output_file.txt'

    results = _evaluate_grid_hyperparameters_sarimax(
                  forecaster  = forecaster,
                  y           = y_datetime,
                  cv          = cv,
                  param_grid  = param_grid,
                  metric      = [mean_absolute_error, 'mean_squared_error'],
                  return_best = False,
                  verbose     = False,
                  output_file = output_file
              )
    results  = results.astype({'params': str, 'order': str})

    def convert_none(val):  # pragma: no cover
        if val == 'None': 
            return None
        return val

    assert os.path.isfile(output_file)
    output_file_content = pd.read_csv(output_file, sep='\t', low_memory=False, converters={'trend': convert_none})
    output_file_content = output_file_content.sort_values(by='mean_squared_error').reset_index(drop=True)
    output_file_content = output_file_content.astype({'params': str, 'order': str})
    pd.testing.assert_frame_equal(results, output_file_content)
    os.remove(output_file)
