# Unit test _evaluate_grid_hyperparameters
# ==============================================================================
import re
import os
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from skforecast.metrics import mean_absolute_scaled_error
from skforecast.metrics import root_mean_squared_scaled_error
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from skforecast.model_selection._search import _evaluate_grid_hyperparameters
from skforecast.model_selection._split import TimeSeriesFold, OneStepAheadFold

# Fixtures
from ..fixtures_model_selection import y
from ..fixtures_model_selection import y_feature_selection
from ..fixtures_model_selection import exog_feature_selection

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # hide progress bar


def test_ValueError_evaluate_grid_hyperparameters_when_return_best_and_len_y_exog_different():
    """
    Test ValueError is raised in _evaluate_grid_hyperparameters when return_best 
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
            refit                 = False,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    err_msg = re.escape(
        (f"`exog` must have same number of samples as `y`. "
         f"length `exog`: ({len(exog)}), length `y`: ({len(y)})")
    )
    with pytest.raises(ValueError, match = err_msg):
        _evaluate_grid_hyperparameters(
            forecaster  = forecaster,
            y           = y,
            exog        = exog,
            cv          = cv,
            lags_grid   = [2, 4],
            param_grid  = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            metric      = 'mean_absolute_error',
            return_best = True,
            verbose     = False
        )


def test_ValueError_evaluate_grid_hyperparameters_metric_list_duplicate_names():
    """
    Test ValueError is raised in _evaluate_grid_hyperparameters when a `list` of 
    metrics is used with duplicate names.
    """
    forecaster = ForecasterRecursive(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y[:-12]),
            window_size           = None,
            differentiation       = None,
            refit                 = False,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    err_msg = re.escape('When `metric` is a `list`, each metric name must be unique.')
    with pytest.raises(ValueError, match = err_msg):
        _evaluate_grid_hyperparameters(
            forecaster  = forecaster,
            y           = y,
            cv          = cv,
            lags_grid   = [2, 4],
            param_grid  = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            metric      = ['mean_absolute_error', mean_absolute_error],
            return_best = False,
            verbose     = False
        )


def test_output_evaluate_grid_hyperparameters_ForecasterAutoreg_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterRecursive with mocked
    (mocked done in Skforecast v0.4.3).
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
            refit                 = False,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]
    idx = len(lags_grid) * len(param_grid)

    results = _evaluate_grid_hyperparameters(
                  forecaster  = forecaster,
                  y           = y,
                  cv          = cv,
                  lags_grid   = lags_grid,
                  param_grid  = param_grid,
                  metric      = 'mean_squared_error',
                  return_best = False,
                  verbose     = False
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


def test_output_evaluate_grid_hyperparameters_ForecasterAutoreg_with_diferentiation_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterRecursive with mocked
    (mocked done in Skforecast v0.10.0) when differentiation is used.
    """
    forecaster = ForecasterRecursive(
                     regressor       = Ridge(random_state=123),
                     lags            = 2,
                     differentiation = 1
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = 1,
            refit                 = False,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False
         )

    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]
    idx = len(lags_grid) * len(param_grid)

    results = _evaluate_grid_hyperparameters(
                  forecaster  = forecaster,
                  y           = y,
                  cv          = cv,
                  lags_grid   = lags_grid,
                  param_grid  = param_grid,
                  metric      = 'mean_squared_error',
                  return_best = False,
                  verbose     = False
              ).reset_index(drop=True)

    expected_results = pd.DataFrame(
        {
            "lags": [[1, 2, 3, 4], [1, 2], [1, 2, 3, 4], [1, 2], [1, 2], [1, 2, 3, 4]],
            "lags_label": [
                [1, 2, 3, 4],
                [1, 2],
                [1, 2, 3, 4],
                [1, 2],
                [1, 2],
                [1, 2, 3, 4],
            ],
            "params": [
                {"alpha": 1},
                {"alpha": 1},
                {"alpha": 0.1},
                {"alpha": 0.1},
                {"alpha": 0.01},
                {"alpha": 0.01},
            ],
            "mean_squared_error": np.array(
                [0.09168123, 0.09300068, 0.09930084, 0.09960109, 0.10102995, 0.1012931]
            ),
            "alpha": np.array([1.0, 1.0, 0.1, 0.1, 0.01, 0.01]),
        },
        index=pd.RangeIndex(start=0, stop=idx, step=1),
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterAutoreg_lags_grid_dict_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterRecursive when 
    `lags_grid` is a dict with mocked (mocked done in Skforecast v0.4.3).
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
            refit                 = False,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False
         )
    lags_grid = {'lags_1': 2, 'lags_2': 4}
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]
    idx = len(lags_grid) * len(param_grid)

    results = _evaluate_grid_hyperparameters(
                  forecaster  = forecaster,
                  y           = y,
                  cv          = cv,
                  lags_grid   = lags_grid,
                  param_grid  = param_grid,
                  metric      = 'mean_squared_error',
                  return_best = False,
                  verbose     = False
              )

    expected_results = pd.DataFrame(
        {
            "lags": [[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            "lags_label": ["lags_1", "lags_1", "lags_1", "lags_2", "lags_2", "lags_2"],
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


def test_output_evaluate_grid_hyperparameters_ForecasterAutoreg_lags_grid_is_None_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterRecursive when 
    `lags_grid` is None with mocked (mocked done in Skforecast v0.4.3), 
    should use forecaster.lags as lags_grid.
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
            refit                 = False,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False
         )
    lags_grid = None
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters(
                  forecaster  = forecaster,
                  y           = y,
                  cv          = cv,
                  lags_grid   = lags_grid,
                  param_grid  = param_grid,
                  metric      = 'mean_squared_error',
                  return_best = False,
                  verbose     = False
              )

    expected_results = pd.DataFrame(
        {
            "lags": [[1, 2], [1, 2], [1, 2]],
            "lags_label": [[1, 2], [1, 2], [1, 2]],
            "params": [{"alpha": 0.01}, {"alpha": 0.1}, {"alpha": 1}],
            "mean_squared_error": np.array([0.06464646, 0.06502362, 0.06745534]),
            "alpha": np.array([0.01, 0.1, 1.0]),
        },
        index=pd.RangeIndex(start=0, stop=3, step=1),
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterAutoreg_metric_list_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterRecursive with mocked
    and multiple metrics (mocked done in Skforecast v0.4.3).
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
            refit                 = False,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False
         )
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]
    idx = len(lags_grid) * len(param_grid)

    results = _evaluate_grid_hyperparameters(
                  forecaster  = forecaster,
                  y           = y,
                  cv          = cv,
                  lags_grid   = lags_grid,
                  param_grid  = param_grid,
                  metric      = ['mean_squared_error', mean_absolute_error],
                  return_best = False,
                  verbose     = False
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
            "mean_absolute_error": np.array(
                [0.20278812, 0.20314819, 0.20519952, 0.20601567, 0.206323, 0.20747017]
            ),
            "alpha": np.array([0.01, 0.1, 1.0, 0.01, 0.1, 1.0]),
        },
        index=pd.RangeIndex(start=0, stop=idx, step=1),
    )

    pd.testing.assert_frame_equal(results, expected_results)


@pytest.mark.parametrize("lags_grid", 
                         [[2, 4], {'lags_1': 2, 'lags_2': 4}], 
                         ids=lambda lg: f'lags_grid: {lg}')
def test_evaluate_grid_hyperparameters_when_return_best_ForecasterAutoreg(lags_grid):
    """
    Test forecaster is refitted when return_best=True in 
    _evaluate_grid_hyperparameters with ForecasterRecursive.
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
            refit                 = False,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False
         )
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    _evaluate_grid_hyperparameters(
        forecaster  = forecaster,
        y           = y,
        cv          = cv,
        lags_grid   = lags_grid,
        param_grid  = param_grid,
        metric      = 'mean_squared_error',
        return_best = True,
        verbose     = False
    )
    
    expected_lags = np.array([1, 2])
    expected_alpha = 0.01
    
    assert (expected_lags == forecaster.lags).all()
    assert expected_alpha == forecaster.regressor.alpha


@pytest.mark.parametrize("lags_grid", 
                         [[2, 4], {'lags_1': 2, 'lags_2': 4}], 
                         ids=lambda lg: f'lags_grid: {lg}')
def test_evaluate_grid_hyperparameters_when_return_best_and_list_metrics(lags_grid):
    """
    Test forecaster is refitted when return_best=True in _evaluate_grid_hyperparameters
    and multiple metrics.
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
            refit                 = False,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False
         )
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    _evaluate_grid_hyperparameters(
        forecaster    = forecaster,
        y             = y,
        cv            = cv,
        lags_grid     = lags_grid,
        param_grid    = param_grid,
        metric        = [mean_absolute_percentage_error, 'mean_squared_error'],
        return_best   = True,
        verbose       = False,
        show_progress = False
    )
    
    expected_lags = np.array([1, 2])
    expected_alpha = 1.
    
    assert (expected_lags == forecaster.lags).all()
    assert expected_alpha == forecaster.regressor.alpha


def test_evaluate_grid_hyperparameters_output_file_when_single_metric():
    """
    Test output file is created when output_file is passed to
    _evaluate_grid_hyperparameters and single metric.
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
            refit                 = False,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False
         )
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]
    output_file = 'test_evaluate_grid_hyperparameters_output_file.txt'

    results = _evaluate_grid_hyperparameters(
                  forecaster    = forecaster,
                  y             = y,
                  cv            = cv,
                  lags_grid     = lags_grid,
                  param_grid    = param_grid,
                  metric        = 'mean_squared_error',
                  return_best   = False,
                  verbose       = False,
                  show_progress = False,
                  output_file   = output_file
              )
    results  = results.astype({'lags': str, 'lags_label': str, 'params': str})

    assert os.path.isfile(output_file)
    output_file_content = pd.read_csv(output_file, sep='\t', low_memory=False)
    output_file_content = output_file_content.sort_values(by='mean_squared_error')
    output_file_content = output_file_content.astype({'lags': str, 'lags_label': str, 'params': str})
    pd.testing.assert_frame_equal(results, output_file_content)
    os.remove(output_file)


def test_evaluate_grid_hyperparameters_output_file_when_single_metric_as_list():
    """ 
    Test output file is created when output_file is passed to 
    _evaluate_grid_hyperparameters and single metric as list.
    """
    forecaster = ForecasterRecursive(
                     regressor = Ridge(random_state=123),
                     lags      = 2 
                 )

    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]
    output_file = 'test_evaluate_grid_hyperparameters_output_file.txt'
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
            return_all_indexes    = False,
        )

    results = _evaluate_grid_hyperparameters(
                  forecaster         = forecaster,
                  y                  = y,
                  cv                 = cv,
                  lags_grid          = lags_grid,
                  param_grid         = param_grid,
                  metric             = ['mean_squared_error'],
                  return_best        = False,
                  verbose            = False,
                  show_progress      = False,
                  output_file        = output_file
              )
    results  = results.astype({'lags': str, 'lags_label': str, 'params': str})

    assert os.path.isfile(output_file)
    output_file_content = pd.read_csv(output_file, sep='\t', low_memory=False)
    output_file_content = output_file_content.sort_values(by='mean_squared_error')
    output_file_content = output_file_content.astype({'lags': str, 'lags_label': str, 'params': str})
    pd.testing.assert_frame_equal(results, output_file_content)
    os.remove(output_file)


def test_evaluate_grid_hyperparameters_output_file_when_2_metrics_as_list():
    """
    Test output file is created when output_file is passed to 
    _evaluate_grid_hyperparameters and 2 metrics as list.
    """
    forecaster = ForecasterRecursive(
                     regressor = Ridge(random_state=123),
                     lags      = 2 
                 )

    n_validation = 12
    y_train = y[:-n_validation]
    lags_grid = [2, 4]
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]
    output_file = 'test_evaluate_grid_hyperparameters_output_file.txt'

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
            return_all_indexes    = False,
        )

    results = _evaluate_grid_hyperparameters(
                  forecaster         = forecaster,
                  y                  = y,
                  cv                 = cv,
                  lags_grid          = lags_grid,
                  param_grid         = param_grid,
                  metric             = ['mean_squared_error', 'mean_absolute_error'],
                  return_best        = False,
                  verbose            = False,
                  show_progress      = False,
                  output_file        = output_file
              )
    results  = results.astype({'lags': str, 'lags_label': str, 'params': str})

    assert os.path.isfile(output_file)
    output_file_content = pd.read_csv(output_file, sep='\t', low_memory=False)
    output_file_content = output_file_content.sort_values(by='mean_squared_error')
    output_file_content = output_file_content.astype({'lags': str, 'lags_label': str, 'params': str})
    pd.testing.assert_frame_equal(results, output_file_content)
    os.remove(output_file)


forecasters = [
    ForecasterRecursive(regressor=Ridge(random_state=678), lags=3),
    ForecasterDirect(regressor=Ridge(random_state=678), lags=3, steps=1),
    ForecasterRecursive(
        regressor=Ridge(random_state=678),
        lags=3,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler()
    ),
    ForecasterDirect(
        regressor=Ridge(random_state=678),
        lags=3,
        steps=1,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler()
    )
]
@pytest.mark.parametrize("forecaster", forecasters)
def test_evaluate_grid_hyperparameters_equivalent_outputs_backtesting_one_step_ahead(
    forecaster,
):

    metrics = [
        "mean_absolute_error",
        "mean_squared_error",
        mean_absolute_percentage_error,
        mean_absolute_scaled_error,
        root_mean_squared_scaled_error,
    ]
    param_grid = {
        "alpha": np.logspace(-3, 3, 2),
    }
    lags_grid = [3, 5, 7]
    param_grid = list(ParameterGrid(param_grid))
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
    results_backtesting = _evaluate_grid_hyperparameters(
        forecaster         = forecaster,
        y                  = y_feature_selection,
        exog               = exog_feature_selection,
        cv                 = cv_backtesnting,
        param_grid         = param_grid,
        lags_grid          = lags_grid,
        metric             = metrics,
        return_best        = False,
        n_jobs             = 'auto',
        verbose            = False,
        show_progress      = False
    )
    results_one_step_ahead = _evaluate_grid_hyperparameters(
        forecaster         = forecaster,
        y                  = y_feature_selection,
        exog               = exog_feature_selection,
        cv                 = cv_one_step_ahead,
        param_grid         = param_grid,
        lags_grid          = lags_grid,
        metric             = metrics,
        return_best        = False,
        verbose            = False,
        show_progress      = False
    )

    pd.testing.assert_frame_equal(results_backtesting, results_one_step_ahead)
