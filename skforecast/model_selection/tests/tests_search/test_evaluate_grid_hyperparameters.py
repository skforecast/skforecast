# Unit test _evaluate_grid_hyperparameters
# ==============================================================================
import re
import os
import warnings
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from skforecast.metrics import mean_absolute_scaled_error, root_mean_squared_scaled_error
from skforecast.exceptions import (
    LongTrainingWarning, MissingValuesWarning, OneStepAheadValidationWarning
)
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection._search import _evaluate_grid_hyperparameters
from skforecast.model_selection._split import TimeSeriesFold, OneStepAheadFold
from skforecast.preprocessing import RollingFeatures

# Fixtures
from ..fixtures_model_selection import y
from ..fixtures_model_selection import y_feature_selection
from ..fixtures_model_selection import exog_feature_selection

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # hide progress bar


def test_TypeError_evaluate_grid_hyperparameters_when_cv_not_valid():
    """
    Test TypeError is raised in _evaluate_grid_hyperparameters when cv is not
    a valid splitter.
    """
    class DummyCV:
        pass

    cv = DummyCV()
    forecaster = ForecasterRecursive(
                     estimator = Ridge(random_state=123),
                     lags      = 2
                 )
    
    err_msg = re.escape(
        f"`cv` must be an instance of `TimeSeriesFold` or `OneStepAheadFold`. "
        f"Got {type(cv)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        _evaluate_grid_hyperparameters(
            forecaster  = forecaster,
            y           = y,
            cv          = cv,
            lags_grid   = [2, 4],
            param_grid  = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            metric      = 'mean_absolute_error',
            return_best = True,
            verbose     = False
        )


def test_ValueError_evaluate_grid_hyperparameters_when_return_best_and_len_y_exog_different():
    """
    Test ValueError is raised in _evaluate_grid_hyperparameters when return_best 
    and length of `y` and `exog` do not match.
    """
    forecaster = ForecasterRecursive(
                     estimator = Ridge(random_state=123),
                     lags      = 2
                 )
    exog_short = y[:30]
    
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y[:-12]),
            window_size           = None,
            refit                 = False,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    
    err_msg = re.escape(
        f"`exog` must have same number of samples as `y`. "
        f"length `exog`: ({len(exog_short)}), length `y`: ({len(y)})"
    )
    with pytest.raises(ValueError, match = err_msg):
        _evaluate_grid_hyperparameters(
            forecaster  = forecaster,
            y           = y,
            exog        = exog_short,
            cv          = cv,
            lags_grid   = [2, 4],
            param_grid  = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}],
            metric      = 'mean_absolute_error',
            return_best = True,
            verbose     = True
        )


def test_ValueError_evaluate_grid_hyperparameters_when_lags_grid_has_None_and_no_window_features():
    """
    Test ValueError is raised in _evaluate_grid_hyperparameters when `lags_grid`
    contains `None` and the forecaster has no window features.
    """
    forecaster = ForecasterRecursive(
                     estimator = Ridge(random_state=123),
                     lags      = 2
                 )
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=False)

    err_msg = re.escape(
        "`lags_grid` contains `None`, a candidate without lags, but the "
        "forecaster has no `window_features` to create the predictors. Set "
        "`window_features` in the forecaster or remove `None` from `lags_grid`."
    )
    with pytest.raises(ValueError, match=err_msg):
        _evaluate_grid_hyperparameters(
            forecaster  = forecaster,
            y           = y,
            cv          = cv,
            lags_grid   = [None],
            param_grid  = [{'alpha': 0.01}, {'alpha': 0.1}],
            metric      = 'mean_squared_error',
            return_best = False,
            verbose     = False
        )


def test_ValueError_evaluate_grid_hyperparameters_metric_list_duplicate_names():
    """
    Test ValueError is raised in _evaluate_grid_hyperparameters when a `list` of 
    metrics is used with duplicate names.
    """
    forecaster = ForecasterRecursive(
                     estimator = Ridge(random_state=123),
                     lags      = 2
                 )
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y[:-12]),
            window_size           = None,
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


def test_ValueError_evaluate_grid_hyperparameters_when_metric_named_n_evaluated():
    """
    Test ValueError is raised in _evaluate_grid_hyperparameters, before
    evaluating any candidate, when a metric is named `n_evaluated`, the column
    of the backtesting metrics with the number of points evaluated.
    """
    forecaster = ForecasterRecursive(estimator=Ridge(random_state=123), lags=2)
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y[:-12]), refit=False)

    def n_evaluated(y_true, y_pred):  # pragma: no cover
        return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

    err_msg = re.escape(
        "The backtesting metrics include the column 'n_evaluated' with the "
        "number of points evaluated. Rename the metric 'n_evaluated' passed in "
        "`metric`."
    )
    with pytest.raises(ValueError, match = err_msg):
        _evaluate_grid_hyperparameters(
            forecaster  = forecaster,
            y           = y,
            cv          = cv,
            lags_grid   = [2, 4],
            param_grid  = [{'alpha': 0.01}, {'alpha': 0.1}],
            metric      = ['mean_absolute_error', n_evaluated],
            return_best = False,
            verbose     = False
        )


def test_grid_hyperparameters_results_warn_when_non_valid_params():
    """
    Test that a single warning, grouped by error, is raised when non valid
    params are included in param_grid, and the candidate is skipped.
    """

    param_grid = {
        "alpha": [0.1],
        "l1_ratio": [0.5, 10],  # 10 is not valid for ElasticNet
    }
    param_grid = list(ParameterGrid(param_grid))
    cv = TimeSeriesFold(steps=12, initial_train_size=30, refit=False)
    forecaster = ForecasterRecursive(estimator=ElasticNet(), lags=5)

    msg = re.escape(
        "Parameters skipped: 1 of 2 candidates raises an error, so it is not in "
        "the results.\n"
        "\n"
        "InvalidParameterError: The 'l1_ratio' parameter of ElasticNet must be a "
        "float in the range [0.0, 1.0]. Got 10 instead.\n"
        "  - lags: [1, 2, 3, 4, 5], params: {'alpha': 0.1, 'l1_ratio': 10}"
    )
    with pytest.warns(RuntimeWarning, match=msg):
        results = _evaluate_grid_hyperparameters(
            forecaster=forecaster,
            y=y,
            param_grid=param_grid,
            cv=cv,
            metric="mean_squared_error",
            return_best=True,
            n_jobs="auto",
            verbose=True,
            show_progress=False,
        )

    expected_results = pd.DataFrame(
        {
            "lags": {0: np.array([1, 2, 3, 4, 5])},
            "lags_label": {0: np.array([1, 2, 3, 4, 5])},
            "params": {0: {"alpha": 0.1, "l1_ratio": 0.5}},
            "mean_squared_error": {0: 0.061178815481733415},
            "alpha": {0: 0.1},
            "l1_ratio": {0: 0.5},
        }
    )
    expected_results.insert(4, 'n_evaluated', [20] * 1)
    pd.testing.assert_frame_equal(results, expected_results)


def test_evaluate_grid_hyperparameters_warns_when_all_params_raise_exceptions():
    """
    Test that a RuntimeWarning is raised and an empty DataFrame is returned
    when all parameter combinations in param_grid are invalid and raise
    exceptions during evaluation.
    """
    # All l1_ratio values are invalid for ElasticNet (must be in [0.0, 1.0])
    param_grid = [
        {"alpha": 0.1, "l1_ratio": 10},
        {"alpha": 0.5, "l1_ratio": 20},
    ]
    cv = TimeSeriesFold(steps=12, initial_train_size=30, refit=False)
    forecaster = ForecasterRecursive(estimator=ElasticNet(), lags=5)

    warn_msg = re.escape(
        "Parameters skipped: 2 of 2 candidates raise an error, so the results "
        "are empty."
    )
    with pytest.warns(RuntimeWarning, match=warn_msg):
        results = _evaluate_grid_hyperparameters(
            forecaster    = forecaster,
            y             = y,
            param_grid    = param_grid,
            cv            = cv,
            metric        = "mean_squared_error",
            return_best   = False,
            n_jobs        = "auto",
            verbose       = False,
            show_progress = False,
        )

    assert results.empty
    assert isinstance(results, pd.DataFrame)


def test_evaluate_grid_hyperparameters_warns_when_all_params_raise_exceptions_OneStepAheadFold():
    """
    Test that a RuntimeWarning is raised and an empty DataFrame is returned
    when all parameter combinations raise exceptions with OneStepAheadFold.
    """
    param_grid = [
        {"alpha": 0.1, "l1_ratio": 10},
        {"alpha": 0.5, "l1_ratio": 20},
    ]
    cv = OneStepAheadFold(initial_train_size=30)
    forecaster = ForecasterRecursive(estimator=ElasticNet(), lags=5)

    warn_msg = re.escape(
        "Parameters skipped: 2 of 2 candidates raise an error, so the results "
        "are empty."
    )
    with pytest.warns(RuntimeWarning, match=warn_msg):
        results = _evaluate_grid_hyperparameters(
            forecaster    = forecaster,
            y             = y,
            param_grid    = param_grid,
            cv            = cv,
            metric        = "mean_squared_error",
            return_best   = False,
            verbose       = False,
            show_progress = False,
        )

    assert results.empty
    assert isinstance(results, pd.DataFrame)


def test_output_evaluate_grid_hyperparameters_ForecasterRecursive_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterRecursive with mocked
    (mocked done in Skforecast v0.4.3).
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

    expected_results.insert(4, 'n_evaluated', [12] * 6)
    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterRecursive_with_window_features():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterRecursive with 
    window features (mocked done in Skforecast v0.14.0).
    """
    window_features = RollingFeatures(
        stats = ['mean', 'std', 'min', 'max', 'sum', 'median', 'ratio_min_max', 'coef_variation'],
        window_sizes = 3,
    )
    forecaster = ForecasterRecursive(
                     estimator       = Ridge(random_state=123),
                     lags            = 2, 
                     window_features = window_features,
                 )
    
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
            refit                 = False,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    lags_grid = [2, 4]
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
        {'lags': [np.array([1, 2]),
                np.array([1, 2, 3, 4]),
                np.array([1, 2, 3, 4]),
                np.array([1, 2]),
                np.array([1, 2, 3, 4]),
                np.array([1, 2])],
        'lags_label': [np.array([1, 2]),
                np.array([1, 2, 3, 4]),
                np.array([1, 2, 3, 4]),
                np.array([1, 2]),
                np.array([1, 2, 3, 4]),
                np.array([1, 2])],
        'params': [{'alpha': 1},
                {'alpha': 1},
                {'alpha': 0.1},
                {'alpha': 0.1},
                {'alpha': 0.01},
                {'alpha': 0.01}],
        'mean_squared_error': [0.08284287269722487,
                0.0832323925238072,
                0.0908332414354649,
                0.09264625639393599,
                0.10408604464751903,
                0.10624605486182163],
        'alpha': [1.0, 1.0, 0.1, 0.1, 0.01, 0.01]}
    )

    expected_results.insert(4, 'n_evaluated', [12] * 6)
    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterRecursive_with_differentiation_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterRecursive with mocked
    (mocked done in Skforecast v0.10.0) when differentiation is used.
    """
    forecaster = ForecasterRecursive(
                     estimator       = Ridge(random_state=123),
                     lags            = 2,
                     differentiation = 1
                 )
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
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

    expected_results.insert(4, 'n_evaluated', [12] * 6)
    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterRecursive_lags_grid_dict_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterRecursive when 
    `lags_grid` is a dict with mocked (mocked done in Skforecast v0.4.3).
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

    expected_results.insert(4, 'n_evaluated', [12] * 6)
    pd.testing.assert_frame_equal(results, expected_results)


def test_output_evaluate_grid_hyperparameters_ForecasterRecursive_lags_grid_is_None_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterRecursive when 
    `lags_grid` is None with mocked (mocked done in Skforecast v0.4.3), 
    should use forecaster.lags as lags_grid.
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

    expected_results.insert(4, 'n_evaluated', [12] * 3)
    pd.testing.assert_frame_equal(results, expected_results)


@pytest.mark.parametrize(
    "cv, expected_metric",
    [(TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=False),
      np.array([0.07236308467323382, 0.07375229571270249, 0.07413626926165669])),
     (OneStepAheadFold(initial_train_size=len(y) - 12),
      np.array([0.07178242992279948, 0.0727094177919783, 0.07301468528616072]))],
    ids=['TimeSeriesFold', 'OneStepAheadFold']
)
@pytest.mark.parametrize("lags_grid",
                         [None, [None]],
                         ids=lambda lg: f'lags_grid: {lg}')
def test_output_evaluate_grid_hyperparameters_when_forecaster_lags_is_None(
    cv, expected_metric, lags_grid
):
    """
    Test output of _evaluate_grid_hyperparameters when the forecaster has
    `lags=None` and only window features: `lags_grid=None` uses the lags of
    the forecaster and `lags_grid=[None]` evaluates the forecaster without lags.
    With `return_best=True`, the forecaster is refitted without lags.
    """
    forecaster = ForecasterRecursive(
                     estimator       = Ridge(random_state=123),
                     lags            = None,
                     window_features = RollingFeatures(stats=['mean'], window_sizes=3)
                 )
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters(
                  forecaster  = forecaster,
                  y           = y,
                  cv          = cv,
                  lags_grid   = lags_grid,
                  param_grid  = param_grid,
                  metric      = 'mean_squared_error',
                  return_best = True,
                  verbose     = False
              )

    expected_results = pd.DataFrame(
        {
            "lags": [None, None, None],
            "lags_label": [None, None, None],
            "params": [{"alpha": 1}, {"alpha": 0.1}, {"alpha": 0.01}],
            "mean_squared_error": expected_metric,
            "alpha": np.array([1.0, 0.1, 0.01]),
        },
        index=pd.RangeIndex(start=0, stop=3, step=1),
    )

    expected_results.insert(4, 'n_evaluated', [12] * 3)
    pd.testing.assert_frame_equal(results, expected_results)
    assert forecaster.lags is None
    assert forecaster.estimator.alpha == 1
    assert forecaster.is_fitted


def test_evaluate_grid_hyperparameters_when_lags_grid_has_None_and_return_best():
    """
    Test that `None` in `lags_grid` evaluates the forecaster without lags (only
    window features) next to the other candidates, and that the forecaster is
    refitted with the best lags when `return_best=True`.
    """
    forecaster = ForecasterDirect(
                     estimator       = Ridge(random_state=123),
                     steps           = 3,
                     lags            = None,
                     window_features = RollingFeatures(stats=['mean'], window_sizes=3)
                 )
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=False)
    param_grid = [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}]

    results = _evaluate_grid_hyperparameters(
                  forecaster  = forecaster,
                  y           = y,
                  cv          = cv,
                  lags_grid   = [None, 2],
                  param_grid  = param_grid,
                  metric      = 'mean_squared_error',
                  return_best = True,
                  verbose     = False
              )

    expected_lags = [[1, 2], [1, 2], [1, 2], None, None, None]
    expected_params = [
        {'alpha': 1}, {'alpha': 0.1}, {'alpha': 0.01},
        {'alpha': 1}, {'alpha': 0.1}, {'alpha': 0.01}
    ]
    expected_metric = np.array(
        [0.069079, 0.069753, 0.072559, 0.073967, 0.076073, 0.076646]
    )

    results_lags = [
        None if lags is None else lags.tolist() for lags in results['lags']
    ]
    results_lags_label = [
        None if lags is None else lags.tolist() for lags in results['lags_label']
    ]
    assert results_lags == expected_lags
    assert results_lags_label == expected_lags
    assert results['params'].to_list() == expected_params
    np.testing.assert_array_almost_equal(
        results['mean_squared_error'].to_numpy(), expected_metric
    )
    np.testing.assert_array_equal(forecaster.lags, np.array([1, 2]))
    for i in forecaster.steps:
        assert forecaster.estimators_[i].alpha == 1
    assert forecaster.is_fitted


def test_output_evaluate_grid_hyperparameters_ForecasterRecursive_metric_list_with_mocked():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterRecursive with mocked
    and multiple metrics (mocked done in Skforecast v0.4.3).
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

    expected_results.insert(5, 'n_evaluated', [12] * 6)
    pd.testing.assert_frame_equal(results, expected_results)


@pytest.mark.parametrize("lags_grid", 
                         [[2, 4], {'lags_1': 2, 'lags_2': 4}], 
                         ids=lambda lg: f'lags_grid: {lg}')
def test_evaluate_grid_hyperparameters_when_return_best_ForecasterRecursive(lags_grid):
    """
    Test forecaster is refitted when return_best=True in 
    _evaluate_grid_hyperparameters with ForecasterRecursive.
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
        verbose     = True
    )
    
    expected_lags = np.array([1, 2])
    expected_alpha = 0.01
    
    assert (expected_lags == forecaster.lags).all()
    assert expected_alpha == forecaster.estimator.alpha


@pytest.mark.parametrize("lags_grid", 
                         [[2, 4], {'lags_1': 2, 'lags_2': 4}], 
                         ids=lambda lg: f'lags_grid: {lg}')
def test_evaluate_grid_hyperparameters_when_return_best_and_list_metrics(lags_grid):
    """
    Test forecaster is refitted when return_best=True in _evaluate_grid_hyperparameters
    and multiple metrics.
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
    assert expected_alpha == forecaster.estimator.alpha


def test_evaluate_grid_hyperparameters_output_file_when_single_metric():
    """
    Test output file is created when output_file is passed to
    _evaluate_grid_hyperparameters and single metric.
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
                     estimator = Ridge(random_state=123),
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
                     estimator = Ridge(random_state=123),
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


def test_output_evaluate_grid_hyperparameters_ForecasterDirect_with_window_features():
    """
    Test output of _evaluate_grid_hyperparameters in ForecasterDirect with 
    window features (mocked done in Skforecast v0.14.0).
    """
    window_features = RollingFeatures(
        stats = ['mean', 'std', 'min', 'max', 'sum', 'median', 'ratio_min_max', 'coef_variation'],
        window_sizes = 3,
    )
    forecaster = ForecasterDirect(
                     estimator       = Ridge(random_state=123),
                     steps           = 3,
                     lags            = 2, 
                     window_features = window_features,
                 )
    
    n_validation = 12
    y_train = y[:-n_validation]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
            refit                 = False,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    lags_grid = [2, 4]
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
        {'lags': [np.array([1, 2, 3, 4]),
                np.array([1, 2]),
                np.array([1, 2, 3, 4]),
                np.array([1, 2]),
                np.array([1, 2, 3, 4]),
                np.array([1, 2])],
        'lags_label': [np.array([1, 2, 3, 4]),
                np.array([1, 2]),
                np.array([1, 2, 3, 4]),
                np.array([1, 2]),
                np.array([1, 2, 3, 4]),
                np.array([1, 2])],
        'params': [{'alpha': 1},
                {'alpha': 1},
                {'alpha': 0.1},
                {'alpha': 0.1},
                {'alpha': 0.01},
                {'alpha': 0.01}],
        'mean_squared_error': [0.07424336814199613,
                0.07576112822191301,
                0.07677225532446876,
                0.08046370393440734,
                0.08411584414110274,
                0.08911485459058603],
        'alpha': [1.0, 1.0, 0.1, 0.1, 0.01, 0.01]}
    )

    expected_results.insert(4, 'n_evaluated', [12] * 6)
    pd.testing.assert_frame_equal(results, expected_results)


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
def test_evaluate_grid_hyperparameters_equivalent_outputs_backtesting_one_step_ahead(
    forecaster, initial_train_size
):
    """
    Test that the outputs of _evaluate_grid_hyperparameters are equivalent when
    using backtesting and one-step-ahead.
    """
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
    lags_grid = [3, 7]
    param_grid = list(ParameterGrid(param_grid))
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

    warn_msg = re.escape(
        "One-step-ahead predictions are used for faster model comparison, but they "
        "may not fully represent multi-step prediction performance. It is recommended "
        "to backtest the final model for a more accurate multi-step performance "
        "estimate."
    )
    with pytest.warns(OneStepAheadValidationWarning, match = warn_msg):
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


def test_evaluate_grid_hyperparameters_metrics_equal_to_backtesting_by_name():
    """
    Test that the metrics of each candidate are those of its backtesting,
    selected by name, followed by the column `n_evaluated` of the backtesting,
    and that the output file has the same columns.
    """
    forecaster = ForecasterRecursive(estimator=Ridge(random_state=123), lags=3)
    cv = TimeSeriesFold(steps=5, initial_train_size=len(y) - 12, refit=False)
    metric = ['mean_absolute_error', mean_absolute_scaled_error]
    param_grid = [{'alpha': 0.1}, {'alpha': 10}]
    output_file = 'test_evaluate_grid_hyperparameters_metrics_by_name.txt'

    results = _evaluate_grid_hyperparameters(
                  forecaster    = forecaster,
                  y             = y,
                  cv            = cv,
                  param_grid    = param_grid,
                  metric        = metric,
                  return_best   = False,
                  verbose       = False,
                  show_progress = False,
                  output_file   = output_file
              )
    output_file_header = pd.read_csv(output_file, sep='\t', nrows=0).columns.tolist()
    os.remove(output_file)

    assert output_file_header == [
        'lags', 'lags_label', 'params', 'mean_absolute_error',
        'mean_absolute_scaled_error', 'n_evaluated', 'alpha'
    ]
    assert results.columns.tolist() == output_file_header
    for _, row in results.iterrows():
        forecaster.set_params(row['params'])
        metric_backtesting, _ = backtesting_forecaster(
            forecaster=forecaster, y=y, cv=cv, metric=metric, show_progress=False
        )
        for name in ['mean_absolute_error', 'mean_absolute_scaled_error', 'n_evaluated']:
            assert row[name] == metric_backtesting.loc[0, name]


def test_evaluate_grid_hyperparameters_LongTrainingWarning_once_with_fits_of_the_search():
    """
    Test that LongTrainingWarning is raised once, with the number of fits of
    each candidate and of the whole search, instead of once per candidate.
    """
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=2)
    cv = TimeSeriesFold(steps=1, initial_train_size=440, refit=True)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _evaluate_grid_hyperparameters(
            forecaster    = forecaster,
            y             = y_feature_selection,
            cv            = cv,
            param_grid    = [{'alpha': 0.1}, {'alpha': 1.0}],
            metric        = 'mean_absolute_error',
            lags_grid     = [2, 3, 4],
            return_best   = False,
            verbose       = False,
            show_progress = False
        )

    messages = [
        w.message.message for w in record if w.category is LongTrainingWarning
    ]
    assert messages == [
        "Each candidate fits the forecaster about 60 times, so the search (6 "
        "candidates) fits it about 360 times. This can take substantial amounts "
        "of time. If not feasible, try with `refit = False`."
    ]


def test_evaluate_grid_hyperparameters_OneStepAheadFold_skips_candidates_when_split_fails():
    """
    Test that, with a `OneStepAheadFold`, the candidates whose lags leave no
    data to train with (`window_size` not smaller than `initial_train_size`)
    are skipped with the warning of the candidates skipped, instead of
    aborting the search.
    """
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=2)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results = _evaluate_grid_hyperparameters(
                      forecaster    = forecaster,
                      y             = y,
                      cv            = OneStepAheadFold(initial_train_size=30),
                      param_grid    = [{'alpha': 0.1}, {'alpha': 1}],
                      metric        = 'mean_absolute_error',
                      lags_grid     = [2, 40],
                      return_best   = True,
                      verbose       = False,
                      show_progress = False
                  )

    assert results['lags_label'].map(len).tolist() == [2, 2]
    assert forecaster.max_lag == 2
    messages = [str(w.message) for w in record if w.category is RuntimeWarning]
    assert messages == [
        "Parameters skipped: 2 of 4 candidates raise an error, so they are not in "
        "the results.\n\nValueError: `initial_train_size` must be greater than the "
        "`window_size` of the forecaster (40), so that the first training set has "
        "data to train with. Got 30.\n  - lags: [1, 2, ..., 40] (2 candidates)"
    ]


def test_evaluate_grid_hyperparameters_OneStepAheadFold_window_size_of_forecaster_not_checked():
    """
    Test that, with a `OneStepAheadFold`, the `window_size` of the forecaster
    passed (70 lags) is not checked against `initial_train_size` (60) when
    `lags_grid` does not use it: the search evaluates the candidates of
    `lags_grid` (3 lags) without errors, as with a `TimeSeriesFold`.
    """
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=70)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results = _evaluate_grid_hyperparameters(
                      forecaster    = forecaster,
                      y             = y_feature_selection.iloc[:80],
                      cv            = OneStepAheadFold(initial_train_size=60),
                      param_grid    = [{'alpha': 0.1}, {'alpha': 1}],
                      metric        = 'mean_absolute_error',
                      lags_grid     = [3],
                      return_best   = True,
                      verbose       = False,
                      show_progress = False
                  )

    assert len(results) == 2
    assert results['lags'].map(len).tolist() == [3, 3]
    assert forecaster.max_lag == 3
    assert [w for w in record if w.category is RuntimeWarning] == []


def test_evaluate_grid_hyperparameters_MissingValuesWarning_n_evaluated_single_series():
    """
    Test that, with a `OneStepAheadFold`, `dropna_from_series=True` and NaN in
    `y`, the candidates with larger windows are evaluated on fewer points and
    a MissingValuesWarning is raised once at the end of the search, with the
    cause and the action for a single series.
    """
    y_nan = y.copy()
    y_nan.iloc[[35, 40, 45]] = np.nan
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=2, dropna_from_series=True)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results = _evaluate_grid_hyperparameters(
                      forecaster    = forecaster,
                      y             = y_nan,
                      cv            = OneStepAheadFold(initial_train_size=30),
                      param_grid    = [{'alpha': 0.1}],
                      metric        = 'mean_absolute_error',
                      lags_grid     = [2, 10],
                      return_best   = False,
                      verbose       = False,
                      show_progress = False
                  )

    assert results['n_evaluated'].tolist() == [11, 5]
    messages = [
        w.message.message for w in record
        if w.category is MissingValuesWarning and "comparable" in str(w.message)
    ]
    assert messages == [
        "1 of 2 candidates is evaluated on fewer points than the most complete "
        "one, so its metrics are not comparable. It is not the best candidate of "
        "the ranking. The rows with missing values in `y` or in its lags are "
        "dropped (`dropna_from_series=True`), more with larger windows. To "
        "compare the candidates on the same points, fill the gaps of `y` or "
        "remove the candidates with larger windows. See the column "
        "`n_evaluated` of the results.\n\nCandidates with less than 95% of the "
        "11 points of the most complete one:\n"
        "  - lags: [1, 2, ..., 10], row 1: 5 (45.5%)"
    ]
