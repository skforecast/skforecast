# Unit test grid_search_forecaster
# ==============================================================================
import warnings
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from joblib import Parallel, cpu_count
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score, zero_one_loss
from skforecast.metrics import mean_absolute_scaled_error
from skforecast.metrics import root_mean_squared_scaled_error
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveClassifier
from skforecast.direct import ForecasterDirect
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection._search import grid_search_forecaster
from skforecast.model_selection._split import TimeSeriesFold
from skforecast.model_selection._split import OneStepAheadFold
from skforecast.exceptions import MissingValuesWarning

from tqdm import tqdm
from functools import partial, partialmethod

# Fixtures
from ..fixtures_model_selection import y, y_clf, exog

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # hide progress bar

 
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

    expected_results.insert(4, 'n_evaluated', [12] * 6)
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
    ForecasterRecursive(
        estimator=Ridge(random_state=678),
        lags=3,
        transformer_y=StandardScaler(),
        differentiation=1,
    ),
    ForecasterRecursive(estimator=Ridge(random_state=678), lags=3, differentiation=2),
    ForecasterDirect(
        estimator=Ridge(random_state=678),
        lags=3,
        steps=1,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
        differentiation=1,
    ),
]
@pytest.mark.parametrize("forecaster", forecasters)
def test_grid_search_forecaster_equivalence_backtesting_one_step_ahead(
    forecaster,
):
    """
    Test that output of grid_search_forecaster is equivalent with backtesting and
    one step ahead one step=1, fixed_train_size=True retran=False. With
    differentiation, each one-step-ahead prediction is reverted from the
    observed values, as in the backtesting.
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


@pytest.mark.parametrize(
    "cv_class", [TimeSeriesFold, OneStepAheadFold], ids=lambda cv: cv.__name__
)
def test_grid_search_forecaster_cv_differentiation_is_deprecated_and_ignored(cv_class):
    """
    Test that `differentiation` of the cv is deprecated and has no effect in
    grid_search_forecaster: a value different from the one of the forecaster
    raises a FutureWarning and gives the same results as `None`. The forecaster
    is refitted and the cv is not modified.
    """
    kwargs = {'steps': 4, 'refit': False} if cv_class is TimeSeriesFold else {}
    cv = cv_class(initial_train_size=30, verbose=False, **kwargs)
    with pytest.warns(FutureWarning, match="`differentiation` is deprecated"):
        cv_explicit = cv_class(
            initial_train_size=30, differentiation=2, verbose=False, **kwargs
        )
    param_grid = {'alpha': [0.01, 1]}
    lags_grid = [2, 4]

    forecaster = ForecasterRecursive(estimator=Ridge(), lags=2, differentiation=1)
    results = grid_search_forecaster(
        forecaster=forecaster, y=y, cv=cv, param_grid=param_grid,
        lags_grid=lags_grid, metric='mean_absolute_error', return_best=True,
        show_progress=False
    )
    forecaster_explicit = ForecasterRecursive(
        estimator=Ridge(), lags=2, differentiation=1
    )
    results_explicit = grid_search_forecaster(
        forecaster=forecaster_explicit, y=y, cv=cv_explicit, param_grid=param_grid,
        lags_grid=lags_grid, metric='mean_absolute_error', return_best=True,
        show_progress=False
    )

    pd.testing.assert_frame_equal(results, results_explicit)
    assert len(results) == 4
    assert forecaster.is_fitted
    assert cv.differentiation is None
    assert cv.window_size is None


def r2_custom(y_true, y_pred):
    """
    Custom metric without a known direction.
    """
    return r2_score(y_true, y_pred)


def r2_custom_greater_is_better(y_true, y_pred, y_train=None):
    """
    Custom metric with the attribute `greater_is_better`.
    """
    return r2_score(y_true, y_pred)


r2_custom_greater_is_better.greater_is_better = True


@pytest.mark.parametrize(
    "cv",
    [
        TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=False),
        OneStepAheadFold(initial_train_size=len(y) - 12),
    ],
    ids=lambda cv: f"cv: {type(cv).__name__}",
)
@pytest.mark.parametrize(
    "metric",
    [r2_score, partial(r2_score), r2_custom_greater_is_better],
    ids=["r2_score", "partial", "greater_is_better"],
)
def test_grid_search_forecaster_maximizes_metric_when_greater_is_better(cv, metric):
    """
    Test that, with a metric in which greater values are better, the results
    are sorted in descending order and `return_best` refits the forecaster
    with the candidate with the greatest value. Before the fix, the order
    depended only on the task of the forecaster and the worst candidate was
    selected.
    """
    forecaster = ForecasterRecursive(estimator=Ridge(random_state=123), lags=2)
    results = grid_search_forecaster(
        forecaster=forecaster,
        y=y,
        cv=cv,
        param_grid={"alpha": [0.01, 1, 100]},
        lags_grid=[2, 4],
        metric=[metric, "mean_absolute_error"],
        return_best=True,
    )

    metric_name = results.columns[3]
    assert results[metric_name].nunique() > 1
    assert results[metric_name].is_monotonic_decreasing
    assert forecaster.estimator.alpha == results.loc[0, "alpha"]
    np.testing.assert_array_equal(forecaster.lags, results.loc[0, "lags"])


@pytest.mark.parametrize(
    "metric",
    [r2_custom, lambda y_true, y_pred: r2_score(y_true, y_pred)],
    ids=["custom", "lambda"],
)
def test_grid_search_forecaster_minimizes_metric_when_direction_unknown(metric):
    """
    Test that a custom metric without a known direction is minimized in a
    regression task, as before the fix.
    """
    forecaster = ForecasterRecursive(estimator=Ridge(random_state=123), lags=2)
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=False)
    results = grid_search_forecaster(
        forecaster=forecaster,
        y=y,
        cv=cv,
        param_grid={"alpha": [0.01, 1, 100]},
        lags_grid=[2, 4],
        metric=metric,
        return_best=True,
    )

    metric_name = results.columns[3]
    assert results[metric_name].nunique() > 1
    assert results[metric_name].is_monotonic_increasing
    assert forecaster.estimator.alpha == results.loc[0, "alpha"]


@pytest.mark.parametrize(
    "metric, greater_is_better",
    [(zero_one_loss, False), ("accuracy_score", True)],
    ids=["zero_one_loss", "accuracy_score"],
)
def test_grid_search_forecaster_ForecasterRecursiveClassifier_follows_metric_direction(
    metric, greater_is_better
):
    """
    Test that a classification forecaster minimizes a loss (`zero_one_loss`)
    and maximizes a score (`accuracy_score`). Before the fix, every metric was
    maximized in classification tasks.
    """
    forecaster = ForecasterRecursiveClassifier(
        estimator=LogisticRegression(max_iter=1000), lags=2
    )
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y_clf) - 12, refit=False)
    results = grid_search_forecaster(
        forecaster=forecaster,
        y=y_clf,
        cv=cv,
        param_grid={"C": [1e-4, 0.1, 10]},
        lags_grid=[2, 4],
        metric=metric,
        return_best=True,
    )

    metric_name = results.columns[3]
    if greater_is_better:
        assert results[metric_name].is_monotonic_decreasing
    else:
        assert results[metric_name].is_monotonic_increasing
    assert results[metric_name].iloc[0] != results[metric_name].iloc[1]
    assert forecaster.estimator.C == 0.1
    np.testing.assert_array_equal(forecaster.lags, np.array([1, 2, 3, 4]))


@pytest.mark.parametrize(
    "estimator, refit, n_jobs, expected_n_jobs",
    [(Ridge(), True, 'auto', 1),
     (HistGradientBoostingRegressor(max_iter=5), True, 'auto', max(1, cpu_count() - 1)),
     (HistGradientBoostingRegressor(max_iter=5), False, 'auto', max(1, cpu_count() - 1)),
     (DecisionTreeRegressor(max_depth=2), False, 'auto', 1),
     (HistGradientBoostingRegressor(max_iter=5), True, 2, 2)],
    ids=[
        'auto, Ridge', 'auto, HistGradientBoostingRegressor',
        'auto, HistGradientBoostingRegressor, no refit',
        'auto, DecisionTreeRegressor, no refit', 'int'
    ]
)
def test_grid_search_forecaster_n_jobs_auto_resolved_with_the_rules_of_each_candidate(
    estimator, refit, n_jobs, expected_n_jobs
):
    """
    Test that grid_search_forecaster resolves `n_jobs='auto'` with
    `select_n_jobs_backtesting` for each candidate and passes the integer to
    `backtesting_forecaster`, so the work threshold of a single backtesting does not
    reduce it. Without refit, the rules give one job, except with
    `HistGradientBoostingRegressor`. `Parallel` is mocked with a
    sequential one, so no workers are started.
    """
    forecaster = ForecasterRecursive(estimator=estimator, lags=3)
    cv = TimeSeriesFold(
             steps              = 10,
             initial_train_size = 10,
             refit              = refit,
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
        grid_search_forecaster(
            forecaster    = forecaster,
            y             = y,
            cv            = cv,
            param_grid    = {'random_state': [1, 2]},
            metric        = 'mean_absolute_error',
            return_best   = False,
            n_jobs        = n_jobs,
            show_progress = False
        )

    assert mock_backtesting.call_count == 2
    for call in mock_backtesting.call_args_list:
        assert call.kwargs['n_jobs'] == expected_n_jobs
    for call in mock_parallel.call_args_list:
        assert call.kwargs == {'n_jobs': expected_n_jobs}


def test_grid_search_forecaster_warnings_of_folds_in_processes_same_as_n_jobs_1():
    """
    Test that the warnings raised in the folds of the backtesting of each
    candidate that run in worker processes (`n_jobs=2`) are collected by the
    search and shown once at the end, as with `n_jobs=1`, instead of once per
    fold and candidate by the worker processes.
    """
    exog_nan = exog.copy()
    exog_nan.iloc[[42, 47]] = np.nan

    def warnings_shown(n_jobs):
        forecaster = ForecasterRecursive(
            HistGradientBoostingRegressor(max_iter=5, random_state=123), lags=3
        )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            grid_search_forecaster(
                forecaster    = forecaster,
                y             = y,
                exog          = exog_nan,
                cv            = TimeSeriesFold(steps=2, initial_train_size=40, refit=False),
                param_grid    = {'learning_rate': [0.1, 0.2]},
                lags_grid     = [2, 3],
                metric        = 'mean_absolute_error',
                return_best   = False,
                n_jobs        = n_jobs,
                show_progress = False,
            )

        return [(w.category, str(w.message), w.filename, w.lineno) for w in record]

    expected = warnings_shown(n_jobs=1)
    results = warnings_shown(n_jobs=2)

    assert [category for category, _, _, _ in expected].count(MissingValuesWarning) == 1
    assert results == expected


class RidgeWarnsThenFailsLargeAlpha(Ridge):
    """
    Ridge that, with `alpha` greater than 0.5 and when it predicts one row (the
    steps of a recursive forecast), warns and then raises an error. It is not a
    class of scikit-learn, so the forecaster calls its `predict`.
    """

    def predict(self, X):
        if len(X) == 1 and self.alpha > 0.5:
            warnings.warn("Warning before the error of the fold.", UserWarning)
            raise ValueError("Error of the fold.")
        return super().predict(X)


def test_grid_search_forecaster_warning_of_fold_that_fails_in_process_kept_with_candidate():
    """
    Test that, when a fold of a candidate that runs in a worker process warns
    and then raises an error, the candidate is skipped and its warning is
    collected with it ("in 1 of 2 candidates"), as with `n_jobs=1`.
    """
    def search(n_jobs):
        forecaster = ForecasterRecursive(RidgeWarnsThenFailsLargeAlpha(), lags=3)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            results = grid_search_forecaster(
                forecaster    = forecaster,
                y             = y,
                cv            = TimeSeriesFold(steps=2, initial_train_size=40, refit=False),
                param_grid    = {'alpha': [0.1, 1.0]},
                metric        = 'mean_absolute_error',
                return_best   = False,
                n_jobs        = n_jobs,
                show_progress = False,
            )

        warnings_shown = [(w.category, str(w.message), w.filename, w.lineno) for w in record]

        return results['params'].tolist(), warnings_shown

    expected = search(n_jobs=1)
    results = search(n_jobs=2)

    assert expected[0] == [{'alpha': 0.1}]
    assert expected[1][0][:2] == (
        UserWarning, "Warning before the error of the fold (in 1 of 2 candidates)."
    )
    assert expected[1][1][0] is RuntimeWarning
    assert "ValueError: Error of the fold." in expected[1][1][1]
    assert results == expected
