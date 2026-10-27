# Unit test grid_search_forecaster
# ==============================================================================
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score, zero_one_loss
from skforecast.metrics import mean_absolute_scaled_error
from skforecast.metrics import root_mean_squared_scaled_error
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveClassifier
from skforecast.direct import ForecasterDirect
from skforecast.model_selection._search import grid_search_forecaster
from skforecast.model_selection._split import TimeSeriesFold
from skforecast.model_selection._split import OneStepAheadFold

from tqdm import tqdm
from functools import partial, partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # hide progress bar

# Fixtures
from ..fixtures_model_selection import y, y_clf

 
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
