# Unit test backtesting_forecaster
# ==============================================================================
import re
import sys
import warnings
import pytest
from unittest.mock import patch
import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, parallel_config
from scipy.stats import norm
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
from skforecast.recursive import ForecasterRecursive
from skforecast.recursive import ForecasterRecursiveClassifier
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirect
from skforecast.metrics import (
    mean_absolute_scaled_error,
    root_mean_squared_scaled_error,
)
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection._split import TimeSeriesFold
from skforecast.model_selection._validation import _fit_predict_forecaster
from skforecast.exceptions import MissingValuesWarning

# Fixtures
from ..fixtures_model_selection import y, y_clf, exog


def test_backtesting_forecaster_TypeError_when_forecaster_not_supported_types():
    """
    Test TypeError is raised in backtesting_forecaster if Forecaster is not one 
    of the types supported by the function.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                    estimator = Ridge(random_state=123),
                    lags      = 2
                 )
    forecaters_allowed = [
        'ForecasterRecursive', 
        'ForecasterDirect',
        'ForecasterEquivalentDate',
        'ForecasterRecursiveClassifier'
    ]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y[:-12]),
            window_size           = None,
            refit                 = False,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    err_msg = re.escape(
        f"`forecaster` must be of type {forecaters_allowed}. For all other "
        f"types of forecasters use the other functions available in the "
        f"`model_selection` module."
    )
    
    with pytest.raises(TypeError, match = err_msg):
        backtesting_forecaster(
            forecaster            = forecaster,
            y                     = y,
            cv                    = cv,
            metric                = 'mean_absolute_error',
            exog                  = None,
            verbose               = False,
            show_progress         = False
        )


@pytest.mark.parametrize(
    "interval",
    [None, [0.1, 0.3, 0.9], [0.05, 0.1, 0.9, 0.95], 'bootstrapping', norm],
    ids = lambda interval: f'interval: {interval}'
)
def test_backtesting_forecaster_no_interval_metrics_when_interval_has_not_two_bounds(interval):
    """
    Test that the interval metrics (`coverage`, `interval_width`) are not added 
    to the metrics when `interval` is None, a list of quantiles that is not 2
    quantiles (optionally with the median), 'bootstrapping' or a scipy.stats
    distribution.
    """
    forecaster = ForecasterRecursive(
                     estimator = Ridge(random_state=123),
                     lags      = 2
                 )
    cv = TimeSeriesFold(
             steps              = 3,
             initial_train_size = len(y[:-12]),
             refit              = False
         )

    metric, _ = backtesting_forecaster(
                    forecaster    = forecaster,
                    y             = y,
                    cv            = cv,
                    metric        = 'mean_absolute_error',
                    interval      = interval,
                    n_boot        = 10,
                    show_progress = False
                )

    assert list(metric.columns) == ['mean_absolute_error', 'n_evaluated']


def test_backtesting_forecaster_cv_differentiation_is_deprecated_and_ignored():
    """
    Test that `differentiation` of the cv is deprecated and has no effect: a
    value different from the one of the forecaster raises a FutureWarning and
    gives the same predictions as `None`. Backtesting does not raise the
    warning, the same cv object works with forecasters with different
    `differentiation` and it is not modified.
    """
    forecaster_diff = ForecasterRecursive(estimator=Ridge(), lags=3, differentiation=1)
    forecaster_no_diff = ForecasterRecursive(estimator=Ridge(), lags=3)
    cv = TimeSeriesFold(steps=4, initial_train_size=30, refit=True, verbose=False)
    warn_msg = re.escape("`differentiation` is deprecated since version 0.27.0")
    with pytest.warns(FutureWarning, match=warn_msg):
        cv_explicit = TimeSeriesFold(
            steps=4, initial_train_size=30, refit=True, differentiation=2,
            verbose=False
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=warn_msg, category=FutureWarning)
        metric, predictions = backtesting_forecaster(
            forecaster=forecaster_diff, y=y, cv=cv, metric='mean_absolute_error',
            show_progress=False
        )
        metric_no_diff, _ = backtesting_forecaster(
            forecaster=forecaster_no_diff, y=y, cv=cv, metric='mean_absolute_error',
            show_progress=False
        )
    metric_explicit, predictions_explicit = backtesting_forecaster(
        forecaster=forecaster_diff, y=y, cv=cv_explicit, metric='mean_absolute_error',
        show_progress=False
    )

    pd.testing.assert_frame_equal(metric, metric_explicit)
    pd.testing.assert_frame_equal(predictions, predictions_explicit)
    assert not metric.equals(metric_no_diff)
    assert cv.differentiation is None
    assert cv.window_size is None


# Series with a missing value in `exog` at position 41: with `gap=2`, the first
# fold predicts positions 40 to 46 and scores 42 to 46, so 41 is a gap row.
rng_gap = np.random.default_rng(0)
idx_gap = pd.date_range("2020-01-01", periods=60, freq="D")
y_gap = pd.Series(rng_gap.normal(size=60).cumsum(), index=idx_gap, name="y")
exog_gap = pd.DataFrame({"x": rng_gap.normal(size=60)}, index=idx_gap)
exog_gap.iloc[41, 0] = np.nan

nan_predictions_msg = (
    "5 backtesting predictions are NaN at dates with observed values (first: "
    "2020-02-12 00:00:00), so the metrics cannot be calculated. The most likely "
    "cause is a missing value in the series or in `exog`, used as a predictor "
    "of these dates, with an estimator that does not accept NaN (for example, "
    "a linear model). Fill the missing values, or use an estimator that "
    "accepts them, for example HistGradientBoostingRegressor or LGBMRegressor."
)


def nan_predictions_differentiation_msg(n_predictions, first_date):
    return (
        f"{n_predictions} backtesting predictions are NaN at dates with observed "
        f"values (first: {first_date}), so the metrics cannot be calculated. The "
        f"most likely causes are a missing value in the series or in `exog`, "
        f"used as a predictor of these dates, with an estimator that does not "
        f"accept NaN (for example, a linear model), or a missing last value of "
        f"the series before the fold, from which the differentiation is "
        f"reverted (with any estimator). Fill the missing values, or, in the "
        f"first case, use an estimator that accepts them, for example "
        f"HistGradientBoostingRegressor or LGBMRegressor."
    )


@pytest.mark.parametrize(
    "forecaster, expected_metric",
    [(ForecasterRecursive(Ridge(), lags=[10]), 7.409637384641371),
     (ForecasterDirect(Ridge(), lags=[10], steps=7), 8.101181293781048)],
    ids=lambda x: f"{type(x).__name__}"
)
def test_backtesting_forecaster_output_when_exog_has_NaN_only_in_gap_steps(
    forecaster, expected_metric
):
    """
    Test backtesting_forecaster output when `exog` has a missing value in a
    step of the `gap`, which is predicted and discarded, and the estimator is
    a linear model. The missing value does not reach any lag of the scored
    steps (`lags=[10]`), so the backtesting completes without NaN predictions
    and with the same metric as before the check of NaN predictors. Before
    the fix of the gap, ForecasterRecursive raised when predicting the gap
    step. ForecasterDirect does not predict the steps of the gap.
    """
    cv = TimeSeriesFold(steps=5, initial_train_size=40, gap=2, refit=False)
    metric, predictions = backtesting_forecaster(
                              forecaster    = forecaster,
                              y             = y_gap,
                              exog          = exog_gap,
                              cv            = cv,
                              metric        = 'mean_absolute_error',
                              verbose       = False,
                              show_progress = False
                          )

    assert len(predictions) == 18
    assert not predictions['pred'].isna().any()
    np.testing.assert_almost_equal(metric.iloc[0, 0], expected_metric)


@pytest.mark.parametrize(
    "forecaster, err_msg",
    [(ForecasterRecursive(Ridge(), lags=[1]), nan_predictions_msg),
     (ForecasterRecursive(Ridge(), lags=[10], differentiation=1),
      nan_predictions_differentiation_msg(5, "2020-02-12 00:00:00")),
     (ForecasterDirect(Ridge(), lags=[10], steps=7, differentiation=1),
      nan_predictions_differentiation_msg(5, "2020-02-12 00:00:00"))],
    ids=["recursive_lags_1", "recursive_differentiation", "direct_differentiation"]
)
def test_backtesting_forecaster_ValueError_when_exog_NaN_in_gap_reaches_scored_steps(
    forecaster, err_msg
):
    """
    Test backtesting_forecaster raises a ValueError before the metrics, with
    the number of NaN predictions, the first date, the likely cause and the
    remedy, when a missing value of `exog` in a step of the `gap` reaches the
    scored steps: as a lag of the following steps (`lags=[1]`), or when the
    differentiation is reverted. Before, the metric raised 'Input contains
    NaN.'. The forecaster of the user is not modified, so its `predict` still
    raises with the missing value.
    """
    cv = TimeSeriesFold(steps=5, initial_train_size=40, gap=2, refit=False)

    with pytest.raises(ValueError, match=re.escape(err_msg)):
        backtesting_forecaster(
            forecaster    = forecaster,
            y             = y_gap,
            exog          = exog_gap,
            cv            = cv,
            metric        = 'mean_absolute_error',
            verbose       = False,
            show_progress = False
        )

    assert forecaster._allow_nan_predictions is False
    forecaster.fit(y=y_gap.iloc[:40], exog=exog_gap.iloc[:40])
    err_msg = re.escape(
        "Input X contains NaN in the predictors ['x'] at step 2 "
        "(2020-02-11 00:00:00). Ridge does not accept missing values encoded "
        "as NaN, so its predictions would be NaN."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.predict(
            steps=7, exog=exog_gap.iloc[40:], suppress_warnings=True
        )


# Series with a missing value at position 44, the last value before the second
# fold (positions 45 to 49) when `initial_train_size=40` and `steps=5`.
y_nan_before_fold = y_gap.copy()
y_nan_before_fold.iloc[44] = np.nan


@pytest.mark.parametrize(
    "forecaster",
    [ForecasterRecursive(Ridge(), lags=[3], differentiation=1),
     ForecasterRecursive(
         HistGradientBoostingRegressor(max_iter=10, random_state=123),
         lags=[3], differentiation=1
     ),
     ForecasterDirect(Ridge(), lags=[3], steps=5, differentiation=1),
     ForecasterDirect(
         HistGradientBoostingRegressor(max_iter=10, random_state=123),
         lags=[3], steps=5, differentiation=1
     )],
    ids=["recursive_Ridge", "recursive_HGBR", "direct_Ridge", "direct_HGBR"]
)
def test_backtesting_forecaster_ValueError_when_NaN_last_value_before_fold_and_differentiation(
    forecaster
):
    """
    Test backtesting_forecaster raises the ValueError of the NaN predictions,
    with differentiation as a cause, when the last value of the series before
    a fold is NaN and `differentiation=1`, with any estimator. No predictor
    of the fold reads it (`lags=[3]`), but the differentiation is reverted
    from it: `predict` of the copy used by backtesting does not raise, the
    predictions of the fold are NaN and the error is raised before the
    metrics.
    """
    cv = TimeSeriesFold(steps=5, initial_train_size=40, refit=False)

    err_msg = nan_predictions_differentiation_msg(5, "2020-02-15 00:00:00")
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        backtesting_forecaster(
            forecaster    = forecaster,
            y             = y_nan_before_fold,
            cv            = cv,
            metric        = 'mean_absolute_error',
            verbose       = False,
            show_progress = False
        )

    assert forecaster._allow_nan_predictions is False



@pytest.mark.parametrize(
    "forecaster",
    [ForecasterRecursive(estimator=Ridge(), lags=3),
     ForecasterDirect(estimator=Ridge(), lags=3, steps=6)],
    ids=lambda forecaster: f"forecaster: {type(forecaster).__name__}"
)
@pytest.mark.parametrize(
    "cv_kwargs, expected_index, expected_fold, expected_step",
    [
        # gap=2 and an incomplete last fold: the steps start at gap + 1.
        ({"steps": 4, "initial_train_size": 38, "gap": 2},
         range(40, 50),
         [0, 0, 0, 0, 1, 1, 1, 1, 2, 2],
         [3, 4, 5, 6, 3, 4, 5, 6, 3, 4]),
        # fold_stride < steps: the predictions of overlapping folds are kept.
        ({"steps": 4, "initial_train_size": 42, "fold_stride": 2},
         [42, 43, 44, 45, 44, 45, 46, 47, 46, 47, 48, 49, 48, 49],
         [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
         [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2]),
        # skip_folds: the folds kept have all their steps.
        ({"steps": 3, "initial_train_size": 38, "skip_folds": 2},
         [38, 39, 40, 44, 45, 46],
         [0, 0, 0, 2, 2, 2],
         [1, 2, 3, 1, 2, 3]),
    ],
    ids=["gap and incomplete fold", "fold_stride < steps", "skip_folds"]
)
def test_backtesting_forecaster_step_y_true_and_n_evaluated(
    forecaster, cv_kwargs, expected_index, expected_fold, expected_step
):
    """
    Test the columns `step` (position of the date in the test set of its fold,
    gap included) and `y_true` of the predictions, and `n_evaluated` of the
    metrics (dates evaluated once, with the prediction of the last fold that
    predicts them), with `gap`, an incomplete last fold, overlapping folds
    and `skip_folds`, for recursive and direct forecasters.
    """
    cv = TimeSeriesFold(refit=False, allow_incomplete_fold=True, **cv_kwargs)
    metric, backtest_predictions = backtesting_forecaster(
        forecaster    = forecaster,
        y             = y,
        cv            = cv,
        metric        = "mean_absolute_error",
        show_progress = False
    )

    assert backtest_predictions.columns.tolist() == ["fold", "step", "y_true", "pred"]
    np.testing.assert_array_equal(backtest_predictions.index, expected_index)
    np.testing.assert_array_equal(backtest_predictions["fold"], expected_fold)
    np.testing.assert_array_equal(backtest_predictions["step"], expected_step)
    np.testing.assert_array_equal(
        backtest_predictions["y_true"], y.loc[expected_index].to_numpy()
    )

    last_predictions = backtest_predictions.loc[
        ~backtest_predictions.index.duplicated(keep="last")
    ]
    assert metric.columns.tolist() == ["mean_absolute_error", "n_evaluated"]
    assert metric.loc[0, "n_evaluated"] == len(set(expected_index))
    assert metric.loc[0, "mean_absolute_error"] == pytest.approx(
        mean_absolute_error(y.loc[last_predictions.index], last_predictions["pred"])
    )


def test_backtesting_forecaster_columns_order_when_interval_and_return_predictors():
    """
    Test the order of the columns of the predictions with an interval and
    `return_predictors=True`: `fold`, `step` and `y_true` first, then the
    predictions and the predictors.
    """
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=3)
    cv = TimeSeriesFold(steps=4, initial_train_size=38, gap=1)
    metric, backtest_predictions = backtesting_forecaster(
        forecaster        = forecaster,
        y                 = y,
        cv                = cv,
        metric            = "mean_absolute_error",
        interval          = [0.1, 0.9],
        interval_method   = "conformal",
        return_predictors = True,
        show_progress     = False
    )

    assert backtest_predictions.columns.tolist() == [
        "fold", "step", "y_true", "pred", "lower_bound", "upper_bound",
        "lag_1", "lag_2", "lag_3"
    ]
    assert metric.columns.tolist() == [
        "mean_absolute_error", "coverage", "interval_width", "n_evaluated"
    ]
    assert backtest_predictions["step"].tolist() == [2, 3, 4, 5] * 2 + [2, 3, 4]
    assert metric.loc[0, "n_evaluated"] == 11


@pytest.mark.parametrize(
    "refit",
    [False, True],
    ids=lambda x: f"refit: {x}"
)
def test_backtesting_forecaster_metrics_leave_out_dates_without_true_values(
    refit
):
    """
    Test that the dates without a true value are left out of the metrics (as
    in the multi-series backtesting), also of the scaled metrics, whose scale
    in the second fold (forecast origin 45, with or without refit) is
    calculated with a missing value: the metrics are the ones calculated by
    hand without those dates, and they never receive a missing value. The
    predictions keep the rows of those dates, with `step` and `y_true` NaN,
    and they are not counted in `n_evaluated`.
    """
    y_nan = y.copy()
    y_nan.iloc[[42, 47]] = np.nan

    received = []

    def spy_metric(y_true, y_pred):
        received.append(np.asarray(y_true))
        return 0.

    forecaster = ForecasterRecursive(
        estimator=HistGradientBoostingRegressor(max_iter=5, random_state=123), lags=3
    )
    cv = TimeSeriesFold(steps=5, initial_train_size=40, refit=refit)
    metric, backtest_predictions = backtesting_forecaster(
        forecaster    = forecaster,
        y             = y_nan,
        cv            = cv,
        metric        = [
            "mean_absolute_error",
            mean_absolute_scaled_error,
            root_mean_squared_scaled_error,
            spy_metric,
        ],
        show_progress = False
    )

    assert backtest_predictions["y_true"].isna().tolist() == (
        [False, False, True] + [False] * 4 + [True] + [False] * 2
    )
    assert backtest_predictions["step"].tolist() == [1, 2, 3, 4, 5] * 2
    assert backtest_predictions["pred"].notna().all()

    is_observed = backtest_predictions["y_true"].notna().to_numpy()
    errors = (
        backtest_predictions["y_true"].to_numpy()[is_observed]
        - backtest_predictions["pred"].to_numpy()[is_observed]
    )
    # NOTE: Each error is scaled with the naive forecast over the values before
    # the forecast origin of its fold (40 and 45, the end of the last window).
    origins = np.where(backtest_predictions["fold"].to_numpy() == 0, 40, 45)[is_observed]
    scale_abs = np.array(
        [np.nanmean(np.abs(np.diff(y_nan.to_numpy()[:o]))) for o in origins]
    )
    scale_sq = np.array(
        [np.nanmean(np.diff(y_nan.to_numpy()[:o]) ** 2) for o in origins]
    )
    expected_metric = pd.DataFrame({
        "mean_absolute_error": [np.mean(np.abs(errors))],
        "mean_absolute_scaled_error": [np.mean(np.abs(errors) / scale_abs)],
        "root_mean_squared_scaled_error": [np.sqrt(np.mean(errors ** 2 / scale_sq))],
        "spy_metric": [0.],
        "n_evaluated": [8],
    })
    pd.testing.assert_frame_equal(metric, expected_metric)
    assert len(received) == 1
    assert len(received[0]) == 8
    assert not np.isnan(received[0]).any()


def test_backtesting_forecaster_ValueError_when_no_true_values_in_test_set():
    """
    Test ValueError is raised when no date of the test set has a true value,
    so the metrics cannot be calculated.
    """
    y_nan = y.copy()
    y_nan.iloc[40:] = np.nan

    forecaster = ForecasterRecursive(
        estimator=HistGradientBoostingRegressor(max_iter=5, random_state=123), lags=3
    )
    cv = TimeSeriesFold(steps=5, initial_train_size=40)

    err_msg = re.escape(
        "The series has no observed values in the test set of the folds, so the "
        "metrics cannot be calculated. Review `initial_train_size` of `cv`, or "
        "the span of the series."
    )
    with pytest.raises(ValueError, match=err_msg):
        backtesting_forecaster(
            forecaster    = forecaster,
            y             = y_nan,
            cv            = cv,
            metric        = "mean_absolute_error",
            show_progress = False
        )


def test_backtesting_forecaster_ForecasterRecursiveClassifier_metrics_without_true_values():
    """
    Test that, with ForecasterRecursiveClassifier, the dates without a true
    value are left out of the metrics: the metric is the one calculated by
    hand without those dates, which previously raised a `TypeError` when
    sorting the classes with NaN.
    """
    y_clf_nan = y_clf.copy()
    y_clf_nan.iloc[[42, 47]] = np.nan

    forecaster = ForecasterRecursiveClassifier(
        estimator=HistGradientBoostingClassifier(max_iter=5, random_state=123), lags=3
    )
    cv = TimeSeriesFold(steps=5, initial_train_size=40)
    metric, backtest_predictions = backtesting_forecaster(
        forecaster    = forecaster,
        y             = y_clf_nan,
        cv            = cv,
        metric        = "accuracy_score",
        show_progress = False
    )

    assert backtest_predictions["y_true"].isna().sum() == 2
    observed = backtest_predictions.loc[backtest_predictions["y_true"].notna()]
    expected_accuracy = np.mean(
        observed["y_true"].to_numpy() == observed["pred"].to_numpy()
    )
    assert metric.loc[0, "accuracy_score"] == pytest.approx(expected_accuracy)
    assert metric.loc[0, "n_evaluated"] == 8


def test_backtesting_forecaster_y_true_ForecasterRecursiveClassifier():
    """
    Test that `y_true` contains the classes of the series with
    ForecasterRecursiveClassifier, before the predicted class and the
    probabilities.
    """
    forecaster = ForecasterRecursiveClassifier(estimator=LogisticRegression(), lags=3)
    cv = TimeSeriesFold(steps=5, initial_train_size=40)
    metric, backtest_predictions = backtesting_forecaster(
        forecaster    = forecaster,
        y             = y_clf,
        cv            = cv,
        metric        = "accuracy_score",
        show_progress = False
    )

    assert backtest_predictions.columns.tolist()[:4] == ["fold", "step", "y_true", "pred"]
    assert backtest_predictions["y_true"].tolist() == y_clf.iloc[40:].tolist()
    assert metric.loc[0, "n_evaluated"] == 10


@pytest.mark.parametrize(
    "n_jobs, refit, min_work, expected_n_jobs",
    [('auto', True, 0.0, min(max(1, cpu_count() - 1), 4)),
     ('auto', True, 1e6, 1),
     ('auto', False, 0.0, 1),
     (2, True, 1e6, 2),
     (10, True, 1e6, 4),
     (-1, False, 1e6, min(cpu_count(), 4))],
    ids=['auto, enough work', 'auto, too little work', 'auto, no refit', 'int',
         'int above folds', '-1']
)
def test_backtesting_forecaster_n_jobs_auto_limited_by_the_workload(
    n_jobs, refit, min_work, expected_n_jobs
):
    """
    Test that, with `n_jobs='auto'`, backtesting_forecaster runs the folds with one job
    when the work estimated from the initial fit is below the minimum of the
    platform (`min_work`, patched) or there is no refit, and otherwise with
    the jobs of `select_n_jobs_backtesting` limited to the number of folds
    (4). An integer `n_jobs` (`-1` is all the processors) is only limited to
    the number of folds. `Parallel` is mocked with a sequential one, so no
    workers are started.
    """
    forecaster = ForecasterRecursive(
                     estimator=HistGradientBoostingRegressor(max_iter=5), lags=3
                 )
    cv = TimeSeriesFold(
             steps              = 10,
             initial_train_size = 10,
             refit              = refit,
             fixed_train_size   = False,
             verbose            = False
         )

    with (
        patch.dict(
            "skforecast.model_selection._utils._MIN_PARALLEL_WORKLOAD_SECONDS",
            {sys.platform: min_work}
        ),
        patch(
            "skforecast.model_selection._utils._MIN_PARALLEL_WORKLOAD_SECONDS_DEFAULT",
            min_work
        ),
        patch(
            "skforecast.model_selection._validation.Parallel",
            side_effect=lambda **kwargs: Parallel(n_jobs=1)
        ) as mock_parallel
    ):
        backtesting_forecaster(
            forecaster    = forecaster,
            y             = y,
            cv            = cv,
            metric        = 'mean_absolute_error',
            n_jobs        = n_jobs,
            show_progress = False
        )

    assert mock_parallel.call_args.kwargs == {'n_jobs': expected_n_jobs}


def _backtesting_exog_with_nan(n_jobs):
    """
    Backtesting of a `HistGradientBoostingRegressor` (its folds run in parallel
    with `n_jobs` other than 1) with a missing value in `exog` in the test set,
    so `predict` warns in the folds that use it.
    """
    exog_nan = exog.copy()
    exog_nan.iloc[[42, 47]] = np.nan
    forecaster = ForecasterRecursive(
        HistGradientBoostingRegressor(max_iter=5, random_state=123), lags=3
    )
    backtesting_forecaster(
        forecaster    = forecaster,
        y             = y,
        exog          = exog_nan,
        cv            = TimeSeriesFold(steps=2, initial_train_size=40, refit=False),
        metric        = "mean_absolute_error",
        n_jobs        = n_jobs,
        show_progress = False,
    )


@pytest.mark.parametrize(
    "backend",
    ["loky", "threading"],
    ids=lambda backend: f"backend: {backend}",
)
def test_backtesting_forecaster_warnings_of_folds_in_parallel_same_as_n_jobs_1(backend):
    """
    Test that the warnings raised in the folds that run in parallel (worker
    processes of loky, or threads) reach the main process once each, with the
    same category, text, location and order as with `n_jobs=1`.
    """
    def warnings_shown(n_jobs):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _backtesting_exog_with_nan(n_jobs=n_jobs)

        return [(w.category, str(w.message), w.filename, w.lineno) for w in record]

    expected = warnings_shown(n_jobs=1)
    with parallel_config(backend=backend):
        results = warnings_shown(n_jobs=2)

    assert len(expected) == 4
    assert {category for category, _, _, _ in expected} == {MissingValuesWarning}
    assert results == expected


def test_backtesting_forecaster_warnings_of_folds_in_processes_go_through_filters():
    """
    Test that the warnings of the folds that run in worker processes go through
    the filters of the user: ignored with 'ignore', and raised as an error,
    after the folds, with 'error'.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", category=MissingValuesWarning)
        _backtesting_exog_with_nan(n_jobs=2)

    assert [w for w in record if w.category is MissingValuesWarning] == []

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=MissingValuesWarning)
        with pytest.raises(MissingValuesWarning, match="`exog` has missing values"):
            _backtesting_exog_with_nan(n_jobs=2)


class RidgeWarnsThenFailsPredictingOneRow(Ridge):
    """
    Ridge that, when it predicts one row (the steps of a recursive forecast, not
    the in-sample residuals of `fit`), warns and then raises an error. It is
    not a class of scikit-learn, so the forecaster calls its `predict`.
    """

    def predict(self, X):
        if len(X) == 1:
            warnings.warn("Warning before the error of the fold.", UserWarning)
            raise ValueError("Error of the fold.")
        return super().predict(X)


def test_backtesting_forecaster_warnings_of_fold_that_fails_in_process_same_as_n_jobs_1():
    """
    Test that, when a fold that runs in a worker process warns and then raises
    an error, its warning is shown in the main process and the error is the
    same as with `n_jobs=1`.
    """
    def warnings_and_error(n_jobs):
        forecaster = ForecasterRecursive(RidgeWarnsThenFailsPredictingOneRow(), lags=3)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            with pytest.raises(ValueError) as exc_info:
                backtesting_forecaster(
                    forecaster    = forecaster,
                    y             = y,
                    cv            = TimeSeriesFold(steps=2, initial_train_size=40, refit=False),
                    metric        = "mean_absolute_error",
                    n_jobs        = n_jobs,
                    show_progress = False,
                )

        warnings_shown = [(w.category, str(w.message), w.filename, w.lineno) for w in record]

        return warnings_shown, type(exc_info.value), str(exc_info.value)

    expected = warnings_and_error(n_jobs=1)
    results = warnings_and_error(n_jobs=2)

    assert expected[0][0][:2] == (UserWarning, "Warning before the error of the fold.")
    assert expected[1:] == (ValueError, "Error of the fold.")
    assert results == expected


def expected_scaled_metrics_per_origin(y, backtest_predictions, folds):
    """
    Reference MASE and RMSSE: the error of each prediction is scaled with the
    error of the one-step naive forecast over the values of `y` before its
    forecast origin, the end of the last window of its fold (`fold[2][1]`).
    The repeated dates of overlapping folds keep the prediction of the last
    fold.
    """
    origin_by_fold = {fold[0]: fold[2][1] for fold in folds}
    predictions = backtest_predictions.loc[
        ~backtest_predictions.index.duplicated(keep="last")
    ]
    predictions = predictions.loc[predictions["y_true"].notna()]
    values = y.to_numpy()
    errors = (predictions["y_true"] - predictions["pred"]).to_numpy()
    origins = predictions["fold"].map(origin_by_fold).to_numpy()
    scale_abs = np.array([np.nanmean(np.abs(np.diff(values[:o]))) for o in origins])
    scale_sq = np.array([np.nanmean(np.diff(values[:o]) ** 2) for o in origins])

    return (
        np.mean(np.abs(errors) / scale_abs),
        np.sqrt(np.mean(errors ** 2 / scale_sq))
    )


@pytest.mark.parametrize(
    "forecaster, cv_kwargs",
    [
        (ForecasterRecursive(Ridge(random_state=123), lags=3), {}),
        (ForecasterRecursive(Ridge(random_state=123), lags=10), {}),
        (ForecasterRecursive(Ridge(random_state=123), lags=10, differentiation=1), {}),
        (ForecasterDirect(Ridge(random_state=123), lags=10, steps=5), {}),
        (ForecasterRecursive(Ridge(random_state=123), lags=10), {"refit": True}),
        (
            ForecasterRecursive(Ridge(random_state=123), lags=10),
            {"refit": True, "fixed_train_size": True}
        ),
        (ForecasterRecursive(Ridge(random_state=123), lags=10), {"gap": 2}),
        (ForecasterRecursive(Ridge(random_state=123), lags=10), {"fold_stride": 3}),
        (
            ForecasterRecursive(Ridge(random_state=123), lags=10),
            {"steps": 4, "allow_incomplete_fold": True}
        ),
    ],
    ids=[
        "lags 3", "lags 10", "differentiation", "direct", "refit",
        "refit fixed_train_size", "gap", "fold_stride", "incomplete last fold"
    ]
)
def test_backtesting_forecaster_scaled_metrics_scaled_by_forecast_origin(
    forecaster, cv_kwargs
):
    """
    Test that `mean_absolute_scaled_error` and `root_mean_squared_scaled_error`
    scale the error of each prediction with the naive forecast over the values
    before its forecast origin (the end of the last window of its fold), the
    same for every forecaster (`window_size`), `refit`, `fixed_train_size`,
    `gap`, overlapping folds and incomplete last fold.
    """
    kwargs = {"steps": 5, "initial_train_size": 30, "refit": False}
    kwargs.update(cv_kwargs)
    cv = TimeSeriesFold(**kwargs)
    metric, backtest_predictions = backtesting_forecaster(
        forecaster    = forecaster,
        y             = y,
        cv            = cv,
        metric        = [mean_absolute_scaled_error, root_mean_squared_scaled_error],
        show_progress = False
    )

    folds = TimeSeriesFold(**kwargs, window_size=forecaster.window_size).split(
        X=y, as_pandas=False
    )
    expected_mase, expected_rmsse = expected_scaled_metrics_per_origin(
        y=y, backtest_predictions=backtest_predictions, folds=folds
    )
    np.testing.assert_almost_equal(
        metric.loc[0, "mean_absolute_scaled_error"], expected_mase
    )
    np.testing.assert_almost_equal(
        metric.loc[0, "root_mean_squared_scaled_error"], expected_rmsse
    )


def test_backtesting_forecaster_scaled_metrics_same_predictions_same_value():
    """
    Test that the scaled metrics of two forecasters with different
    `window_size` are equal when their predictions are equal: the scale does
    not depend on the forecaster. An estimator that predicts the first lag
    (the naive forecast) gives the same predictions with 3 and 10 lags.
    """

    class LastValue(BaseEstimator, RegressorMixin):
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.asarray(X)[:, 0]

    results = []
    for lags in [3, 10]:
        metric, predictions = backtesting_forecaster(
            forecaster    = ForecasterRecursive(LastValue(), lags=lags),
            y             = y,
            cv            = TimeSeriesFold(steps=5, initial_train_size=30),
            metric        = [
                "mean_absolute_error",
                mean_absolute_scaled_error,
                root_mean_squared_scaled_error
            ],
            show_progress = False
        )
        results.append((metric, predictions))

    pd.testing.assert_frame_equal(results[0][1], results[1][1])
    pd.testing.assert_frame_equal(results[0][0], results[1][0])


def test_backtesting_forecaster_metric_with_y_train_receives_values_before_first_origin():
    """
    Test that a metric with the argument `y_train`, other than the scaled
    metrics of skforecast, receives the values of `y` before the first
    forecast origin (`initial_train_size`), with any `refit`.
    """
    received = []

    def custom_metric(y_true, y_pred, y_train):
        received.append(y_train)
        return 0.

    for refit in [False, True]:
        backtesting_forecaster(
            forecaster    = ForecasterRecursive(Ridge(random_state=123), lags=10),
            y             = y,
            cv            = TimeSeriesFold(steps=5, initial_train_size=30, refit=refit),
            metric        = custom_metric,
            show_progress = False
        )

    for y_train in received:
        pd.testing.assert_series_equal(y_train, y.iloc[:30])


def test_backtesting_forecaster_RuntimeWarning_and_NaN_when_scale_is_zero():
    """
    Test that the scaled metrics leave out the points whose scale is zero
    (constant series before the forecast origin), warning about it, and are
    NaN when no point is left.
    """
    y_constant = pd.Series(np.full(50, 5.0))
    warn_msg = re.escape(
        "The scale of `mean_absolute_scaled_error` and "
        "`root_mean_squared_scaled_error` (the error of the one-step naive "
        "forecast over the observations before the forecast origin) is zero "
        "or missing for 20 of the 20 points evaluated"
    )
    with pytest.warns(RuntimeWarning, match=warn_msg):
        metric, _ = backtesting_forecaster(
            forecaster    = ForecasterRecursive(Ridge(random_state=123), lags=3),
            y             = y_constant,
            cv            = TimeSeriesFold(steps=5, initial_train_size=30),
            metric        = [
                "mean_absolute_error",
                mean_absolute_scaled_error,
                root_mean_squared_scaled_error
            ],
            show_progress = False
        )

    assert metric.loc[0, "mean_absolute_error"] == 0.
    assert np.isnan(metric.loc[0, "mean_absolute_scaled_error"])
    assert np.isnan(metric.loc[0, "root_mean_squared_scaled_error"])
    assert metric.loc[0, "n_evaluated"] == 20


@pytest.mark.parametrize(
    "refit",
    [True, False],
    ids=lambda refit: f"refit: {refit}",
)
def test_backtesting_forecaster_folds_in_processes_same_output_as_n_jobs_1(refit):
    """
    Test that the folds that run in worker processes, which receive the
    forecaster serialized once in the main process, return the same metrics
    and predictions (intervals included) as with `n_jobs=1`. The forecaster
    has a lambda as `weight_func`, which only `cloudpickle` can serialize.
    """
    def run_backtesting(n_jobs):
        forecaster = ForecasterRecursive(
            estimator   = Ridge(random_state=123),
            lags        = 3,
            weight_func = lambda index: np.where(index < 20, 0.5, 1.0)
        )
        cv = TimeSeriesFold(
                 steps              = 5,
                 initial_train_size = 30,
                 refit              = refit,
                 verbose            = False
             )

        return backtesting_forecaster(
                   forecaster    = forecaster,
                   y             = y,
                   exog          = exog,
                   cv            = cv,
                   metric        = "mean_absolute_error",
                   interval      = [0.1, 0.9],
                   n_jobs        = n_jobs,
                   show_progress = False
               )

    expected_metric, expected_predictions = run_backtesting(n_jobs=1)
    metric, predictions = run_backtesting(n_jobs=2)

    pd.testing.assert_frame_equal(metric, expected_metric)
    pd.testing.assert_frame_equal(predictions, expected_predictions)


def test_backtesting_forecaster_serializes_forecaster_once_only_with_n_jobs_greater_than_1():
    """
    Test that, with `n_jobs` greater than 1, the folds receive the forecaster
    serialized once, as the same bytes object, and that, with `n_jobs=1`, they
    receive the forecaster itself. `Parallel` is mocked with a sequential one,
    so no workers are started.
    """
    received = []

    def fit_predict_recording(forecaster, **kwargs):
        received.append(forecaster)
        return _fit_predict_forecaster(forecaster=forecaster, **kwargs)

    cv = TimeSeriesFold(steps=10, initial_train_size=30, refit=True, verbose=False)
    for n_jobs in [1, 2]:
        with (
            patch(
                "skforecast.model_selection._validation.Parallel",
                side_effect=lambda **kwargs: Parallel(n_jobs=1)
            ),
            patch(
                "skforecast.model_selection._validation._fit_predict_forecaster",
                side_effect=fit_predict_recording
            )
        ):
            backtesting_forecaster(
                forecaster    = ForecasterRecursive(estimator=Ridge(), lags=3),
                y             = y,
                cv            = cv,
                metric        = "mean_absolute_error",
                n_jobs        = n_jobs,
                show_progress = False
            )

    received_n_jobs_1, received_n_jobs_2 = received[:2], received[2:]
    assert len(received_n_jobs_1) == len(received_n_jobs_2) == 2
    assert all(isinstance(f, ForecasterRecursive) for f in received_n_jobs_1)
    assert all(isinstance(f, bytes) for f in received_n_jobs_2)
    assert received_n_jobs_2[0] is received_n_jobs_2[1]


@pytest.mark.parametrize(
    "min_work, expected_n_jobs",
    [(0.0, min(max(1, cpu_count() - 1), 4)),
     (1e6, 1)],
    ids=["enough work", "too little work"]
)
def test_backtesting_forecaster_n_jobs_auto_no_refit_limited_by_the_prediction_workload(
    min_work, expected_n_jobs
):
    """
    Test that, with `n_jobs='auto'` and no refit, backtesting_forecaster with a
    `HistGradientBoostingRegressor` predicts the first fold in the main process,
    and runs the other 4 folds in parallel only if their estimated work is
    above the minimum of the platform (`min_work`, patched), with at most one
    job per fold. The predictions are the same as with `n_jobs=1`. `Parallel`
    is mocked with a sequential one, so no workers are started.
    """
    cv = TimeSeriesFold(steps=4, initial_train_size=30, refit=False, verbose=False)

    def run_backtesting(n_jobs):
        forecaster = ForecasterRecursive(
                         estimator=HistGradientBoostingRegressor(max_iter=5), lags=3
                     )
        return backtesting_forecaster(
                   forecaster    = forecaster,
                   y             = y,
                   cv            = cv,
                   metric        = 'mean_absolute_error',
                   n_jobs        = n_jobs,
                   show_progress = False
               )

    expected_metric, expected_predictions = run_backtesting(n_jobs=1)
    with (
        patch.dict(
            "skforecast.model_selection._utils._MIN_PARALLEL_PREDICTION_SECONDS",
            {sys.platform: min_work}
        ),
        patch(
            "skforecast.model_selection._utils._MIN_PARALLEL_PREDICTION_SECONDS_DEFAULT",
            min_work
        ),
        patch(
            "skforecast.model_selection._validation.Parallel",
            side_effect=lambda **kwargs: Parallel(n_jobs=1)
        ) as mock_parallel
    ):
        metric, predictions = run_backtesting(n_jobs='auto')

    assert mock_parallel.call_args.kwargs == {'n_jobs': expected_n_jobs}
    pd.testing.assert_frame_equal(metric, expected_metric)
    pd.testing.assert_frame_equal(predictions, expected_predictions)
