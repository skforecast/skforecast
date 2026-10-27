# Unit test backtesting_forecaster
# ==============================================================================
import re
import warnings
import pytest
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirect
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection._split import TimeSeriesFold

# Fixtures
from ..fixtures_model_selection import y


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

    assert list(metric.columns) == ['mean_absolute_error']


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

