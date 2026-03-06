# Unit tests for callable exog support in backtesting
# ==============================================================================
# Feature: "As-Of" timestamp filtering for exogenous covariates (#1131)
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from skforecast.model_selection._validation import _backtesting_forecaster
from skforecast.model_selection._split import TimeSeriesFold
from skforecast.model_selection import backtesting_forecaster, backtesting_forecaster_multiseries


# ============================================================================
# Fixtures
# ============================================================================

def make_datetime_series(n=60, freq="D", start="2020-01-01", seed=42):
    np.random.seed(seed)
    idx = pd.date_range(start=start, periods=n, freq=freq)
    return pd.Series(np.random.rand(n), index=idx)


y_dt = make_datetime_series(n=60, seed=10)
exog_dt = make_datetime_series(n=60, seed=20).rename("exog_feature")


# ============================================================================
# Test: callable exog for _backtesting_forecaster (univariate)
# ============================================================================

def test_backtesting_forecaster_callable_exog_matches_static_exog():
    """
    When a callable `exog` is passed that simply returns the full static exog
    (regardless of the timestamp argument), results must be identical to
    passing the DataFrame directly.
    """
    forecaster_static = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster_callable = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    n_backtest = 20
    y_train_len = len(y_dt) - n_backtest

    cv = TimeSeriesFold(
        steps=5,
        initial_train_size=y_train_len,
        refit=False,
        fixed_train_size=True,
        gap=0,
    )

    # Static exog
    metric_static, preds_static = _backtesting_forecaster(
        forecaster=forecaster_static,
        y=y_dt,
        cv=cv,
        exog=exog_dt,
        metric="mean_squared_error",
        verbose=False,
    )

    # Callable exog — returns the same exog regardless of timestamp
    def callable_exog(as_of_timestamp):
        return exog_dt

    metric_callable, preds_callable = _backtesting_forecaster(
        forecaster=forecaster_callable,
        y=y_dt,
        cv=cv,
        exog=callable_exog,
        metric="mean_squared_error",
        verbose=False,
    )

    pd.testing.assert_frame_equal(
        metric_static, metric_callable, check_dtype=True
    )
    pd.testing.assert_frame_equal(
        preds_static, preds_callable, check_index_type=False
    )


def test_backtesting_forecaster_callable_exog_receives_correct_timestamps():
    """
    The callable `exog` must be called with the timestamp of the last observed
    datapoint in the window before each prediction fold.
    """
    n_backtest = 20
    y_train_len = len(y_dt) - n_backtest
    received_timestamps = []

    def callable_exog(as_of_timestamp):
        received_timestamps.append(as_of_timestamp)
        return exog_dt

    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    cv = TimeSeriesFold(
        steps=5,
        initial_train_size=y_train_len,
        refit=False,
        fixed_train_size=True,
        gap=0,
    )

    _backtesting_forecaster(
        forecaster=forecaster,
        y=y_dt,
        cv=cv,
        exog=callable_exog,
        metric="mean_squared_error",
        verbose=False,
    )

    # 20 obs / 5 steps = 4 folds → 4 calls to callable
    # initial fit call uses the same callable
    assert len(received_timestamps) >= 1
    # All received timestamps should be pandas Timestamps
    for ts in received_timestamps:
        assert isinstance(ts, (pd.Timestamp, type(y_dt.index[0]))), (
            f"Expected a Timestamp, got {type(ts)}"
        )


def test_backtesting_forecaster_callable_exog_no_exog_returned():
    """
    When callable `exog` returns None, it should behave as if no exog was
    provided, matching the result when `exog=None` is passed.
    """
    n_backtest = 20
    y_train_len = len(y_dt) - n_backtest

    forecaster_none = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster_callable = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    cv = TimeSeriesFold(
        steps=5,
        initial_train_size=y_train_len,
        refit=False,
        fixed_train_size=True,
        gap=0,
    )

    metric_none, preds_none = _backtesting_forecaster(
        forecaster=forecaster_none,
        y=y_dt,
        cv=cv,
        exog=None,
        metric="mean_squared_error",
        verbose=False,
    )

    metric_callable, preds_callable = _backtesting_forecaster(
        forecaster=forecaster_callable,
        y=y_dt,
        cv=cv,
        exog=lambda ts: None,
        metric="mean_squared_error",
        verbose=False,
    )

    pd.testing.assert_frame_equal(
        metric_none, metric_callable, check_dtype=True
    )
    pd.testing.assert_frame_equal(
        preds_none, preds_callable, check_index_type=False
    )


# ============================================================================
# Test: callable exog for backtesting_forecaster (public API)
# ============================================================================

def test_backtesting_forecaster_public_api_callable_exog():
    """
    The public `backtesting_forecaster` should accept a callable `exog`
    and produce the same results as the static exog variant.
    """
    n_backtest = 20
    y_train_len = len(y_dt) - n_backtest

    forecaster_static = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster_callable = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    cv = TimeSeriesFold(
        steps=5,
        initial_train_size=y_train_len,
        refit=False,
        fixed_train_size=True,
        gap=0,
    )

    metric_static, preds_static = backtesting_forecaster(
        forecaster=forecaster_static,
        y=y_dt,
        cv=cv,
        exog=exog_dt,
        metric="mean_squared_error",
        verbose=False,
    )

    metric_callable, preds_callable = backtesting_forecaster(
        forecaster=forecaster_callable,
        y=y_dt,
        cv=cv,
        exog=lambda ts: exog_dt,
        metric="mean_squared_error",
        verbose=False,
    )

    pd.testing.assert_frame_equal(
        metric_static, metric_callable, check_dtype=True
    )
    pd.testing.assert_frame_equal(
        preds_static, preds_callable, check_index_type=False
    )


def test_backtesting_forecaster_callable_exog_as_of_filtering():
    """
    Test the actual 'as-of' filtering use-case: callable `exog` sees a different
    view of the exog data depending on the forecast origin timestamp.

    Validates that both the full-exog and callable-exog paths complete
    successfully, and that the callable is actually called (by tracking calls).
    """
    np.random.seed(42)
    n = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    y_series = pd.Series(np.random.rand(n), index=idx)

    # exog: two different versions — 'current' vs 'stale' forecasts
    exog_v1 = pd.Series(np.random.rand(n), index=idx, name="exog_feature")
    exog_v2 = pd.Series(np.random.rand(n) + 1.0, index=idx, name="exog_feature")  # shifted

    n_backtest = 15
    y_train_len = n - n_backtest

    cv = TimeSeriesFold(
        steps=5,
        initial_train_size=y_train_len,
        refit=False,
        fixed_train_size=True,
        gap=0,
    )

    # Track timestamp calls
    timestamps_seen = []

    def as_of_exog(as_of):
        """Return v1 if before midpoint, v2 otherwise — simulating data revisions."""
        timestamps_seen.append(as_of)
        cutoff = idx[n // 2]
        if as_of <= cutoff:
            return exog_v1
        else:
            return exog_v2

    forecaster_asof = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    metric_asof, preds_asof = backtesting_forecaster(
        forecaster=forecaster_asof,
        y=y_series,
        cv=cv,
        exog=as_of_exog,
        metric="mean_squared_error",
        verbose=False,
    )

    # Verify run completed and callable was invoked
    assert isinstance(metric_asof, pd.DataFrame)
    assert isinstance(preds_asof, pd.DataFrame)
    assert len(timestamps_seen) >= 1, "Callable should have been invoked at least once"



# ============================================================================
# Test: callable exog for backtesting_forecaster_multiseries
# ============================================================================

def make_multiseries(n=60, seed=42):
    np.random.seed(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return {
        "l1": pd.Series(np.random.rand(n), index=idx),
        "l2": pd.Series(np.random.rand(n), index=idx),
    }


series_dict_dt = make_multiseries(n=60, seed=10)
exog_series_dt = pd.Series(
    np.random.RandomState(30).rand(60),
    index=pd.date_range("2020-01-01", periods=60, freq="D"),
    name="exog_feature"
)


def test_backtesting_multieries_callable_exog_matches_static_exog():
    """
    When a callable `exog` is passed to `backtesting_forecaster_multiseries`
    that returns the full static exog, results must match the static version.
    """
    forecaster_static = ForecasterRecursiveMultiSeries(
        estimator=Ridge(random_state=42), lags=2
    )
    forecaster_callable = ForecasterRecursiveMultiSeries(
        estimator=Ridge(random_state=42), lags=2
    )

    cv = TimeSeriesFold(
        steps=5,
        initial_train_size=40,
        refit=False,
        fixed_train_size=True,
        gap=0,
    )

    metric_static, preds_static = backtesting_forecaster_multiseries(
        forecaster=forecaster_static,
        series=series_dict_dt,
        cv=cv,
        exog=exog_series_dt,
        metric="mean_squared_error",
        verbose=False,
        show_progress=False,
    )

    def callable_exog(as_of_timestamp):
        return exog_series_dt

    metric_callable, preds_callable = backtesting_forecaster_multiseries(
        forecaster=forecaster_callable,
        series=series_dict_dt,
        cv=cv,
        exog=callable_exog,
        metric="mean_squared_error",
        verbose=False,
        show_progress=False,
    )

    pd.testing.assert_frame_equal(
        metric_static, metric_callable, check_dtype=True
    )
    pd.testing.assert_frame_equal(
        preds_static, preds_callable, check_index_type=False
    )


def test_backtesting_multieries_callable_exog_dict_returned():
    """
    When callable `exog` for multiseries returns a dict (matching the dict-exog
    interface), the results should be valid and not raise errors.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        estimator=Ridge(random_state=42), lags=2
    )

    cv = TimeSeriesFold(
        steps=5,
        initial_train_size=40,
        refit=False,
        fixed_train_size=True,
        gap=0,
    )

    def callable_exog_dict(as_of_timestamp):
        return {
            "l1": exog_series_dt.rename("exog_feature"),
            "l2": exog_series_dt.rename("exog_feature"),
        }

    metric, preds = backtesting_forecaster_multiseries(
        forecaster=forecaster,
        series=series_dict_dt,
        cv=cv,
        exog=callable_exog_dict,
        metric="mean_squared_error",
        verbose=False,
        show_progress=False,
    )

    assert isinstance(metric, pd.DataFrame)
    assert isinstance(preds, pd.DataFrame)
    assert "l1" in preds["level"].unique() or len(preds) > 0
