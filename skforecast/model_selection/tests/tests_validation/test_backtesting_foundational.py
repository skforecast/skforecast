# Unit test backtesting_foundational
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from copy import deepcopy
from unittest.mock import patch
from sklearn.linear_model import Ridge

from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection._split import TimeSeriesFold
from skforecast.model_selection import backtesting_foundational

# Fixtures — reuse FakePipeline and series fixtures from foundational tests
from ....foundational.tests.tests_forecaster_foundational.fixtures_forecaster_foundational import (
    FakePipeline,
    make_forecaster,
    y,
    series_wide,
    exog,
)

# ---------------------------------------------------------------------------
# Additional local fixtures
# ---------------------------------------------------------------------------

# Wide-format series (identical to series_wide but explicitly named)
_index = pd.date_range("2020-01-01", periods=50, freq="ME")

series_dict = {
    "series_1": pd.Series(np.arange(50, dtype=float), index=_index, name="series_1"),
    "series_2": pd.Series(np.arange(50, 100, dtype=float), index=_index, name="series_2"),
}

exog_dict = {
    "series_1": pd.DataFrame({"feat_a": np.arange(50, dtype=float)}, index=_index),
    "series_2": pd.DataFrame({"feat_a": np.arange(50, dtype=float) * 2}, index=_index),
}

# Expected test-split index (initial_train_size=38, steps=3, no gap)
_test_index = pd.date_range("2023-03-31", periods=12, freq="ME")


# ===========================================================================
# Input validation
# ===========================================================================

def test_backtesting_foundational_TypeError_when_wrong_forecaster_type():
    """
    Test that TypeError is raised when forecaster is not ForecasterFoundational.
    """
    forecaster = ForecasterRecursive(
        estimator=Ridge(random_state=123),
        lags=2,
    )
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        fixed_train_size=False,
        gap=0,
        allow_incomplete_fold=True,
    )
    err_msg = re.escape(
        "`forecaster` must be of type `ForecasterFoundational`. For all "
        "other types of forecasters use the other functions available in "
        "the `model_selection` module."
    )
    with pytest.raises(TypeError, match=err_msg):
        backtesting_foundational(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
        )


def test_backtesting_foundational_ValueError_when_interval_and_quantiles_both_provided():
    """
    Test that ValueError is raised when both interval and quantiles are provided.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        verbose=False,
    )
    err_msg = re.escape(
        "`interval` and `quantiles` cannot be provided simultaneously."
    )
    with pytest.raises(ValueError, match=err_msg):
        backtesting_foundational(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            interval=[10, 90],
            quantiles=[0.1, 0.5, 0.9],
        )


def test_backtesting_foundational_TypeError_when_quantiles_invalid():
    """
    Test that TypeError is raised when quantiles contains a value outside [0, 1].
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        verbose=False,
    )
    err_msg = re.escape(
        "`quantiles` must be a list or tuple of floats in the range [0, 1]."
    )
    with pytest.raises(TypeError, match=err_msg):
        backtesting_foundational(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            quantiles=[0.1, 1.5],
        )


# ===========================================================================
# Single-series, point forecast
# ===========================================================================

@pytest.mark.parametrize(
    "initial_train_size",
    [38, "2023-02-28"],
    ids=lambda v: f'initial_train_size: {v}',
)
def test_output_backtesting_foundational_single_no_refit_no_exog_no_remainder(initial_train_size):
    """
    Test output of backtesting_foundational for single series, no refit, no exog,
    steps=3 (no remainder). FakePipeline returns 0.5 for every step, so
    MAE = mean(|38 - 0.5|, ..., |49 - 0.5|) = 43.0.
    Also covers initial_train_size passed as a string date.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=initial_train_size,
        refit=False,
        fixed_train_size=False,
        gap=0,
        allow_incomplete_fold=True,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, backtest_predictions = backtesting_foundational(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            verbose=True,
            show_progress=False,
        )

    expected_metric = pd.DataFrame({"mean_absolute_error": [43.0]})
    expected_preds = pd.DataFrame(
        data=np.full(12, 0.5),
        columns=["pred"],
        index=_test_index,
    )
    expected_preds.insert(0, "fold", [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions)


@pytest.mark.parametrize(
    "initial_train_size",
    [38, "2023-02-28"],
    ids=lambda v: f'initial_train_size: {v}',
)
def test_output_backtesting_foundational_single_no_refit_no_exog_remainder(initial_train_size):
    """
    Test output of backtesting_foundational for single series, no refit, no exog,
    steps=5 (remainder — 12 obs: 2 full folds of 5 and 1 partial fold of 2).
    Also covers initial_train_size passed as a string date.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=5,
        initial_train_size=initial_train_size,
        refit=False,
        fixed_train_size=False,
        gap=0,
        allow_incomplete_fold=True,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, backtest_predictions = backtesting_foundational(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            verbose=False,
            show_progress=False,
        )

    expected_metric = pd.DataFrame({"mean_absolute_error": [43.0]})
    assert backtest_predictions.shape == (12, 2)
    assert backtest_predictions["fold"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2]
    assert backtest_predictions.columns.tolist() == ["fold", "pred"]
    pd.testing.assert_frame_equal(expected_metric, metric)


def test_output_backtesting_foundational_single_no_refit_yes_exog():
    """
    Test output of backtesting_foundational for single series, no refit, with exog.
    FakePipeline ignores input data and always returns 0.5, so MAE is the same.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        fixed_train_size=False,
        gap=0,
        allow_incomplete_fold=True,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, backtest_predictions = backtesting_foundational(
            forecaster=forecaster,
            series=y,
            exog=exog,
            cv=cv,
            metric="mean_absolute_error",
            verbose=False,
            show_progress=False,
        )

    expected_metric = pd.DataFrame({"mean_absolute_error": [43.0]})
    pd.testing.assert_frame_equal(expected_metric, metric)
    assert backtest_predictions.shape == (12, 2)
    assert (backtest_predictions["pred"] == 0.5).all()


def test_output_backtesting_foundational_single_refit_no_exog_no_remainder():
    """
    Test output of backtesting_foundational for single series, with refit, no exog,
    steps=3 (no remainder). FakePipeline predictions are constant regardless
    of training data, so metric value matches the no-refit case.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=True,
        fixed_train_size=False,
        gap=0,
        allow_incomplete_fold=True,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, backtest_predictions = backtesting_foundational(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            verbose=False,
            show_progress=False,
        )

    expected_metric = pd.DataFrame({"mean_absolute_error": [43.0]})
    pd.testing.assert_frame_equal(expected_metric, metric)
    assert backtest_predictions.shape == (12, 2)
    assert (backtest_predictions["pred"] == 0.5).all()
    assert backtest_predictions["fold"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_output_backtesting_foundational_single_refit_fixed_train_size_no_exog():
    """
    Test output of backtesting_foundational for single series, refit with fixed
    training window, no exog, steps=3.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=True,
        fixed_train_size=True,
        gap=0,
        allow_incomplete_fold=True,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, backtest_predictions = backtesting_foundational(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            verbose=False,
            show_progress=False,
        )

    expected_metric = pd.DataFrame({"mean_absolute_error": [43.0]})
    pd.testing.assert_frame_equal(expected_metric, metric)
    assert backtest_predictions.shape == (12, 2)


def test_output_backtesting_foundational_single_refit_yes_exog():
    """
    Test output of backtesting_foundational for single series, refit, with exog.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=True,
        fixed_train_size=False,
        gap=0,
        allow_incomplete_fold=True,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, backtest_predictions = backtesting_foundational(
            forecaster=forecaster,
            series=y,
            exog=exog,
            cv=cv,
            metric="mean_absolute_error",
            verbose=False,
            show_progress=False,
        )

    expected_metric = pd.DataFrame({"mean_absolute_error": [43.0]})
    pd.testing.assert_frame_equal(expected_metric, metric)
    assert backtest_predictions.shape == (12, 2)


# ===========================================================================
# Single-series, metrics
# ===========================================================================

def test_output_backtesting_foundational_single_callable_metric():
    """
    Test output of backtesting_foundational with a custom callable metric.
    """

    def my_mae(y_true, y_pred, y_train=None):
        return float(np.abs(y_true.values - y_pred.values).mean())

    my_mae.__name__ = "my_mae"

    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, _ = backtesting_foundational(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric=my_mae,
            verbose=False,
            show_progress=False,
        )

    expected_metric = pd.DataFrame({"my_mae": [43.0]})
    pd.testing.assert_frame_equal(expected_metric, metric)


def test_output_backtesting_foundational_single_list_of_metrics():
    """
    Test output of backtesting_foundational with a list of metrics. The returned
    DataFrame must have one column per metric.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, _ = backtesting_foundational(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric=["mean_absolute_error", "mean_squared_error"],
            verbose=False,
            show_progress=False,
        )

    assert list(metric.columns) == ["mean_absolute_error", "mean_squared_error"]
    assert metric["mean_absolute_error"].iloc[0] == pytest.approx(43.0)
    assert metric["mean_squared_error"].iloc[0] == pytest.approx(1860.9166666666667)


# ===========================================================================
# Single-series, prediction intervals and quantiles
# ===========================================================================

def test_output_backtesting_foundational_single_interval_no_refit():
    """
    Test output columns and values for backtesting with interval=[10, 90].
    FakePipeline: lower_bound=0.1, pred=0.5, upper_bound=0.9 for all steps.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, backtest_predictions = backtesting_foundational(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            interval=[10, 90],
            verbose=False,
            show_progress=False,
        )

    assert backtest_predictions.columns.tolist() == [
        "fold", "pred", "lower_bound", "upper_bound"
    ]
    assert backtest_predictions.shape == (12, 4)
    np.testing.assert_array_almost_equal(
        backtest_predictions["pred"].values, np.full(12, 0.5)
    )
    np.testing.assert_array_almost_equal(
        backtest_predictions["lower_bound"].values, np.full(12, 0.1)
    )
    np.testing.assert_array_almost_equal(
        backtest_predictions["upper_bound"].values, np.full(12, 0.9)
    )
    expected_metric = pd.DataFrame({"mean_absolute_error": [43.0]})
    pd.testing.assert_frame_equal(expected_metric, metric)


def test_output_backtesting_foundational_single_interval_refit():
    """
    Test output columns and values for backtesting with interval=[10, 90] and refit.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=True,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, backtest_predictions = backtesting_foundational(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            interval=[10, 90],
            verbose=False,
            show_progress=False,
        )

    assert backtest_predictions.columns.tolist() == [
        "fold", "pred", "lower_bound", "upper_bound"
    ]
    assert backtest_predictions.shape == (12, 4)
    np.testing.assert_array_almost_equal(
        backtest_predictions["lower_bound"].values, np.full(12, 0.1)
    )
    np.testing.assert_array_almost_equal(
        backtest_predictions["upper_bound"].values, np.full(12, 0.9)
    )


def test_output_backtesting_foundational_single_quantiles_no_refit():
    """
    Test output columns and values for backtesting with quantiles=[0.1, 0.5, 0.9].
    FakePipeline returns quantile level as value, so q_0.1=0.1, q_0.5=0.5, q_0.9=0.9.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, backtest_predictions = backtesting_foundational(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            quantiles=[0.1, 0.5, 0.9],
            verbose=False,
            show_progress=False,
        )

    assert backtest_predictions.columns.tolist() == [
        "fold", "q_0.1", "q_0.5", "q_0.9"
    ]
    assert backtest_predictions.shape == (12, 4)
    np.testing.assert_array_almost_equal(
        backtest_predictions["q_0.1"].values, np.full(12, 0.1)
    )
    np.testing.assert_array_almost_equal(
        backtest_predictions["q_0.5"].values, np.full(12, 0.5)
    )
    np.testing.assert_array_almost_equal(
        backtest_predictions["q_0.9"].values, np.full(12, 0.9)
    )
    expected_metric = pd.DataFrame({"mean_absolute_error": [43.0]})
    pd.testing.assert_frame_equal(expected_metric, metric)


def test_output_backtesting_foundational_single_quantiles_refit():
    """
    Test backtesting with quantiles=[0.1, 0.5, 0.9] and refit=True.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=True,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, backtest_predictions = backtesting_foundational(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            quantiles=[0.1, 0.5, 0.9],
            verbose=False,
            show_progress=False,
        )

    assert backtest_predictions.columns.tolist() == [
        "fold", "q_0.1", "q_0.5", "q_0.9"
    ]
    np.testing.assert_array_almost_equal(
        backtest_predictions["q_0.1"].values, np.full(12, 0.1)
    )
    np.testing.assert_array_almost_equal(
        backtest_predictions["q_0.5"].values, np.full(12, 0.5)
    )
    np.testing.assert_array_almost_equal(
        backtest_predictions["q_0.9"].values, np.full(12, 0.9)
    )


# ===========================================================================
# Single-series, fold options
# ===========================================================================

def test_output_backtesting_foundational_single_fold_stride():
    """
    Test backtesting with fold_stride < steps (overlapping folds).
    fold_stride=2, steps=3 → overlapping test windows; more than 12 rows in output.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        fold_stride=2,
        refit=False,
        fixed_train_size=False,
        gap=0,
        allow_incomplete_fold=True,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, backtest_predictions = backtesting_foundational(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            verbose=False,
            show_progress=False,
        )

    # fold_stride=2, steps=3, 12 available test obs → overlapping folds
    # produce 17 total rows (confirmed with exact expected value)
    assert backtest_predictions.shape == (17, 2)
    np.testing.assert_array_almost_equal(
        backtest_predictions["pred"].values, np.full(17, 0.5)
    )
    assert "fold" in backtest_predictions.columns


def test_output_backtesting_foundational_single_gap():
    """
    Test backtesting with gap=2 between train end and test start.
    With gap=2 and steps=3, the effective y_true values are from obs 40 onwards.
    MAE = mean(|40-0.5|, ..., |49-0.5|) = 44.0 (10 rows, last fold truncated).
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        fixed_train_size=False,
        gap=2,
        allow_incomplete_fold=True,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, backtest_predictions = backtesting_foundational(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            verbose=False,
            show_progress=False,
        )

    expected_metric = pd.DataFrame({"mean_absolute_error": [44.0]})
    pd.testing.assert_frame_equal(expected_metric, metric)
    assert backtest_predictions.shape == (10, 2)


# ===========================================================================
# Multi-series, point forecast
# ===========================================================================

def test_output_backtesting_foundational_multiseries_dataframe_no_refit():
    """
    Test backtesting with wide DataFrame input (two series), no refit.
    Expected MAE: series_1=43.0, series_2=93.0; with aggregated rows.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, backtest_predictions = backtesting_foundational(
            forecaster=forecaster,
            series=series_wide,
            cv=cv,
            metric="mean_absolute_error",
            add_aggregated_metric=True,
            verbose=False,
            show_progress=False,
        )

    # Metric DataFrame: series_1, series_2 + 3 aggregated rows
    assert "levels" in metric.columns
    assert "mean_absolute_error" in metric.columns
    assert metric.shape[0] == 5
    s1_mae = metric.loc[metric["levels"] == "series_1", "mean_absolute_error"].iloc[0]
    s2_mae = metric.loc[metric["levels"] == "series_2", "mean_absolute_error"].iloc[0]
    assert s1_mae == pytest.approx(43.0)
    assert s2_mae == pytest.approx(93.0)

    # Predictions: 2 series × 12 steps = 24 rows; includes 'level' column
    assert backtest_predictions.shape == (24, 3)
    assert "level" in backtest_predictions.columns
    assert set(backtest_predictions["level"].unique()) == {"series_1", "series_2"}
    assert (backtest_predictions["pred"] == 0.5).all()


def test_output_backtesting_foundational_multiseries_dataframe_refit():
    """
    Test backtesting with wide DataFrame input, refit=True.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=True,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, backtest_predictions = backtesting_foundational(
            forecaster=forecaster,
            series=series_wide,
            cv=cv,
            metric="mean_absolute_error",
            add_aggregated_metric=False,
            verbose=False,
            show_progress=False,
        )

    # 2 series without aggregated rows
    assert metric.shape[0] == 2
    s1_mae = metric.loc[metric["levels"] == "series_1", "mean_absolute_error"].iloc[0]
    s2_mae = metric.loc[metric["levels"] == "series_2", "mean_absolute_error"].iloc[0]
    assert s1_mae == pytest.approx(43.0)
    assert s2_mae == pytest.approx(93.0)
    assert backtest_predictions.shape == (24, 3)


def test_output_backtesting_foundational_multiseries_dict_no_refit():
    """
    Test that dict input produces the same result as equivalent wide DataFrame input.
    """
    forecaster_df = make_forecaster()
    forecaster_dict = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric_df, preds_df = backtesting_foundational(
            forecaster=forecaster_df,
            series=series_wide,
            cv=cv,
            metric="mean_absolute_error",
            add_aggregated_metric=False,
            verbose=False,
            show_progress=False,
        )
        metric_dict, preds_dict = backtesting_foundational(
            forecaster=forecaster_dict,
            series=series_dict,
            cv=cv,
            metric="mean_absolute_error",
            add_aggregated_metric=False,
            verbose=False,
            show_progress=False,
        )

    pd.testing.assert_frame_equal(metric_df, metric_dict, check_like=True)
    assert preds_df.shape == preds_dict.shape


# ===========================================================================
# Multi-series, levels filter
# ===========================================================================

def test_output_backtesting_foundational_multiseries_levels_filter():
    """
    Test that passing levels=['series_1'] restricts predictions and metric
    evaluation to that series only.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, backtest_predictions = backtesting_foundational(
            forecaster=forecaster,
            series=series_wide,
            cv=cv,
            metric="mean_absolute_error",
            levels=["series_1"],
            add_aggregated_metric=False,
            verbose=False,
            show_progress=False,
        )

    assert metric.shape[0] == 1
    assert metric["levels"].iloc[0] == "series_1"
    assert metric["mean_absolute_error"].iloc[0] == pytest.approx(43.0)
    assert backtest_predictions.shape == (12, 3)
    assert set(backtest_predictions["level"].unique()) == {"series_1"}


# ===========================================================================
# Multi-series, intervals and quantiles
# ===========================================================================

def test_output_backtesting_foundational_multiseries_interval():
    """
    Test backtesting with interval=[10, 90] in multi-series mode.
    Output DataFrame must have columns level, fold, pred, lower_bound, upper_bound.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, backtest_predictions = backtesting_foundational(
            forecaster=forecaster,
            series=series_wide,
            cv=cv,
            metric="mean_absolute_error",
            interval=[10, 90],
            add_aggregated_metric=False,
            verbose=False,
            show_progress=False,
        )

    assert backtest_predictions.columns.tolist() == [
        "level", "fold", "pred", "lower_bound", "upper_bound"
    ]
    assert backtest_predictions.shape == (24, 5)
    np.testing.assert_array_almost_equal(
        backtest_predictions["lower_bound"].values, np.full(24, 0.1)
    )
    np.testing.assert_array_almost_equal(
        backtest_predictions["upper_bound"].values, np.full(24, 0.9)
    )


def test_output_backtesting_foundational_multiseries_quantiles():
    """
    Test backtesting with quantiles=[0.1, 0.5, 0.9] in multi-series mode.
    Output DataFrame must have columns level, fold, q_0.1, q_0.5, q_0.9.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, backtest_predictions = backtesting_foundational(
            forecaster=forecaster,
            series=series_wide,
            cv=cv,
            metric="mean_absolute_error",
            quantiles=[0.1, 0.5, 0.9],
            add_aggregated_metric=False,
            verbose=False,
            show_progress=False,
        )

    assert backtest_predictions.columns.tolist() == [
        "level", "fold", "q_0.1", "q_0.5", "q_0.9"
    ]
    assert backtest_predictions.shape == (24, 5)
    np.testing.assert_array_almost_equal(
        backtest_predictions["q_0.1"].values, np.full(24, 0.1)
    )
    np.testing.assert_array_almost_equal(
        backtest_predictions["q_0.9"].values, np.full(24, 0.9)
    )


# ===========================================================================
# Multi-series, aggregated metrics
# ===========================================================================

def test_output_backtesting_foundational_multiseries_add_aggregated_metric_true():
    """
    Test that add_aggregated_metric=True includes average, weighted_average, and
    pooling rows in the returned metric DataFrame.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, _ = backtesting_foundational(
            forecaster=forecaster,
            series=series_wide,
            cv=cv,
            metric="mean_absolute_error",
            add_aggregated_metric=True,
            verbose=False,
            show_progress=False,
        )

    # 2 series + 3 aggregated
    assert metric.shape[0] == 5
    assert set(metric["levels"]) == {
        "series_1", "series_2", "average", "weighted_average", "pooling"
    }


def test_output_backtesting_foundational_multiseries_add_aggregated_metric_false():
    """
    Test that add_aggregated_metric=False returns only the per-series metric rows.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, _ = backtesting_foundational(
            forecaster=forecaster,
            series=series_wide,
            cv=cv,
            metric="mean_absolute_error",
            add_aggregated_metric=False,
            verbose=False,
            show_progress=False,
        )

    # Exactly 2 series rows, no aggregated rows
    assert metric.shape[0] == 2
    assert set(metric["levels"]) == {"series_1", "series_2"}
    # average, weighted_average, pooling must NOT be present
    assert "average" not in metric["levels"].values


# ===========================================================================
# Multi-series, exogenous variables
# ===========================================================================

def test_output_backtesting_foundational_multiseries_exog():
    """
    Test backtesting in multi-series mode with exog as a per-series dict.
    FakePipeline ignores input data, so metric values are the same as without exog.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, backtest_predictions = backtesting_foundational(
            forecaster=forecaster,
            series=series_wide,
            exog=exog_dict,
            cv=cv,
            metric="mean_absolute_error",
            add_aggregated_metric=False,
            verbose=False,
            show_progress=False,
        )

    assert metric.shape[0] == 2
    s1_mae = metric.loc[metric["levels"] == "series_1", "mean_absolute_error"].iloc[0]
    s2_mae = metric.loc[metric["levels"] == "series_2", "mean_absolute_error"].iloc[0]
    assert s1_mae == pytest.approx(43.0)
    assert s2_mae == pytest.approx(93.0)
    assert backtest_predictions.shape == (24, 3)
