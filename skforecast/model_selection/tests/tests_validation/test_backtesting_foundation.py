# Unit test backtesting_foundation
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
from skforecast.model_selection import backtesting_foundation
from skforecast.exceptions import IgnoredArgumentWarning, MissingValuesWarning

# Fixtures — reuse the FakePipeline-backed forecaster and series fixtures
# from foundation tests
from ....foundation.tests.tests_forecaster_foundation.fixtures_forecaster_foundation import (
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

def test_backtesting_foundation_TypeError_when_wrong_forecaster_type():
    """
    Test that TypeError is raised when forecaster is not ForecasterFoundation.
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
        "`forecaster` must be of type `ForecasterFoundation`. For all "
        "other types of forecasters use the other functions available in "
        "the `model_selection` module."
    )
    with pytest.raises(TypeError, match=err_msg):
        backtesting_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
        )


def test_backtesting_foundation_warns_IgnoredArgumentWarning_when_quantiles_missing_median():
    """
    Test that IgnoredArgumentWarning is raised when quantiles does not
    include 0.5 (median), because it is auto-added for metric computation.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=True,
        fixed_train_size=False,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        with pytest.warns(
            IgnoredArgumentWarning,
            match="The median quantile",
        ):
            backtesting_foundation(
                forecaster=forecaster,
                series=y,
                cv=cv,
                metric="mean_absolute_error",
                quantiles=[0.1, 0.9],
                show_progress=False,
            )


def test_backtesting_foundation_ValueError_when_quantiles_invalid():
    """
    Test that ValueError is raised when quantiles contains a value outside [0, 1].
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        verbose=False,
    )
    err_msg = re.escape(
        "All elements in `quantiles` must be >= 0 and <= 1."
    )
    with pytest.raises(ValueError, match=err_msg):
        backtesting_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            quantiles=[0.1, 1.5],
        )


@pytest.mark.parametrize(
    "levels",
    ["does_not_exist", ["series_1", "does_not_exist"]],
    ids=lambda levels: f"levels: {levels}",
)
def test_backtesting_foundation_ValueError_when_levels_not_in_series(levels):
    """
    Test that ValueError is raised when `levels` contains a name that is not
    a series, mirroring `backtesting_forecaster_multiseries`.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        verbose=False,
    )
    levels_list = [levels] if isinstance(levels, str) else levels
    err_msg = re.escape(
        f"Levels {levels_list} not found in `series`, available levels are "
        f"['series_1', 'series_2']. Review `levels` argument."
    )
    with pytest.raises(ValueError, match=err_msg):
        backtesting_foundation(
            forecaster=forecaster,
            series=series_wide,
            cv=cv,
            metric="mean_absolute_error",
            levels=levels,
            show_progress=False,
        )


# ===========================================================================
# Single-series, point forecast
# ===========================================================================

@pytest.mark.parametrize(
    "initial_train_size",
    [38, "2023-02-28"],
    ids=lambda v: f'initial_train_size: {v}',
)
def test_output_backtesting_foundation_single_no_refit_no_exog_no_remainder(initial_train_size):
    """
    Test output of backtesting_foundation for single series, no refit, no exog,
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
        metric, backtest_predictions = backtesting_foundation(
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
    expected_preds.insert(0, "level", "y")
    expected_preds.insert(1, "fold", [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions)


@pytest.mark.parametrize(
    "initial_train_size",
    [38, "2023-02-28"],
    ids=lambda v: f'initial_train_size: {v}',
)
def test_output_backtesting_foundation_single_no_refit_no_exog_remainder(initial_train_size):
    """
    Test output of backtesting_foundation for single series, no refit, no exog,
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
        metric, backtest_predictions = backtesting_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            verbose=False,
            show_progress=False,
        )

    expected_metric = pd.DataFrame({"mean_absolute_error": [43.0]})
    assert backtest_predictions.shape == (12, 3)
    assert backtest_predictions["fold"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2]
    assert backtest_predictions.columns.tolist() == ["level", "fold", "pred"]
    pd.testing.assert_frame_equal(expected_metric, metric)


def test_output_backtesting_foundation_single_no_refit_yes_exog():
    """
    Test output of backtesting_foundation for single series, no refit, with exog.
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
        metric, backtest_predictions = backtesting_foundation(
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
    assert backtest_predictions.shape == (12, 3)
    assert (backtest_predictions["pred"] == 0.5).all()


def test_output_backtesting_foundation_single_refit_no_exog_no_remainder():
    """
    Test output of backtesting_foundation for single series, with refit, no exog,
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
        metric, backtest_predictions = backtesting_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            verbose=False,
            show_progress=False,
        )

    expected_metric = pd.DataFrame({"mean_absolute_error": [43.0]})
    pd.testing.assert_frame_equal(expected_metric, metric)
    assert backtest_predictions.shape == (12, 3)
    assert (backtest_predictions["pred"] == 0.5).all()
    assert backtest_predictions["fold"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_output_backtesting_foundation_single_refit_fixed_train_size_no_exog():
    """
    Test output of backtesting_foundation for single series, refit with fixed
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
        metric, backtest_predictions = backtesting_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            verbose=False,
            show_progress=False,
        )

    expected_metric = pd.DataFrame({"mean_absolute_error": [43.0]})
    pd.testing.assert_frame_equal(expected_metric, metric)
    assert backtest_predictions.shape == (12, 3)


def test_output_backtesting_foundation_single_refit_yes_exog():
    """
    Test output of backtesting_foundation for single series, refit, with exog.
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
        metric, backtest_predictions = backtesting_foundation(
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
    assert backtest_predictions.shape == (12, 3)


# ===========================================================================
# Single-series, metrics
# ===========================================================================

def test_output_backtesting_foundation_single_callable_metric():
    """
    Test output of backtesting_foundation with a custom callable metric.
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
        metric, _ = backtesting_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric=my_mae,
            verbose=False,
            show_progress=False,
        )

    expected_metric = pd.DataFrame({"my_mae": [43.0]})
    pd.testing.assert_frame_equal(expected_metric, metric)


def test_output_backtesting_foundation_single_list_of_metrics():
    """
    Test output of backtesting_foundation with a list of metrics. The returned
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
        metric, _ = backtesting_foundation(
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

def test_output_backtesting_foundation_single_quantiles_auto_add_median_no_refit():
    """
    Test output columns and values for backtesting with quantiles=[0.1, 0.9].
    The median (0.5) is auto-added. FakePipeline returns quantile level as
    value, so q_0.1=0.1, q_0.5=0.5, q_0.9=0.9.
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
        metric, backtest_predictions = backtesting_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            quantiles=[0.1, 0.9],
            verbose=False,
            show_progress=False,
        )

    assert backtest_predictions.columns.tolist() == [
        "level", "fold", "q_0.1", "q_0.5", "q_0.9"
    ]
    assert backtest_predictions.shape == (12, 5)
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


def test_output_backtesting_foundation_single_quantiles_auto_add_median_refit():
    """
    Test output columns and values for backtesting with quantiles=[0.1, 0.9]
    and refit=True. The median (0.5) is auto-added.
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
        metric, backtest_predictions = backtesting_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            quantiles=[0.1, 0.9],
            verbose=False,
            show_progress=False,
        )

    assert backtest_predictions.columns.tolist() == [
        "level", "fold", "q_0.1", "q_0.5", "q_0.9"
    ]
    assert backtest_predictions.shape == (12, 5)
    np.testing.assert_array_almost_equal(
        backtest_predictions["q_0.1"].values, np.full(12, 0.1)
    )
    np.testing.assert_array_almost_equal(
        backtest_predictions["q_0.9"].values, np.full(12, 0.9)
    )


def test_output_backtesting_foundation_single_quantiles_no_refit():
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
        metric, backtest_predictions = backtesting_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            quantiles=[0.1, 0.5, 0.9],
            verbose=False,
            show_progress=False,
        )

    assert backtest_predictions.columns.tolist() == [
        "level", "fold", "q_0.1", "q_0.5", "q_0.9"
    ]
    assert backtest_predictions.shape == (12, 5)
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


def test_output_backtesting_foundation_single_quantiles_refit():
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
        metric, backtest_predictions = backtesting_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            quantiles=[0.1, 0.5, 0.9],
            verbose=False,
            show_progress=False,
        )

    assert backtest_predictions.columns.tolist() == [
        "level", "fold", "q_0.1", "q_0.5", "q_0.9"
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

def test_output_backtesting_foundation_single_fold_stride():
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
        metric, backtest_predictions = backtesting_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            verbose=False,
            show_progress=False,
        )

    # fold_stride=2, steps=3, 12 available test obs → overlapping folds
    # produce 17 total rows (confirmed with exact expected value)
    assert backtest_predictions.shape == (17, 3)
    np.testing.assert_array_almost_equal(
        backtest_predictions["pred"].values, np.full(17, 0.5)
    )
    assert "fold" in backtest_predictions.columns


def test_output_backtesting_foundation_single_gap():
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
        metric, backtest_predictions = backtesting_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            metric="mean_absolute_error",
            verbose=False,
            show_progress=False,
        )

    expected_metric = pd.DataFrame({"mean_absolute_error": [44.0]})
    pd.testing.assert_frame_equal(expected_metric, metric)
    assert backtest_predictions.shape == (10, 3)


# ===========================================================================
# Multi-series, point forecast
# ===========================================================================

def test_output_backtesting_foundation_multiseries_dataframe_no_refit():
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
        metric, backtest_predictions = backtesting_foundation(
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


def test_output_backtesting_foundation_multiseries_dataframe_refit():
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
        metric, backtest_predictions = backtesting_foundation(
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


def test_output_backtesting_foundation_multiseries_dict_no_refit():
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
        metric_df, preds_df = backtesting_foundation(
            forecaster=forecaster_df,
            series=series_wide,
            cv=cv,
            metric="mean_absolute_error",
            add_aggregated_metric=False,
            verbose=False,
            show_progress=False,
        )
        metric_dict, preds_dict = backtesting_foundation(
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

def test_output_backtesting_foundation_multiseries_levels_filter():
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
        metric, backtest_predictions = backtesting_foundation(
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

def test_output_backtesting_foundation_multiseries_quantiles_auto_add_median():
    """
    Test backtesting with quantiles=[0.1, 0.9] in multi-series mode.
    The median (0.5) is auto-added. Output DataFrame must have columns
    level, fold, q_0.1, q_0.5, q_0.9.
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
        metric, backtest_predictions = backtesting_foundation(
            forecaster=forecaster,
            series=series_wide,
            cv=cv,
            metric="mean_absolute_error",
            quantiles=[0.1, 0.9],
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


def test_output_backtesting_foundation_multiseries_quantiles():
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
        metric, backtest_predictions = backtesting_foundation(
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

def test_output_backtesting_foundation_multiseries_add_aggregated_metric_true():
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
        metric, _ = backtesting_foundation(
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


def test_output_backtesting_foundation_multiseries_add_aggregated_metric_false():
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
        metric, _ = backtesting_foundation(
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
# Multi-series, fold options
# ===========================================================================

# series_2 is NaN from position 46 onwards, so it is skipped in the folds
# whose test window falls entirely in that range.
_series_2_tail_nan = np.arange(50, 100, dtype=float)
_series_2_tail_nan[46:] = np.nan
series_dict_tail_nan = {
    "series_1": pd.Series(np.arange(50, dtype=float), index=_index, name="series_1"),
    "series_2": pd.Series(_series_2_tail_nan, index=_index, name="series_2"),
}


def test_output_backtesting_foundation_multiseries_gap():
    """
    Test multi-series backtesting with gap=2. The rows of each fold are
    step-major (all levels for step 1, then step 2, ...) and the first
    `len(levels_predict) * gap` rows are dropped, also in the folds where
    only series_1 is predicted (series_2 has no observed value in the test
    window of folds 2 and 3).
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        gap=2,
        allow_incomplete_fold=True,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, backtest_predictions = backtesting_foundation(
            forecaster=forecaster,
            series=series_dict_tail_nan,
            cv=cv,
            metric="mean_absolute_error",
            add_aggregated_metric=False,
            verbose=False,
            show_progress=False,
        )

    expected_metric = pd.DataFrame(
        {"levels": ["series_1", "series_2"], "mean_absolute_error": [44.0, 92.0]}
    )
    expected_index = pd.DatetimeIndex(
        ["2023-05-31"] * 2 + ["2023-06-30"] * 2 + ["2023-07-31"] * 2
        + ["2023-08-31"] * 2 + ["2023-09-30"] * 2 + ["2023-10-31"] * 2
        + ["2023-11-30", "2023-12-31", "2024-01-31", "2024-02-29"]
    )
    expected_predictions = pd.DataFrame(
        {
            "level": ["series_1", "series_2"] * 6 + ["series_1"] * 4,
            "fold": [0] * 6 + [1] * 6 + [2] * 3 + [3],
            "pred": [0.5] * 16,
        },
        index=expected_index,
    )

    pd.testing.assert_frame_equal(metric, expected_metric)
    pd.testing.assert_frame_equal(backtest_predictions, expected_predictions)


def test_output_backtesting_foundation_multiseries_fold_stride():
    """
    Test multi-series backtesting with fold_stride=2 < steps=3 (overlapping
    test windows). Overlapping dates appear once per fold in the output, the
    metrics keep the last prediction of each date, and a series is skipped
    in the folds whose test window has no observed value (series_2 in folds
    4 and 5) while its NaN dates in earlier folds are masked.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        fold_stride=2,
        refit=False,
        gap=0,
        allow_incomplete_fold=True,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        metric, backtest_predictions = backtesting_foundation(
            forecaster=forecaster,
            series=series_dict_tail_nan,
            cv=cv,
            metric="mean_absolute_error",
            add_aggregated_metric=False,
            verbose=False,
            show_progress=False,
        )

    expected_metric = pd.DataFrame(
        {"levels": ["series_1", "series_2"], "mean_absolute_error": [43.0, 91.0]}
    )
    expected_index = pd.DatetimeIndex(
        ["2023-03-31"] * 2 + ["2023-04-30"] * 2 + ["2023-05-31"] * 2
        + ["2023-05-31"] * 2 + ["2023-06-30"] * 2 + ["2023-07-31"] * 2
        + ["2023-07-31"] * 2 + ["2023-08-31"] * 2 + ["2023-09-30"] * 2
        + ["2023-09-30"] * 2 + ["2023-10-31"] * 2 + ["2023-11-30"] * 2
        + ["2023-11-30", "2023-12-31", "2024-01-31"]
        + ["2024-01-31", "2024-02-29"]
    )
    expected_predictions = pd.DataFrame(
        {
            "level": ["series_1", "series_2"] * 12 + ["series_1"] * 5,
            "fold": [0] * 6 + [1] * 6 + [2] * 6 + [3] * 6 + [4] * 3 + [5] * 2,
            "pred": [0.5] * 23 + [np.nan] + [0.5] * 5,
        },
        index=expected_index,
    )

    pd.testing.assert_frame_equal(metric, expected_metric)
    pd.testing.assert_frame_equal(backtest_predictions, expected_predictions)


# ===========================================================================
# Multi-series, exogenous variables
# ===========================================================================

def test_output_backtesting_foundation_multiseries_exog():
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
        metric, backtest_predictions = backtesting_foundation(
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


# ===========================================================================
# IgnoredArgumentWarning
# ===========================================================================

def test_backtesting_foundation_warns_IgnoredArgumentWarning_when_refit_True():
    """
    Test that an IgnoredArgumentWarning is raised when refit=True is passed,
    because foundation models are zero-shot and do not use refit.
    The warning fires when the user explicitly sets non-default values
    (refit=True or fixed_train_size=False), signalling they expect specific
    behaviour that does not apply to foundation models.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=True,
        fixed_train_size=False,
        verbose=False,
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ):
        with pytest.warns(
            IgnoredArgumentWarning,
            match="`refit` and `fixed_train_size` have no effect",
        ):
            backtesting_foundation(
                forecaster=forecaster,
                series=y,
                cv=cv,
                metric="mean_absolute_error",
                show_progress=False,
            )



# ===========================================================================
# Multi-series, heterogeneous series (different lengths, NaN, short exog)
# ===========================================================================

# Four monthly series over a 50-period span: `s1` is complete, `s2` has a NaN
# block at positions 40..43 (inside the test windows of folds 0 and 1 and at
# the end of the train span of fold 2), `s3` ends at position 41 and its exog
# ends with it, `s4` ends at position 29 (before the first test window).
_hetero_index = pd.date_range("2020-01-31", periods=50, freq="ME")
_s2_values = np.arange(50, 100, dtype=float)
_s2_values[40:44] = np.nan
series_dict_hetero = {
    "s1": pd.Series(np.arange(50, dtype=float), index=_hetero_index, name="s1"),
    "s2": pd.Series(_s2_values, index=_hetero_index, name="s2"),
    "s3": pd.Series(np.arange(100, 142, dtype=float), index=_hetero_index[:42], name="s3"),
    "s4": pd.Series(np.arange(200, 230, dtype=float), index=_hetero_index[:30], name="s4"),
}
exog_dict_hetero = {
    name: pd.DataFrame({"feat_a": np.arange(len(s), dtype=float)}, index=s.index)
    for name, s in series_dict_hetero.items()
}


def test_output_backtesting_foundation_multiseries_heterogeneous_series():
    """
    Test backtesting with series of different lengths, a NaN block and an
    exog that ends before the horizon. Every prediction is dated inside its
    fold, a series is predicted in a fold only if it has an actual value in
    the test window (`s2` is skipped in fold 1, `s3` after fold 1, `s4`
    always), the exog ending early is reindexed with a MissingValuesWarning,
    and the metric of a never predicted series is NaN.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        verbose=False,
    )
    warn_msg = re.escape(
        "`exog` for series ['s3'] has been reindexed to match the expected "
        "forecast horizon. Missing timestamps were filled with NaN."
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ), pytest.warns(MissingValuesWarning, match=warn_msg):
        metric, backtest_predictions = backtesting_foundation(
            forecaster=forecaster,
            series=series_dict_hetero,
            exog=exog_dict_hetero,
            cv=cv,
            metric="mean_absolute_error",
            add_aggregated_metric=False,
            verbose=False,
            show_progress=False,
        )

    expected_metric = pd.DataFrame(
        {
            "levels": ["s1", "s2", "s3", "s4"],
            "mean_absolute_error": [43.0, 94.0, 139.0, np.nan],
        }
    )
    expected_index = pd.DatetimeIndex(
        ["2023-03-31"] * 3 + ["2023-04-30"] * 3 + ["2023-05-31"] * 3
        + ["2023-06-30"] * 2 + ["2023-07-31"] * 2 + ["2023-08-31"] * 2
        + ["2023-09-30"] * 2 + ["2023-10-31"] * 2 + ["2023-11-30"] * 2
        + ["2023-12-31"] * 2 + ["2024-01-31"] * 2 + ["2024-02-29"] * 2
    )
    expected_predictions = pd.DataFrame(
        {
            "level": (
                ["s1", "s2", "s3"] * 3 + ["s1", "s3"] * 3 + ["s1", "s2"] * 6
            ),
            "fold": [0] * 9 + [1] * 6 + [2] * 6 + [3] * 6,
            "pred": [
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, np.nan, 0.5,
                0.5, 0.5, 0.5, np.nan, 0.5, np.nan,
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
            ],
        },
        index=expected_index,
    )

    pd.testing.assert_frame_equal(metric, expected_metric)
    pd.testing.assert_frame_equal(backtest_predictions, expected_predictions)


def test_output_backtesting_foundation_fold_skipped_when_no_levels_to_predict():
    """
    Test that a fold where none of the requested levels has an observed value
    in the test window is skipped with a MissingValuesWarning and contributes
    no rows, while the other folds are predicted and the metric is computed
    (`s2` has a NaN block covering the whole test window of fold 1).
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(
        steps=3,
        initial_train_size=38,
        refit=False,
        verbose=False,
    )
    warn_msg = re.escape(
        "Fold 1 has been skipped because none of the levels to predict ['s2'] "
        "have observed values in both its context window and its test window. "
        "No predictions are generated for this fold."
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ), pytest.warns(MissingValuesWarning, match=warn_msg):
        metric, backtest_predictions = backtesting_foundation(
            forecaster=forecaster,
            series=series_dict_hetero,
            cv=cv,
            levels="s2",
            metric="mean_absolute_error",
            add_aggregated_metric=False,
            verbose=False,
            show_progress=False,
        )

    expected_metric = pd.DataFrame(
        {"levels": ["s2"], "mean_absolute_error": [94.0]}
    )
    expected_predictions = pd.DataFrame(
        {
            "level": ["s2"] * 9,
            "fold": [0, 0, 0, 2, 2, 2, 3, 3, 3],
            "pred": [0.5, 0.5, np.nan, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        },
        index=pd.DatetimeIndex(
            ["2023-03-31", "2023-04-30", "2023-05-31",
             "2023-09-30", "2023-10-31", "2023-11-30",
             "2023-12-31", "2024-01-31", "2024-02-29"]
        ),
    )

    pd.testing.assert_frame_equal(metric, expected_metric)
    pd.testing.assert_frame_equal(backtest_predictions, expected_predictions)


def test_output_backtesting_foundation_all_folds_skipped():
    """
    Test that, when the requested level ends before every test window, every
    fold is skipped with a MissingValuesWarning, the predictions DataFrame is
    empty but keeps its columns and the metric is None, as in
    backtesting_forecaster_multiseries.
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
    ), pytest.warns(MissingValuesWarning, match="Fold 0 has been skipped"):
        metric, backtest_predictions = backtesting_foundation(
            forecaster=forecaster,
            series=series_dict_hetero,
            cv=cv,
            levels="s4",
            metric="mean_absolute_error",
            add_aggregated_metric=False,
            verbose=False,
            show_progress=False,
        )

    expected_metric = pd.DataFrame(
        {"levels": ["s4"], "mean_absolute_error": [None]}
    )

    pd.testing.assert_frame_equal(metric, expected_metric)
    assert backtest_predictions.empty
    assert list(backtest_predictions.columns) == ["level", "fold", "pred"]


def test_output_backtesting_foundation_series_skipped_when_context_window_all_nan():
    """
    Test that a series is not predicted in a fold when the last
    `context_length` observations before the train end are all NaN, even if
    it has observed values in the test window (`s5` has NaN at positions
    20..37 and `context_length=10`, so its fold 0 context is entirely NaN),
    that it is predicted in the later folds once its context has values, and
    that the fold is skipped with a MissingValuesWarning when it was the only
    level to predict.
    """
    _s5_values = np.arange(100, 150, dtype=float)
    _s5_values[20:38] = np.nan
    series_dict_nan_context = {
        "s1": series_dict_hetero["s1"],
        "s5": pd.Series(_s5_values, index=_hetero_index, name="s5"),
    }
    forecaster = make_forecaster(context_length=10)
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
        metric, backtest_predictions = backtesting_foundation(
            forecaster=forecaster,
            series=series_dict_nan_context,
            cv=cv,
            metric="mean_absolute_error",
            add_aggregated_metric=False,
            verbose=False,
            show_progress=False,
        )

    expected_metric = pd.DataFrame(
        {"levels": ["s1", "s5"], "mean_absolute_error": [43.0, 144.5]}
    )
    expected_index = pd.DatetimeIndex(
        ["2023-03-31", "2023-04-30", "2023-05-31"]
        + ["2023-06-30"] * 2 + ["2023-07-31"] * 2 + ["2023-08-31"] * 2
        + ["2023-09-30"] * 2 + ["2023-10-31"] * 2 + ["2023-11-30"] * 2
        + ["2023-12-31"] * 2 + ["2024-01-31"] * 2 + ["2024-02-29"] * 2
    )
    expected_predictions = pd.DataFrame(
        {
            "level": ["s1"] * 3 + ["s1", "s5"] * 9,
            "fold": [0] * 3 + [1] * 6 + [2] * 6 + [3] * 6,
            "pred": [0.5] * 21,
        },
        index=expected_index,
    )

    pd.testing.assert_frame_equal(metric, expected_metric)
    pd.testing.assert_frame_equal(backtest_predictions, expected_predictions)

    warn_msg = re.escape(
        "Fold 0 has been skipped because none of the levels to predict ['s5'] "
        "have observed values in both its context window and its test window. "
        "No predictions are generated for this fold."
    )
    with patch(
        "skforecast.model_selection._validation.deepcopy_forecaster",
        side_effect=deepcopy,
    ), pytest.warns(MissingValuesWarning, match=warn_msg):
        backtesting_foundation(
            forecaster=forecaster,
            series=series_dict_nan_context,
            cv=cv,
            levels="s5",
            metric="mean_absolute_error",
            add_aggregated_metric=False,
            verbose=False,
            show_progress=False,
        )
