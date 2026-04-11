# Unit test predict FoundationModel
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.exceptions import IgnoredArgumentWarning
from skforecast.foundation._foundation_model import FoundationModel
from .fixtures_adapters import (
    y, data, y_dict,
    FakePipeline, FakeTimesFM25Model,
)


# Tests predict — errors
# ==============================================================================
def test_predict_ValueError_when_not_fitted_and_no_last_window():
    """
    Test predict raises ValueError when model is not fitted and no
    last_window is provided.
    """
    m = FoundationModel("autogluon/chronos-2-small")
    err_msg = re.escape("Call `fit` before `predict`, or pass `last_window`.")
    with pytest.raises(ValueError, match=err_msg):
        m.predict(steps=5)


@pytest.mark.parametrize(
    "steps",
    [0, -1, -10, "abc", 1.5],
    ids=lambda x: f"steps: {x}",
)
def test_predict_ValueError_when_steps_not_positive(steps):
    """
    Test predict raises ValueError when steps is 0, negative, or not an
    integer type.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y)
    err_msg = re.escape("`steps` must be a positive integer.")
    with pytest.raises(ValueError, match=err_msg):
        m.predict(steps=steps)


@pytest.mark.parametrize(
    "bad_quantile",
    [-0.1, 1.1, 2.0],
    ids=lambda x: f"quantile: {x}",
)
def test_predict_ValueError_when_quantile_out_of_range(bad_quantile):
    """
    Test predict raises ValueError when a quantile level is outside [0, 1].
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y)
    err_msg = re.escape(
        f"All quantiles must be between 0 and 1. Got {bad_quantile}."
    )
    with pytest.raises(ValueError, match=err_msg):
        m.predict(steps=3, quantiles=[0.5, bad_quantile])


def test_predict_TypeError_when_last_window_index_type_mismatch():
    """
    Test predict raises TypeError when last_window has a different index
    type than the fitted series.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y)  # DatetimeIndex

    last_window = pd.Series(
        np.arange(10, dtype=float),
        index=pd.RangeIndex(start=0, stop=10, step=1),
        name="sales",
    )
    err_msg = re.escape("Expected index of type DatetimeIndex")
    with pytest.raises(TypeError, match=err_msg):
        m.predict(steps=3, last_window=last_window)


def test_predict_TypeError_when_last_window_index_freq_mismatch():
    """
    Test predict raises TypeError when last_window has a different
    frequency than the fitted series.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y)  # freq="ME"

    last_window = pd.Series(
        np.arange(10, dtype=float),
        index=pd.date_range("2025-01-01", periods=10, freq="D"),
        name="sales",
    )
    err_msg = re.escape("Expected frequency")
    with pytest.raises(TypeError, match=err_msg):
        m.predict(steps=3, last_window=last_window)


# Tests predict — single-series output
# ==============================================================================
def test_predict_output_point_forecast():
    """
    Test that predict returns a long-format DataFrame with columns
    ["level", "pred"], correct length, correct index, and expected values
    (FakePipeline returns median = 0.5).
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=data)
    result = m.predict(steps=12)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["level", "pred"]
    assert len(result) == 12

    expected_index = pd.date_range("1961-01-01", periods=12, freq="MS")
    pd.testing.assert_index_equal(result.index, expected_index)
    np.testing.assert_array_almost_equal(
        result["pred"].to_numpy(), np.full(12, 0.5)
    )


def test_predict_output_quantile_forecast():
    """
    Test that predict returns a DataFrame with quantile columns, correct
    index, and each quantile column equals its level (FakePipeline property).
    """
    quantiles = [0.1, 0.5, 0.9]
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=data)
    result = m.predict(steps=12, quantiles=quantiles)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["level", "q_0.1", "q_0.5", "q_0.9"]
    assert len(result) == 12

    expected_index = pd.date_range("1961-01-01", periods=12, freq="MS")
    pd.testing.assert_index_equal(result.index, expected_index)
    for q in quantiles:
        np.testing.assert_array_almost_equal(
            result[f"q_{q}"].to_numpy(), np.full(12, q)
        )


# Tests predict — multi-series output
# ==============================================================================
def test_predict_output_multiseries_point_forecast():
    """
    Test multi-series point forecast: returns a long DataFrame with columns
    ["level", "pred"], correct length (steps * n_series), correct level
    values, and all predictions equal 0.5.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y_dict)
    result = m.predict(steps=5)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["level", "pred"]
    assert len(result) == 5 * 2  # steps * n_series

    for name in ["s1", "s2"]:
        subset = result[result["level"] == name]
        np.testing.assert_array_almost_equal(
            subset["pred"].to_numpy(), np.full(5, 0.5)
        )


def test_predict_output_multiseries_quantile_forecast():
    """
    Test multi-series quantile forecast: returns a long DataFrame with
    columns ["level", "q_0.1", "q_0.5", "q_0.9"] and correct length.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y_dict)
    result = m.predict(steps=5, quantiles=[0.1, 0.5, 0.9])

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["level", "q_0.1", "q_0.5", "q_0.9"]
    assert len(result) == 5 * 2  # steps * n_series


# Tests predict — last_window
# ==============================================================================
def test_predict_output_when_last_window_single_series():
    """
    Test that predict with last_window produces a forecast index that
    immediately follows the last_window index.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y)

    last_window = pd.Series(
        np.arange(10, dtype=float),
        index=pd.date_range("2025-01-01", periods=10, freq="ME"),
        name="sales",
    )
    result = m.predict(steps=3, last_window=last_window)

    expected_start = last_window.index[-1] + last_window.index.freq
    expected_index = pd.date_range(
        start=expected_start, periods=3, freq=last_window.index.freq
    )
    pd.testing.assert_index_equal(result.index, expected_index)


@pytest.mark.parametrize(
    "last_window_input",
    [
        pd.DataFrame(
            {
                "s1": np.arange(10, dtype=float),
                "s2": np.arange(10, 20, dtype=float),
            },
            index=pd.date_range("2025-01-01", periods=10, freq="ME"),
        ),
        {
            "s1": pd.Series(
                np.arange(10, dtype=float),
                index=pd.date_range("2025-01-01", periods=10, freq="ME"),
                name="s1",
            ),
            "s2": pd.Series(
                np.arange(10, 20, dtype=float),
                index=pd.date_range("2025-01-01", periods=10, freq="ME"),
                name="s2",
            ),
        },
    ],
    ids=["wide_dataframe", "dict"],
)
def test_predict_output_when_last_window_multiseries(last_window_input):
    """
    Test that a wide DataFrame or dict[str, pd.Series] passed as
    last_window produces a long DataFrame with the correct forecast index.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y_dict)
    result = m.predict(steps=4, last_window=last_window_input)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["level", "pred"]

    new_idx = pd.date_range("2025-01-01", periods=10, freq="ME")
    expected_start = new_idx[-1] + new_idx.freq
    expected_index = pd.date_range(
        start=expected_start, periods=4, freq=new_idx.freq
    )
    pd.testing.assert_index_equal(result.index.unique(), expected_index)


def test_predict_output_when_last_window_without_fit():
    """
    Test that predict works on an unfitted model when last_window is
    provided.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    assert m.is_fitted is False

    last_window = pd.Series(
        np.arange(20, dtype=float),
        index=pd.date_range("2025-01-01", periods=20, freq="ME"),
        name="sales",
    )
    result = m.predict(steps=5, last_window=last_window)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["level", "pred"]
    assert len(result) == 5


# Tests predict — exog forwarding
# ==============================================================================
def test_predict_passes_future_exog_to_pipeline():
    """
    Test that future exog passed to predict is forwarded as
    future_covariates to the pipeline.
    """
    pipeline = FakePipeline()
    m = FoundationModel("autogluon/chronos-2-small", pipeline=pipeline)
    m.fit(series=y)

    future = pd.DataFrame(
        {"feat_a": np.arange(6, dtype=float)},
        index=pd.date_range("2024-03-01", periods=6, freq="ME"),
    )
    m.predict(steps=6, exog=future)
    assert "future_covariates" in pipeline.last_inputs[0]


def test_predict_cross_learning_forwarded_to_pipeline():
    """
    Test that cross_learning=True is forwarded all the way to
    predict_quantiles in multi-series mode.
    """
    pipeline = FakePipeline()
    m = FoundationModel(
        "autogluon/chronos-2-small", pipeline=pipeline, cross_learning=True
    )
    m.fit(series=y_dict)
    m.predict(steps=3)
    assert pipeline.last_kwargs.get("cross_learning") is True


def test_predict_IgnoredArgumentWarning_when_adapter_no_exog():
    """
    Test that predict issues IgnoredArgumentWarning when exog is passed
    to an adapter that does not support exogenous variables.
    """
    m = FoundationModel("google/timesfm-2.5-200m-pytorch")
    m.adapter._model = FakeTimesFM25Model()
    m.fit(series=y)

    future = pd.DataFrame(
        {"feat_a": np.arange(6, dtype=float)},
        index=pd.date_range("2024-03-01", periods=6, freq="ME"),
    )
    warn_msg = re.escape("does not currently support covariates")
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        m.predict(steps=6, exog=future)


# Tests predict — does not modify input
# ==============================================================================
def test_predict_does_not_modify_last_window():
    """
    Test that predict does not modify last_window.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y)

    last_window = pd.Series(
        np.arange(10, dtype=float),
        index=pd.date_range("2025-01-01", periods=10, freq="ME"),
        name="sales",
    )
    lw_copy = last_window.copy()
    m.predict(steps=3, last_window=last_window)

    pd.testing.assert_series_equal(last_window, lw_copy)
