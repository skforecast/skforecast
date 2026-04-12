# Unit test predict_quantiles ForecasterFoundation
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.foundation import ForecasterFoundation

# Fixtures
from .fixtures_forecaster_foundation import (
    make_forecaster,
    y,
    y_lw,
    exog,
    exog_lw,
    exog_predict,
    exog_predict_lw,
    series_df,
    series_dict,
    lw_dict,
)


# Tests predict_quantiles — errors
# ==============================================================================

def test_predict_quantiles_NotFittedError_when_not_fitted():
    """
    Raise NotFittedError when forecaster is not fitted, regardless of whether
    last_window is provided.
    """
    forecaster = make_forecaster()

    err_msg = re.escape(
        "This forecaster is not fitted yet. Call `fit` with appropriate "
        "arguments before using `predict_quantiles()`."
    )
    with pytest.raises(NotFittedError, match=err_msg):
        forecaster.predict_quantiles(steps=5)

    with pytest.raises(NotFittedError):
        forecaster.predict_quantiles(steps=3, last_window=y)


# Tests predict_quantiles — single series basic output
# ==============================================================================

def test_predict_quantiles_output_when_single_series():
    """
    predict_quantiles() returns a DataFrame with correct columns, length,
    values, and index continuation from the training series.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_quantiles(steps=5, quantiles=[0.1, 0.5, 0.9])

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["level", "q_0.1", "q_0.5", "q_0.9"]
    assert len(result) == 5
    np.testing.assert_array_almost_equal(result["q_0.1"].values, [0.1] * 5)
    np.testing.assert_array_almost_equal(result["q_0.5"].values, [0.5] * 5)
    np.testing.assert_array_almost_equal(result["q_0.9"].values, [0.9] * 5)

    expected_index = pd.date_range(
        start=y.index[-1] + y.index.freq,
        periods=5,
        freq=y.index.freq,
    )
    pd.testing.assert_index_equal(result.index, expected_index)


@pytest.mark.parametrize(
    "quantiles, expected_columns",
    [
        ([0.05, 0.5, 0.95], ["level", "q_0.05", "q_0.5", "q_0.95"]),
        ([0.5], ["level", "q_0.5"]),
        ([0.25, 0.75], ["level", "q_0.25", "q_0.75"]),
    ],
    ids=["custom_three", "single_quantile", "custom_two"],
)
def test_predict_quantiles_custom_quantiles(quantiles, expected_columns):
    """
    Custom quantiles produce the matching column names and correct length.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_quantiles(steps=4, quantiles=quantiles)
    assert list(result.columns) == expected_columns
    assert len(result) == 4


@pytest.mark.parametrize(
    "quantiles",
    [[0.1, 0.9], [0.25, 0.5, 0.75], [0.05, 0.5, 0.95]],
    ids=lambda v: f"quantiles={v}",
)
def test_predict_quantiles_monotone_columns(quantiles):
    """
    For FakePipeline outputs: quantile columns are monotonically non-decreasing.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_quantiles(steps=5, quantiles=quantiles)

    q_cols = [c for c in result.columns if c.startswith("q_")]
    for i in range(len(q_cols) - 1):
        assert (result[q_cols[i]] <= result[q_cols[i + 1]]).all()


# Tests predict_quantiles — multi-series basic output
# ==============================================================================

def test_predict_quantiles_output_when_multiseries():
    """
    predict_quantiles() with a multi-series fit returns correct columns,
    row count, level values, and quantile values matching the quantile levels.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=series_df)
    result = forecaster.predict_quantiles(steps=5)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["level", "q_0.1", "q_0.5", "q_0.9"]
    assert len(result) == 5 * 2
    expected_levels = np.tile(["s1", "s2"], 5)
    np.testing.assert_array_equal(result["level"].values, expected_levels)


@pytest.mark.parametrize(
    "levels, expected_levels",
    [(["s1"], {"s1"}), ("s2", {"s2"})],
    ids=["levels_list", "levels_str"],
)
def test_predict_quantiles_with_levels_filter(levels, expected_levels):
    """
    predict_quantiles() with `levels` filters the output to the specified series.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=series_df)
    result = forecaster.predict_quantiles(steps=5, levels=levels)
    assert set(result["level"].unique()) == expected_levels
    assert len(result) == 5


def test_predict_quantiles_values_match_quantile_levels():
    """
    FakePipeline sets q_<x> = x for every step, so we can verify values
    exactly.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=series_df)
    result = forecaster.predict_quantiles(steps=3, quantiles=[0.1, 0.9])
    np.testing.assert_allclose(result["q_0.1"].values, 0.1)
    np.testing.assert_allclose(result["q_0.9"].values, 0.9)


def test_predict_quantiles_dict_input():
    """
    Dict-based fit also routes through multi-series path.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=series_dict)
    result = forecaster.predict_quantiles(steps=5)
    assert list(result.columns) == ["level", "q_0.1", "q_0.5", "q_0.9"]
    assert len(result) == 10


# Tests predict_quantiles — last_window and exog
# ==============================================================================

def test_predict_quantiles_with_last_window_and_exog():
    """
    predict_quantiles() with last_window, last_window_exog, and exog returns
    the correct output. Exog longer than steps is trimmed automatically.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y, exog=exog)

    # With exog
    result = forecaster.predict_quantiles(steps=5, exog=exog_predict)
    assert len(result) == 5
    assert "q_0.5" in result.columns

    # With last_window + last_window_exog
    result_lw = forecaster.predict_quantiles(
        steps=3,
        exog=exog_predict_lw.iloc[:3],
        last_window=y_lw,
        last_window_exog=exog_lw,
    )
    assert len(result_lw) == 3
    assert isinstance(result_lw, pd.DataFrame)

    # Exog trimmed to steps
    long_exog = pd.DataFrame(
        {"feat_a": np.arange(70, 80, dtype=float)},
        index=pd.date_range("2024-03-31", periods=10, freq="ME"),
    )
    result_trimmed = forecaster.predict_quantiles(steps=5, exog=long_exog)
    assert len(result_trimmed) == 5

    # last_window index and values (no exog)
    forecaster_no_exog = make_forecaster()
    forecaster_no_exog.fit(series=y)
    result_idx = forecaster_no_exog.predict_quantiles(
        steps=4, quantiles=[0.1, 0.5, 0.9], last_window=y_lw
    )
    expected_index = pd.date_range(
        start=y_lw.index[-1] + y_lw.index.freq,
        periods=4,
        freq=y_lw.index.freq,
    )
    pd.testing.assert_index_equal(result_idx.index, expected_index)
    np.testing.assert_array_almost_equal(result_idx["q_0.1"].values, [0.1] * 4)
    np.testing.assert_array_almost_equal(result_idx["q_0.5"].values, [0.5] * 4)
    np.testing.assert_array_almost_equal(result_idx["q_0.9"].values, [0.9] * 4)


def test_predict_quantiles_with_last_window_multiseries():
    """
    predict_quantiles() with multi-series last_window (dict) returns
    correct output.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=series_df)
    result = forecaster.predict_quantiles(steps=5, last_window=lw_dict)
    assert "level" in result.columns
