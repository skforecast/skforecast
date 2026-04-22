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
def test_predict_ValueError_when_not_fitted_and_no_context():
    """
    Test predict raises ValueError when model is not fitted and no
    context is provided.
    """
    m = FoundationModel("autogluon/chronos-2-small")
    err_msg = re.escape("Call `fit` before `predict`, or pass `context`.")
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


@pytest.mark.parametrize(
    "bad_quantiles",
    [0.5, np.float64(0.5), {0.1, 0.5, 0.9}],
    ids=["float", "np.float64", "set"],
)
def test_predict_TypeError_when_quantiles_not_list_or_tuple(bad_quantiles):
    """
    Test predict raises TypeError when `quantiles` is not a list or tuple
    (e.g. a bare float, a numpy scalar, or a set).
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y)
    err_msg = re.escape(
        "`quantiles` must be a `list` or `tuple`. For example, quantiles "
        "0.1, 0.5, and 0.9 should be as `quantiles = [0.1, 0.5, 0.9]`."
    )
    with pytest.raises(TypeError, match=err_msg):
        m.predict(steps=3, quantiles=bad_quantiles)


def test_predict_ValueError_when_steps_is_bool_false():
    """
    Test predict raises ValueError when steps is False. bool is a subclass
    of int in Python, so False (== 0) fails the `steps < 1` check.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y)
    err_msg = re.escape("`steps` must be a positive integer.")
    with pytest.raises(ValueError, match=err_msg):
        m.predict(steps=False)


def test_predict_IgnoredArgumentWarning_when_context_exog_without_context():
    """
    Test predict issues IgnoredArgumentWarning when context_exog is
    provided but context is not, because context_exog is silently
    replaced by the stored context_exog_.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y)
    dummy_exog = pd.DataFrame(
        {"feat": np.arange(50, dtype=float)},
        index=y.index,
    )
    warn_msg = re.escape(
        "`context_exog` is ignored when `context` is not provided."
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        m.predict(steps=5, context_exog=dummy_exog)


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


# Tests predict — context
# ==============================================================================
def test_predict_output_when_context_single_series():
    """
    Test that predict with context produces a forecast index that
    immediately follows the context index.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y)

    context = pd.Series(
        np.arange(10, dtype=float),
        index=pd.date_range("2025-01-01", periods=10, freq="ME"),
        name="sales",
    )
    result = m.predict(steps=3, context=context)

    expected_start = context.index[-1] + context.index.freq
    expected_index = pd.date_range(
        start=expected_start, periods=3, freq=context.index.freq
    )
    pd.testing.assert_index_equal(result.index, expected_index)


@pytest.mark.parametrize(
    "context_input",
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
def test_predict_output_when_context_multiseries(context_input):
    """
    Test that a wide DataFrame or dict[str, pd.Series] passed as
    context produces a long DataFrame with the correct forecast index.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y_dict)
    result = m.predict(steps=4, context=context_input)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["level", "pred"]

    new_idx = pd.date_range("2025-01-01", periods=10, freq="ME")
    expected_start = new_idx[-1] + new_idx.freq
    expected_index = pd.date_range(
        start=expected_start, periods=4, freq=new_idx.freq
    )
    pd.testing.assert_index_equal(result.index.unique(), expected_index)


def test_predict_output_when_context_without_fit():
    """
    Test that predict works on an unfitted model when context is
    provided.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    assert m.is_fitted is False

    context = pd.Series(
        np.arange(20, dtype=float),
        index=pd.date_range("2025-01-01", periods=20, freq="ME"),
        name="sales",
    )
    result = m.predict(steps=5, context=context)

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
def test_predict_does_not_modify_context():
    """
    Test that predict does not modify context.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y)

    context = pd.Series(
        np.arange(10, dtype=float),
        index=pd.date_range("2025-01-01", periods=10, freq="ME"),
        name="sales",
    )
    lw_copy = context.copy()
    m.predict(steps=3, context=context)

    pd.testing.assert_series_equal(context, lw_copy)
