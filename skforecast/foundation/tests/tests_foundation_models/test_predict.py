# Unit test predict FoundationModel
# ==============================================================================
import re
import warnings
import pytest
import numpy as np
import pandas as pd
from skforecast.exceptions import IgnoredArgumentWarning, MissingValuesWarning
from skforecast.foundation._foundation_model import FoundationModel
from .fixtures_adapters import (
    y, data, y_dict,
    FakePipeline, FakeTimesFM25Model, FakeTimesFM3Forecaster, FakeT0Forecaster,
    FakeTimestralForecaster,
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


@pytest.mark.parametrize("steps", [True, False], ids=["True", "False"])
def test_predict_ValueError_when_steps_is_bool(steps):
    """
    Test predict rejects boolean `steps` rather than silently treating
    True as steps=1 (bool is an int subclass).
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


def test_predict_ValueError_when_context_is_empty_dict():
    """
    Test predict raises ValueError when `context` is an empty dict,
    both when `check_inputs=True` and `check_inputs=False`.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y)

    err_msg1 = re.escape("`series` cannot be an empty dictionary or an empty DataFrame.")
    with pytest.raises(ValueError, match=err_msg1):
        m.predict(steps=3, context={}, check_inputs=True)

    err_msg2 = re.escape("`context` cannot be an empty dictionary.")
    with pytest.raises(ValueError, match=err_msg2):
        m.predict(steps=3, context={}, check_inputs=False)


@pytest.mark.parametrize(
    "index",
    [
        pd.DatetimeIndex(["2020-01-01", "2020-01-03"]),
        pd.DatetimeIndex(["2020-01-01", "2020-01-03", "2020-01-10"]),
    ],
    ids=["fewer_than_3_observations", "irregularly_spaced"],
)
def test_predict_ValueError_when_check_inputs_False_and_context_freq_not_inferable(index):
    """
    Test predict raises a clear ValueError, instead of an unhandled
    TypeError, when `check_inputs=False` skips the usual freq validation
    and the context's DatetimeIndex has no freq that pandas can infer.
    This is the path `backtesting_foundation` uses internally, and it
    applies to every adapter (not just the ones with their own
    timestamp-building logic) because it goes through the shared
    `expand_index` call that builds the output prediction index.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    context = {"sales": pd.Series(np.arange(len(index), dtype=float), index=index, name="sales")}

    err_msg = re.escape("Could not infer a frequency from `index`.")
    with pytest.raises(ValueError, match=err_msg):
        m.predict(steps=3, context=context, check_inputs=False)


# Tests predict — exog forwarding
# ==============================================================================
def test_predict_passes_future_exog_to_pipeline():
    """
    Test that future exog passed to predict is forwarded as
    future_covariates to the pipeline.
    """
    pipeline = FakePipeline()
    m = FoundationModel("autogluon/chronos-2-small", pipeline=pipeline)
    exog_fit = pd.DataFrame(
        {"feat_a": np.arange(len(y), dtype=float)}, index=y.index
    )
    m.fit(series=y, exog=exog_fit)

    future = pd.DataFrame(
        {"feat_a": np.arange(6, dtype=float)},
        index=pd.date_range("2024-03-01", periods=6, freq="ME"),
    )
    m.predict(steps=6, exog=future)
    assert "future_covariates" in pipeline.last_inputs[0]


def test_predict_ValueError_when_future_exog_column_has_no_history():
    """
    Test predict raises ValueError through the user-facing path
    (check_inputs=True) when a future exog column has no historical values
    in the context_exog of the same series.
    """
    pipeline = FakePipeline()
    m = FoundationModel("autogluon/chronos-2-small", pipeline=pipeline)
    exog_fit = pd.DataFrame(
        {"feat_a": np.arange(len(y), dtype=float)}, index=y.index
    )
    m.fit(series=y, exog=exog_fit)

    future = pd.DataFrame(
        {"feat_b": np.arange(6, dtype=float)},
        index=pd.date_range("2024-03-01", periods=6, freq="ME"),
    )
    err_msg = re.escape(
        "`exog` contains columns with no historical values in the context "
        "for series {'sales': ['feat_b']}."
    )
    with pytest.raises(ValueError, match=err_msg):
        m.predict(steps=6, exog=future)


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


# Tests predict: levels filtering
# ==============================================================================
def test_predict_ValueError_when_levels_is_empty():
    """
    Test predict raises ValueError when `levels` is an empty list.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y_dict)

    err_msg = re.escape("`levels` must be a single string or a list-like of strings, but cannot be empty.")
    with pytest.raises(ValueError, match=err_msg):
        m.predict(steps=3, levels=[])


def test_predict_levels_filters_before_adapter_inference():        
    """
    Test that passing `levels` filters the input sent to the adapter, so the
    underlying pipeline receives only the requested series (avoiding
    unnecessary inference work).
    """
    pipe = FakePipeline()
    m = FoundationModel("autogluon/chronos-2-small", pipeline=pipe)
    m.fit(series=y_dict)  # 2 series: s1, s2

    result = m.predict(steps=5, levels=["s1"])

    # Pipeline must have received exactly 1 input, not 2
    assert len(pipe.last_inputs) == 1
    assert list(result["level"].unique()) == ["s1"]
    assert len(result) == 5


def test_predict_levels_preserves_requested_order():
    """
    Test that when `levels` is provided, the output preserves the order of
    the levels as requested by the user, regardless of the order in which
    series were fitted.
    """
    pipe = FakePipeline()
    m = FoundationModel("autogluon/chronos-2-small", pipeline=pipe)
    m.fit(series=y_dict)  # fitted order: s1, s2

    result = m.predict(steps=3, levels=["s2", "s1"])

    # Each timestamp appears twice (once per level); the first occurrence of
    # each timestamp must correspond to the first level in `levels`.
    level_order = result.groupby(result.index, sort=False)["level"].first().tolist()
    assert all(lv == "s2" for lv in level_order)


def test_predict_levels_accepts_string():
    """
    Test that `levels` accepts a single string, not only a list.
    """
    pipe = FakePipeline()
    m = FoundationModel("autogluon/chronos-2-small", pipeline=pipe)
    m.fit(series=y_dict)

    result = m.predict(steps=3, levels="s2")

    assert len(pipe.last_inputs) == 1
    assert list(result["level"].unique()) == ["s2"]


def test_predict_levels_ValueError_when_unknown_level():
    """
    Test predict raises ValueError with a clear message when `levels`
    contains a name not present in the fitted series.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y_dict)  # series: s1, s2

    err_msg = re.escape("`levels` ['foo'] not found in available series")
    with pytest.raises(ValueError, match=err_msg):
        m.predict(steps=3, levels=["s1", "foo"])


# Tests predict: heterogeneous exog across series
# ==============================================================================
_HETERO_INDEX = pd.date_range("2020-01-31", periods=12, freq="ME")
_HETERO_FUTURE_INDEX = pd.date_range("2021-01-31", periods=3, freq="ME")
_p = np.arange(12, dtype=float)


def _make_heterogeneous_inputs():
    """
    Four series with different exog columns: 'full' and 'full2' share the
    same (past-only, future) columns, 'past' has only a past-only column and
    'none' has no exog.
    """
    context = {
        name: pd.Series(_p * k, index=_HETERO_INDEX, name=name)
        for k, name in enumerate(["full", "none", "past", "full2"], start=1)
    }
    context_exog = {
        "full":  pd.DataFrame({"p": _p, "k": _p * 2}, index=_HETERO_INDEX),
        "none":  None,
        "past":  pd.DataFrame({"p": _p * 3}, index=_HETERO_INDEX),
        "full2": pd.DataFrame({"k": _p * 4, "p": _p * 5}, index=_HETERO_INDEX),
    }
    exog = {
        "full":  pd.DataFrame({"k": np.arange(3, dtype=float)}, index=_HETERO_FUTURE_INDEX),
        "full2": pd.DataFrame({"k": np.arange(3, dtype=float)}, index=_HETERO_FUTURE_INDEX),
    }
    return context, context_exog, exog


def test_predict_groups_series_by_exog_signature_when_adapter_requires_it():
    """
    Test that, for an adapter with supports_heterogeneous_covariates=False
    (Chronos), predict calls the backend once per distinct set of exog
    columns, every call receives homogeneous covariate keys, and the output
    contains every series in the input order.
    """
    context, context_exog, exog = _make_heterogeneous_inputs()
    pipeline = FakePipeline()
    m = FoundationModel("autogluon/chronos-2-small", pipeline=pipeline)
    m.fit(series=context, exog=context_exog)

    predictions = m.predict(steps=3, exog=exog)

    assert len(pipeline.calls) == 3
    calls_keys = [
        [
            (sorted(d.get("past_covariates", {})), sorted(d.get("future_covariates", {})))
            for d in call["inputs"]
        ]
        for call in pipeline.calls
    ]
    assert calls_keys == [
        [(["k", "p"], ["k"]), (["k", "p"], ["k"])],
        [([], [])],
        [(["p"], [])],
    ]
    assert list(predictions["level"].unique()) == ["full", "none", "past", "full2"]
    assert predictions.shape == (12, 2)
    np.testing.assert_array_almost_equal(predictions["pred"].to_numpy(), np.full(12, 0.5))


def test_predict_single_call_when_adapter_supports_heterogeneous_covariates(monkeypatch):
    """
    Test that an adapter with supports_heterogeneous_covariates=True receives
    every series in a single backend call regardless of their exog columns.
    """
    context, context_exog, exog = _make_heterogeneous_inputs()
    pipeline = FakePipeline()
    m = FoundationModel("autogluon/chronos-2-small", pipeline=pipeline)
    monkeypatch.setattr(m.adapter, "supports_heterogeneous_covariates", True)
    m.fit(series=context, exog=context_exog)

    predictions = m.predict(steps=3, exog=exog)

    assert len(pipeline.calls) == 1
    assert len(pipeline.calls[0]["inputs"]) == 4
    assert list(predictions["level"].unique()) == ["full", "none", "past", "full2"]


def test_predict_timesfm_v3_one_predict_batch_call_per_exog_signature():
    """
    Test that TimesFM 3.0, whose predict_batch stacks the covariate arrays
    of the batch, receives one predict_batch call per distinct set of exog
    columns and that the output keeps the input series order.
    """
    context, context_exog, exog = _make_heterogeneous_inputs()
    fake_model = FakeTimesFM3Forecaster()
    m = FoundationModel("google/timesfm-3.0-pytorch", model=fake_model)
    m.fit(series=context, exog=context_exog)

    predictions = m.predict(steps=3, exog=exog)

    assert len(fake_model.calls) == 3
    assert [len(call["contexts"]) for call in fake_model.calls] == [2, 1, 1]
    assert [call["padding_mode"] for call in fake_model.calls] == ["edge", "none", "edge"]
    assert list(predictions["level"].unique()) == ["full", "none", "past", "full2"]
    assert predictions.shape == (12, 2)


def test_predict_aligns_context_exog_and_exog_when_check_inputs_False():
    """
    Test that, with check_inputs=False (internal backtesting path), the
    historical exog is aligned to the context index, the future exog is
    reindexed to the forecast horizon with NaN for missing timestamps (with
    a MissingValuesWarning), and a series without a key in exog is
    forwarded without future covariates.
    """
    long_index = pd.date_range("2019-07-31", periods=18, freq="ME")
    context = {
        "s1": pd.Series(_p, index=_HETERO_INDEX, name="s1"),
        "s2": pd.Series(_p, index=_HETERO_INDEX, name="s2"),
    }
    context_exog = {
        "s1": pd.DataFrame({"a": np.arange(18, dtype=float)}, index=long_index),
        "s2": pd.DataFrame({"a": np.arange(18, dtype=float)}, index=long_index),
    }
    exog = {
        "s1": pd.DataFrame({"a": [1.0]}, index=_HETERO_FUTURE_INDEX[:1]),
    }
    pipeline = FakePipeline()
    m = FoundationModel("autogluon/chronos-2-small", pipeline=pipeline)

    warn_msg = re.escape(
        "`exog` for series ['s1'] has been reindexed to match the expected "
        "forecast horizon. Missing timestamps were filled with NaN."
    )
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        predictions = m.predict(
            steps        = 3,
            context      = context,
            context_exog = context_exog,
            exog         = exog,
            check_inputs = False,
        )

    assert len(pipeline.calls) == 2
    input_s1 = pipeline.calls[0]["inputs"][0]
    input_s2 = pipeline.calls[1]["inputs"][0]
    np.testing.assert_array_almost_equal(
        input_s1["past_covariates"]["a"], np.arange(6, 18, dtype=float)
    )
    np.testing.assert_array_almost_equal(
        input_s1["future_covariates"]["a"], np.array([1.0, np.nan, np.nan])
    )
    np.testing.assert_array_almost_equal(
        input_s2["past_covariates"]["a"], np.arange(6, 18, dtype=float)
    )
    assert "future_covariates" not in input_s2
    assert predictions.shape == (6, 2)


def test_predict_ValueError_when_context_has_nan_and_adapter_does_not_support_it(
    monkeypatch
):
    """
    Test that predict raises ValueError naming the series with NaN when the
    adapter declares supports_nan_in_series=False.
    """
    context = {
        "s1": pd.Series(_p, index=_HETERO_INDEX, name="s1"),
        "s2": pd.Series(np.where(_p > 5, np.nan, _p), index=_HETERO_INDEX, name="s2"),
    }
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    monkeypatch.setattr(m.adapter, "supports_nan_in_series", False)
    m.fit(series=context)

    err_msg = re.escape(
        "ChronosAdapter does not accept NaN values in the series used as "
        "context. Series with NaN: ['s2']. Impute or drop them before "
        "predicting."
    )
    with pytest.raises(ValueError, match=err_msg):
        m.predict(steps=3)


# Tests predict: left-padding of the context
# ==============================================================================
def _record_adapter_predict(monkeypatch, model):
    """
    Wrap `model.adapter.predict` so that every call stores the keyword
    arguments it receives, and return the list of calls.
    """
    calls = []
    original_predict = model.adapter.predict

    def recording_predict(**kwargs):
        calls.append(kwargs)
        return original_predict(**kwargs)

    monkeypatch.setattr(model.adapter, "predict", recording_predict)

    return calls


def _make_different_length_inputs():
    """
    Two daily series in America/Santiago of different length and end date,
    on both sides of the daylight saving change at midnight of 2024-09-08,
    with one historical exog column and its future values.
    """
    idx_long = pd.date_range("2024-08-30", periods=6, freq="D", tz="America/Santiago")
    idx_short = pd.date_range("2024-09-09", periods=3, freq="D", tz="America/Santiago")
    future_long = pd.date_range("2024-09-05", periods=2, freq="D", tz="America/Santiago")
    future_short = pd.date_range("2024-09-12", periods=2, freq="D", tz="America/Santiago")
    context = {
        "long": pd.Series(np.arange(1.0, 7.0), index=idx_long, name="long"),
        "short": pd.Series([10.0, 11.0, 12.0], index=idx_short, name="short"),
    }
    context_exog = {
        "long": pd.DataFrame({"a": np.arange(6.0)}, index=idx_long),
        "short": pd.DataFrame({"a": [20.0, 21.0, 22.0]}, index=idx_short),
    }
    exog = {
        "long": pd.DataFrame({"a": [6.0, 7.0]}, index=future_long),
        "short": pd.DataFrame({"a": [23.0, 24.0]}, index=future_short),
    }

    return context, context_exog, exog, future_long, future_short


@pytest.mark.parametrize(
    "check_inputs",
    [True, False],
    ids=lambda check_inputs: f"check_inputs: {check_inputs}",
)
def test_predict_left_pads_groups_when_adapter_requires_equal_length_context(
    monkeypatch, check_inputs
):
    """
    Test that, for an adapter with requires_equal_length_context=True (T0),
    predict left-pads the context and the historical exog of every series
    with NaN to the length of the longest one, on a common positional
    RangeIndex (also across a daylight saving change and on the
    check_inputs=False path), forwards the future exog with the index of the
    forecast horizon, and builds the index of the predictions from the
    unpadded context of each series.
    """
    context, context_exog, exog, future_long, future_short = (
        _make_different_length_inputs()
    )
    m = FoundationModel("theforecastingcompany/t0-alpha", model=FakeT0Forecaster())
    calls = _record_adapter_predict(monkeypatch, m)

    if check_inputs:
        m.fit(series=context, exog=context_exog)
        predictions = m.predict(steps=2, exog=exog)
    else:
        predictions = m.predict(
            steps        = 2,
            context      = context,
            context_exog = context_exog,
            exog         = exog,
            check_inputs = False,
        )

    assert len(calls) == 1
    index = pd.RangeIndex(6)
    padded_context = calls[0]["context"]
    padded_context_exog = calls[0]["context_exog"]
    pd.testing.assert_series_equal(
        padded_context["long"], pd.Series(np.arange(1.0, 7.0), index=index, name="long")
    )
    pd.testing.assert_series_equal(
        padded_context["short"],
        pd.Series([np.nan, np.nan, np.nan, 10.0, 11.0, 12.0], index=index, name="short"),
    )
    pd.testing.assert_frame_equal(
        padded_context_exog["long"], pd.DataFrame({"a": np.arange(6.0)}, index=index)
    )
    pd.testing.assert_frame_equal(
        padded_context_exog["short"],
        pd.DataFrame({"a": [np.nan, np.nan, np.nan, 20.0, 21.0, 22.0]}, index=index),
    )
    pd.testing.assert_index_equal(calls[0]["exog"]["long"].index, future_long)
    pd.testing.assert_index_equal(calls[0]["exog"]["short"].index, future_short)
    np.testing.assert_array_almost_equal(
        m.adapter._model.last_context,
        np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [np.nan, np.nan, np.nan, 10.0, 11.0, 12.0],
        ]),
    )
    # The long-format index of several series has no `freq`.
    pd.testing.assert_index_equal(
        predictions.loc[predictions["level"] == "long"].index,
        pd.DatetimeIndex(future_long, freq=None),
    )
    pd.testing.assert_index_equal(
        predictions.loc[predictions["level"] == "short"].index,
        pd.DatetimeIndex(future_short, freq=None),
    )


def test_predict_does_not_pad_when_adapter_does_not_require_equal_length_context(
    monkeypatch
):
    """
    Test that, for an adapter with requires_equal_length_context=False
    (Chronos), predict forwards every series and its historical exog with its
    own length and index.
    """
    context, context_exog, exog, _, _ = _make_different_length_inputs()
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    calls = _record_adapter_predict(monkeypatch, m)
    m.fit(series=context, exog=context_exog)

    m.predict(steps=2, exog=exog)

    assert len(calls) == 1
    for series_name in context:
        pd.testing.assert_series_equal(
            calls[0]["context"][series_name], context[series_name]
        )
        pd.testing.assert_frame_equal(
            calls[0]["context_exog"][series_name], context_exog[series_name]
        )


# Tests predict: chunks of max_series_per_call series
# ==============================================================================
def _make_chunk_inputs():
    """
    Five daily series of different length ending on the same date. Series
    s0, s1 and s2 have the exog column `a` and s3 and s4 the column `b`, so
    an adapter without heterogeneous covariates groups them in two.
    """
    lengths = {"s0": 6, "s1": 3, "s2": 5, "s3": 4, "s4": 2}
    columns = {"s0": "a", "s1": "a", "s2": "a", "s3": "b", "s4": "b"}
    future_index = pd.date_range("2024-01-11", periods=2, freq="D")
    context = {}
    context_exog = {}
    exog = {}
    for i, (series_name, length) in enumerate(lengths.items()):
        index = pd.date_range(end="2024-01-10", periods=length, freq="D")
        context[series_name] = pd.Series(
            np.arange(length, dtype=float) + 10 * i, index=index, name=series_name
        )
        context_exog[series_name] = pd.DataFrame(
            {columns[series_name]: np.arange(length, dtype=float)}, index=index
        )
        exog[series_name] = pd.DataFrame(
            {columns[series_name]: [100.0, 101.0]}, index=future_index
        )

    return context, context_exog, exog


@pytest.mark.parametrize(
    "check_inputs",
    [True, False],
    ids=lambda check_inputs: f"check_inputs: {check_inputs}",
)
def test_predict_splits_groups_into_chunks_of_max_series_per_call(
    monkeypatch, check_inputs
):
    """
    Test that predict splits each group of series (Timestral groups the
    series by exog columns) into consecutive chunks of at most
    max_series_per_call series, calls the adapter once per chunk, left-pads
    each chunk to the length of its own longest series, and returns the same
    predictions as one call per group.
    """
    context, context_exog, exog = _make_chunk_inputs()

    def predict(max_series_per_call):
        m = FoundationModel(
            "mistralai/Timestral-28M", model=FakeTimestralForecaster()
        )
        monkeypatch.setattr(m.adapter, "max_series_per_call", max_series_per_call)
        calls = _record_adapter_predict(monkeypatch, m)
        if check_inputs:
            m.fit(series=context, exog=context_exog)
            predictions = m.predict(steps=2, exog=exog, quantiles=[0.1, 0.5, 0.9])
        else:
            predictions = m.predict(
                steps        = 2,
                context      = context,
                context_exog = context_exog,
                exog         = exog,
                quantiles    = [0.1, 0.5, 0.9],
                check_inputs = False,
            )

        return predictions, calls

    predictions, calls = predict(max_series_per_call=2)
    expected, calls_one_per_group = predict(max_series_per_call=None)

    assert [list(call["context"]) for call in calls] == [
        ["s0", "s1"], ["s2"], ["s3", "s4"]
    ]
    assert [list(call["context"]) for call in calls_one_per_group] == [
        ["s0", "s1", "s2"], ["s3", "s4"]
    ]
    for call, width in zip(calls, [6, 5, 4]):
        for series_name, series in call["context"].items():
            pd.testing.assert_index_equal(series.index, pd.RangeIndex(width))
            np.testing.assert_array_equal(
                series.to_numpy()[-len(context[series_name]):],
                context[series_name].to_numpy(),
            )
            assert list(call["context_exog"][series_name].columns) == list(
                context_exog[series_name].columns
            )
            assert list(call["exog"][series_name].columns) == list(
                exog[series_name].columns
            )
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_TimestralAdapter_calls_backend_in_chunks_of_50_series():
    """
    Test that, with the default max_series_per_call of TimestralAdapter (50),
    predict sends 230 series to the backend in calls of 50, 50, 50, 50 and
    30 series, and returns one prediction per series.
    """
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    series = {
        f"s{i}": pd.Series(np.arange(10, dtype=float) + i, index=index)
        for i in range(230)
    }
    fake_model = FakeTimestralForecaster()
    m = FoundationModel("mistralai/Timestral-28M", model=fake_model)
    m.fit(series=series)

    predictions = m.predict(steps=3)

    assert m.adapter.max_series_per_call == 50
    assert [call["context"].shape[0] for call in fake_model.calls] == [
        50, 50, 50, 50, 30
    ]
    assert predictions.shape == (230 * 3, 2)
    assert predictions["level"].tolist() == list(series) * 3


# Tests predict: past-only covariates
# ==============================================================================
def _stub_adapter_predict(monkeypatch, model):
    """
    Replace `model.adapter.predict` with a stub that stores the keyword
    arguments of every call and returns zeros, so that no backend is needed,
    and return the list of calls.
    """
    calls = []

    def stub_predict(**kwargs):
        calls.append(kwargs)
        n_quantiles = 1 if kwargs["quantiles"] is None else len(kwargs["quantiles"])
        return {
            series_name: np.zeros((kwargs["steps"], n_quantiles))
            for series_name in kwargs["context"]
        }

    monkeypatch.setattr(model.adapter, "predict", stub_predict)

    return calls


def _count_past_only_warnings(record):
    """
    Number of IgnoredArgumentWarning about historical exog columns without
    future values in a list of recorded warnings.
    """
    return sum(
        issubclass(w.category, IgnoredArgumentWarning)
        and "Historical exog columns without future values are ignored"
        in str(w.message)
        for w in record
    )


@pytest.mark.parametrize(
    "check_inputs",
    [True, False],
    ids=lambda check_inputs: f"check_inputs: {check_inputs}",
)
@pytest.mark.parametrize(
    "model_id, expected_columns",
    [
        ("autogluon/chronos-2-small", ["p", "k"]),
        ("google/timesfm-3.0-pytorch", ["p", "k"]),
        ("taharnbl/TS-ICL", ["p", "k"]),
        ("mistralai/Timestral-28M", ["p", "k"]),
        ("NX-AI/TiRex-2", ["p", "k"]),
        ("soda-inria/tabicl", ["k"]),
        ("priorlabs/tabpfn-ts", ["k"]),
        ("theforecastingcompany/t0-alpha", ["k"]),
        ("Synthefy/Nori", ["k"]),
        ("google/timesfm-2.5-200m-pytorch", None),
        ("Salesforce/moirai-2.0-R-small", None),
    ],
    ids=lambda value: f"{value}",
)
def test_predict_adapter_receives_past_only_exog_only_when_it_supports_it(
    monkeypatch, model_id, expected_columns, check_inputs
):
    """
    Test the historical exog that every adapter receives when a column has
    no future values ('p'): adapters with supports_past_only_covariates=True
    receive it, adapters with supports_past_only_covariates=False only
    receive the columns with future values ('k'), and adapters with
    allow_exog=False receive no exog. The column is removed also with
    check_inputs=False, while the IgnoredArgumentWarning that names it is
    issued once and only with check_inputs=True.
    """
    context = {"s1": pd.Series(_p, index=_HETERO_INDEX, name="s1")}
    context_exog = {
        "s1": pd.DataFrame({"p": _p * 2, "k": _p * 3}, index=_HETERO_INDEX)
    }
    exog = {
        "s1": pd.DataFrame({"k": np.arange(3, dtype=float)}, index=_HETERO_FUTURE_INDEX)
    }
    m = FoundationModel(model_id)
    calls = _stub_adapter_predict(monkeypatch, m)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        predictions = m.predict(
            steps        = 3,
            context      = context,
            context_exog = context_exog,
            exog         = exog,
            check_inputs = check_inputs,
        )

    assert len(calls) == 1
    if expected_columns is None:
        assert m.allow_exog is False
        assert calls[0]["context_exog"] is None
        assert calls[0]["exog"] is None
    else:
        assert m.supports_past_only_covariates is (expected_columns == ["p", "k"])
        assert list(calls[0]["context_exog"]["s1"].columns) == expected_columns
        np.testing.assert_array_almost_equal(
            calls[0]["context_exog"]["s1"]["k"].to_numpy(), _p * 3
        )
        assert list(calls[0]["exog"]["s1"].columns) == ["k"]
    expected_warnings = 1 if expected_columns == ["k"] and check_inputs else 0
    assert _count_past_only_warnings(record) == expected_warnings
    if expected_warnings:
        messages = [str(w.message) for w in record]
        assert any("ignored for series {'s1': ['p']}." in msg for msg in messages)
    assert list(context_exog["s1"].columns) == ["p", "k"]
    assert predictions.shape == (3, 2)


@pytest.mark.parametrize(
    "check_inputs",
    [True, False],
    ids=lambda check_inputs: f"check_inputs: {check_inputs}",
)
@pytest.mark.parametrize(
    "context_from",
    ["fit", "predict"],
    ids=lambda context_from: f"context_from: {context_from}",
)
def test_predict_adapter_receives_no_exog_when_exog_None_and_no_past_only_support(
    monkeypatch, context_from, check_inputs
):
    """
    Test that, for an adapter with supports_past_only_covariates=False
    (TabICL), predicting without future exog removes every historical exog
    column, both with the context stored in fit and with the one passed to
    predict, also with check_inputs=False. The exog stored in fit is not
    modified, and the IgnoredArgumentWarning is issued once and only with
    check_inputs=True.
    """
    context = {
        "s1": pd.Series(_p, index=_HETERO_INDEX, name="s1"),
        "s2": pd.Series(_p * 2, index=_HETERO_INDEX, name="s2"),
    }
    context_exog = {
        "s1": pd.DataFrame({"a": _p, "b": _p * 2}, index=_HETERO_INDEX),
        "s2": pd.DataFrame({"a": _p * 3}, index=_HETERO_INDEX),
    }
    m = FoundationModel("soda-inria/tabicl")
    calls = _stub_adapter_predict(monkeypatch, m)
    if context_from == "fit":
        m.fit(series=context, exog=context_exog)
        predict_kwargs = {}
    else:
        predict_kwargs = {"context": context, "context_exog": context_exog}

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        predictions = m.predict(steps=3, check_inputs=check_inputs, **predict_kwargs)

    assert len(calls) == 1
    assert calls[0]["context_exog"] == {"s1": None, "s2": None}
    assert calls[0]["exog"] == {"s1": None, "s2": None}
    assert _count_past_only_warnings(record) == (1 if check_inputs else 0)
    if check_inputs:
        messages = [str(w.message) for w in record]
        assert any(
            "ignored for series {'s1': ['a', 'b'], 's2': ['a']}." in msg
            for msg in messages
        )
    if context_from == "fit":
        assert list(m.context_exog_["s1"].columns) == ["a", "b"]
        assert list(m.context_exog_["s2"].columns) == ["a"]
    assert predictions.shape == (6, 2)


def test_predict_groups_series_after_dropping_past_only_exog(monkeypatch):
    """
    Test that, for an adapter with supports_past_only_covariates=False and
    supports_heterogeneous_covariates=False (TabICL), the historical columns
    without future values are removed before grouping the series by exog
    columns: the series whose only exog column is past-only is forecast in
    the same call as the series without exog, and no call receives a
    past-only column.
    """
    context, context_exog, exog = _make_heterogeneous_inputs()
    m = FoundationModel("soda-inria/tabicl")
    calls = _stub_adapter_predict(monkeypatch, m)
    m.fit(series=context, exog=context_exog)

    warn_msg = re.escape(
        "ignored for series {'full': ['p'], 'past': ['p'], 'full2': ['p']}."
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        predictions = m.predict(steps=3, exog=exog)

    assert [list(call["context"]) for call in calls] == [
        ["full", "full2"], ["none", "past"]
    ]
    assert [list(df.columns) for df in calls[0]["context_exog"].values()] == [
        ["k"], ["k"]
    ]
    assert calls[1]["context_exog"] == {"none": None, "past": None}
    assert [list(df.columns) for df in calls[0]["exog"].values()] == [["k"], ["k"]]
    assert calls[1]["exog"] == {"none": None, "past": None}
    assert list(predictions["level"].unique()) == ["full", "none", "past", "full2"]
    assert predictions.shape == (12, 2)
