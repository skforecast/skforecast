# Unit test predict FoundationModel
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.exceptions import IgnoredArgumentWarning, MissingValuesWarning
from skforecast.foundation._foundation_model import FoundationModel
from .fixtures_adapters import (
    y, data, y_dict,
    FakePipeline, FakeTimesFM25Model, FakeTimesFM3Forecaster,
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
