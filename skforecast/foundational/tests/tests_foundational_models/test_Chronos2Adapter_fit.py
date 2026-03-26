# Unit test Chronos2Adapter fit
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.foundational._foundational_model import Chronos2Adapter


# Fixtures
# ==============================================================================
y = pd.Series(
    data=np.arange(50, dtype=float),
    index=pd.date_range("2020-01-01", periods=50, freq="ME"),
    name="y",
)

exog = pd.DataFrame(
    {"feat_a": np.arange(50, dtype=float), "feat_b": np.arange(50, dtype=float) * 2},
    index=y.index,
)


# Tests Chronos2Adapter.fit
# ==============================================================================
def test_Chronos2Adapter_fit_sets_is_fitted():
    """
    Test that fit sets _is_fitted to True.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    assert adapter._is_fitted is False
    adapter.fit(series=y)
    assert adapter._is_fitted is True


def test_Chronos2Adapter_fit_returns_self():
    """
    Test that fit returns the adapter instance itself.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    result = adapter.fit(series=y)
    assert result is adapter


def test_Chronos2Adapter_init_raises_ValueError_when_context_length_is_None():
    """
    Test that Chronos2Adapter raises ValueError when context_length is None.
    """
    with pytest.raises(ValueError, match="`context_length` must be a positive integer"):
        Chronos2Adapter(model_id="autogluon/chronos-2-small", context_length=None)


def test_Chronos2Adapter_init_raises_ValueError_when_context_length_not_positive():
    """
    Test that Chronos2Adapter raises ValueError when context_length <= 0.
    """
    with pytest.raises(ValueError, match="`context_length` must be a positive integer"):
        Chronos2Adapter(model_id="autogluon/chronos-2-small", context_length=0)


def test_Chronos2Adapter_fit_stores_none_when_no_exog():
    """
    Test that _history_exog is None when no exog is passed to fit.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    adapter.fit(series=y)
    assert adapter._history_exog is None


def test_Chronos2Adapter_fit_trims_history_to_context_length():
    """
    Test that fit trims history to the last context_length observations.
    """
    context_length = 20
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", context_length=context_length)
    adapter.fit(series=y)
    assert len(adapter._history) == context_length
    pd.testing.assert_series_equal(adapter._history, y.iloc[-context_length:])


def test_Chronos2Adapter_fit_trims_exog_to_context_length():
    """
    Test that fit trims exog to the last context_length rows when context_length is set.
    """
    context_length = 15
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", context_length=context_length)
    adapter.fit(series=y, exog=exog)
    assert len(adapter._history_exog) == context_length
    pd.testing.assert_frame_equal(adapter._history_exog, exog.iloc[-context_length:])


def test_Chronos2Adapter_fit_exog_none_when_context_length_set_but_no_exog():
    """
    Test that _history_exog remains None when context_length is set but no exog is provided.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", context_length=20)
    adapter.fit(series=y)
    assert adapter._history_exog is None


def test_Chronos2Adapter_fit_does_not_modify_input_series():
    """
    Test that fit does not modify the original input series.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", context_length=20)
    y_copy = y.copy()
    adapter.fit(series=y)
    pd.testing.assert_series_equal(y, y_copy)


def test_Chronos2Adapter_fit_raises_TypeError_when_series_is_invalid_type():
    """
    Test that fit raises TypeError when y is not a pd.Series, pd.DataFrame, or dict.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    err_msg = re.escape(
        "`series` must be a pd.Series, a wide pd.DataFrame, or a "
        f"dict[str, pd.Series]. Got {type(np.array([1, 2, 3]))}."
    )
    with pytest.raises(TypeError, match=err_msg):
        adapter.fit(series=np.array([1, 2, 3]))


def test_Chronos2Adapter_fit_raises_ValueError_when_series_has_nan():
    """
    Test that fit raises ValueError when y contains NaN values.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    y_nan = y.copy()
    y_nan.iloc[5] = np.nan
    err_msg = re.escape("`series` has missing values.")
    with pytest.raises(ValueError, match=err_msg):
        adapter.fit(series=y_nan)


@pytest.mark.parametrize(
    "context_length, expected_len",
    [(10, 10), (25, 25), (50, 50), (100, 50)],
    ids=lambda x: f"context_length, expected_len: {x}",
)
def test_Chronos2Adapter_fit_context_length_parametrize(context_length, expected_len):
    """
    Test that fit trims history to min(context_length, len(y)).
    When context_length >= len(y), the full series is stored.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", context_length=context_length)
    adapter.fit(series=y)
    assert len(adapter._history) == expected_len


# Fixtures for multi-series tests
# ==============================================================================
_idx_ms = pd.date_range("2020-01-01", periods=30, freq="ME")
_y_s1 = pd.Series(np.arange(30, dtype=float), index=_idx_ms, name="s1")
_y_s2 = pd.Series(np.arange(30, 60, dtype=float), index=_idx_ms, name="s2")
_y_wide = pd.DataFrame({"s1": _y_s1, "s2": _y_s2})
_y_dict = {"s1": _y_s1.copy(), "s2": _y_s2.copy()}
_exog_shared = pd.DataFrame({"feat": np.arange(30, dtype=float)}, index=_idx_ms)


# Tests Chronos2Adapter._normalize_exog_to_dict
# ==============================================================================
def test_normalize_exog_to_dict_none_maps_all_to_none():
    """
    Test that None exog maps every series name to None.
    """
    result = Chronos2Adapter._normalize_exog_to_dict(None, ["a", "b", "c"])
    assert result == {"a": None, "b": None, "c": None}


def test_normalize_exog_to_dict_dataframe_broadcast_to_all_series():
    """
    Test that a flat DataFrame is broadcast (same reference) to every series name.
    """
    df = pd.DataFrame({"feat": [1.0, 2.0, 3.0]})
    result = Chronos2Adapter._normalize_exog_to_dict(df, ["s1", "s2"])
    assert result["s1"] is df
    assert result["s2"] is df


def test_normalize_exog_to_dict_series_broadcast_to_all_series():
    """
    Test that a pandas Series is broadcast to every series name.
    """
    s = pd.Series([1.0, 2.0, 3.0], name="feat")
    result = Chronos2Adapter._normalize_exog_to_dict(s, ["s1", "s2", "s3"])
    assert all(v is s for v in result.values())
    assert list(result.keys()) == ["s1", "s2", "s3"]


def test_normalize_exog_to_dict_dict_keeps_per_series_values():
    """
    Test that a dict input keeps per-series values for keys present in series_names.
    """
    df_a = pd.DataFrame({"feat": [1.0]})
    df_b = pd.DataFrame({"feat": [2.0]})
    result = Chronos2Adapter._normalize_exog_to_dict({"a": df_a, "b": df_b}, ["a", "b"])
    assert result["a"] is df_a
    assert result["b"] is df_b


def test_normalize_exog_to_dict_dict_missing_key_gets_none():
    """
    Test that series names absent from the exog dict are mapped to None.
    """
    df_a = pd.DataFrame({"feat": [1.0]})
    result = Chronos2Adapter._normalize_exog_to_dict({"a": df_a}, ["a", "b", "c"])
    assert result["a"] is df_a
    assert result["b"] is None
    assert result["c"] is None


# Tests Chronos2Adapter.fit — multi-series
# ==============================================================================
def test_Chronos2Adapter_fit_multiseries_wide_dataframe_sets_is_multiseries():
    """
    Test that fitting on a wide DataFrame sets _is_multiseries to True.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    adapter.fit(series=_y_wide)
    assert adapter._is_multiseries is True


def test_Chronos2Adapter_fit_multiseries_dict_sets_is_multiseries():
    """
    Test that fitting on a dict of Series sets _is_multiseries to True.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    adapter.fit(series=_y_dict)
    assert adapter._is_multiseries is True


def test_Chronos2Adapter_fit_multiseries_stores_dict_in_history():
    """
    Test that _history is a dict after fitting on multi-series input.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    adapter.fit(series=_y_dict)
    assert isinstance(adapter._history, dict)


def test_Chronos2Adapter_fit_multiseries_series_names_from_dataframe_columns():
    """
    Test that the keys of _history match the column names of the input DataFrame.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    adapter.fit(series=_y_wide)
    assert list(adapter._history.keys()) == list(_y_wide.columns)


def test_Chronos2Adapter_fit_multiseries_series_names_from_dict_keys():
    """
    Test that the keys of _history match the keys of the input dict.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    adapter.fit(series=_y_dict)
    assert list(adapter._history.keys()) == list(_y_dict.keys())


def test_Chronos2Adapter_fit_multiseries_each_value_is_series():
    """
    Test that each value stored in _history is a pandas Series.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    adapter.fit(series=_y_dict)
    for name, s in adapter._history.items():
        assert isinstance(s, pd.Series), f"Expected pd.Series for '{name}', got {type(s)}"


def test_Chronos2Adapter_fit_multiseries_sets_is_fitted():
    """
    Test that _is_fitted is True after multi-series fit.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    assert adapter._is_fitted is False
    adapter.fit(series=_y_dict)
    assert adapter._is_fitted is True


def test_Chronos2Adapter_fit_multiseries_context_length_trims_each_series():
    """
    Test that context_length trims each individual series to the last
    context_length observations.
    """
    context_length = 10
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", context_length=context_length)
    adapter.fit(series=_y_dict)
    for name, s in adapter._history.items():
        assert len(s) == context_length, f"'{name}' has {len(s)} rows, expected {context_length}"


def test_Chronos2Adapter_fit_multiseries_shared_exog_broadcast_to_all_series():
    """
    Test that a shared (flat) DataFrame exog is broadcast to all series in
    _history_exog so each series gets the same exog values.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    adapter.fit(series=_y_dict, exog=_exog_shared)
    assert isinstance(adapter._history_exog, dict)
    for name in _y_dict:
        assert adapter._history_exog[name] is not None
        pd.testing.assert_frame_equal(
            adapter._history_exog[name],
            _exog_shared,
        )


def test_Chronos2Adapter_fit_multiseries_per_series_exog_stored_correctly():
    """
    Test that a per-series exog dict is stored correctly: s1 has exog, s2 has None.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    adapter.fit(series=_y_dict, exog={"s1": _exog_shared.copy()})
    assert adapter._history_exog["s1"] is not None
    assert adapter._history_exog["s2"] is None


def test_Chronos2Adapter_fit_multiseries_missing_exog_key_gets_none():
    """
    Test that a series absent from the per-series exog dict stores None in
    _history_exog.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    adapter.fit(series=_y_dict, exog={"s1": _exog_shared})
    assert adapter._history_exog["s2"] is None


def test_Chronos2Adapter_fit_multiseries_context_length_trims_shared_exog():
    """
    Test that context_length also trims shared exog stored per series.
    """
    context_length = 8
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", context_length=context_length)
    adapter.fit(series=_y_dict, exog=_exog_shared)
    for name in _y_dict:
        assert len(adapter._history_exog[name]) == context_length


def test_Chronos2Adapter_fit_multiseries_raises_ValueError_for_empty_dict():
    """
    Test that fit raises ValueError when an empty dict is passed as y.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    with pytest.raises(ValueError, match=re.escape("`series` must contain at least one series.")):
        adapter.fit(series={})


def test_Chronos2Adapter_fit_multiseries_raises_TypeError_for_non_series_value():
    """
    Test that fit raises TypeError when a dict value is not a pd.Series.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    with pytest.raises(TypeError, match="All values in `series` must be pd.Series"):
        adapter.fit(series={"s1": np.array([1.0, 2.0, 3.0])})


def test_Chronos2Adapter_fit_multiseries_raises_ValueError_for_nan_in_series():
    """
    Test that fit raises ValueError when any series in the dict contains NaN.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    y_nan_dict = {"s1": _y_s1.copy(), "s2": _y_s2.copy()}
    y_nan_dict["s2"].iloc[5] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        adapter.fit(series=y_nan_dict)


def test_Chronos2Adapter_fit_multiseries_refitting_single_series_resets_flag():
    """
    Test that re-fitting on a single Series after a multi-series fit resets
    _is_multiseries to False and stores a Series in _history.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    adapter.fit(series=_y_dict)
    assert adapter._is_multiseries is True
    adapter.fit(series=y)
    assert adapter._is_multiseries is False
    assert isinstance(adapter._history, pd.Series)

