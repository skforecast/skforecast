# Unit test skforecast.foundation._utils
# ==============================================================================
import re
import pytest
import warnings
import numpy as np
import pandas as pd

from skforecast.foundation._utils import (
    check_preprocess_series_foundation,
    check_preprocess_exog_type,
    normalize_exog_to_dict,
    validate_context_exog,
    validate_exog_predict,
)
from skforecast.exceptions import (
    IgnoredArgumentWarning,
    InputTypeWarning,
    MissingExogWarning,
    MissingValuesWarning,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_index = pd.date_range("2020-01-01", periods=4, freq="ME")


def _make_long_df_datetime(cols=("value",)):
    """Build a long-format MultiIndex DataFrame with a DatetimeIndex second level."""
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    idx = pd.MultiIndex.from_product([["s1", "s2"], dates])
    data = {col: np.arange(1, 7, dtype=float) for col in cols}
    return pd.DataFrame(data, index=idx)


def _make_long_df_non_datetime():
    """Build a long-format MultiIndex DataFrame with a non-DatetimeIndex second level."""
    idx = pd.MultiIndex.from_arrays([["s1", "s1", "s2", "s2"], [1, 2, 1, 2]])
    return pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]}, index=idx)


# ===========================================================================
# check_preprocess_series_foundation
# ===========================================================================

def test_check_preprocess_series_foundation_named_series():
    """
    pd.Series with a name → dict with one entry keyed by the name.
    """
    s = pd.Series([1.0, 2.0, 3.0], index=_index[:3], name="sales")
    result, indexes = check_preprocess_series_foundation(s)
    assert list(result.keys()) == ["sales"]
    assert list(indexes.keys()) == ["sales"]
    pd.testing.assert_index_equal(indexes["sales"], s.index)


def test_check_preprocess_series_foundation_unnamed_series_returns_y():
    """
    pd.Series without a name → key defaults to 'y'.
    """
    s = pd.Series([1.0, 2.0], index=_index[:2], name=None)
    result, indexes = check_preprocess_series_foundation(s)
    assert list(result.keys()) == ["y"]
    assert list(indexes.keys()) == ["y"]


def test_check_preprocess_series_foundation_wide_dataframe():
    """
    Wide (flat-index) DataFrame → dict with one entry per column.
    """
    df = pd.DataFrame({"s1": [1.0, 2.0], "s2": [3.0, 4.0]}, index=_index[:2])
    result, indexes = check_preprocess_series_foundation(df)
    assert list(result.keys()) == ["s1", "s2"]
    assert list(indexes.keys()) == ["s1", "s2"]


def test_check_preprocess_series_foundation_dict():
    """
    dict[str, pd.Series] → validated and returned.
    """
    d = {
        "s1": pd.Series([1.0, 2.0], index=_index[:2], name="s1"),
        "s2": pd.Series([3.0, 4.0], index=_index[:2], name="s2"),
    }
    result, indexes = check_preprocess_series_foundation(d)
    assert list(result.keys()) == ["s1", "s2"]
    assert list(indexes.keys()) == ["s1", "s2"]


def test_check_preprocess_series_foundation_long_format_converts_to_dict():
    """
    A long-format MultiIndex DataFrame with a DatetimeIndex second level should
    be converted to a dict[str, pd.Series] and an InputTypeWarning should be
    issued.
    """
    df_long = _make_long_df_datetime()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result, indexes = check_preprocess_series_foundation(df_long)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"s1", "s2"}
    assert any(issubclass(warning.category, InputTypeWarning) for warning in w)


def test_check_preprocess_series_foundation_long_format_multicol_warns_IgnoredArgumentWarning():
    """
    A long-format MultiIndex DataFrame with multiple columns should warn that
    extra columns are ignored in addition to the InputTypeWarning.
    """
    df_long = _make_long_df_datetime(cols=("value", "extra"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_preprocess_series_foundation(df_long)
    warning_categories = [warning.category for warning in w]
    assert IgnoredArgumentWarning in warning_categories


def test_check_preprocess_series_foundation_long_format_non_datetime_second_level_raises_TypeError():
    """
    A long-format MultiIndex DataFrame where the second level is NOT a DatetimeIndex
    should raise TypeError.
    """
    df_bad = _make_long_df_non_datetime()
    err_msg = re.escape(
        "The second level of the MultiIndex in `series` must be a "
        "pandas DatetimeIndex with the same frequency for each series."
    )
    with pytest.raises(TypeError, match=err_msg):
        check_preprocess_series_foundation(df_bad)


def test_check_preprocess_series_foundation_invalid_type_raises_TypeError():
    """
    An unsupported type (e.g., list) should raise TypeError.
    """
    with pytest.raises(TypeError):
        check_preprocess_series_foundation([1.0, 2.0, 3.0])


# ===========================================================================
# check_preprocess_exog_type
# ===========================================================================

def test_check_preprocess_exog_type_none_returns_none():
    """
    None input should be returned unchanged.
    """
    result = check_preprocess_exog_type(None)
    assert result is None


def test_check_preprocess_exog_type_dict_returns_dict_unchanged():
    """
    dict input should be returned unchanged.
    """
    d = {"s1": pd.DataFrame({"feat": [1.0, 2.0]}, index=_index[:2])}
    result = check_preprocess_exog_type(d)
    assert result is d


def test_check_preprocess_exog_type_flat_series_returns_unchanged():
    """
    A flat-index pd.Series should be returned unchanged (broadcast to all series).
    """
    s = pd.Series([1.0, 2.0, 3.0], index=_index[:3], name="feat")
    result = check_preprocess_exog_type(s)
    assert result is s


def test_check_preprocess_exog_type_wide_dataframe_returns_unchanged():
    """
    A wide (flat-index) DataFrame should be returned unchanged (broadcast to all series).
    """
    df = pd.DataFrame({"feat_a": [1.0, 2.0], "feat_b": [3.0, 4.0]}, index=_index[:2])
    result = check_preprocess_exog_type(df)
    assert result is df


def test_check_preprocess_exog_type_long_format_converts_to_dict():
    """
    A long-format MultiIndex DataFrame with DatetimeIndex second level should
    be converted to a dict[str, pd.DataFrame] and an InputTypeWarning issued.
    """
    df_long = _make_long_df_datetime()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = check_preprocess_exog_type(df_long)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"s1", "s2"}
    assert any(issubclass(warning.category, InputTypeWarning) for warning in w)


def test_check_preprocess_exog_type_long_format_missing_series_warns_MissingExogWarning():
    """
    When series_names_in_ contains series that are not in the long-format exog,
    a MissingExogWarning should be issued for the absent series.
    """
    df_long = _make_long_df_datetime()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = check_preprocess_exog_type(
            df_long, series_names_in_=["s1", "s2", "s3"]
        )
    warning_categories = [warning.category for warning in w]
    assert MissingExogWarning in warning_categories
    missing_warning = next(
        warning for warning in w if issubclass(warning.category, MissingExogWarning)
    )
    assert "s3" in str(missing_warning.message)


def test_check_preprocess_exog_type_invalid_type_raises_TypeError():
    """
    An unsupported type should raise TypeError.
    """
    err_msg = re.escape(
        "`exog` must be a pandas Series, a pandas DataFrame, or a "
        "dict."
    )
    with pytest.raises(TypeError, match=err_msg):
        check_preprocess_exog_type([1.0, 2.0, 3.0])


# ===========================================================================
# validate_context_exog
# ===========================================================================


def test_validate_context_exog_none_context_returns_immediately():
    """
    _validate_context_exog: context=None means no validation is needed.
    """
    # Should not raise or warn
    validate_context_exog(
        context_exog=pd.Series([1.0, 2.0], name="x"),
        context=None,
        exog_in_=True,
    )


def test_validate_context_exog_exog_in_none_warns_IgnoredArgumentWarning():
    """
    _validate_context_exog: when exog_in_=True and context_exog=None,
    an IgnoredArgumentWarning must be issued (improvement A).
    """
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    lw  = pd.Series(np.ones(3), index=idx, name="y")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_context_exog(
            context_exog=None,
            context=lw,
            exog_in_=True,
        )

    assert any(issubclass(warning.category, IgnoredArgumentWarning) for warning in w)


def test_validate_context_exog_single_length_mismatch_raises_ValueError():
    """
    _validate_context_exog single-series: length mismatch raises ValueError.
    """
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    lw   = pd.Series(np.ones(5), index=idx, name="y")
    lwe  = pd.DataFrame({"x": np.ones(3)}, index=idx[:3])

    with pytest.raises(ValueError, match="same number of observations"):
        validate_context_exog(
            context_exog=lwe,
            context=lw,
            exog_in_=True,
        )


def test_validate_context_exog_single_datetimeindex_mismatch_raises_ValueError():
    """
    _validate_context_exog single-series: DatetimeIndex mismatch raises ValueError.
    """
    idx_lw  = pd.date_range("2020-01-01", periods=5, freq="D")
    idx_lwe = pd.date_range("2020-01-02", periods=5, freq="D")
    lw  = pd.Series(np.ones(5), index=idx_lw,  name="y")
    lwe = pd.DataFrame({"x": np.ones(5)}, index=idx_lwe)

    with pytest.raises(ValueError, match="index of `context_exog`.*aligned"):
        validate_context_exog(
            context_exog=lwe,
            context=lw,
            exog_in_=True,
        )


def test_validate_context_exog_multi_length_mismatch_raises_ValueError():
    """
    _validate_context_exog multi-series dict: length mismatch for one series
    raises ValueError.
    """
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    lw = {
        "A": pd.Series(np.ones(5), index=idx, name="A"),
        "B": pd.Series(np.ones(5), index=idx, name="B"),
    }
    lwe = {
        "A": pd.DataFrame({"x": np.ones(5)}, index=idx),
        "B": pd.DataFrame({"x": np.ones(3)}, index=idx[:3]),  # wrong length
    }

    with pytest.raises(ValueError, match="series 'B'.*same number"):
        validate_context_exog(
            context_exog=lwe,
            context=lw,
            exog_in_=True,
        )


def test_validate_context_exog_multi_missing_series_warns_MissingExogWarning():
    """
    _validate_context_exog multi-series dict: absent series in context_exog
    triggers a MissingExogWarning.
    """
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    lw = {
        "A": pd.Series(np.ones(5), index=idx, name="A"),
        "B": pd.Series(np.ones(5), index=idx, name="B"),
    }
    lwe = {"A": pd.DataFrame({"x": np.ones(5)}, index=idx)}  # "B" missing

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_context_exog(
            context_exog=lwe,
            context=lw,
            exog_in_=True,
        )

    assert MissingExogWarning in [warning.category for warning in w]


# ===========================================================================
# validate_exog_predict — RangeIndex start check
# ===========================================================================


def _make_validate_predict_kwargs(exog, steps=3, context_end=9):
    """
    Build a minimal kwargs dict for validate_exog_predict using a RangeIndex.
    context_range_ is set so the reference end-point is context_end.
    """
    return dict(
        exog=exog,
        steps=steps,
        context=None,
        exog_names_in_=["x"],
        exog_in_=True,
        index_freq_=1,
        is_multiseries=False,
        context_range_={"y": pd.RangeIndex(start=0, stop=context_end + 1, step=1)[[0, -1]]},
        series_names_in_=["y"],
    )


def test_validate_exog_predict_rangeindex_wrong_start_raises_ValueError():
    """
    _validate_exog_predict: RangeIndex exog that does not start immediately after
    context must raise ValueError.
    """
    # context_end = 9, so expected start = 10; exog starts at 15
    exog = pd.Series(
        np.ones(3),
        index=pd.RangeIndex(start=15, stop=18, step=1),
        name="x",
    )
    with pytest.raises(ValueError, match="must start one step ahead"):
        validate_exog_predict(**_make_validate_predict_kwargs(exog))


def test_validate_exog_predict_rangeindex_correct_start_passes():
    """
    _validate_exog_predict: RangeIndex exog that starts correctly passes without error.
    """
    exog = pd.Series(
        np.ones(3),
        index=pd.RangeIndex(start=10, stop=13, step=1),  # correct: 9 + 1 = 10
        name="x",
    )
    result = validate_exog_predict(**_make_validate_predict_kwargs(exog))
    assert len(result) == 3


# ===========================================================================
# validate_exog_predict — per-series column check with exog_names_in_per_series_
# ===========================================================================


def test_validate_exog_predict_per_series_wrong_column_raises_ValueError():
    """
    _validate_exog_predict dict exog: column not present in the per-series expected
    columns raises ValueError.
    """
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    exog = {
        "A": pd.DataFrame({"bad_col": np.ones(3)}, index=idx),
    }
    with pytest.raises(ValueError, match="bad_col"):
        validate_exog_predict(
            exog=exog,
            steps=3,
            context=None,
            exog_names_in_=["x"],
            exog_in_=True,
            index_freq_=pd.tseries.frequencies.to_offset("D"),
            is_multiseries=True,
            context_range_={"A": pd.DatetimeIndex(["2019-12-29", "2019-12-31"])},
            series_names_in_=["A"],
            exog_names_in_per_series_={"A": ["x"]},
        )


def test_validate_exog_predict_per_series_correct_columns_passes():
    """
    _validate_exog_predict dict exog: correct per-series columns pass validation
    and exog is aligned to the forecast horizon.
    """
    idx_train_end = pd.Timestamp("2019-12-31")
    idx_pred = pd.date_range("2020-01-01", periods=3, freq="D")
    exog = {
        "A": pd.DataFrame({"x": np.ones(3)}, index=idx_pred),
        "B": pd.DataFrame({"wind": np.ones(3)}, index=idx_pred),
    }
    result = validate_exog_predict(
        exog=exog,
        steps=3,
        context=None,
        exog_names_in_=["x", "wind"],
        exog_in_=True,
        index_freq_=pd.tseries.frequencies.to_offset("D"),
        is_multiseries=True,
        context_range_={
            "A": pd.DatetimeIndex([idx_train_end - pd.Timedelta(days=2), idx_train_end]),
            "B": pd.DatetimeIndex([idx_train_end - pd.Timedelta(days=2), idx_train_end]),
        },
        series_names_in_=["A", "B"],
        exog_names_in_per_series_={"A": ["x"], "B": ["wind"]},
    )
    assert set(result.keys()) == {"A", "B"}
    assert list(result["A"].columns) == ["x"]
    assert list(result["B"].columns) == ["wind"]


# ===========================================================================
# normalize_exog_to_dict
# ===========================================================================

def test_normalize_exog_to_dict_none_maps_all_to_none():
    """
    Test that None exog maps every series name to None.
    """
    result = normalize_exog_to_dict(None, ["a", "b", "c"])
    assert result == {"a": None, "b": None, "c": None}


def test_normalize_exog_to_dict_dataframe_broadcast_to_all_series():
    """
    Test that a flat DataFrame is broadcast (same reference) to every series name.
    """
    df = pd.DataFrame({"feat": [1.0, 2.0, 3.0]})
    result = normalize_exog_to_dict(df, ["s1", "s2"])
    assert result["s1"] is df
    assert result["s2"] is df


def test_normalize_exog_to_dict_series_broadcast_to_all_series():
    """
    Test that a pandas Series is broadcast to every series name.
    """
    s = pd.Series([1.0, 2.0, 3.0], name="feat")
    result = normalize_exog_to_dict(s, ["s1", "s2", "s3"])
    assert all(v is s for v in result.values())
    assert list(result.keys()) == ["s1", "s2", "s3"]


def test_normalize_exog_to_dict_dict_keeps_per_series_values():
    """
    Test that a dict input keeps per-series values for keys present in
    series_names.
    """
    df_a = pd.DataFrame({"feat": [1.0]})
    df_b = pd.DataFrame({"feat": [2.0]})
    result = normalize_exog_to_dict({"a": df_a, "b": df_b}, ["a", "b"])
    assert result["a"] is df_a
    assert result["b"] is df_b


def test_normalize_exog_to_dict_dict_missing_key_gets_none():
    """
    Test that series names absent from the exog dict are mapped to None.
    """
    df_a = pd.DataFrame({"feat": [1.0]})
    result = normalize_exog_to_dict({"a": df_a}, ["a", "b", "c"])
    assert result["a"] is df_a
    assert result["b"] is None
    assert result["c"] is None
