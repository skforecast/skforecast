# Unit test skforecast.foundational._utils
# ==============================================================================
import re
import pytest
import warnings
import numpy as np
import pandas as pd

from skforecast.foundational._utils import (
    check_preprocess_series_type,
    check_preprocess_exog_type,
    align_exog_to_series,
    validate_exog_fit,
    validate_last_window_exog,
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
    idx = pd.MultiIndex.from_arrays(
        [
            ["s1", "s1", "s2", "s2"],
            pd.DatetimeIndex(
                ["2020-01-31", "2020-02-29", "2020-01-31", "2020-02-29"]
            ),
        ]
    )
    data = {col: np.arange(1, 5, dtype=float) for col in cols}
    return pd.DataFrame(data, index=idx)


def _make_long_df_non_datetime():
    """Build a long-format MultiIndex DataFrame with a non-DatetimeIndex second level."""
    idx = pd.MultiIndex.from_arrays([["s1", "s1", "s2", "s2"], [1, 2, 1, 2]])
    return pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]}, index=idx)


# ===========================================================================
# check_preprocess_series_type
# ===========================================================================

def test_check_preprocess_series_type_named_series():
    """
    pd.Series with a name → (False, [name], series) and the series is returned
    unchanged.
    """
    s = pd.Series([1.0, 2.0, 3.0], index=_index[:3], name="sales")
    is_multi, names, out = check_preprocess_series_type(s)
    assert is_multi is False
    assert names == ["sales"]
    assert out is s


def test_check_preprocess_series_type_unnamed_series_returns_y():
    """
    pd.Series without a name → name is replaced with 'y'.
    """
    s = pd.Series([1.0, 2.0], index=_index[:2], name=None)
    is_multi, names, out = check_preprocess_series_type(s)
    assert is_multi is False
    assert names == ["y"]
    assert out is s


def test_check_preprocess_series_type_wide_dataframe():
    """
    Wide (flat-index) DataFrame → (True, column_names, df) and the DataFrame is
    returned unchanged.
    """
    df = pd.DataFrame({"s1": [1.0, 2.0], "s2": [3.0, 4.0]}, index=_index[:2])
    is_multi, names, out = check_preprocess_series_type(df)
    assert is_multi is True
    assert names == ["s1", "s2"]
    assert out is df


def test_check_preprocess_series_type_dict():
    """
    dict[str, pd.Series] → (True, dict_keys, dict) and the dict is returned
    unchanged.
    """
    d = {
        "s1": pd.Series([1.0, 2.0], index=_index[:2], name="s1"),
        "s2": pd.Series([3.0, 4.0], index=_index[:2], name="s2"),
    }
    is_multi, names, out = check_preprocess_series_type(d)
    assert is_multi is True
    assert names == ["s1", "s2"]
    assert out is d


def test_check_preprocess_series_type_long_format_converts_to_dict():
    """
    A long-format MultiIndex DataFrame with a DatetimeIndex second level should
    be converted to a dict[str, pd.Series] and an InputTypeWarning should be
    issued.
    """
    df_long = _make_long_df_datetime()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        is_multi, names, out = check_preprocess_series_type(df_long)
    assert is_multi is True
    assert set(names) == {"s1", "s2"}
    assert isinstance(out, dict)
    assert set(out.keys()) == {"s1", "s2"}
    assert any(issubclass(warning.category, InputTypeWarning) for warning in w)


def test_check_preprocess_series_type_long_format_multicol_warns_IgnoredArgumentWarning():
    """
    A long-format MultiIndex DataFrame with multiple columns should warn that
    extra columns are ignored in addition to the InputTypeWarning.
    """
    df_long = _make_long_df_datetime(cols=("value", "extra"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_preprocess_series_type(df_long)
    warning_categories = [warning.category for warning in w]
    assert IgnoredArgumentWarning in warning_categories


def test_check_preprocess_series_type_long_format_non_datetime_second_level_raises_TypeError():
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
        check_preprocess_series_type(df_bad)


def test_check_preprocess_series_type_invalid_type_raises_TypeError():
    """
    An unsupported type (e.g., list) should raise TypeError.
    """
    err_msg = re.escape(
        "`series` must be a pandas Series, a wide pandas DataFrame, a "
        "long-format pandas DataFrame (MultiIndex), or a "
        "dict[str, pd.Series]."
    )
    with pytest.raises(TypeError, match=err_msg):
        check_preprocess_series_type([1.0, 2.0, 3.0])


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
        "dict[str, pd.Series | pd.DataFrame | None]."
    )
    with pytest.raises(TypeError, match=err_msg):
        check_preprocess_exog_type([1.0, 2.0, 3.0])


# ===========================================================================
# validate_exog_fit — single-series
# ===========================================================================


def test_validate_exog_fit_single_length_mismatch_raises_ValueError():
    """
    _validate_exog_fit single-series: length mismatch between exog and series
    must raise ValueError.
    """
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    series = pd.Series(np.ones(5), index=idx, name="y")
    exog   = pd.Series(np.ones(4), index=idx[:4], name="x")

    with pytest.raises(ValueError, match="same number of observations"):
        validate_exog_fit(series=series, exog=exog, is_multiseries=False)


def test_validate_exog_fit_single_datetimeindex_mismatch_raises_ValueError():
    """
    _validate_exog_fit single-series: same length but different DatetimeIndex
    timestamps must raise ValueError.
    """
    idx_series = pd.date_range("2020-01-01", periods=5, freq="D")
    idx_exog   = pd.date_range("2020-01-02", periods=5, freq="D")  # shifted by 1 day
    series = pd.Series(np.ones(5), index=idx_series, name="y")
    exog   = pd.Series(np.ones(5), index=idx_exog, name="x")

    with pytest.raises(ValueError, match="index of `exog` must be aligned"):
        validate_exog_fit(series=series, exog=exog, is_multiseries=False)


def test_validate_exog_fit_single_valid_returns_correct_dict():
    """
    _validate_exog_fit single-series: valid inputs return {series_name: [col_names]}.
    """
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    series = pd.Series(np.ones(5), index=idx, name="y")
    exog   = pd.DataFrame({"a": np.ones(5), "b": np.zeros(5)}, index=idx)

    result = validate_exog_fit(series=series, exog=exog, is_multiseries=False)

    assert result == {"y": ["a", "b"]}


def test_validate_exog_fit_single_none_exog_returns_none_mapping():
    """
    _validate_exog_fit single-series with exog=None returns {series_name: None}.
    """
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    series = pd.Series(np.ones(3), index=idx, name="y")

    result = validate_exog_fit(series=series, exog=None, is_multiseries=False)

    assert result == {"y": None}


def test_validate_exog_fit_single_rangeindex_length_mismatch_raises_ValueError():
    """
    _validate_exog_fit single-series with RangeIndex: length mismatch raises ValueError.
    """
    series = pd.Series(np.ones(5), name="y")
    exog   = pd.Series(np.ones(3), name="x")

    with pytest.raises(ValueError, match="same number of observations"):
        validate_exog_fit(series=series, exog=exog, is_multiseries=False)


# ===========================================================================
# validate_exog_fit — multi-series
# ===========================================================================


def test_validate_exog_fit_multi_length_mismatch_raises_ValueError():
    """
    _validate_exog_fit multi-series dict: length mismatch for one series raises ValueError.
    """
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    series = {
        "A": pd.Series(np.ones(5), index=idx, name="A"),
        "B": pd.Series(np.ones(5), index=idx, name="B"),
    }
    exog = {
        "A": pd.DataFrame({"x": np.ones(5)}, index=idx),
        "B": pd.DataFrame({"x": np.ones(3)}, index=idx[:3]),  # wrong length
    }

    with pytest.raises(ValueError, match="series 'B'.*same number"):
        validate_exog_fit(series=series, exog=exog, is_multiseries=True)


def test_validate_exog_fit_multi_datetimeindex_mismatch_raises_ValueError():
    """
    _validate_exog_fit multi-series dict: DatetimeIndex mismatch for one series
    raises ValueError.
    """
    idx_series = pd.date_range("2020-01-01", periods=5, freq="D")
    idx_exog   = pd.date_range("2020-01-02", periods=5, freq="D")
    series = {"A": pd.Series(np.ones(5), index=idx_series, name="A")}
    exog   = {"A": pd.DataFrame({"x": np.ones(5)}, index=idx_exog)}

    with pytest.raises(ValueError, match="index of `exog` for series 'A'"):
        validate_exog_fit(series=series, exog=exog, is_multiseries=True)


def test_validate_exog_fit_multi_missing_series_in_dict_warns_MissingExogWarning():
    """
    _validate_exog_fit multi-series dict: series absent from the exog dict must
    trigger a MissingExogWarning.
    """
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    series = {
        "A": pd.Series(np.ones(5), index=idx, name="A"),
        "B": pd.Series(np.ones(5), index=idx, name="B"),
    }
    exog = {"A": pd.DataFrame({"x": np.ones(5)}, index=idx)}  # "B" absent

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = validate_exog_fit(series=series, exog=exog, is_multiseries=True)

    assert MissingExogWarning in [warning.category for warning in w]
    assert result["B"] is None
    assert result["A"] == ["x"]


def test_validate_exog_fit_multi_heterogeneous_columns_stored_correctly():
    """
    _validate_exog_fit multi-series dict: heterogeneous exog columns across series
    are allowed; each series gets its own column list.
    """
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    series = {
        "A": pd.Series(np.ones(5), index=idx, name="A"),
        "B": pd.Series(np.ones(5), index=idx, name="B"),
    }
    exog = {
        "A": pd.DataFrame({"temp": np.ones(5), "humidity": np.zeros(5)}, index=idx),
        "B": pd.DataFrame({"temp": np.ones(5), "wind": np.zeros(5)}, index=idx),
    }

    result = validate_exog_fit(series=series, exog=exog, is_multiseries=True)

    assert result["A"] == ["temp", "humidity"]
    assert result["B"] == ["temp", "wind"]


def test_validate_exog_fit_multi_broadcast_exog_validates_against_each_series():
    """
    _validate_exog_fit multi-series with broadcast (flat) exog: the same exog is
    validated against each series; returns the same column list for all series.
    """
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    series = {
        "A": pd.Series(np.ones(5), index=idx, name="A"),
        "B": pd.Series(np.ones(5), index=idx, name="B"),
    }
    exog = pd.DataFrame({"x": np.ones(5)}, index=idx)  # broadcast

    result = validate_exog_fit(series=series, exog=exog, is_multiseries=True)

    assert result == {"A": ["x"], "B": ["x"]}


# ===========================================================================
# validate_last_window_exog
# ===========================================================================


def test_validate_last_window_exog_none_last_window_returns_immediately():
    """
    _validate_last_window_exog: last_window=None means no validation is needed.
    """
    # Should not raise or warn
    validate_last_window_exog(
        last_window_exog=pd.Series([1.0, 2.0], name="x"),
        last_window=None,
        exog_in_=True,
    )


def test_validate_last_window_exog_exog_in_none_warns_IgnoredArgumentWarning():
    """
    _validate_last_window_exog: when exog_in_=True and last_window_exog=None,
    an IgnoredArgumentWarning must be issued (improvement A).
    """
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    lw  = pd.Series(np.ones(3), index=idx, name="y")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_last_window_exog(
            last_window_exog=None,
            last_window=lw,
            exog_in_=True,
        )

    assert any(issubclass(warning.category, IgnoredArgumentWarning) for warning in w)


def test_validate_last_window_exog_single_length_mismatch_raises_ValueError():
    """
    _validate_last_window_exog single-series: length mismatch raises ValueError.
    """
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    lw   = pd.Series(np.ones(5), index=idx, name="y")
    lwe  = pd.DataFrame({"x": np.ones(3)}, index=idx[:3])

    with pytest.raises(ValueError, match="same number of observations"):
        validate_last_window_exog(
            last_window_exog=lwe,
            last_window=lw,
            exog_in_=True,
        )


def test_validate_last_window_exog_single_datetimeindex_mismatch_raises_ValueError():
    """
    _validate_last_window_exog single-series: DatetimeIndex mismatch raises ValueError.
    """
    idx_lw  = pd.date_range("2020-01-01", periods=5, freq="D")
    idx_lwe = pd.date_range("2020-01-02", periods=5, freq="D")
    lw  = pd.Series(np.ones(5), index=idx_lw,  name="y")
    lwe = pd.DataFrame({"x": np.ones(5)}, index=idx_lwe)

    with pytest.raises(ValueError, match="index of `last_window_exog`.*aligned"):
        validate_last_window_exog(
            last_window_exog=lwe,
            last_window=lw,
            exog_in_=True,
        )


def test_validate_last_window_exog_multi_length_mismatch_raises_ValueError():
    """
    _validate_last_window_exog multi-series dict: length mismatch for one series
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
        validate_last_window_exog(
            last_window_exog=lwe,
            last_window=lw,
            exog_in_=True,
        )


def test_validate_last_window_exog_multi_missing_series_warns_MissingExogWarning():
    """
    _validate_last_window_exog multi-series dict: absent series in last_window_exog
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
        validate_last_window_exog(
            last_window_exog=lwe,
            last_window=lw,
            exog_in_=True,
        )

    assert MissingExogWarning in [warning.category for warning in w]


# ===========================================================================
# validate_exog_predict — RangeIndex start check
# ===========================================================================


def _make_validate_predict_kwargs(exog, steps=3, last_window_end=9):
    """
    Build a minimal kwargs dict for validate_exog_predict using a RangeIndex.
    training_range_ is set so the reference end-point is last_window_end.
    """
    return dict(
        exog=exog,
        steps=steps,
        last_window=None,
        exog_names_in_=["x"],
        exog_in_=True,
        index_freq_=1,
        is_multiseries=False,
        training_range_=pd.RangeIndex(start=0, stop=last_window_end + 1, step=1)[[0, -1]],
        series_names_in_=["y"],
    )


def test_validate_exog_predict_rangeindex_wrong_start_raises_ValueError():
    """
    _validate_exog_predict: RangeIndex exog that does not start immediately after
    last_window must raise ValueError.
    """
    # last_window_end = 9, so expected start = 10; exog starts at 15
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
            last_window=None,
            exog_names_in_=["x"],
            exog_in_=True,
            index_freq_=pd.tseries.frequencies.to_offset("D"),
            is_multiseries=True,
            training_range_={"A": pd.DatetimeIndex(["2019-12-29", "2019-12-31"])},
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
        last_window=None,
        exog_names_in_=["x", "wind"],
        exog_in_=True,
        index_freq_=pd.tseries.frequencies.to_offset("D"),
        is_multiseries=True,
        training_range_={
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
# align_exog_to_series
# ===========================================================================


def test_align_exog_to_series_none_returns_none_single():
    """
    exog=None → returns None regardless of series type (single-series).
    """
    s = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3, freq="D"), name="y")
    assert align_exog_to_series(series=s, exog=None, is_multiseries=False) is None


def test_align_exog_to_series_none_returns_none_multiseries():
    """
    exog=None → returns None regardless of series type (multi-series).
    """
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    series = {"A": pd.Series([1.0, 2.0, 3.0], index=idx, name="A")}
    assert align_exog_to_series(series=series, exog=None, is_multiseries=True) is None


def test_align_exog_to_series_single_rangeindex_is_noop():
    """
    Single-series with RangeIndex → exog returned unchanged (same object).
    """
    series = pd.Series([1.0, 2.0, 3.0], name="y")
    exog = pd.Series([10.0, 20.0, 30.0], name="x")
    result = align_exog_to_series(series=series, exog=exog, is_multiseries=False)
    assert result is exog


def test_align_exog_to_series_single_already_aligned_returns_same_object():
    """
    Single-series with DatetimeIndex, exog already has the same index →
    the same exog object is returned without copying.
    """
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    series = pd.Series(np.ones(4), index=idx, name="y")
    exog = pd.DataFrame({"x": np.ones(4)}, index=idx)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = align_exog_to_series(series=series, exog=exog, is_multiseries=False)
    assert result is exog
    assert not any(issubclass(warning.category, MissingValuesWarning) for warning in w)


def test_align_exog_to_series_single_exog_longer_clipped_to_series():
    """
    Single-series: exog covers a superset of the series index → after reindex
    the result has exactly the series index (extra dates dropped).
    """
    idx_series = pd.date_range("2020-01-02", periods=3, freq="D")
    idx_exog   = pd.date_range("2020-01-01", periods=5, freq="D")  # 1 extra on each side
    series = pd.Series(np.ones(3), index=idx_series, name="y")
    exog = pd.DataFrame({"x": np.arange(5, dtype=float)}, index=idx_exog)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = align_exog_to_series(series=series, exog=exog, is_multiseries=False)
    assert list(result.index) == list(idx_series)
    assert not any(issubclass(warning.category, MissingValuesWarning) for warning in w)


def test_align_exog_to_series_single_missing_dates_fills_nan_and_warns():
    """
    Single-series: exog is missing some dates that the series covers → those
    positions are NaN and a MissingValuesWarning is issued.
    """
    idx_series = pd.date_range("2020-01-01", periods=5, freq="D")
    # exog only covers 3 of the 5 dates
    idx_exog = idx_series[[0, 1, 4]]
    series = pd.Series(np.ones(5), index=idx_series, name="y")
    exog = pd.DataFrame({"x": [1.0, 2.0, 5.0]}, index=idx_exog)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = align_exog_to_series(series=series, exog=exog, is_multiseries=False)
    missing_warnings = [warning for warning in w if issubclass(warning.category, MissingValuesWarning)]
    assert len(missing_warnings) == 1
    assert list(result.index) == list(idx_series)
    assert result["x"].isna().sum() == 2
    assert pd.notna(result.loc[idx_series[0], "x"])
    assert pd.notna(result.loc[idx_series[1], "x"])
    assert pd.isna(result.loc[idx_series[2], "x"])
    assert pd.isna(result.loc[idx_series[3], "x"])
    assert pd.notna(result.loc[idx_series[4], "x"])


def test_align_exog_to_series_single_warning_message_contains_counts():
    """
    MissingValuesWarning message must report the number of missing positions
    and the total length of the series.
    """
    idx_series = pd.date_range("2020-01-01", periods=5, freq="D")
    idx_exog = idx_series[[0, 1, 2]]  # 2 missing
    series = pd.Series(np.ones(5), index=idx_series, name="y")
    exog = pd.Series([1.0, 2.0, 3.0], index=idx_exog, name="x")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        align_exog_to_series(series=series, exog=exog, is_multiseries=False)
    msg = str(w[0].message)
    assert "2" in msg   # n_missing
    assert "5" in msg   # total len


def test_align_exog_to_series_single_series_is_pd_series_input():
    """
    exog can be a pd.Series (not only a DataFrame) for single-series mode.
    """
    idx_series = pd.date_range("2020-01-01", periods=4, freq="D")
    idx_exog   = idx_series[[0, 1, 3]]  # missing index[2]
    series = pd.Series(np.ones(4), index=idx_series, name="y")
    exog = pd.Series([10.0, 20.0, 40.0], index=idx_exog, name="x")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = align_exog_to_series(series=series, exog=exog, is_multiseries=False)
    assert isinstance(result, pd.Series)
    assert list(result.index) == list(idx_series)
    assert pd.isna(result.iloc[2])
    assert len([warning for warning in w if issubclass(warning.category, MissingValuesWarning)]) == 1


def test_align_exog_to_series_multiseries_rangeindex_is_noop():
    """
    Multi-series: when the dict series has a RangeIndex, exog is returned unchanged.
    """
    series = {
        "A": pd.Series([1.0, 2.0, 3.0], name="A"),
    }
    exog = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    result = align_exog_to_series(series=series, exog=exog, is_multiseries=True)
    assert result is exog


def test_align_exog_to_series_multiseries_dict_exog_already_aligned_noop():
    """
    Multi-series dict exog: each entry already matches its series index →
    same objects returned, no warnings.
    """
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    series = {
        "A": pd.Series(np.ones(3), index=idx, name="A"),
        "B": pd.Series(np.ones(3), index=idx, name="B"),
    }
    exog_a = pd.DataFrame({"x": np.ones(3)}, index=idx)
    exog_b = pd.DataFrame({"x": np.ones(3)}, index=idx)
    exog = {"A": exog_a, "B": exog_b}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = align_exog_to_series(series=series, exog=exog, is_multiseries=True)
    assert result["A"] is exog_a
    assert result["B"] is exog_b
    assert not any(issubclass(warning.category, MissingValuesWarning) for warning in w)


def test_align_exog_to_series_multiseries_dict_exog_missing_dates_warns_per_series():
    """
    Multi-series dict exog: when one series entry has missing dates, a
    MissingValuesWarning is issued (once per series with missing data).
    """
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    series = {
        "A": pd.Series(np.ones(5), index=idx, name="A"),
        "B": pd.Series(np.ones(5), index=idx, name="B"),
    }
    exog = {
        "A": pd.DataFrame({"x": np.ones(5)}, index=idx),          # fully aligned
        "B": pd.DataFrame({"x": np.ones(3)}, index=idx[:3]),       # missing 2 dates
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = align_exog_to_series(series=series, exog=exog, is_multiseries=True)
    missing_warnings = [warning for warning in w if issubclass(warning.category, MissingValuesWarning)]
    assert len(missing_warnings) == 1
    assert result["A"] is exog["A"]
    assert result["B"]["x"].isna().sum() == 2


def test_align_exog_to_series_multiseries_dict_exog_none_entry_passed_through():
    """
    Multi-series dict exog: None entries (no exog for that series) pass through
    unchanged.
    """
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    series = {
        "A": pd.Series(np.ones(3), index=idx, name="A"),
        "B": pd.Series(np.ones(3), index=idx, name="B"),
    }
    exog = {"A": pd.DataFrame({"x": np.ones(3)}, index=idx), "B": None}
    result = align_exog_to_series(series=series, exog=exog, is_multiseries=True)
    assert result["B"] is None


def test_align_exog_to_series_multiseries_broadcast_dataframe_series_reindexed():
    """
    Multi-series with broadcast DataFrame exog and a wide DataFrame for series:
    exog is reindexed once to series.index; missing positions are NaN.
    """
    idx_series = pd.date_range("2020-01-01", periods=5, freq="D")
    idx_exog   = pd.date_range("2020-01-01", periods=3, freq="D")  # missing last 2
    series = pd.DataFrame(
        {"A": np.ones(5), "B": np.ones(5)}, index=idx_series
    )
    exog = pd.DataFrame({"feat": [1.0, 2.0, 3.0]}, index=idx_exog)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = align_exog_to_series(series=series, exog=exog, is_multiseries=True)
    assert isinstance(result, pd.DataFrame)
    assert list(result.index) == list(idx_series)
    assert result["feat"].isna().sum() == 2
    assert len([warning for warning in w if issubclass(warning.category, MissingValuesWarning)]) == 1


def test_align_exog_to_series_multiseries_broadcast_dict_series_expands_to_dict():
    """
    Multi-series with broadcast DataFrame exog and dict series: broadcast exog
    is expanded to a per-series dict so each entry is reindexed independently.
    """
    idx_common = pd.date_range("2020-01-01", periods=4, freq="D")
    series = {
        "A": pd.Series(np.ones(4), index=idx_common, name="A"),
        "B": pd.Series(np.ones(4), index=idx_common, name="B"),
    }
    exog = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}, index=idx_common)
    result = align_exog_to_series(series=series, exog=exog, is_multiseries=True)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"A", "B"}
    pd.testing.assert_frame_equal(result["A"], exog.reindex(idx_common))
    pd.testing.assert_frame_equal(result["B"], exog.reindex(idx_common))


def test_align_exog_to_series_multiseries_broadcast_dict_series_heterogeneous_end_dates():
    """
    Multi-series broadcast exog with dict series that have different end dates:
    each per-series entry is reindexed to its own (shorter) index, filling NaN
    for the dates the exog has but the series doesn't, and issuing no warning
    because the series index is the target (no dates in the series that are
    missing from the exog).
    """
    idx_full  = pd.date_range("2020-01-01", periods=5, freq="D")
    idx_short = pd.date_range("2020-01-01", periods=3, freq="D")
    series = {
        "A": pd.Series(np.ones(5), index=idx_full,  name="A"),
        "B": pd.Series(np.ones(3), index=idx_short, name="B"),
    }
    # exog covers all 5 days
    exog = pd.DataFrame({"x": np.arange(5, dtype=float)}, index=idx_full)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = align_exog_to_series(series=series, exog=exog, is_multiseries=True)
    assert isinstance(result, dict)
    assert list(result["A"].index) == list(idx_full)
    assert list(result["B"].index) == list(idx_short)
    # No MissingValuesWarning: exog covers all series dates
    assert not any(issubclass(warning.category, MissingValuesWarning) for warning in w)


def test_align_exog_to_series_multiseries_broadcast_dict_series_shorter_exog_warns():
    """
    Multi-series broadcast exog with dict series: when broadcast exog does not
    cover all dates in a series, NaN is introduced and a MissingValuesWarning
    per affected series is issued.
    """
    idx_full  = pd.date_range("2020-01-01", periods=5, freq="D")
    idx_exog  = pd.date_range("2020-01-01", periods=3, freq="D")  # missing last 2
    series = {
        "A": pd.Series(np.ones(5), index=idx_full, name="A"),
        "B": pd.Series(np.ones(5), index=idx_full, name="B"),
    }
    exog = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=idx_exog)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = align_exog_to_series(series=series, exog=exog, is_multiseries=True)
    missing_warnings = [warning for warning in w if issubclass(warning.category, MissingValuesWarning)]
    assert len(missing_warnings) == 2  # one per series
    assert result["A"]["x"].isna().sum() == 2
    assert result["B"]["x"].isna().sum() == 2


def test_align_exog_to_series_multiseries_dataframe_series_rangeindex_is_noop():
    """
    Multi-series with wide DataFrame series that has a RangeIndex → exog
    returned unchanged.
    """
    series = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    exog = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    result = align_exog_to_series(series=series, exog=exog, is_multiseries=True)
    assert result is exog
