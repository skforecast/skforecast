# Unit test skforecast.foundational._utils
# ==============================================================================
import re
import pytest
import warnings
import numpy as np
import pandas as pd

from skforecast.foundational._utils import (
    _check_preprocess_series_type,
    _check_preprocess_exog_type,
)
from skforecast.exceptions import (
    IgnoredArgumentWarning,
    InputTypeWarning,
    MissingExogWarning,
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
# _check_preprocess_series_type
# ===========================================================================

def test_check_preprocess_series_type_named_series():
    """
    pd.Series with a name → (False, [name], series) and the series is returned
    unchanged.
    """
    s = pd.Series([1.0, 2.0, 3.0], index=_index[:3], name="sales")
    is_multi, names, out = _check_preprocess_series_type(s)
    assert is_multi is False
    assert names == ["sales"]
    assert out is s


def test_check_preprocess_series_type_unnamed_series_returns_y():
    """
    pd.Series without a name → name is replaced with 'y'.
    """
    s = pd.Series([1.0, 2.0], index=_index[:2], name=None)
    is_multi, names, out = _check_preprocess_series_type(s)
    assert is_multi is False
    assert names == ["y"]
    assert out is s


def test_check_preprocess_series_type_wide_dataframe():
    """
    Wide (flat-index) DataFrame → (True, column_names, df) and the DataFrame is
    returned unchanged.
    """
    df = pd.DataFrame({"s1": [1.0, 2.0], "s2": [3.0, 4.0]}, index=_index[:2])
    is_multi, names, out = _check_preprocess_series_type(df)
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
    is_multi, names, out = _check_preprocess_series_type(d)
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
        is_multi, names, out = _check_preprocess_series_type(df_long)
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
        _check_preprocess_series_type(df_long)
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
        _check_preprocess_series_type(df_bad)


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
        _check_preprocess_series_type([1.0, 2.0, 3.0])


# ===========================================================================
# _check_preprocess_exog_type
# ===========================================================================

def test_check_preprocess_exog_type_none_returns_none():
    """
    None input should be returned unchanged.
    """
    result = _check_preprocess_exog_type(None)
    assert result is None


def test_check_preprocess_exog_type_dict_returns_dict_unchanged():
    """
    dict input should be returned unchanged.
    """
    d = {"s1": pd.DataFrame({"feat": [1.0, 2.0]}, index=_index[:2])}
    result = _check_preprocess_exog_type(d)
    assert result is d


def test_check_preprocess_exog_type_flat_series_returns_unchanged():
    """
    A flat-index pd.Series should be returned unchanged (broadcast to all series).
    """
    s = pd.Series([1.0, 2.0, 3.0], index=_index[:3], name="feat")
    result = _check_preprocess_exog_type(s)
    assert result is s


def test_check_preprocess_exog_type_wide_dataframe_returns_unchanged():
    """
    A wide (flat-index) DataFrame should be returned unchanged (broadcast to all series).
    """
    df = pd.DataFrame({"feat_a": [1.0, 2.0], "feat_b": [3.0, 4.0]}, index=_index[:2])
    result = _check_preprocess_exog_type(df)
    assert result is df


def test_check_preprocess_exog_type_long_format_converts_to_dict():
    """
    A long-format MultiIndex DataFrame with DatetimeIndex second level should
    be converted to a dict[str, pd.DataFrame] and an InputTypeWarning issued.
    """
    df_long = _make_long_df_datetime()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _check_preprocess_exog_type(df_long)
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
        result = _check_preprocess_exog_type(
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
        _check_preprocess_exog_type([1.0, 2.0, 3.0])
