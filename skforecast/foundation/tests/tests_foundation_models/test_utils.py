# Unit test skforecast.foundation._utils
# ==============================================================================
import re
import pytest
import warnings
import numpy as np
import pandas as pd

from skforecast.foundation._utils import (
    check_preprocess_series_foundation,
)
from skforecast.exceptions import (
    IgnoredArgumentWarning,
    InputTypeWarning,
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
