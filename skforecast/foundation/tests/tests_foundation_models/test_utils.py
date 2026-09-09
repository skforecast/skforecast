# Unit test skforecast.foundation._utils
# ==============================================================================
import re
import pytest
import warnings
import numpy as np
import pandas as pd

from skforecast.foundation._utils import (
    check_preprocess_series_foundation,
    get_exog_signature,
    group_series_by_exog_signature,
    align_context_exog,
    _warn_if_non_commercial,
    _NON_COMMERCIAL_LICENSES,
)
from skforecast.exceptions import (
    IgnoredArgumentWarning,
    InputTypeWarning,
    LicenseWarning,
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
# _warn_if_non_commercial
# ===========================================================================

@pytest.mark.parametrize(
    "model_id",
    [
        "google/timesfm-3.0-pytorch",
        "Salesforce/moirai-2.0-R-small",
        "priorlabs/tabpfn-ts",
        "taharnbl/TS-ICL",
    ],
    ids=["timesfm-3.0", "moirai", "tabpfn-ts", "tsicl"],
)
def test_warn_if_non_commercial_warns_for_registered_prefixes(model_id):
    """
    _warn_if_non_commercial should raise a LicenseWarning naming the
    model_id and its license for every registered non-commercial prefix.
    """
    license_name, license_url = next(
        info for prefix, info in _NON_COMMERCIAL_LICENSES.items()
        if model_id.startswith(prefix)
    )
    warn_msg = re.escape(
        f"The weights for '{model_id}' are released under {license_name}"
    )
    with pytest.warns(LicenseWarning, match=warn_msg) as record:
        _warn_if_non_commercial(model_id)
    assert license_url in str(record[0].message)


@pytest.mark.parametrize(
    "model_id",
    [
        "autogluon/chronos-2-small",
        "amazon/chronos-2",
        "google/timesfm-2.5-200m-pytorch",
        "soda-inria/tabicl",
        "theforecastingcompany/t0-alpha",
        "Synthefy/Nori",
        "unknown/some-model",
    ],
    ids=["chronos-autogluon", "chronos-amazon", "timesfm-2.5", "tabicl", "t0", "nori", "unknown"],
)
def test_warn_if_non_commercial_no_warning_for_unmatched_prefixes(model_id):
    """
    _warn_if_non_commercial should be a no-op (no warning) for model ids
    that are not registered as non-commercial.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_if_non_commercial(model_id)
    assert not any(issubclass(w.category, LicenseWarning) for w in caught)


def test_warn_if_non_commercial_uses_longest_prefix_match(monkeypatch):
    """
    _warn_if_non_commercial should resolve the most specific (longest)
    matching prefix when several registered prefixes share a root, so a more
    specific prefix is not shadowed by a shorter one. Two overlapping
    prefixes are injected into the registry to actually exercise the
    tie-break (the shipped registry has none that overlap).
    """
    monkeypatch.setitem(
        _NON_COMMERCIAL_LICENSES, "vendor/model",
        ("Short License", "https://example.com/short"),
    )
    monkeypatch.setitem(
        _NON_COMMERCIAL_LICENSES, "vendor/model-pro",
        ("Long License", "https://example.com/long"),
    )

    # A model_id matching both prefixes must resolve to the longer one.
    with pytest.warns(LicenseWarning, match=re.escape("Long License")):
        _warn_if_non_commercial("vendor/model-pro-v1")

    # A model_id matching only the shorter prefix resolves to it.
    with pytest.warns(LicenseWarning, match=re.escape("Short License")):
        _warn_if_non_commercial("vendor/model-basic")


def test_warn_if_non_commercial_matches_registered_and_skips_unregistered():
    """
    A registered non-commercial prefix (TimesFM 3.0) warns, while an
    unregistered id (TimesFM 2.5) does not.
    """
    with pytest.warns(LicenseWarning, match=re.escape("google/timesfm-3.0-pytorch")):
        _warn_if_non_commercial("google/timesfm-3.0-pytorch")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_if_non_commercial("google/timesfm-2.5-200m-pytorch")
    assert not any(issubclass(w.category, LicenseWarning) for w in caught)


def test_warn_if_non_commercial_suppressible_via_simplefilter():
    """
    The LicenseWarning issued by _warn_if_non_commercial should be
    suppressible through the standard warnings.simplefilter mechanism, the
    same one used by the `suppress_warnings` argument across skforecast.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore", category=LicenseWarning)
        _warn_if_non_commercial("google/timesfm-3.0-pytorch")
    assert len(caught) == 0


# ===========================================================================
# get_exog_signature
# ===========================================================================

@pytest.mark.parametrize(
    "context_exog, exog, expected",
    [
        (None, None, ((), ())),
        (pd.DataFrame({"b": [0.0], "a": [0.0]}), None, (("a", "b"), ())),
        (pd.DataFrame({"b": [0.0], "a": [0.0]}), pd.DataFrame({"b": [1.0]}), (("a",), ("b",))),
        (pd.Series([0.0], name="f"), pd.Series([1.0], name="f"), ((), ("f",))),
        (None, pd.DataFrame({"b": [1.0]}), ((), ("b",))),
    ],
    ids=["no_exog", "past_only", "mixed", "series_blocks", "future_without_history"],
)
def test_get_exog_signature_output(context_exog, exog, expected):
    """
    Test that get_exog_signature returns sorted (past_only_cols, fut_cols)
    tuples for one series, treating pandas Series blocks by name and never
    raising for a future column without history.
    """
    assert get_exog_signature(context_exog, exog) == expected


# ===========================================================================
# group_series_by_exog_signature
# ===========================================================================

def test_group_series_by_exog_signature_output():
    """
    Test that series are grouped by their (past-only, future) exog columns,
    that groups keep the first-seen order and input order within a group,
    and that missing keys, None values and None dicts mean no exog.
    """
    p = np.arange(3, dtype=float)
    context_exog = {
        "full":  pd.DataFrame({"p": p, "k": p}),
        "none":  None,
        "past":  pd.DataFrame({"p": p}),
        "full2": pd.DataFrame({"k": p, "p": p}),
    }
    exog = {
        "full":  pd.DataFrame({"k": p}),
        "past":  None,
        "full2": pd.DataFrame({"k": p}),
    }
    series_names_in = ["full", "none", "past", "full2", "missing"]

    results = group_series_by_exog_signature(series_names_in, context_exog, exog)
    assert results == [["full", "full2"], ["none", "missing"], ["past"]]

    results = group_series_by_exog_signature(series_names_in, None, None)
    assert results == [series_names_in]


# ===========================================================================
# align_context_exog
# ===========================================================================

def test_align_context_exog_output():
    """
    Test that align_context_exog reindexes each series' historical exog to
    the index of its context: aligned exog is returned unchanged, extra rows
    are dropped, a Series is coerced to a DataFrame, missing keys or None
    values stay None, and only the requested series are returned.
    """
    ctx_idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    long_idx = pd.date_range("2019-11-30", periods=8, freq="ME")
    context = {
        "s1": pd.Series(np.arange(4, dtype=float), index=ctx_idx),
        "s2": pd.Series(np.arange(4, dtype=float), index=ctx_idx),
        "s3": pd.Series(np.arange(4, dtype=float), index=ctx_idx),
        "s4": pd.Series(np.arange(4, dtype=float), index=ctx_idx),
        "s5": pd.Series(np.arange(4, dtype=float), index=ctx_idx),
    }
    context_exog = {
        "s1": pd.DataFrame({"a": np.arange(4, dtype=float)}, index=ctx_idx),
        "s2": pd.DataFrame({"a": np.arange(8, dtype=float)}, index=long_idx),
        "s3": pd.Series(np.arange(4, dtype=float), index=ctx_idx, name="a"),
        "s4": None,
    }

    results = align_context_exog(context, context_exog, ["s1", "s2", "s3", "s4", "s5"])

    expected_s2 = pd.DataFrame({"a": np.arange(2, 6, dtype=float)}, index=ctx_idx)
    assert list(results.keys()) == ["s1", "s2", "s3", "s4", "s5"]
    pd.testing.assert_frame_equal(results["s1"], context_exog["s1"])
    pd.testing.assert_frame_equal(results["s2"], expected_s2)
    pd.testing.assert_frame_equal(results["s3"], context_exog["s3"].to_frame())
    assert results["s4"] is None
    assert results["s5"] is None


def test_align_context_exog_MissingValuesWarning_when_exog_shorter_than_context():
    """
    Test that context timestamps missing from the historical exog are added
    as NaN rows and reported once with a MissingValuesWarning naming the
    affected series.
    """
    ctx_idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    context = {
        "s1": pd.Series(np.arange(4, dtype=float), index=ctx_idx),
        "s2": pd.Series(np.arange(4, dtype=float), index=ctx_idx),
    }
    context_exog = {
        "s1": pd.DataFrame({"a": np.arange(2, dtype=float)}, index=ctx_idx[:2]),
        "s2": pd.DataFrame({"a": np.arange(3, dtype=float)}, index=ctx_idx[1:]),
    }

    warn_msg = re.escape(
        "`context_exog` for series ['s1', 's2'] has been reindexed to match "
        "the index of `context`. Missing timestamps were filled with NaN."
    )
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        results = align_context_exog(context, context_exog, ["s1", "s2"])

    expected_s1 = pd.DataFrame({"a": [0.0, 1.0, np.nan, np.nan]}, index=ctx_idx)
    expected_s2 = pd.DataFrame({"a": [np.nan, 0.0, 1.0, 2.0]}, index=ctx_idx)
    pd.testing.assert_frame_equal(results["s1"], expected_s1)
    pd.testing.assert_frame_equal(results["s2"], expected_s2)
