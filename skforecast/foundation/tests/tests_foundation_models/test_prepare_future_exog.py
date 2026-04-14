# Unit test _prepare_future_exog FoundationModel
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.exceptions import InputTypeWarning, MissingValuesWarning
from skforecast.foundation._foundation_model import FoundationModel
from .fixtures_adapters import FakePipeline


# -- Fixtures ------------------------------------------------------------------
_idx = pd.date_range("2020-01-01", periods=30, freq="MS")
_ctx_s1 = pd.Series(np.arange(30, dtype=float), index=_idx, name="s1")
_ctx_s2 = pd.Series(np.arange(30, 60, dtype=float), index=_idx, name="s2")
_context_single = {"s1": _ctx_s1}
_context_multi = {"s1": _ctx_s1, "s2": _ctx_s2}
_series_single = ["s1"]
_series_multi = ["s1", "s2"]

_future_idx = pd.date_range("2022-07-01", periods=5, freq="MS")
_exog_df = pd.DataFrame(
    {"feat_a": np.arange(5, dtype=float), "feat_b": np.arange(5, 10, dtype=float)},
    index=_future_idx,
)
_exog_series = pd.Series(np.arange(5, dtype=float), index=_future_idx, name="feat_a")


def _make_model():
    return FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())


# Tests _exog_to_dict
# ==============================================================================
def test_exog_to_dict_with_dict_input():
    """
    Test _exog_to_dict returns correct keys when exog is a dict.
    Missing keys are filled with None.
    """
    exog = {"s1": _exog_df}
    result = FoundationModel._exog_to_dict(exog, _series_multi)

    assert list(result.keys()) == _series_multi
    pd.testing.assert_frame_equal(result["s1"], _exog_df)
    assert result["s2"] is None


def test_exog_to_dict_with_flat_series():
    """
    Test _exog_to_dict broadcasts a flat Series to all series.
    """
    result = FoundationModel._exog_to_dict(_exog_series, _series_multi)

    assert list(result.keys()) == _series_multi
    for name in _series_multi:
        pd.testing.assert_series_equal(result[name], _exog_series)


def test_exog_to_dict_with_flat_dataframe():
    """
    Test _exog_to_dict broadcasts a flat DataFrame to all series.
    """
    result = FoundationModel._exog_to_dict(_exog_df, _series_multi)

    assert list(result.keys()) == _series_multi
    for name in _series_multi:
        pd.testing.assert_frame_equal(result[name], _exog_df)


def test_exog_to_dict_with_multiindex_dataframe():
    """
    Test _exog_to_dict splits a long-format MultiIndex DataFrame into
    per-series dict and raises InputTypeWarning.
    """
    idx = pd.date_range("2020-01-01", periods=3, freq="MS")
    mi = pd.MultiIndex.from_arrays(
        [["s1"] * 3 + ["s2"] * 3, idx.tolist() * 2]
    )
    df_long = pd.DataFrame({"feat": np.arange(6, dtype=float)}, index=mi)

    warn_msg = re.escape(
        "Passing a long-format DataFrame as `exog` requires "
        "additional internal transformations"
    )
    with pytest.warns(InputTypeWarning, match=warn_msg):
        result = FoundationModel._exog_to_dict(df_long, _series_multi)

    assert list(result.keys()) == _series_multi
    assert len(result["s1"]) == 3
    assert len(result["s2"]) == 3


def test_exog_to_dict_TypeError_multiindex_not_datetime():
    """
    Test _exog_to_dict raises TypeError when the second MultiIndex level
    is not a DatetimeIndex.
    """
    mi = pd.MultiIndex.from_arrays(
        [["s1", "s1", "s2", "s2"], [0, 1, 0, 1]]
    )
    df_long = pd.DataFrame({"feat": [1, 2, 3, 4]}, index=mi)

    err_msg = re.escape(
        "The second level of the MultiIndex in `exog` must be a "
        "pandas DatetimeIndex."
    )
    with pytest.raises(TypeError, match=err_msg):
        FoundationModel._exog_to_dict(df_long, _series_multi)


def test_exog_to_dict_with_multiindex_series():
    """
    Test _exog_to_dict handles a MultiIndex Series by converting it to
    DataFrame first.
    """
    idx = pd.date_range("2020-01-01", periods=3, freq="MS")
    mi = pd.MultiIndex.from_arrays(
        [["s1"] * 3 + ["s2"] * 3, idx.tolist() * 2]
    )
    s_long = pd.Series(np.arange(6, dtype=float), index=mi, name="feat")

    warn_msg = re.escape(
        "Passing a long-format DataFrame as `exog` requires "
        "additional internal transformations"
    )
    with pytest.warns(InputTypeWarning, match=warn_msg):
        result = FoundationModel._exog_to_dict(s_long, _series_multi)

    assert list(result.keys()) == _series_multi
    assert len(result["s1"]) == 3
    assert len(result["s2"]) == 3


# Tests _prepare_future_exog — None and type errors
# ==============================================================================
def test_prepare_future_exog_returns_none_dict_when_exog_is_none():
    """
    Test _prepare_future_exog returns dict of Nones when exog is None.
    """
    m = _make_model()
    result = m._prepare_future_exog(
        steps=5, context=_context_multi, exog=None,
        series_names_in=_series_multi,
    )

    assert list(result.keys()) == _series_multi
    for v in result.values():
        assert v is None


def test_prepare_future_exog_TypeError_when_unsupported_type():
    """
    Test _prepare_future_exog raises TypeError when exog is an unsupported
    type (e.g. list).
    """
    m = _make_model()
    err_msg = re.escape(
        "`exog` must be a pandas Series, DataFrame, dict, or None."
    )
    with pytest.raises(TypeError, match=err_msg):
        m._prepare_future_exog(
            steps=5, context=_context_single, exog=[1, 2, 3],
            series_names_in=_series_single,
        )


# Tests _prepare_future_exog — DatetimeIndex alignment
# ==============================================================================
def test_prepare_future_exog_aligns_datetime_exog():
    """
    Test _prepare_future_exog aligns a DatetimeIndex exog to the forecast
    horizon derived from context.
    """
    m = _make_model()
    steps = 3
    expected_idx = pd.date_range("2022-07-01", periods=steps, freq="MS")
    future_exog = pd.DataFrame(
        {"feat": [10.0, 20.0, 30.0]}, index=expected_idx
    )

    result = m._prepare_future_exog(
        steps=steps, context=_context_single, exog=future_exog,
        series_names_in=_series_single,
    )

    assert result["s1"] is not None
    assert len(result["s1"]) == steps
    pd.testing.assert_frame_equal(result["s1"], future_exog)


def test_prepare_future_exog_MissingValuesWarning_when_nan_gaps():
    """
    Test _prepare_future_exog emits MissingValuesWarning when reindexing
    creates NaN gaps and batches the warning for all affected series.
    """
    m = _make_model()
    steps = 3
    # Provide exog that only covers 1 of 3 expected timestamps
    first_date = _ctx_s1.index[-1] + pd.DateOffset(months=1)
    partial_exog = pd.DataFrame(
        {"feat": [10.0]},
        index=pd.DatetimeIndex([first_date], freq="MS"),
    )

    warn_msg = re.escape("Missing timestamps were filled with NaN.")
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        result = m._prepare_future_exog(
            steps=steps, context=_context_single, exog=partial_exog,
            series_names_in=_series_single,
        )

    assert result["s1"] is not None
    assert len(result["s1"]) == steps
    assert result["s1"].isnull().any(axis=None)


def test_prepare_future_exog_broadcasts_dataframe_to_all_series():
    """
    Test _prepare_future_exog broadcasts a flat DataFrame to all series
    and aligns each independently.
    """
    m = _make_model()
    steps = 3
    expected_idx = pd.date_range("2022-07-01", periods=steps, freq="MS")
    future_exog = pd.DataFrame(
        {"feat": [10.0, 20.0, 30.0]}, index=expected_idx
    )

    result = m._prepare_future_exog(
        steps=steps, context=_context_multi, exog=future_exog,
        series_names_in=_series_multi,
    )

    for name in _series_multi:
        assert result[name] is not None
        assert len(result[name]) == steps


# Tests _prepare_future_exog — RangeIndex
# ==============================================================================
def test_prepare_future_exog_aligns_range_index_exog():
    """
    Test _prepare_future_exog correctly slices a RangeIndex exog to
    `steps` rows.
    """
    m = _make_model()
    range_ctx = {"s1": pd.Series(np.arange(10, dtype=float))}
    steps = 3
    exog_ri = pd.DataFrame(
        {"feat": [100.0, 200.0, 300.0, 400.0, 500.0]},
        index=pd.RangeIndex(start=10, stop=15, step=1),
    )

    result = m._prepare_future_exog(
        steps=steps, context=range_ctx, exog=exog_ri,
        series_names_in=["s1"],
    )

    assert len(result["s1"]) == steps
    np.testing.assert_array_equal(result["s1"]["feat"].values, [100.0, 200.0, 300.0])


def test_prepare_future_exog_ValueError_when_too_few_rows():
    """
    Test _prepare_future_exog raises ValueError when non-DatetimeIndex
    exog has fewer than `steps` rows.
    """
    m = _make_model()
    range_ctx = {"s1": pd.Series(np.arange(10, dtype=float))}
    short_exog = pd.DataFrame({"feat": [1.0, 2.0]}, index=pd.RangeIndex(start=10, stop=12))

    err_msg = re.escape(
        "`exog` for series 's1' must have at least 5 values. Got 2."
    )
    with pytest.raises(ValueError, match=err_msg):
        m._prepare_future_exog(
            steps=5, context=range_ctx, exog=short_exog,
            series_names_in=["s1"],
        )


def test_prepare_future_exog_ValueError_when_range_index_wrong_start():
    """
    Test _prepare_future_exog raises ValueError when RangeIndex exog does not
    start at the expected position.
    """
    m = _make_model()
    range_ctx = {"s1": pd.Series(np.arange(10, dtype=float))}
    bad_start_exog = pd.DataFrame(
        {"feat": np.arange(5, dtype=float)},
        index=pd.RangeIndex(start=20, stop=25),
    )

    err_msg = re.escape(
        "To make predictions `exog` for series 's1' must start one step "
        "ahead of `context`."
    )
    with pytest.raises(ValueError, match=err_msg):
        m._prepare_future_exog(
            steps=5, context=range_ctx, exog=bad_start_exog,
            series_names_in=["s1"],
        )


# Tests _prepare_future_exog — dict exog with missing keys
# ==============================================================================
def test_prepare_future_exog_dict_with_missing_keys_fills_none():
    """
    Test _prepare_future_exog fills None for series keys missing from the
    exog dict.
    """
    m = _make_model()
    steps = 3
    expected_idx = pd.date_range("2022-07-01", periods=steps, freq="MS")
    exog_dict = {"s1": pd.DataFrame({"feat": [1.0, 2.0, 3.0]}, index=expected_idx)}

    result = m._prepare_future_exog(
        steps=steps, context=_context_multi, exog=exog_dict,
        series_names_in=_series_multi,
    )

    assert result["s1"] is not None
    assert result["s2"] is None


# Tests _prepare_future_exog — Series coercion
# ==============================================================================
def test_prepare_future_exog_coerces_series_to_dataframe():
    """
    Test _prepare_future_exog coerces a flat pandas Series exog value into
    a single-column DataFrame.
    """
    m = _make_model()
    steps = 3
    expected_idx = pd.date_range("2022-07-01", periods=steps, freq="MS")
    exog_s = pd.Series([1.0, 2.0, 3.0], index=expected_idx, name="feat")

    result = m._prepare_future_exog(
        steps=steps, context=_context_single, exog=exog_s,
        series_names_in=_series_single,
    )

    assert isinstance(result["s1"], pd.DataFrame)
    assert len(result["s1"]) == steps
    assert list(result["s1"].columns) == ["feat"]
