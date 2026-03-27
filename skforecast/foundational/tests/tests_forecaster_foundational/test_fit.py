# Unit test fit ForecasterFoundational
# ==============================================================================
import re
import pytest
import warnings
import numpy as np
import pandas as pd
from skforecast.foundational import ForecasterFoundational
from skforecast.exceptions import InputTypeWarning, IgnoredArgumentWarning, MissingExogWarning

# Fixtures
from .fixtures_forecaster_foundational import (
    make_forecaster,
    y,
    y_range,
    exog,
    df_exog,
    series_wide,
    series_wide_range,
    series_long,
    exog_long,
)


# Tests fit
# ==============================================================================

def test_fit_returns_self():
    """
    fit() returns the ForecasterFoundational instance itself.
    """
    forecaster = make_forecaster()
    result = forecaster.fit(series=y)
    assert result is forecaster


def test_fit_sets_is_fitted_to_True():
    """
    is_fitted is set to True after a successful fit.
    """
    forecaster = make_forecaster()
    assert forecaster.is_fitted is False
    forecaster.fit(series=y)
    assert forecaster.is_fitted is True


def test_fit_stores_series_name_in_():
    """
    series_name_in_ stores the name of the training series.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    assert forecaster.series_name_in_ == "y"


def test_fit_stores_series_name_in_fallback_to_y_when_name_is_None():
    """
    series_name_in_ falls back to 'y' when the Series has no name.
    """
    forecaster = make_forecaster()
    y_unnamed = y.rename(None)
    forecaster.fit(series=y_unnamed)
    assert forecaster.series_name_in_ == "y"


def test_fit_stores_training_range_():
    """
    training_range_ contains the first and last index values of the training series.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    expected = y.index[[0, -1]]
    pd.testing.assert_index_equal(forecaster.training_range_, expected)


def test_fit_stores_index_type_for_DatetimeIndex():
    """
    index_type_ stores the type of the training index (DatetimeIndex).
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    assert forecaster.index_type_ == pd.DatetimeIndex


def test_fit_stores_index_type_for_RangeIndex():
    """
    index_type_ stores the type of the training index (RangeIndex).
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y_range)
    assert forecaster.index_type_ == pd.RangeIndex


def test_fit_stores_index_freq_for_DatetimeIndex():
    """
    index_freq_ stores the frequency object for a DatetimeIndex.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    assert forecaster.index_freq_ == y.index.freq


def test_fit_stores_index_freq_for_RangeIndex():
    """
    index_freq_ stores the step for a RangeIndex.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y_range)
    assert forecaster.index_freq_ == y_range.index.step


def test_fit_stores_fit_date():
    """
    fit_date is set to a non-None string after fit.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    assert forecaster.fit_date is not None
    assert isinstance(forecaster.fit_date, str)


def test_fit_exog_in_False_when_no_exog():
    """
    exog_in_ is False and exog_names_in_/exog_type_in_ are None when no exog passed.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    assert forecaster.exog_in_ is False
    assert forecaster.exog_names_in_ is None
    assert forecaster.exog_type_in_ is None


def test_fit_exog_metadata_stored_for_Series_exog():
    """
    exog_in_, exog_names_in_, and exog_type_in_ are set when a Series exog is passed.
    """
    exog_series = exog["feat_a"]
    forecaster = make_forecaster()
    forecaster.fit(series=y, exog=exog_series)

    assert forecaster.exog_in_ is True
    assert forecaster.exog_names_in_ == ["feat_a"]
    assert forecaster.exog_type_in_ == pd.Series


def test_fit_exog_metadata_stored_for_DataFrame_exog():
    """
    exog_in_, exog_names_in_, and exog_type_in_ are set when a DataFrame exog is passed.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y, exog=df_exog)

    assert forecaster.exog_in_ is True
    assert forecaster.exog_names_in_ == ["feat_a", "feat_b"]
    assert forecaster.exog_type_in_ == pd.DataFrame


def test_fit_ValueError_when_len_exog_differs_from_len_y():
    """
    Raise ValueError when exog and y have different lengths.
    """
    forecaster = make_forecaster()
    exog_short = exog.iloc[:10]

    err_msg = re.escape(
        f"`exog` must have same number of samples as `series`. "
        f"length `exog`: ({len(exog_short)}), "
        f"length `series`: ({len(y)})"
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.fit(series=y, exog=exog_short)


def test_fit_resets_state_on_refit():
    """
    Calling fit a second time clears all previous metadata and re-populates it.
    """
    forecaster = make_forecaster()
    y2 = pd.Series(
        np.arange(30, dtype=float),
        index=pd.date_range("2022-01-01", periods=30, freq="ME"),
        name="z",
    )
    forecaster.fit(series=y)
    forecaster.fit(series=y2)

    assert forecaster.series_name_in_ == "z"
    assert forecaster.exog_in_ is False


def test_fit_does_not_modify_y():
    """
    fit() must not modify the original y series.
    """
    forecaster = make_forecaster()
    y_copy = y.copy()
    forecaster.fit(series=y)
    pd.testing.assert_series_equal(y, y_copy)


def test_fit_does_not_modify_exog():
    """
    fit() must not modify the original exog DataFrame.
    """
    forecaster = make_forecaster()
    exog_copy = df_exog.copy()
    forecaster.fit(series=y, exog=df_exog)
    pd.testing.assert_frame_equal(df_exog, exog_copy)


# Tests fit — long-format DataFrame
# ==============================================================================

def test_fit_long_format_dataframe_works():
    """
    fit() accepts a long-format (MultiIndex) DataFrame and stores correct metadata.
    """
    forecaster = make_forecaster()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster.fit(series=series_long)

    assert forecaster.is_fitted is True
    assert forecaster.series_names_in_ == ["series_1", "series_2"]
    assert forecaster._is_multiseries is True
    assert forecaster.series_name_in_ is None
    assert forecaster.index_type_ == pd.DatetimeIndex
    # training_range_ is a dict with one entry per series
    assert set(forecaster.training_range_.keys()) == {"series_1", "series_2"}


def test_fit_long_format_issues_InputTypeWarning():
    """
    fit() raises an InputTypeWarning when a long-format DataFrame is passed.
    """
    forecaster = make_forecaster()
    with pytest.warns(InputTypeWarning):
        forecaster.fit(series=series_long)


def test_fit_long_format_multiple_columns_issues_IgnoredArgumentWarning():
    """
    fit() raises an IgnoredArgumentWarning when the long-format DataFrame has
    more than one column, and only uses the first column values.
    """
    forecaster = make_forecaster()
    series_long_multicol = series_long.copy()
    series_long_multicol["extra"] = 99.0

    with pytest.warns(IgnoredArgumentWarning, match="first column"):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            forecaster.fit(series=series_long_multicol)

    # Only the first column ('value') should have been used.
    assert forecaster.series_names_in_ == ["series_1", "series_2"]


def test_fit_long_format_non_datetime_second_level_raises_TypeError():
    """
    fit() raises TypeError when a long-format DataFrame has a non-DatetimeIndex
    as the second level of the MultiIndex (e.g. RangeIndex).
    """
    # Build a long-format DataFrame whose second level is NOT a DatetimeIndex.
    arrays = [
        ["s1", "s1", "s2", "s2"],
        [0, 1, 0, 1],
    ]
    idx = pd.MultiIndex.from_arrays(arrays, names=["series_id", "time"])
    series_bad = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]}, index=idx)

    forecaster = make_forecaster()
    with pytest.raises(TypeError, match="second level of the MultiIndex"):
        forecaster.fit(series=series_bad)


# Tests fit — RangeIndex regression
# ==============================================================================

def test_fit_range_index_single_series_sets_index_freq_correctly():
    """
    index_freq_ is set to the RangeIndex step (not a freqstr) for a single
    series with a RangeIndex.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y_range)
    assert forecaster.index_freq_ == y_range.index.step


def test_fit_range_index_multiseries_sets_index_freq_correctly():
    """
    index_freq_ is set to the RangeIndex step for a wide DataFrame with a
    RangeIndex.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=series_wide_range)
    assert forecaster.index_freq_ == series_wide_range.index.step
    assert forecaster.index_type_ == pd.RangeIndex


# Tests fit — long-format exog DataFrame
# ==============================================================================

def test_fit_long_format_exog_works():
    """
    fit() accepts a long-format (MultiIndex) DataFrame as `exog` and stores
    correct metadata. exog_type_in_ records the original type (pd.DataFrame).
    """
    forecaster = make_forecaster()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster.fit(series=series_long, exog=exog_long)

    assert forecaster.exog_in_ is True
    assert forecaster.exog_type_in_ == pd.DataFrame
    assert forecaster.exog_names_in_ == ["feat_a"]
    assert forecaster.is_fitted is True


def test_fit_long_format_exog_issues_InputTypeWarning():
    """
    fit() raises an InputTypeWarning when a long-format DataFrame is passed
    as `exog`.
    """
    forecaster = make_forecaster()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with pytest.warns(InputTypeWarning, match="long-format DataFrame"):
            forecaster.fit(series=series_long, exog=exog_long)


def test_fit_long_format_exog_non_datetime_second_level_raises_TypeError():
    """
    fit() raises TypeError when a long-format exog DataFrame has a
    non-DatetimeIndex as the second MultiIndex level.
    """
    arrays = [
        ["series_1", "series_1", "series_2", "series_2"],
        [0, 1, 0, 1],
    ]
    idx = pd.MultiIndex.from_arrays(arrays, names=["series_id", "time"])
    exog_bad = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0, 4.0]}, index=idx)

    forecaster = make_forecaster()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(TypeError, match="second level of the MultiIndex in `exog`"):
            forecaster.fit(series=series_long, exog=exog_bad)


def test_fit_long_format_series_exog_works():
    """
    fit() accepts a long-format MultiIndex pd.Series as `exog` (Fix #2:
    MultiIndex pd.Series is coerced to pd.DataFrame before normalisation).
    """
    exog_series_long = exog_long["feat_a"]  # MultiIndex pd.Series
    assert isinstance(exog_series_long.index, pd.MultiIndex)

    forecaster = make_forecaster()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster.fit(series=series_long, exog=exog_series_long)

    assert forecaster.exog_in_ is True
    assert forecaster.exog_names_in_ == ["feat_a"]
    assert forecaster.exog_type_in_ == pd.Series


def test_fit_long_format_exog_missing_series_issues_MissingExogWarning():
    """
    fit() issues a MissingExogWarning when the long-format exog does not
    cover all series provided in `series`.
    """
    # exog only for series_1 — series_2 is missing
    exog_partial = exog_long.loc[["series_1"]]

    forecaster = make_forecaster()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with pytest.warns(MissingExogWarning, match="series_2"):
            forecaster.fit(series=series_long, exog=exog_partial)


def test_fit_issues_IgnoredArgumentWarning_when_adapter_does_not_support_exog():
    """
    fit() issues IgnoredArgumentWarning and sets exog_in_=False when the
    underlying adapter has allow_exogenous=False.
    """
    forecaster = make_forecaster()  # Chronos2Adapter (allow_exogenous=True)
    # Force allow_exogenous to False to simulate a non-exog adapter
    forecaster.estimator.adapter.allow_exogenous = False

    with pytest.warns(
        IgnoredArgumentWarning,
        match="does not support exogenous variables",
    ):
        forecaster.fit(series=y, exog=exog)

    assert forecaster.exog_in_ is False
    assert forecaster.exog_names_in_ is None
    assert forecaster.exog_type_in_ is None


def test_fit_exog_accepted_when_adapter_supports_exog():
    """
    fit() stores exog metadata normally when allow_exogenous=True (default for
    Chronos2Adapter).
    """
    forecaster = make_forecaster()
    assert forecaster.estimator.allow_exogenous is True

    forecaster.fit(series=y, exog=exog)

    assert forecaster.exog_in_ is True
    assert forecaster.exog_names_in_ == ["feat_a"]


