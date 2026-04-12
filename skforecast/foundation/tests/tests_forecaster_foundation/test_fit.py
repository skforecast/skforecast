# Unit test fit ForecasterFoundation
# ==============================================================================
import re
import pytest
import warnings
import numpy as np
import pandas as pd
from skforecast.foundation import ForecasterFoundation
from skforecast.exceptions import (
    InputTypeWarning,
    IgnoredArgumentWarning,
    MissingExogWarning,
    MissingValuesWarning,
)

# Fixtures
from .fixtures_forecaster_foundation import (
    make_forecaster,
    y,
    y_range,
    exog,
    df_exog,
    series_wide,
    series_wide_range,
    series_long,
    exog_long,
    series_df,
    series_dict,
    exog_dict,
    MULTISERIES_INDEX,
)


# Tests fit — errors and warnings
# ==============================================================================

def test_fit_TypeError_when_series_has_invalid_type():
    """
    Raise TypeError when `series` is not pd.Series, pd.DataFrame, or dict.
    """
    forecaster = make_forecaster()
    with pytest.raises(TypeError, match="`series` must be"):
        forecaster.fit(series=[1, 2, 3])


def test_fit_MissingValuesWarning_when_len_exog_differs_from_len_y_RangeIndex():
    """
    Issue MissingValuesWarning when exog and y have different lengths and a
    RangeIndex is used.
    """
    forecaster = make_forecaster()
    y_range_named = y_range.rename("y")
    exog_short_range = pd.DataFrame(
        {"feat_a": np.arange(10, dtype=float)},
        index=pd.RangeIndex(10),
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        forecaster.fit(series=y_range_named, exog=exog_short_range)

    assert forecaster.is_fitted
    missing_warnings = [
        warning for warning in w
        if issubclass(warning.category, MissingValuesWarning)
    ]
    assert len(missing_warnings) >= 1


def test_fit_MissingValuesWarning_when_short_DatetimeIndex_exog_reindexed():
    """
    When a partial DatetimeIndex exog is shorter than the series, it is
    reindexed to the full series index (NaN-filled) and a MissingValuesWarning
    is issued.
    """
    forecaster = make_forecaster()
    exog_short = exog.iloc[:10]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        forecaster.fit(series=y, exog=exog_short)

    assert forecaster.is_fitted
    missing_warnings = [
        warning for warning in w
        if issubclass(warning.category, MissingValuesWarning)
    ]
    assert len(missing_warnings) == 1
    assert "Missing values" in str(missing_warnings[0].message) or \
           "filled with NaN" in str(missing_warnings[0].message)


def test_fit_issues_IgnoredArgumentWarning_when_adapter_does_not_support_exog():
    """
    fit() issues IgnoredArgumentWarning and sets exog_in_=False when the
    underlying adapter has allow_exogenous=False.
    """
    forecaster = make_forecaster()
    forecaster.estimator.adapter.allow_exogenous = False

    with pytest.warns(
        IgnoredArgumentWarning,
        match="does not support exogenous variables",
    ):
        forecaster.fit(series=y, exog=exog)

    assert forecaster.exog_in_ is False
    assert forecaster.exog_names_in_ is None
    assert forecaster.exog_type_in_ is None


# Tests fit — single series basic output
# ==============================================================================

def test_fit_output_when_single_series():
    """
    fit() returns None, sets is_fitted to True, stores fit_date, and
    stores series_names_in_ correctly for a single series.
    """
    forecaster = make_forecaster()
    result = forecaster.fit(series=y)

    assert result is None
    assert forecaster.is_fitted is True
    assert isinstance(forecaster.fit_date, str)
    assert forecaster.series_names_in_ == ["y"]
    assert forecaster.is_multiple_series_ is False


def test_fit_series_names_in_fallback_to_y_when_name_is_None():
    """
    series_names_in_ falls back to ['y'] when the Series has no name.
    """
    forecaster = make_forecaster()
    y_unnamed = y.rename(None)
    forecaster.fit(series=y_unnamed)
    assert forecaster.series_names_in_ == ["y"]


def test_fit_training_range_correctly_stored():
    """
    training_range_ is a dict with the first and last index values of each
    training series.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    expected = y.index[[0, -1]]
    assert isinstance(forecaster.training_range_, dict)
    assert set(forecaster.training_range_.keys()) == {"y"}
    pd.testing.assert_index_equal(forecaster.training_range_["y"], expected)


@pytest.mark.parametrize(
    "series, expected_type, expected_freq",
    [
        (y, pd.DatetimeIndex, y.index.freq),
        (y_range, pd.RangeIndex, y_range.index.step),
    ],
    ids=["DatetimeIndex", "RangeIndex"],
)
def test_fit_index_metadata_correctly_stored(series, expected_type, expected_freq):
    """
    index_type_ and index_freq_ are correctly stored for both DatetimeIndex
    and RangeIndex.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=series)
    assert forecaster.index_type_ == expected_type
    assert forecaster.index_freq_ == expected_freq


# Tests fit — exog metadata
# ==============================================================================

@pytest.mark.parametrize(
    "exog_input, expected_in, expected_names, expected_type",
    [
        (None, False, None, None),
        (exog["feat_a"], True, ["feat_a"], pd.Series),
        (df_exog, True, ["feat_a", "feat_b"], pd.DataFrame),
    ],
    ids=["no_exog", "Series_exog", "DataFrame_exog"],
)
def test_fit_exog_metadata_correctly_stored(
    exog_input, expected_in, expected_names, expected_type
):
    """
    exog_in_, exog_names_in_, and exog_type_in_ are set correctly for
    no exog, Series exog, and DataFrame exog.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y, exog=exog_input)

    assert forecaster.exog_in_ is expected_in
    assert forecaster.exog_names_in_ == expected_names
    assert forecaster.exog_type_in_ == expected_type


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


# Tests fit — multi-series mode (DataFrame and dict)
# ==============================================================================

@pytest.mark.parametrize(
    "series",
    [series_df, series_dict],
    ids=["DataFrame", "dict"],
)
def test_fit_output_when_multiseries(series):
    """
    fit() with a DataFrame or dict sets is_multiple_series_, is_fitted,
    series_names_in_, index_type_, index_freq_, and training_range_ as dict.
    """
    forecaster = make_forecaster()
    result = forecaster.fit(series=series)

    assert result is None
    assert forecaster.is_fitted is True
    assert forecaster.is_multiple_series_ is True
    assert forecaster.series_names_in_ == ["s1", "s2"]
    assert forecaster.index_type_ == pd.DatetimeIndex
    assert forecaster.index_freq_ == MULTISERIES_INDEX.freq
    assert isinstance(forecaster.training_range_, dict)
    assert set(forecaster.training_range_.keys()) == {"s1", "s2"}


def test_fit_multiseries_exog_dict_stores_metadata():
    """
    fit() with exog as dict stores exog_in_, exog_names_in_, exog_type_in_
    correctly.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=series_df, exog=exog_dict)
    assert forecaster.exog_in_ is True
    assert forecaster.exog_names_in_ == ["feat_a"]
    assert forecaster.exog_type_in_ == dict


def test_fit_multiseries_broadcast_exog_stores_metadata():
    """
    fit() with a single DataFrame exog broadcast to all series stores
    exog metadata correctly.
    """
    forecaster = make_forecaster()
    broadcast_exog = pd.DataFrame(
        {"feat_a": np.arange(50, dtype=float)}, index=MULTISERIES_INDEX
    )
    forecaster.fit(series=series_df, exog=broadcast_exog)
    assert forecaster.exog_in_ is True
    assert forecaster.exog_names_in_ == ["feat_a"]


def test_fit_range_index_multiseries_sets_index_freq_correctly():
    """
    index_freq_ is set to the RangeIndex step for a wide DataFrame with a
    RangeIndex.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=series_wide_range)
    assert forecaster.index_freq_ == series_wide_range.index.step
    assert forecaster.index_type_ == pd.RangeIndex


# Tests fit — refit / state reset
# ==============================================================================

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
    assert forecaster.series_names_in_ == ["z"]
    assert forecaster.exog_in_ is False


def test_fit_refit_clears_multiseries_state():
    """
    Re-fitting with a single series clears is_multiple_series_.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=series_df)
    assert forecaster.is_multiple_series_ is True

    forecaster.fit(series=y)
    assert forecaster.is_multiple_series_ is False
    assert forecaster.series_names_in_ == ["y"]


# Tests fit — long-format DataFrame
# ==============================================================================

def test_fit_long_format_dataframe_works():
    """
    fit() accepts a long-format (MultiIndex) DataFrame and stores correct
    metadata.
    """
    forecaster = make_forecaster()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster.fit(series=series_long)

    assert forecaster.is_fitted is True
    assert forecaster.series_names_in_ == ["series_1", "series_2"]
    assert forecaster.is_multiple_series_ is True
    assert forecaster.index_type_ == pd.DatetimeIndex
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

    assert forecaster.series_names_in_ == ["series_1", "series_2"]


def test_fit_long_format_non_datetime_second_level_raises_TypeError():
    """
    fit() raises TypeError when a long-format DataFrame has a non-DatetimeIndex
    as the second level of the MultiIndex.
    """
    arrays = [
        ["s1", "s1", "s2", "s2"],
        [0, 1, 0, 1],
    ]
    idx = pd.MultiIndex.from_arrays(arrays, names=["series_id", "time"])
    series_bad = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]}, index=idx)

    forecaster = make_forecaster()
    with pytest.raises(TypeError, match="second level of the MultiIndex"):
        forecaster.fit(series=series_bad)


# Tests fit — long-format exog
# ==============================================================================

def test_fit_long_format_exog_works():
    """
    fit() accepts a long-format (MultiIndex) DataFrame as `exog` and stores
    correct metadata.
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
        with pytest.raises(
            TypeError, match="second level.*index must be a.*DatetimeIndex"
        ):
            forecaster.fit(series=series_long, exog=exog_bad)


def test_fit_long_format_series_exog_works():
    """
    fit() accepts a long-format MultiIndex pd.Series as `exog` (MultiIndex
    pd.Series is coerced to pd.DataFrame before normalisation).
    """
    exog_series_long = exog_long["feat_a"]
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
    exog_partial = exog_long.loc[["series_1"]]

    forecaster = make_forecaster()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with pytest.warns(MissingExogWarning, match="series_2"):
            forecaster.fit(series=series_long, exog=exog_partial)


# Tests fit — does not modify input
# ==============================================================================

def test_fit_does_not_modify_input():
    """
    fit() must not modify the original y series or exog DataFrame.
    """
    forecaster = make_forecaster()
    y_copy = y.copy()
    exog_copy = df_exog.copy()
    forecaster.fit(series=y, exog=df_exog)
    pd.testing.assert_series_equal(y, y_copy)
    pd.testing.assert_frame_equal(df_exog, exog_copy)
