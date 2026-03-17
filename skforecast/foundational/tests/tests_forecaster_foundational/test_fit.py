# Unit test fit ForecasterFoundational
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.foundational import ForecasterFoundational

# Fixtures
from .fixtures_forecaster_foundational import (
    make_forecaster,
    y,
    y_range,
    exog,
    df_exog,
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
    index_freq_ stores the frequency string for a DatetimeIndex.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    assert forecaster.index_freq_ == y.index.freqstr


def test_fit_stores_index_freq_for_RangeIndex():
    """
    index_freq_ stores the step for a RangeIndex.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y_range)
    assert forecaster.index_freq_ == y_range.index.step


def test_fit_stores_extended_index_():
    """
    extended_index_ is equal to the training series index after fit.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    pd.testing.assert_index_equal(forecaster.extended_index_, y.index)


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
    assert len(forecaster.extended_index_) == 30
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
