# Unit test reshape_series_wide_to_long
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from ..._preprocessing import reshape_series_wide_to_long


def test_TypeError_when_data_is_not_dataframe():
    """
    Raise TypeError if data is not a pandas DataFrame.
    """
    err_msg = re.escape("`data` must be a pandas DataFrame.")
    with pytest.raises(TypeError, match=err_msg):
        reshape_series_wide_to_long(data='not_a_dataframe')


def test_TypeError_when_data_not_DatetimeIndex():
    """
    Raise TypeError when `data` index is not a pandas DatetimeIndex.
    """
    data = pd.DataFrame({
        'series_1': np.arange(10),
        'series_2': np.arange(10, 20)
    })

    err_msg = re.escape("`data` index must be a pandas DatetimeIndex.")
    with pytest.raises(TypeError, match=err_msg):
        reshape_series_wide_to_long(data=data)


@pytest.mark.parametrize("return_multi_index", 
                         [True, False], 
                         ids = lambda dt: f'return_multi_index: {dt}')
def test_check_output_reshape_series_wide_to_long(return_multi_index):
    """
    Check output of reshape_series_wide_to_long.
    """
    data = pd.DataFrame({
        'series_1': np.arange(10),
        'series_2': np.arange(10, 20)
    })
    data.index = pd.date_range(start='2020-01-01', periods=10, freq='D')

    results = reshape_series_wide_to_long(data=data, return_multi_index=return_multi_index)

    expected = pd.DataFrame({
        'series_id': ['series_1'] * 10 + ['series_2'] * 10,
        'datetime': pd.date_range(start='2020-01-01', periods=10, freq='D').tolist() * 2,
        'value': np.concatenate([np.arange(10), np.arange(10, 20)])
    })
    if return_multi_index:
        expected = expected.groupby("series_id", sort=False).apply(
            lambda x: x.set_index("datetime").asfreq('D'), include_groups=False
        )
    else:
        expected.index = pd.RangeIndex(start=0, stop=20, step=1)

    pd.testing.assert_frame_equal(results, expected)
    if return_multi_index:
        assert results.loc['series_1'].index.freq == expected.loc['series_1'].index.freq
        assert results.loc['series_2'].index.freq == expected.loc['series_2'].index.freq


@pytest.mark.parametrize(
    "start", ["2025-10-01", "2026-03-01"], ids=lambda start: f"start: {start}"
)
def test_reshape_series_wide_to_long_output_when_index_is_tz_aware_and_utc_anchored(start):
    """
    Test reshape_series_wide_to_long when the index is timezone-aware and
    advances in fixed UTC steps (created in UTC and converted to a local
    timezone) across a daylight saving change.
    """
    index = pd.date_range(
        start=start, periods=60, freq="D", tz="UTC"
    ).tz_convert("Europe/Madrid")
    values = np.arange(60, dtype=float)
    data = pd.DataFrame({"A": values, "B": values * 2}, index=index)
    results = reshape_series_wide_to_long(data=data)

    assert len(results) == 120
    assert results["value"].notna().all()
    for k, factor in (("A", 1), ("B", 2)):
        pd.testing.assert_index_equal(
            results.loc[k].index, index, check_names=False
        )
        np.testing.assert_array_equal(
            results.loc[k, "value"].to_numpy(), values * factor
        )


@pytest.mark.parametrize(
    "freq", ["h", "D", "B", "MS"], ids=lambda freq: f"freq: {freq}"
)
def test_reshape_series_wide_to_long_output_when_index_has_no_freq(freq):
    """
    Test the frequency is inferred when the index has none, so the output is
    the same as with the frequency. Before, `asfreq` used a daily frequency:
    hourly data lost rows, and business day and monthly data gained rows of
    NaN. The index of the input DataFrame is not renamed.
    """
    index = pd.date_range(start="2020-01-01", periods=30, freq=freq, name="my_index")
    values = np.arange(30, dtype=float)
    data = pd.DataFrame({"A": values, "B": values * 2}, index=index)
    data_no_freq = data.set_axis(pd.DatetimeIndex(index.to_list(), name="my_index"))
    data_no_freq_copy = data_no_freq.copy()

    results = reshape_series_wide_to_long(data=data_no_freq)
    expected = reshape_series_wide_to_long(data=data)

    pd.testing.assert_frame_equal(results, expected)
    assert len(results) == 60
    assert results["value"].notna().all()
    pd.testing.assert_frame_equal(data_no_freq, data_no_freq_copy)
    assert data_no_freq.index.name == "my_index"
    assert data.index.name == "my_index"


@pytest.mark.parametrize(
    "order", ["descending", "shuffled"], ids=lambda order: f"order: {order}"
)
def test_reshape_series_wide_to_long_output_when_index_has_no_freq_and_is_not_sorted(order):
    """
    Test the frequency is inferred on the sorted dates when the index has no
    frequency and is not sorted, so the output is the same as with the sorted
    data (the frequency of descending dates is negative).
    """
    index = pd.date_range(start="2020-01-01", periods=30, freq="D")
    values = np.arange(30, dtype=float)
    data = pd.DataFrame({"A": values, "B": values * 2}, index=index)
    if order == "descending":
        positions = np.arange(30)[::-1]
    else:
        positions = np.random.default_rng(123).permutation(30)
    data_not_sorted = data.iloc[positions].set_axis(
        pd.DatetimeIndex(index[positions].to_list())
    )

    results = reshape_series_wide_to_long(data=data_not_sorted)
    expected = reshape_series_wide_to_long(data=data)

    pd.testing.assert_frame_equal(results, expected)


@pytest.mark.parametrize(
    "dates",
    [
        ["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"],
        ["2020-01-06", "2020-01-07", "2020-01-09", "2020-01-10", "2020-01-13"],
        ["2020-01-01", "2020-01-02"],
        ["2020-01-01"],
    ],
    ids=[
        "daily with a missing date",
        "business days with a holiday",
        "2 dates",
        "1 date",
    ],
)
def test_reshape_series_wide_to_long_ValueError_when_index_has_no_freq_and_it_cannot_be_inferred(dates):
    """
    Raise ValueError when the index has no frequency and it cannot be
    inferred: the dates are not regularly spaced (daily data with a missing
    date, business days with a holiday) or there are fewer than 3.
    """
    index = pd.DatetimeIndex(dates)
    values = np.arange(len(dates), dtype=float)
    data = pd.DataFrame({"A": values, "B": values}, index=index)

    err_msg = re.escape(
        "`data` index has no frequency and it cannot be inferred: the dates "
        "are not regularly spaced or there are fewer than 3. Set the frequency "
        "of the data first, for example `data = data.asfreq('D')` for daily "
        "data or `data = data.asfreq('B')` for business days. Missing dates "
        "are added as rows of NaN."
    )
    # NOTE: `$` checks the message ends here: the hint of `asfreq('24h')` is
    # only added when the index is timezone-aware.
    with pytest.raises(ValueError, match=f"^{err_msg}$"):
        reshape_series_wide_to_long(data=data)


def test_reshape_series_wide_to_long_ValueError_and_output_when_index_is_tz_aware_and_freq_cannot_be_inferred():
    """
    Raise ValueError when the index is timezone-aware, has no frequency and it
    cannot be inferred: the message also suggests `asfreq('24h')` for dates in
    fixed UTC steps. Following it, only the missing date is added, also across
    a daylight saving change (2021-03-28 in Europe/Madrid).
    """
    index_utc = pd.DatetimeIndex(
        ["2021-03-25", "2021-03-26", "2021-03-27", "2021-03-29", "2021-03-30"],
        tz="UTC",
    )
    data = pd.DataFrame(
        {"A": [0., 1., 2., 3., 4.], "B": [10., 11., 12., 13., 14.]},
        index=index_utc.tz_convert("Europe/Madrid"),
    )

    err_msg = re.escape(
        "`data` index has no frequency and it cannot be inferred: the dates "
        "are not regularly spaced or there are fewer than 3. Set the frequency "
        "of the data first, for example `data = data.asfreq('D')` for daily "
        "data or `data = data.asfreq('B')` for business days. Missing dates "
        "are added as rows of NaN. If the dates advance in fixed UTC steps "
        "(created in UTC and converted to a local timezone), use "
        "`data.asfreq('24h')` for daily data: `asfreq('D')` moves them to "
        "calendar days."
    )
    with pytest.raises(ValueError, match=f"^{err_msg}$"):
        reshape_series_wide_to_long(data=data)

    results = reshape_series_wide_to_long(data=data.asfreq("24h"))

    expected_index = pd.date_range(
        "2021-03-25", periods=6, freq="24h", tz="UTC"
    ).tz_convert("Europe/Madrid")
    expected_values = {
        "A": [0., 1., 2., np.nan, 3., 4.],
        "B": [10., 11., 12., np.nan, 13., 14.],
    }
    assert len(results) == 12
    for k in ("A", "B"):
        pd.testing.assert_index_equal(
            results.loc[k].index, expected_index, check_names=False
        )
        np.testing.assert_array_equal(
            results.loc[k, "value"].to_numpy(), expected_values[k]
        )


def test_reshape_series_wide_to_long_output_when_freq_is_set_as_the_error_suggests():
    """
    Test the output when the frequency cannot be inferred and it is set with
    `asfreq`, as the error message suggests: business days with a holiday only
    get the row of the holiday with `asfreq('B')`, and 2 dates work with
    `asfreq('D')`.
    """
    index = pd.DatetimeIndex(
        ["2020-01-06", "2020-01-07", "2020-01-09", "2020-01-10", "2020-01-13"]
    )
    data = pd.DataFrame(
        {"A": [0., 1., 2., 3., 4.], "B": [10., 11., 12., 13., 14.]}, index=index
    )
    results = reshape_series_wide_to_long(data=data.asfreq("B"))

    expected = pd.DataFrame(
        {"value": [0., 1., np.nan, 2., 3., 4., 10., 11., np.nan, 12., 13., 14.]},
        index=pd.MultiIndex.from_product(
            [["A", "B"], pd.date_range("2020-01-06", periods=6, freq="B")],
            names=["series_id", "datetime"],
        ),
    )
    pd.testing.assert_frame_equal(results, expected)

    data = pd.DataFrame(
        {"A": [0., 1.], "B": [10., 11.]},
        index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"]),
    )
    results = reshape_series_wide_to_long(data=data.asfreq("D"))

    expected = pd.DataFrame(
        {"value": [0., 1., 10., 11.]},
        index=pd.MultiIndex.from_product(
            [["A", "B"], pd.date_range("2020-01-01", periods=2, freq="D")],
            names=["series_id", "datetime"],
        ),
    )
    pd.testing.assert_frame_equal(results, expected)
