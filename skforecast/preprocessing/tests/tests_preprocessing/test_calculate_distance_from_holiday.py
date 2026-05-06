# Unit test calculate_distance_from_holiday
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.preprocessing import calculate_distance_from_holiday
from skforecast.exceptions import IgnoredArgumentWarning

if pd.__version__ < "2.2.0":
    freq_h = "H"
else:
    freq_h = "h"


def test_calculate_distance_from_holiday_daily_index():
    """
    Test output column names, values, and dtype for a daily DatetimeIndex.
    Holidays on day 0 and day 3; expected distances verified manually.
    """
    idx = pd.date_range("2022-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {"value": [1, 2, 3, 4, 5], "is_holiday": [True, False, False, True, False]},
        index=idx,
    )

    result = calculate_distance_from_holiday(df, holiday_column="is_holiday")

    assert list(result.columns) == ["time_to_holiday", "time_since_holiday"]
    assert result.index.equals(df.index)
    np.testing.assert_array_equal(
        result["time_to_holiday"].to_numpy(dtype=float),
        [0, 2, 1, 0, 0],  # last 0 is fill_na (no next holiday after day 3)
    )
    np.testing.assert_array_equal(
        result["time_since_holiday"].to_numpy(dtype=float),
        [0, 1, 2, 0, 1],
    )


def test_calculate_distance_from_holiday_hourly_index():
    """
    Test that hourly data produces `time_*` column names with correct hour values.
    """
    idx = pd.date_range("2022-01-01", periods=4, freq=freq_h)
    df = pd.DataFrame(
        {"is_holiday": [True, False, True, False]},
        index=idx,
    )

    result = calculate_distance_from_holiday(df, holiday_column="is_holiday")

    assert list(result.columns) == ["time_to_holiday", "time_since_holiday"]
    np.testing.assert_array_equal(
        result["time_to_holiday"].to_numpy(dtype=float),
        [0, 1, 0, 0],  # last 0 is fill_na
    )
    np.testing.assert_array_equal(
        result["time_since_holiday"].to_numpy(dtype=float),
        [0, 1, 0, 1],
    )


def test_calculate_distance_from_holiday_date_column():
    """
    Test the `date_column` string path: dates come from a DataFrame column
    and the output unit is always days.
    """
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03"]),
            "is_holiday": [True, False, False],
        }
    )

    result = calculate_distance_from_holiday(
        df, holiday_column="is_holiday", date_column="date"
    )

    assert list(result.columns) == ["time_to_holiday", "time_since_holiday"]
    np.testing.assert_array_equal(
        result["time_to_holiday"].to_numpy(dtype=float),
        [0, 0, 0],  # no next holiday after day 0, so fill_na=0 for rows 1 and 2
    )
    np.testing.assert_array_equal(
        result["time_since_holiday"].to_numpy(dtype=float),
        [0, 1, 2],
    )


def test_calculate_distance_from_holiday_no_holidays():
    """
    Test that when there are no holidays, all output values equal `fill_na`.
    """
    idx = pd.date_range("2022-01-01", periods=3, freq="D")
    df = pd.DataFrame({"is_holiday": [False, False, False]}, index=idx)

    result = calculate_distance_from_holiday(df, holiday_column="is_holiday", fill_na=-1)

    np.testing.assert_array_equal(
        result["time_to_holiday"].to_numpy(dtype=float), [-1, -1, -1]
    )
    np.testing.assert_array_equal(
        result["time_since_holiday"].to_numpy(dtype=float), [-1, -1, -1]
    )


def test_calculate_distance_from_holiday_fill_na():
    """
    Test that `fill_na` controls the value at positions with no prior/next holiday.
    The first row has no prior holiday and the last has no next holiday.
    """
    idx = pd.date_range("2022-01-01", periods=3, freq="D")
    df = pd.DataFrame({"is_holiday": [False, True, False]}, index=idx)

    result = calculate_distance_from_holiday(df, holiday_column="is_holiday", fill_na=99)

    np.testing.assert_array_equal(
        result["time_to_holiday"].to_numpy(dtype=float), [1, 0, 99]
    )
    np.testing.assert_array_equal(
        result["time_since_holiday"].to_numpy(dtype=float), [99, 0, 1]
    )


def test_calculate_distance_from_holiday_no_freq_warns_and_falls_back_to_hours():
    """
    Test that when the index has no frequency and it cannot be inferred (irregular
    spacing), a UserWarning is raised and the output columns are named `time_*`.
    """
    idx = pd.DatetimeIndex(["2022-01-01", "2022-01-03", "2022-01-07"])
    df = pd.DataFrame({"is_holiday": [True, False, False]}, index=idx)

    with pytest.warns(UserWarning, match="Could not determine the frequency"):
        result = calculate_distance_from_holiday(df, holiday_column="is_holiday")

    assert list(result.columns) == ["time_to_holiday", "time_since_holiday"]


def test_calculate_distance_from_holiday_TypeError_when_invalid_index_type():
    """
    Test that a TypeError is raised when `date_column=None` and the index is not
    a pandas DatetimeIndex.
    """
    df = pd.DataFrame({"is_holiday": [True, False]}, index=[0, 1])

    with pytest.raises(
        TypeError,
        match="When `date_column=None`, the index must be a pandas DatetimeIndex",
    ):
        calculate_distance_from_holiday(df, holiday_column="is_holiday")


def test_calculate_distance_from_holiday_does_not_mutate_input():
    """
    Test that the original DataFrame is not modified by the function.
    """
    idx = pd.date_range("2022-01-01", periods=3, freq="D")
    df = pd.DataFrame({"is_holiday": [True, False, False]}, index=idx)
    original_columns = list(df.columns)
    original_values = df["is_holiday"].tolist()

    calculate_distance_from_holiday(df, holiday_column="is_holiday")

    assert list(df.columns) == original_columns
    assert df["is_holiday"].tolist() == original_values


def test_calculate_distance_from_holiday_output_shape():
    """
    Test that the result has exactly 2 columns and the same index as the input.
    """
    idx = pd.date_range("2022-01-01", periods=5, freq="D")
    df = pd.DataFrame({"is_holiday": [True, False, False, True, False]}, index=idx)

    result = calculate_distance_from_holiday(df, holiday_column="is_holiday")

    assert result.shape == (5, 2)
    assert result.index.equals(df.index)


def test_calculate_distance_from_holiday_TypeError_when_invalid_X_type():
    """
    Test that a TypeError is raised when `X` is not a pandas Series or DataFrame.
    """
    with pytest.raises(
        TypeError,
        match="Input `X` must be a pandas Series or pandas DataFrame",
    ):
        calculate_distance_from_holiday([True, False, True], holiday_column="is_holiday")


def test_calculate_distance_from_holiday_ValueError_when_dataframe_without_holiday_column():
    """
    Test that a ValueError is raised when `X` is a DataFrame and `holiday_column`
    is not specified.
    """
    idx = pd.date_range("2022-01-01", periods=3, freq="D")
    df = pd.DataFrame({"is_holiday": [True, False, False]}, index=idx)

    with pytest.raises(
        ValueError,
        match="`holiday_column` must be specified when `X` is a pandas DataFrame",
    ):
        calculate_distance_from_holiday(df)


def test_calculate_distance_from_holiday_series_input():
    """
    Test that a boolean pandas Series is accepted directly as the holiday indicator.
    Output values are identical to passing the same data as a DataFrame column.
    """
    idx = pd.date_range("2022-01-01", periods=5, freq="D")
    s = pd.Series(
        [True, False, False, True, False], index=idx, name="is_holiday"
    )

    result = calculate_distance_from_holiday(s)

    assert list(result.columns) == ["time_to_holiday", "time_since_holiday"]
    assert result.index.equals(idx)
    np.testing.assert_array_equal(
        result["time_to_holiday"].to_numpy(dtype=float),
        [0, 2, 1, 0, 0],
    )
    np.testing.assert_array_equal(
        result["time_since_holiday"].to_numpy(dtype=float),
        [0, 1, 2, 0, 1],
    )


def test_calculate_distance_from_holiday_series_ignores_holiday_column_with_warning():
    """
    Test that passing `holiday_column` alongside a Series input issues an
    IgnoredArgumentWarning and still produces correct output.
    """
    idx = pd.date_range("2022-01-01", periods=3, freq="D")
    s = pd.Series([True, False, False], index=idx, name="is_holiday")

    with pytest.warns(IgnoredArgumentWarning, match="`holiday_column` is ignored"):
        result = calculate_distance_from_holiday(s, holiday_column="some_column")

    assert list(result.columns) == ["time_to_holiday", "time_since_holiday"]
    np.testing.assert_array_equal(
        result["time_since_holiday"].to_numpy(dtype=float), [0, 1, 2]
    )


def test_calculate_distance_from_holiday_series_unnamed():
    """
    Test that an unnamed Series (name=None) is accepted and the function uses
    'is_holiday' as the internal column name without error.
    """
    idx = pd.date_range("2022-01-01", periods=3, freq="D")
    s = pd.Series([True, False, False], index=idx)  # name=None

    result = calculate_distance_from_holiday(s)

    assert list(result.columns) == ["time_to_holiday", "time_since_holiday"]
    np.testing.assert_array_equal(
        result["time_to_holiday"].to_numpy(dtype=float), [0, 0, 0]
    )
    np.testing.assert_array_equal(
        result["time_since_holiday"].to_numpy(dtype=float), [0, 1, 2]
    )


def test_calculate_distance_from_holiday_fill_na_invalid_float_raises():
    """
    Test that a non-NaN float `fill_na` (e.g. 0.5) raises TypeError because
    the output columns have Int64 dtype.
    """
    idx = pd.date_range("2022-01-01", periods=3, freq="D")
    df = pd.DataFrame({"is_holiday": [False, True, False]}, index=idx)
    with pytest.raises(TypeError, match=r"`fill_na` must be an int, np\.integer, or numpy\.nan"):
        calculate_distance_from_holiday(df, holiday_column="is_holiday", fill_na=0.5)


def test_calculate_distance_from_holiday_fill_na_bool_raises():
    """
    Test that a boolean `fill_na` is rejected even though `bool` is a
    subclass of `int` in Python — bools should not be implicitly accepted
    as integer fill values.
    """
    idx = pd.date_range("2022-01-01", periods=3, freq="D")
    df = pd.DataFrame({"is_holiday": [False, True, False]}, index=idx)
    with pytest.raises(TypeError, match=r"`fill_na` must be an int, np\.integer, or numpy\.nan"):
        calculate_distance_from_holiday(df, holiday_column="is_holiday", fill_na=True)


def test_calculate_distance_from_holiday_fill_na_nan_keeps_pd_NA():
    """
    Test that `fill_na=numpy.nan` is accepted and preserves missing entries
    as `pd.NA` in the Int64 output (not coerced to 0 or any int).
    """
    idx = pd.date_range("2022-01-01", periods=3, freq="D")
    df = pd.DataFrame({"is_holiday": [False, False, True]}, index=idx)
    result = calculate_distance_from_holiday(
        df, holiday_column="is_holiday", fill_na=np.nan
    )
    # The first two rows have no prior holiday → time_since_holiday is pd.NA
    assert result["time_since_holiday"].iloc[0] is pd.NA
    assert result["time_since_holiday"].iloc[1] is pd.NA
    assert result["time_since_holiday"].iloc[2] == 0
    assert result["time_since_holiday"].dtype == "Int64"


def test_calculate_distance_from_holiday_invalid_holiday_column_raises():
    """
    Test that passing a `holiday_column` name that does not exist in `X`
    raises ValueError with a clear message listing available columns.
    """
    idx = pd.date_range("2022-01-01", periods=3, freq="D")
    df = pd.DataFrame({"is_holiday": [False, True, False]}, index=idx)
    with pytest.raises(
        ValueError,
        match=r"`holiday_column='isHoliday'` is not a column of `X`",
    ):
        calculate_distance_from_holiday(df, holiday_column="isHoliday")  # typo


def test_calculate_distance_from_holiday_invalid_date_column_raises():
    """
    Test that passing a `date_column` name that does not exist in `X`
    raises ValueError with a clear message listing available columns.
    """
    df = pd.DataFrame(
        {"is_holiday": [False, True, False], "date": pd.date_range("2022-01-01", periods=3)}
    )
    with pytest.raises(
        ValueError,
        match=r"`date_column='dt'` is not a column of `X`",
    ):
        calculate_distance_from_holiday(
            df, holiday_column="is_holiday", date_column="dt"  # typo
        )


def test_calculate_distance_from_holiday_fill_na_np_floating_nan_works():
    """
    Test that numpy floating NaN subclasses (e.g. `np.float32(np.nan)`,
    `np.float64(np.nan)`) are accepted as `fill_na` values, not just plain
    Python `float` NaN.
    """
    idx = pd.date_range("2022-01-01", periods=3, freq="D")
    df = pd.DataFrame({"is_holiday": [False, False, True]}, index=idx)

    for fill_na_value in (np.float32(np.nan), np.float64(np.nan)):
        result = calculate_distance_from_holiday(
            df, holiday_column="is_holiday", fill_na=fill_na_value
        )
        # The first two rows have no prior holiday → time_since_holiday is pd.NA
        assert result["time_since_holiday"].iloc[0] is pd.NA
        assert result["time_since_holiday"].iloc[1] is pd.NA
        assert result["time_since_holiday"].iloc[2] == 0


def test_calculate_distance_from_holiday_warns_and_fills_nan_in_dataframe():
    """
    Test that NaN values in the holiday column trigger a UserWarning, are
    filled with False, the original DataFrame is not mutated, and the output
    matches what would be obtained by pre-filling NaN with False.
    """
    idx = pd.date_range("2022-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {"is_holiday": [True, False, np.nan, True, False]}, index=idx
    )

    with pytest.warns(UserWarning, match="contains NaN values"):
        result = calculate_distance_from_holiday(df, holiday_column="is_holiday")

    df_filled = df.copy()
    with pd.option_context("future.no_silent_downcasting", True):
        df_filled["is_holiday"] = df_filled["is_holiday"].fillna(False).astype(bool)
    expected = calculate_distance_from_holiday(df_filled, holiday_column="is_holiday")
    pd.testing.assert_frame_equal(result, expected)

    # Original input must not be mutated
    assert df["is_holiday"].isna().sum() == 1


def test_calculate_distance_from_holiday_warns_and_fills_nan_in_series():
    """
    Test that NaN values in a Series input trigger a UserWarning and are
    filled with False before computing distances.
    """
    idx = pd.date_range("2022-01-01", periods=4, freq="D")
    s = pd.Series([True, np.nan, False, True], index=idx, name="is_holiday")

    with pytest.warns(UserWarning, match="contains NaN values"):
        result = calculate_distance_from_holiday(s)

    assert list(result.columns) == ["time_to_holiday", "time_since_holiday"]
    # NaN at position 1 → filled with False; nearest holidays at positions 0 and 3
    assert result["time_to_holiday"].iloc[1] == 2
    assert result["time_since_holiday"].iloc[1] == 1
