# Unit test _pandas_compat
# ==============================================================================
import warnings
import pytest
import numpy as np
import pandas as pd
from skforecast.utils._pandas_compat import PANDAS_GE_3, concat, astype, infer_freq


def test_PANDAS_GE_3_matches_the_installed_pandas():
    """
    Test that PANDAS_GE_3 reflects the major version of the installed pandas.
    """
    assert PANDAS_GE_3 == (int(pd.__version__.split(".")[0]) >= 3)


@pytest.mark.parametrize(
    "axis",
    [0, 1],
    ids=lambda axis: f"axis: {axis}"
)
def test_concat_output_without_warnings(axis):
    """
    Test that concat gives the same result as pd.concat, along both axes,
    without any warning in pandas 2 or pandas 3.
    """
    df_1 = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]},
        index=pd.RangeIndex(start=0, stop=3)
    )
    df_2 = pd.DataFrame(
        {"c": [10, 20, 30]},
        index=pd.RangeIndex(start=0, stop=3)
    )
    if axis == 0:
        df_2 = df_2.rename(columns={"c": "a"}).astype(float)
        df_2.index = pd.RangeIndex(start=3, stop=6)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        results = concat([df_1, df_2], axis=axis)

    if axis == 0:
        expected = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
                "b": ["x", "y", "z", np.nan, np.nan, np.nan],
            },
            index=pd.RangeIndex(start=0, stop=6)
        )
    else:
        expected = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"], "c": [10, 20, 30]},
            index=pd.RangeIndex(start=0, stop=3)
        )

    pd.testing.assert_frame_equal(results, expected)


def test_astype_output_without_warnings():
    """
    Test that astype casts a DataFrame and a Series as pandas does, without
    any warning in pandas 2 or pandas 3.
    """
    df = pd.DataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.5, 2.5, 3.5]},
        index=pd.RangeIndex(start=0, stop=3)
    )
    series = pd.Series([1, 2, 3], name="a")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        results_df = astype(df, {"a": float, "b": "category"})
        results_series = astype(series, "float64")

    expected_df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": pd.Categorical(["x", "y", "z"]),
            "c": [1.5, 2.5, 3.5],
        },
        index=pd.RangeIndex(start=0, stop=3)
    )
    expected_series = pd.Series([1.0, 2.0, 3.0], name="a")

    pd.testing.assert_frame_equal(results_df, expected_df)
    pd.testing.assert_series_equal(results_series, expected_series)


def test_astype_changed_only_casts_only_the_columns_that_change():
    """
    Test that astype with `changed_only=True` returns the same object when no
    column changes, and otherwise casts only the columns whose dtype differs,
    sharing the memory of the untouched ones.
    """
    df = pd.DataFrame(
        {"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]},
        index=pd.RangeIndex(start=0, stop=3)
    )

    results_same = astype(df, {"b": float}, changed_only=True)
    results = astype(df, {"a": float, "b": float}, changed_only=True)

    expected = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": [1.5, 2.5, 3.5]},
        index=pd.RangeIndex(start=0, stop=3)
    )

    assert results_same is df
    pd.testing.assert_frame_equal(results, expected)
    assert np.shares_memory(results["b"].to_numpy(), df["b"].to_numpy())
    assert not np.shares_memory(results["a"].to_numpy(), df["a"].to_numpy())


def test_astype_changed_only_output_when_columns_are_duplicated():
    """
    Test that astype with `changed_only=True` casts every column, as pandas
    does, when the DataFrame has duplicated column names.
    """
    df = pd.DataFrame(
        np.array([[1, 1.5, 2], [2, 2.5, 3], [3, 3.5, 4]]),
        columns=["a", "b", "a"],
        index=pd.RangeIndex(start=0, stop=3)
    )

    results = astype(df, {"a": int, "b": float}, changed_only=True)

    expected = pd.DataFrame(
        {"a": [1, 2, 3], "b": [1.5, 2.5, 3.5], "a_": [2, 3, 4]},
        index=pd.RangeIndex(start=0, stop=3)
    )
    expected.columns = ["a", "b", "a"]

    pd.testing.assert_frame_equal(results, expected)


def test_astype_changed_only_KeyError_when_column_is_missing():
    """
    Test that astype with `changed_only=True` raises a KeyError, as pandas
    does, when a column of `dtypes` is not in the DataFrame.
    """
    df = pd.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(KeyError):
        astype(df, {"a": float, "missing": float}, changed_only=True)


@pytest.mark.parametrize(
    "dates, expected",
    [
        (["2020-01-01", "2020-01-02", "2020-01-03"], pd.offsets.Day()),
        (["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00"], pd.offsets.Hour()),
        (["2020-01-01", "2020-02-01", "2020-03-01"], pd.offsets.MonthBegin()),
        (["2020-01-01", "2020-01-02", "2020-01-05"], None),
    ],
    ids=["daily", "hourly", "month start", "irregular"]
)
def test_infer_freq_output_without_warnings(dates, expected):
    """
    Test that infer_freq returns the frequency as an offset, or None when it
    cannot be inferred, without any warning in pandas 2 or pandas 3.
    """
    index = pd.DatetimeIndex(dates)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        results = infer_freq(index)

    if expected is None:
        assert results is None
    else:
        assert isinstance(results, pd.offsets.BaseOffset)
        assert results == expected


def test_infer_freq_ValueError_when_fewer_than_3_dates():
    """
    Test that infer_freq passes on the ValueError of pandas when the index has
    fewer than 3 dates.
    """
    index = pd.DatetimeIndex(["2020-01-01", "2020-01-02"])

    with pytest.raises(ValueError, match="at least 3"):
        infer_freq(index)
