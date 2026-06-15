# Unit test create_calendar_features
# ==============================================================================
import pytest
import re
import pandas as pd
import numpy as np
from skforecast.preprocessing import create_calendar_features
from skforecast.exceptions import IgnoredArgumentWarning

# Fixtures
from ..tests_preprocessing.fixtures_preprocessing import features_all_onehot

if pd.__version__ < '2.2.0':
    freq_h = "H"
else:
    freq_h = "h"


def test_create_calendar_features_invalid_input_type():
    """
    Test that create_calendar_features raises a TypeError when input is not 
    a pandas DataFrame, Series or DatetimeIndex.
    """
    err_msg = re.escape(
        "Input `X` must be a pandas Series, DataFrame or DatetimeIndex"
    )
    with pytest.raises(TypeError, match=err_msg):
        create_calendar_features([1, 2, 3])


def test_create_calendar_features_no_datetime_index():
    """
    Test that create_calendar_features raises a ValueError when input does not 
    have a pandas DatetimeIndex index.
    """
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(TypeError, match="Input `X` must have a pandas DatetimeIndex"):
        create_calendar_features(df, keep_original_columns=False)


def test_create_calendar_features_empty_datetimeindex_raises():
    """
    Test that create_calendar_features raises a ValueError when an empty
    DatetimeIndex is passed.
    """
    with pytest.raises(ValueError, match="Cannot fit on empty input."):
        create_calendar_features(pd.DatetimeIndex([]))


@pytest.mark.parametrize(
    "X",
    [
        pd.DataFrame(
            np.random.rand(5, 3),
            columns=["col_1", "col_2", "col_3"],
            index=pd.date_range(start="2022-01-01", periods=5, freq="D"),
        ),
        pd.Series(
            np.random.rand(5),
            name="y",
            index=pd.date_range(start="2022-01-01", periods=5, freq="D"),
        ),
        pd.date_range(start="2022-01-01", periods=5, freq="D"),
    ],
    ids=["dataframe", "series", "datetimeindex"],
)
def test_create_calendar_features_output_equivalent_across_input_types(X):
    """
    Test that create_calendar_features returns the same output for a DataFrame,
    a Series and a DatetimeIndex input when keep_original_columns=False.
    """
    results = create_calendar_features(
        X,
        features=["year", "month", "weekend"],
        encoding="cyclical",
        keep_original_columns=False,
    )
    expected = pd.DataFrame(
        {
            "year": {
                pd.Timestamp("2022-01-01 00:00:00"): 2022,
                pd.Timestamp("2022-01-02 00:00:00"): 2022,
                pd.Timestamp("2022-01-03 00:00:00"): 2022,
                pd.Timestamp("2022-01-04 00:00:00"): 2022,
                pd.Timestamp("2022-01-05 00:00:00"): 2022,
            },
            "weekend": {
                pd.Timestamp("2022-01-01 00:00:00"): 1,
                pd.Timestamp("2022-01-02 00:00:00"): 1,
                pd.Timestamp("2022-01-03 00:00:00"): 0,
                pd.Timestamp("2022-01-04 00:00:00"): 0,
                pd.Timestamp("2022-01-05 00:00:00"): 0,
            },
            "month_sin": {
                pd.Timestamp("2022-01-01 00:00:00"): 0.49999999999999994,
                pd.Timestamp("2022-01-02 00:00:00"): 0.49999999999999994,
                pd.Timestamp("2022-01-03 00:00:00"): 0.49999999999999994,
                pd.Timestamp("2022-01-04 00:00:00"): 0.49999999999999994,
                pd.Timestamp("2022-01-05 00:00:00"): 0.49999999999999994,
            },
            "month_cos": {
                pd.Timestamp("2022-01-01 00:00:00"): 0.8660254037844387,
                pd.Timestamp("2022-01-02 00:00:00"): 0.8660254037844387,
                pd.Timestamp("2022-01-03 00:00:00"): 0.8660254037844387,
                pd.Timestamp("2022-01-04 00:00:00"): 0.8660254037844387,
                pd.Timestamp("2022-01-05 00:00:00"): 0.8660254037844387,
            },
        }
    ).asfreq("D").astype(
        {'year': int, 'weekend': int}
    )

    pd.testing.assert_frame_equal(results, expected)


def test_create_calendar_features_datetimeindex_keep_original_columns_ignored():
    """
    Test that keep_original_columns has no effect when X is a DatetimeIndex,
    since there are no original columns to keep. Both True and False produce
    the same output and neither raises.
    """
    index = pd.date_range(start="2022-01-01", periods=3, freq="D")

    results_true = create_calendar_features(
        index, features=["month"], encoding=None, keep_original_columns=True
    )
    results_false = create_calendar_features(
        index, features=["month"], encoding=None, keep_original_columns=False
    )
    expected = pd.DataFrame(
        {"month": [1, 1, 1]}, index=index
    ).astype({"month": int})

    pd.testing.assert_frame_equal(results_true, expected)
    pd.testing.assert_frame_equal(results_false, expected)


def test_create_calendar_features_invalid_encoding():
    """
    Test that create_calendar_features raises a ValueError when encoding is not 
    one of 'cyclical', 'onehot' or None.
    """
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.date_range(start="1/1/2022", end="1/5/2022", freq="D"),
    )

    with pytest.raises(
        ValueError, match="Encoding must be one of 'cyclical', 'onehot', 'spline' or None"
    ):
        create_calendar_features(df, encoding="invalid encoding")


def test_create_calendar_features_invalid_feature_name():
    """
    Test that create_calendar_features raises a ValueError when a feature name is not valid.
    """
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.date_range(start="1/1/2022", end="1/5/2022", freq="D"),
    )

    err_msg = re.escape(
        "Features {'invalid_feature'} are not supported. Supported features are "
        "['year', 'month', 'week', 'day_of_week', 'day_of_month', 'day_of_year', "
        "'weekend', 'hour', 'minute', 'second', 'quarter']."
    )
    with pytest.raises(ValueError, match=err_msg):
        create_calendar_features(df, features=["invalid_feature"])


def test_create_calendar_features_output_columns_when_cyclical_encoding():
    """
    Test that create_calendar_features returns the expected columns when encoding is 'cyclical'.
    """
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.date_range(start="1/1/2022", end="1/5/2022", freq="D"),
    )

    results = create_calendar_features(df, encoding="cyclical", keep_original_columns=False)
    expected_features = [
        "year",
        "weekend",
        "month_sin",
        "month_cos",
        "week_sin",
        "week_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "day_of_month_sin",
        "day_of_month_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "hour_sin",
        "hour_cos",
        "minute_sin",
        "minute_cos",
        "second_sin",
        "second_cos",
    ]

    assert all([feature in results.columns for feature in expected_features])
    assert len(results) == len(df)


def test_create_calendar_features_output_columns_when_onehot_encoding():
    """
    Test that create_calendar_features returns the expected columns when encoding is 'onehot'.
    """
    index = pd.date_range(start="2021-01-01", end="2023-01-01", freq=freq_h)
    df = pd.DataFrame(
            np.random.rand(len(index), 3),
            columns=["col_1", "col_2", "col_3"],
            index=index,
        )

    results = create_calendar_features(df, encoding="onehot", keep_original_columns=False)

    assert list(results.columns) == features_all_onehot
    assert len(results) == len(df)


def test_create_calendar_features_output_columns_when_None_encoding():
    """
    Test that create_calendar_features returns the expected columns when encoding is 'None'.
    """
    index = pd.date_range(start="2021-01-01", end="2023-01-01", freq=freq_h)
    df = pd.DataFrame(
        np.random.rand(len(index), 3),
        columns=["col_1", "col_2", "col_3"],
        index=index,
    )
    results = create_calendar_features(df, encoding=None, keep_original_columns=False)
    expected_features = [
        "year",
        "month",
        "week",
        "day_of_week",
        "day_of_month",
        "day_of_year",
        "weekend",
        "hour",
        "minute",
        "second",
        "quarter",
    ]
    assert all(results.columns == expected_features)
    assert len(results) == len(df)


def test_create_calendar_features_output_when_features_year_month_encoding_cyclical():
    """
    Test that create_calendar_features returns the expected columns when features
     is ['year', 'month'] and encoding is 'cyclical'.
    """
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.date_range(start="1/1/2022", end="1/5/2022", freq="D"),
    )

    results = create_calendar_features(
        df, features=["year", "month", "weekend"], encoding="cyclical", keep_original_columns=False
    )
    expected = pd.DataFrame(
        {
            "year": {
                pd.Timestamp("2022-01-01 00:00:00"): 2022,
                pd.Timestamp("2022-01-02 00:00:00"): 2022,
                pd.Timestamp("2022-01-03 00:00:00"): 2022,
                pd.Timestamp("2022-01-04 00:00:00"): 2022,
                pd.Timestamp("2022-01-05 00:00:00"): 2022,
            },
            "weekend": {
                pd.Timestamp("2022-01-01 00:00:00"): 1,
                pd.Timestamp("2022-01-02 00:00:00"): 1,
                pd.Timestamp("2022-01-03 00:00:00"): 0,
                pd.Timestamp("2022-01-04 00:00:00"): 0,
                pd.Timestamp("2022-01-05 00:00:00"): 0,
            },
            "month_sin": {
                pd.Timestamp("2022-01-01 00:00:00"): 0.49999999999999994,
                pd.Timestamp("2022-01-02 00:00:00"): 0.49999999999999994,
                pd.Timestamp("2022-01-03 00:00:00"): 0.49999999999999994,
                pd.Timestamp("2022-01-04 00:00:00"): 0.49999999999999994,
                pd.Timestamp("2022-01-05 00:00:00"): 0.49999999999999994,
            },
            "month_cos": {
                pd.Timestamp("2022-01-01 00:00:00"): 0.8660254037844387,
                pd.Timestamp("2022-01-02 00:00:00"): 0.8660254037844387,
                pd.Timestamp("2022-01-03 00:00:00"): 0.8660254037844387,
                pd.Timestamp("2022-01-04 00:00:00"): 0.8660254037844387,
                pd.Timestamp("2022-01-05 00:00:00"): 0.8660254037844387,
            },
        }
    ).asfreq("D").astype(
        {'year': int, 'weekend': int}
    )

    pd.testing.assert_frame_equal(results, expected)


def test_create_calendar_features_output_when_features_year_month_encoding_onehot():
    """
    Test that create_calendar_features returns the expected columns when features
    is ['year', 'month', 'weekend'] and encoding is 'onehot'. All predefined
    categories must be present even when only January dates are in the data.
    """
    index = pd.date_range(start="1/1/2022", end="1/5/2022", freq="D")
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=index,
    )

    results = create_calendar_features(
        df, features=["year", "month", "weekend"], encoding="onehot", keep_original_columns=False
    )

    # year: kept as raw integer (unbounded, not one-hot encoded)
    assert "year" in results.columns
    assert results["year"].tolist() == [2022, 2022, 2022, 2022, 2022]

    # month: all 12 columns always generated
    month_cols = [c for c in results.columns if c.startswith("month_")]
    assert len(month_cols) == 12
    assert results["month_1"].tolist() == [1, 1, 1, 1, 1]
    assert all(results[f"month_{m}"].tolist() == [0, 0, 0, 0, 0] for m in range(2, 13))

    # weekend: kept as raw integer (binary, not one-hot encoded)
    assert "weekend" in results.columns
    assert results["weekend"].tolist() == [1, 1, 0, 0, 0]

    assert results.shape == (5, 14)


def test_create_calendar_features_output_when_features_year_month_encoding_None():
    """
    Test that create_calendar_features returns the expected columns when features
     is ['year', 'month'] and encoding is None.
    """
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.date_range(start="1/1/2022", end="1/5/2022", freq="D"),
    )

    results = create_calendar_features(
        df, features=["year", "month", "weekend"], encoding=None, keep_original_columns=False
    )
    expected = pd.DataFrame(
        {
            "year": {
                pd.Timestamp("2022-01-01 00:00:00"): 2022,
                pd.Timestamp("2022-01-02 00:00:00"): 2022,
                pd.Timestamp("2022-01-03 00:00:00"): 2022,
                pd.Timestamp("2022-01-04 00:00:00"): 2022,
                pd.Timestamp("2022-01-05 00:00:00"): 2022,
            },
            "month": {
                pd.Timestamp("2022-01-01 00:00:00"): 1,
                pd.Timestamp("2022-01-02 00:00:00"): 1,
                pd.Timestamp("2022-01-03 00:00:00"): 1,
                pd.Timestamp("2022-01-04 00:00:00"): 1,
                pd.Timestamp("2022-01-05 00:00:00"): 1,
            },
            "weekend": {
                pd.Timestamp("2022-01-01 00:00:00"): 1,
                pd.Timestamp("2022-01-02 00:00:00"): 1,
                pd.Timestamp("2022-01-03 00:00:00"): 0,
                pd.Timestamp("2022-01-04 00:00:00"): 0,
                pd.Timestamp("2022-01-05 00:00:00"): 0,
            },
        }
    ).asfreq("D").astype(
        {'year': int, 'month': int, 'weekend': int}
    )

    pd.testing.assert_frame_equal(results, expected)


def test_create_calendar_features_output_when_features_year_month_encoding_cyclical_and_custom_max_values():
    """
    Test that create_calendar_features returns the expected columns when features
    is ['year', 'month'] and encoding is 'cyclical' with custom max values.
    """

    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.DatetimeIndex(
            ["2022-01-31", "2022-02-28", "2022-03-31", "2022-04-30", "2022-05-31"]
        ),
    )

    results = create_calendar_features(
        df,
        features=["year", "month", "weekend"],
        encoding="cyclical",
        max_values={"month": 6},
        keep_original_columns=False
    )

    expected = pd.DataFrame(
        {
            "year": {
                pd.Timestamp("2022-01-31 00:00:00"): 2022,
                pd.Timestamp("2022-02-28 00:00:00"): 2022,
                pd.Timestamp("2022-03-31 00:00:00"): 2022,
                pd.Timestamp("2022-04-30 00:00:00"): 2022,
                pd.Timestamp("2022-05-31 00:00:00"): 2022,
            },
            "weekend": {
                pd.Timestamp("2022-01-31 00:00:00"): 0,
                pd.Timestamp("2022-02-28 00:00:00"): 0,
                pd.Timestamp("2022-03-31 00:00:00"): 0,
                pd.Timestamp("2022-04-30 00:00:00"): 1,
                pd.Timestamp("2022-05-31 00:00:00"): 0,
            },
            "month_sin": {
                pd.Timestamp("2022-01-31 00:00:00"): 0.8660254037844386,
                pd.Timestamp("2022-02-28 00:00:00"): 0.8660254037844387,
                pd.Timestamp("2022-03-31 00:00:00"): 1.2246467991473532e-16,
                pd.Timestamp("2022-04-30 00:00:00"): -0.8660254037844384,
                pd.Timestamp("2022-05-31 00:00:00"): -0.8660254037844386,
            },
            "month_cos": {
                pd.Timestamp("2022-01-31 00:00:00"): 0.5000000000000001,
                pd.Timestamp("2022-02-28 00:00:00"): -0.4999999999999998,
                pd.Timestamp("2022-03-31 00:00:00"): -1.0,
                pd.Timestamp("2022-04-30 00:00:00"): -0.5000000000000004,
                pd.Timestamp("2022-05-31 00:00:00"): 0.5000000000000001,
            },
        }
    ).astype(
        {'year': int, 'weekend': int}
    )

    pd.testing.assert_frame_equal(results, expected)


def test_create_calendar_features_invalid_features_to_encode():
    """
    Test that create_calendar_features raises ValueError when features_to_encode
    contains features not present in features list.
    """
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.DatetimeIndex(
            ["2022-01-31", "2022-02-28", "2022-03-31", "2022-04-30", "2022-05-31"]
        ),
    )

    err_msg = re.escape("Features {'invalid_feature'} are not present in `features`.")
    with pytest.raises(ValueError, match=err_msg):
        create_calendar_features(
            df,
            features=["year", "month"],
            features_to_encode=["month", "invalid_feature"]
        )


def test_create_calendar_features_features_to_encode_cyclical():
    """
    Test that create_calendar_features encodes only features in features_to_encode
    when using cyclical encoding.
    """
    df = pd.DataFrame(
        np.random.rand(2, 1),
        index=pd.DatetimeIndex(["2022-01-31", "2022-02-28"])
    )

    results = create_calendar_features(
        df,
        features=["month", "hour"],
        features_to_encode=["hour"],
        encoding="cyclical",
        keep_original_columns=False
    )
    
    expected = pd.DataFrame({
        "month": [1, 2],
        "hour_sin": [0.0, 0.0],
        "hour_cos": [1.0, 1.0]
    }, index=df.index).astype({"month": int})

    pd.testing.assert_frame_equal(results, expected)


def test_create_calendar_features_features_to_encode_onehot():
    """
    Test that create_calendar_features encodes only features in features_to_encode
    when using onehot encoding. All 24 hour columns are always generated even
    though only hours 1 and 2 appear in the data.
    """
    df = pd.DataFrame(
        np.random.rand(2, 1),
        index=pd.DatetimeIndex(["2022-01-01 01:00:00", "2022-02-01 02:00:00"])
    )

    results = create_calendar_features(
        df,
        features=["month", "hour"],
        features_to_encode=["hour"],
        encoding="onehot",
        keep_original_columns=False
    )

    # month is not encoded — kept as raw integer
    assert results["month"].tolist() == [1, 2]

    # All 24 hour columns are generated regardless of which hours appear
    hour_cols = [c for c in results.columns if c.startswith("hour_")]
    assert len(hour_cols) == 24
    assert results["hour_1"].tolist() == [1, 0]
    assert results["hour_2"].tolist() == [0, 1]
    assert results["hour_0"].tolist() == [0, 0]

    assert results.shape == (2, 25)


def test_create_calendar_features_features_to_encode_spline():
    """
    Test that create_calendar_features encodes only features in features_to_encode
    when using spline encoding.
    """
    df = pd.DataFrame(
        np.random.rand(2, 1),
        index=pd.DatetimeIndex(["2022-01-01 01:00:00", "2022-02-01 02:00:00"])
    )

    results = create_calendar_features(
        df,
        features=["month", "hour"],
        features_to_encode=["hour"],
        encoding="spline",
        keep_original_columns=False
    )

    # Since spline output is complex, we verify the presence and absence of columns
    assert "month" in results.columns
    assert "hour" not in results.columns
    
    # Hour splines should exist
    spline_cols = [c for c in results.columns if "hour_sp_" in c or "hour" in c]
    assert len(spline_cols) > 0
    assert len(results.columns) > 1


def test_create_calendar_features_keep_original_columns_True_dataframe():
    """
    Test that create_calendar_features returns original columns when 
    keep_original_columns=True for a DataFrame.
    """
    df = pd.DataFrame(
        {"exog_1": [1, 2], "exog_2": [3, 4]},
        index=pd.DatetimeIndex(["2022-01-01", "2022-02-01"])
    )

    results = create_calendar_features(
        df,
        features=["month"],
        encoding=None,
        keep_original_columns=True
    )

    assert "exog_1" in results.columns
    assert "exog_2" in results.columns
    assert "month" in results.columns
    assert list(results.columns) == ["exog_1", "exog_2", "month"]
    assert results["exog_1"].tolist() == [1, 2]


def test_create_calendar_features_keep_original_columns_True_series():
    """
    Test that create_calendar_features returns original series as a column when 
    keep_original_columns=True for a Series.
    """
    series = pd.Series(
        [1, 2],
        name="target",
        index=pd.DatetimeIndex(["2022-01-01", "2022-02-01"])
    )

    results = create_calendar_features(
        series,
        features=["month"],
        encoding=None,
        keep_original_columns=True
    )

    assert "target" in results.columns
    assert "month" in results.columns
    assert list(results.columns) == ["target", "month"]
    assert results["target"].tolist() == [1, 2]


def test_create_calendar_features_keep_original_columns_True_overlap_error():
    """
    Test that create_calendar_features raises ValueError when keep_original_columns=True
    and there is a column name overlap with extracted features.
    """
    df = pd.DataFrame(
        {"month": [1, 2], "exog_1": [3, 4]},
        index=pd.DatetimeIndex(["2022-01-01", "2022-02-01"])
    )
    
    err_msg = re.escape(
        "The following extracted feature names already exist in the input DataFrame: "
        "['month']. To avoid duplicate columns, rename the original columns or "
        "avoid extracting these features."
    )
    with pytest.raises(ValueError, match=err_msg):
        create_calendar_features(
            df,
            features=["month"],
            encoding=None,
            keep_original_columns=True
        )

    # Test for series
    series = pd.Series(
        [1, 2],
        name="month",
        index=pd.DatetimeIndex(["2022-01-01", "2022-02-01"])
    )
    err_msg_series = re.escape(
        "The following extracted feature names already exist in the input Series: "
        "['month']. To avoid duplicate columns, rename the original Series or "
        "avoid extracting these features."
    )
    with pytest.raises(ValueError, match=err_msg_series):
        create_calendar_features(
            series,
            features=["month"],
            encoding=None,
            keep_original_columns=True
        )


def test_create_calendar_features_onehot_single_row_generates_all_columns():
    """
    Test that onehot encoding always generates all expected columns even when
    only a single row (or a subset of categories) is present in the input.
    This guards against pd.get_dummies silently dropping unobserved categories,
    which would cause downstream model failures at inference time.
    """
    # A single Tuesday row: day_of_week == 1, should still produce columns 0-6
    index = pd.DatetimeIndex(["2022-01-04"])  # Tuesday
    df = pd.DataFrame({"value": [1.0]}, index=index)

    result = create_calendar_features(
        df,
        features=["day_of_week"],
        encoding="onehot",
        keep_original_columns=False,
    )

    expected_cols = [f"day_of_week_{i}" for i in range(7)]
    assert list(result.columns) == expected_cols
    assert result["day_of_week_1"].iloc[0] == 1  # Tuesday
    assert result["day_of_week_0"].iloc[0] == 0  # not Monday
    assert len(result) == 1


def test_create_calendar_features_onehot_all_columns_present_regardless_of_data():
    """
    Test that onehot encoding produces the full set of columns for known-bounded
    features (month, week, day_of_week, day_of_month, day_of_year, hour, minute,
    second, quarter, weekend) regardless of which values appear in the data.
    A January-only dataset should still produce 12 month columns.
    """
    index = pd.date_range(start="2022-01-01", end="2022-01-31", freq="D")
    df = pd.DataFrame({"value": range(len(index))}, index=index)

    result = create_calendar_features(
        df,
        features=["month", "weekend"],
        encoding="onehot",
        keep_original_columns=False,
    )

    month_cols = [c for c in result.columns if c.startswith("month_")]
    assert len(month_cols) == 12
    assert "weekend" in result.columns
    assert result["weekend"].dtype == np.dtype("int64")


def test_create_calendar_features_onehot_year_and_weekend_never_encoded():
    """
    Test that year and weekend are never one-hot encoded when encoding='onehot',
    regardless of whether they appear in features_to_encode.
    """
    index = pd.date_range(start="2022-01-01", periods=7, freq="D")
    df = pd.DataFrame({"value": range(7)}, index=index)

    result = create_calendar_features(
        df,
        features=["year", "weekend", "month"],
        features_to_encode=["year", "weekend", "month"],
        encoding="onehot",
        keep_original_columns=False,
    )

    assert "year" in result.columns
    assert result["year"].dtype == np.dtype("int64")
    assert not any(c.startswith("year_") for c in result.columns)

    assert "weekend" in result.columns
    assert result["weekend"].dtype == np.dtype("int64")
    assert "weekend_0" not in result.columns
    assert "weekend_1" not in result.columns

    month_cols = [c for c in result.columns if c.startswith("month_")]
    assert len(month_cols) == 12


def test_create_calendar_features_warns_when_features_to_encode_not_encodable():
    """
    Test that IgnoredArgumentWarning is raised when features_to_encode contains
    features that cannot be encoded with the chosen encoding (e.g. 'year' with
    cyclical encoding).
    """
    df = pd.DataFrame(
        {"value": range(7)},
        index=pd.date_range(start="2022-01-01", periods=7, freq="D"),
    )
    with pytest.warns(
        IgnoredArgumentWarning,
        match=r"Features \['year'\] cannot be encoded with encoding='cyclical'",
    ):
        create_calendar_features(
            df,
            features=["year", "month"],
            features_to_encode=["year"],
            encoding="cyclical",
            keep_original_columns=False,
        )


def test_create_calendar_features_max_values_merge_with_defaults():
    """
    Test that user-provided max_values is merged with defaults: keys not in user
    input fall back to defaults, so other cyclical features are still encoded.
    """
    df = pd.DataFrame(
        {"value": range(5)},
        index=pd.date_range(start="2022-01-01", periods=5, freq="h"),
    )
    result = create_calendar_features(
        df,
        features=["month", "hour"],
        encoding="cyclical",
        max_values={"month": 6},  # only month overridden; hour falls back to default 24
        keep_original_columns=False,
    )
    assert "month_sin" in result.columns
    assert "month_cos" in result.columns
    assert "hour_sin" in result.columns
    assert "hour_cos" in result.columns
    # month value at index 0 is 1; with custom period 6, sin(2π·1/6)
    np.testing.assert_allclose(
        result["month_sin"].iloc[0], np.sin(2 * np.pi * 1 / 6)
    )
    # hour value at index 0 is 0; with default period 24, sin(2π·0/24) = 0
    np.testing.assert_allclose(result["hour_sin"].iloc[0], 0.0, atol=1e-12)


def test_create_calendar_features_spline_kwargs_blocked_knots_raises():
    """
    Test that passing 'knots' in spline_kwargs raises ValueError because knots
    are computed internally from max_values.
    """
    df = pd.DataFrame(
        {"value": range(5)},
        index=pd.date_range(start="2022-01-01", periods=5, freq="D"),
    )
    with pytest.raises(
        ValueError,
        match=r"Keys \['knots'\] are not allowed in `spline_kwargs`",
    ):
        create_calendar_features(
            df,
            features=["month"],
            encoding="spline",
            spline_kwargs={"knots": np.array([[1], [2], [3]])},
            keep_original_columns=False,
        )


def test_create_calendar_features_spline_kwargs_blocked_sparse_output_raises():
    """
    Test that passing 'sparse_output' in spline_kwargs raises ValueError because
    it is incompatible with the DataFrame output.
    """
    df = pd.DataFrame(
        {"value": range(5)},
        index=pd.date_range(start="2022-01-01", periods=5, freq="D"),
    )
    with pytest.raises(
        ValueError,
        match=r"Keys \['sparse_output'\] are not allowed in `spline_kwargs`",
    ):
        create_calendar_features(
            df,
            features=["month"],
            encoding="spline",
            spline_kwargs={"sparse_output": True},
            keep_original_columns=False,
        )


def test_create_calendar_features_spline_kwargs_unknown_key_raises():
    """
    Test that passing an unknown key (typo) in spline_kwargs raises ValueError
    listing the allowed keys.
    """
    df = pd.DataFrame(
        {"value": range(5)},
        index=pd.date_range(start="2022-01-01", periods=5, freq="D"),
    )
    with pytest.raises(
        ValueError,
        match=r"Unknown keys in `spline_kwargs`: \['degrees'\]",
    ):
        create_calendar_features(
            df,
            features=["month"],
            encoding="spline",
            spline_kwargs={"degrees": 3},  # typo: should be 'degree'
            keep_original_columns=False,
        )


def test_create_calendar_features_spline_kwargs_extrapolation_forwarded():
    """
    Test that 'extrapolation' is actually forwarded to SplineTransformer.
    Passing an invalid value should raise from sklearn — proving the kwarg is
    no longer silently dropped (was hardcoded to 'periodic' before the fix).
    """
    df = pd.DataFrame(
        {"value": range(5)},
        index=pd.date_range(start="2022-01-01", periods=5, freq="D"),
    )
    with pytest.raises(ValueError):
        create_calendar_features(
            df,
            features=["month"],
            encoding="spline",
            spline_kwargs={"extrapolation": "not_a_valid_extrapolation_mode"},
            keep_original_columns=False,
        )


def test_create_calendar_features_unnamed_series_keep_original_raises():
    """
    Test that passing an unnamed Series with keep_original_columns=True raises
    ValueError, since pd.concat would otherwise produce a column literally named '0'.
    """
    series = pd.Series(
        [1, 2, 3],
        index=pd.date_range(start="2022-01-01", periods=3, freq="D"),
    )
    assert series.name is None
    with pytest.raises(
        ValueError,
        match=r"the input Series must have a name",
    ):
        create_calendar_features(series, keep_original_columns=True)


def test_create_calendar_features_unnamed_series_keep_original_false_ok():
    """
    Test that passing an unnamed Series with keep_original_columns=False works
    without error.
    """
    series = pd.Series(
        [1, 2, 3],
        index=pd.date_range(start="2022-01-01", periods=3, freq="D"),
    )
    assert series.name is None
    result = create_calendar_features(
        series, features=["month"], encoding=None, keep_original_columns=False
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "month" in result.columns


def test_create_calendar_features_onehot_week_53_and_day_of_year_366():
    """
    Test that 2020-12-31 (ISO week 53, day-of-year 366 in leap year 2020)
    produces a 1 in the week_53 and day_of_year_366 onehot columns and 0 in
    the rest, instead of silently producing all-zero rows as before the fix.
    """
    df = pd.DataFrame(
        {"value": [0]},
        index=pd.DatetimeIndex(["2020-12-31"]),
    )
    result = create_calendar_features(
        df,
        features=["week", "day_of_year"],
        encoding="onehot",
        keep_original_columns=False,
    )
    assert "week_53" in result.columns
    assert "day_of_year_366" in result.columns
    assert result["week_53"].iloc[0] == 1
    assert result["day_of_year_366"].iloc[0] == 1
    week_cols = [c for c in result.columns if c.startswith("week_")]
    doy_cols = [c for c in result.columns if c.startswith("day_of_year_")]
    assert sum(result[c].iloc[0] for c in week_cols) == 1
    assert sum(result[c].iloc[0] for c in doy_cols) == 1


def test_create_calendar_features_cyclical_uses_period_53_and_366():
    """
    Test that cyclical encoding for week and day_of_year uses periods 53 and
    366 respectively. For value=period, sin(2π) must be 0 and cos(2π) must
    be 1 (one full cycle). With the previous periods 52/365, the boundary
    values (week=53, day_of_year=366) would not satisfy this — sin would be
    ~0.12 and ~0.017 respectively.
    """
    df = pd.DataFrame(
        {"value": [0]},
        index=pd.DatetimeIndex(["2020-12-31"]),
    )
    result = create_calendar_features(
        df,
        features=["week", "day_of_year"],
        encoding="cyclical",
        keep_original_columns=False,
    )
    # 2020-12-31 has week=53 and day_of_year=366
    np.testing.assert_allclose(result["week_sin"].iloc[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(result["week_cos"].iloc[0], 1.0, atol=1e-12)
    np.testing.assert_allclose(result["day_of_year_sin"].iloc[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(result["day_of_year_cos"].iloc[0], 1.0, atol=1e-12)


def test_create_calendar_features_onehot_non_leap_year_keeps_max_columns_zero():
    """
    Test that in a non-leap year without ISO week 53 (2022), the columns
    week_53 and day_of_year_366 still exist in the output (categorical
    consistency for train/predict) but remain 0 throughout the year.
    """
    df = pd.DataFrame(
        {"value": range(365)},
        index=pd.date_range(start="2022-01-01", periods=365, freq="D"),
    )
    result = create_calendar_features(
        df,
        features=["week", "day_of_year"],
        encoding="onehot",
        keep_original_columns=False,
    )
    assert "week_53" in result.columns
    assert "day_of_year_366" in result.columns
    assert (result["week_53"] == 0).all()
    assert (result["day_of_year_366"] == 0).all()


def test_create_calendar_features_spline_month_12_distinct_from_month_1():
    """
    Regression test for the knot-placement bug. With the old formula
    `linspace(min_val, max_val, n_knots)` the periodic spline period for
    `month` was 11 instead of 12, causing month 12 (December) to map to the
    same coordinates as month 1 (January). Verify that with the corrected
    knot range `[1, 13]`, December and January produce distinct outputs.
    """
    df = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2022-01-15", "2022-12-15"]),
    )
    result = create_calendar_features(
        df, features=["month"], encoding="spline", keep_original_columns=False
    )
    jan = result.iloc[0].to_numpy()
    dec = result.iloc[1].to_numpy()
    assert not np.allclose(jan, dec, atol=1e-6), (
        "December and January spline encodings collapsed to the same value, "
        "indicating the periodic knot placement is wrong."
    )


def test_create_calendar_features_spline_week_53_continuity():
    """
    Verify that with the corrected knot placement, week 53 sits between
    week 52 and week 1 in spline space — the cyclical neighborhood is
    preserved at the year boundary. Both distances must be strictly
    positive (week 53 is not collapsed onto its neighbors) and equal
    (week 53 is one knot step away from each).
    """
    # 2020 contains ISO week 52 (2020-12-21 Mon), week 53 (2020-12-28 Mon),
    # and week 1 of 2021 (2021-01-04 Mon).
    df = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2020-12-21", "2020-12-28", "2021-01-04"]),
    )
    result = create_calendar_features(
        df, features=["week"], encoding="spline", keep_original_columns=False
    )
    week_52 = result.iloc[0].to_numpy()
    week_53 = result.iloc[1].to_numpy()
    week_1 = result.iloc[2].to_numpy()

    dist_53_to_52 = np.linalg.norm(week_53 - week_52)
    dist_53_to_1 = np.linalg.norm(week_53 - week_1)
    dist_52_to_1 = np.linalg.norm(week_52 - week_1)

    # Strictly positive — week 53 is its own point.
    assert dist_53_to_52 > 0
    assert dist_53_to_1 > 0
    # Equidistant — week 53 is one knot step from each of week 52 and week 1.
    np.testing.assert_allclose(dist_53_to_52, dist_53_to_1, atol=1e-6)
    # Going through week 53 (52 -> 53 -> 1) is shorter than the direct chord
    # 52 -> 1 only as inequality — not equality — because spline space is not
    # exactly the unit circle. But the two single-step distances must be
    # equal AND less than the two-step direct chord (this also rules out
    # the bug where week 53 collapsed onto week 1 making dist_53_to_1 ≈ 0).
    assert dist_53_to_52 < dist_52_to_1


def test_create_calendar_features_spline_day_of_year_366_continuity():
    """
    Verify that with the corrected knot placement, day_of_year 366 sits
    between day 365 and day 1 in spline space — analogous to the week 53
    test, but for the leap-year boundary.
    """
    # 2020 is a leap year: day 365 = 2020-12-30, day 366 = 2020-12-31,
    # day 1 of 2021 = 2021-01-01.
    df = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2020-12-30", "2020-12-31", "2021-01-01"]),
    )
    result = create_calendar_features(
        df,
        features=["day_of_year"],
        encoding="spline",
        keep_original_columns=False,
    )
    day_365 = result.iloc[0].to_numpy()
    day_366 = result.iloc[1].to_numpy()
    day_1 = result.iloc[2].to_numpy()

    dist_366_to_365 = np.linalg.norm(day_366 - day_365)
    dist_366_to_1 = np.linalg.norm(day_366 - day_1)

    # Strictly positive — day 366 is its own point, not collapsed.
    assert dist_366_to_365 > 0
    assert dist_366_to_1 > 0
    # Equidistant — day 366 is one knot step from day 365 and one from day 1.
    np.testing.assert_allclose(dist_366_to_365, dist_366_to_1, atol=1e-6)


def test_create_calendar_features_max_values_unknown_key_warns():
    """
    Test that unknown keys in `max_values` (typos like 'mnth' instead of
    'month') trigger an `IgnoredArgumentWarning` listing valid keys, are
    silently dropped, and the rest of the encoding proceeds with defaults.
    """
    df = pd.DataFrame(
        {"value": [1, 2, 3]},
        index=pd.date_range("2022-01-01", periods=3, freq="D"),
    )
    with pytest.warns(
        IgnoredArgumentWarning,
        match=r"Unknown keys in `max_values`: \['mnth'\]",
    ):
        result = create_calendar_features(
            df,
            features=["month"],
            encoding="cyclical",
            max_values={"month": 12, "mnth": 12},  # 'mnth' is a typo
            keep_original_columns=False,
        )
    # `month` is still encoded with the (default-equal-to-passed) value 12.
    assert "month_sin" in result.columns
    assert "month_cos" in result.columns


def test_create_calendar_features_cyclical_respects_features_order():
    """
    Output cyclical-encoded columns must appear in the order given by
    `features`, not in the internal `_DEFAULT_MAX_VALUES` insertion order.
    `features=['hour', 'month']` reverses `_DEFAULT_MAX_VALUES` order
    (where month precedes hour), so the output must lead with `hour`
    sin/cos columns.
    """
    df = pd.DataFrame(
        {"value": [0, 1, 2]},
        index=pd.date_range("2022-01-01", periods=3, freq="h"),
    )
    result = create_calendar_features(
        df,
        features=["hour", "month"],
        encoding="cyclical",
        keep_original_columns=False,
    )
    assert list(result.columns) == [
        "hour_sin", "hour_cos", "month_sin", "month_cos"
    ]


def test_create_calendar_features_onehot_respects_features_order():
    """
    Output onehot dummy columns must appear grouped by feature in the order
    given by `features`. `features=['quarter', 'month']` puts quarter
    (last in `_DEFAULT_MAX_VALUES`) before month (first), so quarter
    dummies must precede month dummies in the output.
    """
    df = pd.DataFrame(
        {"value": [0, 1, 2]},
        index=pd.date_range("2022-01-01", periods=3, freq="h"),
    )
    result = create_calendar_features(
        df,
        features=["quarter", "month"],
        encoding="onehot",
        keep_original_columns=False,
    )
    cols = list(result.columns)
    quarter_cols = [c for c in cols if c.startswith("quarter_")]
    month_cols = [c for c in cols if c.startswith("month_")]
    assert quarter_cols and month_cols
    quarter_last = max(cols.index(c) for c in quarter_cols)
    month_first = min(cols.index(c) for c in month_cols)
    assert quarter_last < month_first


def test_create_calendar_features_spline_respects_features_order():
    """
    Output spline-encoded columns must appear grouped by feature in the
    order given by `features`. `features=['hour', 'month']` reverses the
    internal default order, so hour_sp_* columns must precede month_sp_*.
    """
    df = pd.DataFrame(
        {"value": [0, 1, 2]},
        index=pd.date_range("2022-01-01", periods=3, freq="h"),
    )
    result = create_calendar_features(
        df,
        features=["hour", "month"],
        encoding="spline",
        keep_original_columns=False,
    )
    cols = list(result.columns)
    hour_cols = [c for c in cols if c.startswith("hour_sp_")]
    month_cols = [c for c in cols if c.startswith("month_sp_")]
    assert hour_cols and month_cols
    hour_last = max(cols.index(c) for c in hour_cols)
    month_first = min(cols.index(c) for c in month_cols)
    assert hour_last < month_first


def test_create_calendar_features_onehot_non_encoded_appear_before_dummies():
    """
    With encoding='onehot' and a mix of encodable and non-encodable features,
    non-encoded features (year, weekend) must appear first in `features`
    order, and dummies must follow grouped per encoded feature, also in
    `features` order. This matches the cyclical / spline output layout.
    """
    df = pd.DataFrame(
        {"value": [0, 1, 2]},
        index=pd.date_range("2022-01-01", periods=3, freq="h"),
    )
    result = create_calendar_features(
        df,
        features=["month", "year", "hour", "weekend"],
        encoding="onehot",
        keep_original_columns=False,
    )
    cols = list(result.columns)
    expected_prefix = ["year", "weekend"]
    assert cols[: len(expected_prefix)] == expected_prefix
    month_cols = [c for c in cols if c.startswith("month_")]
    hour_cols = [c for c in cols if c.startswith("hour_")]
    assert len(month_cols) == 12
    assert len(hour_cols) == 24
    month_last = max(cols.index(c) for c in month_cols)
    hour_first = min(cols.index(c) for c in hour_cols)
    assert month_last < hour_first


def test_create_calendar_features_warns_when_max_values_with_onehot():
    """
    Test that passing `max_values` together with `encoding='onehot'` triggers
    an IgnoredArgumentWarning, since onehot uses the fixed known-category set
    and ignores `max_values`.
    """
    idx = pd.date_range("2022-01-01", periods=3, freq="D")
    s = pd.Series([1, 2, 3], index=idx, name="y")

    with pytest.warns(IgnoredArgumentWarning, match=r"max_values.*onehot"):
        result = create_calendar_features(
            s,
            features=["month"],
            encoding="onehot",
            max_values={"month": 6},
            keep_original_columns=False,
        )

    # Onehot ignores max_values: output still has 12 month columns from the
    # fixed known-category set.
    month_cols = [c for c in result.columns if c.startswith("month_")]
    assert len(month_cols) == 12


def test_create_calendar_features_no_onehot_warning_with_cyclical_encoding():
    """
    Test that `max_values` with `encoding='cyclical'` does NOT trigger the
    onehot-specific IgnoredArgumentWarning.
    """
    import warnings as _warnings

    idx = pd.date_range("2022-01-01", periods=3, freq="D")
    s = pd.Series([1, 2, 3], index=idx, name="y")

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", IgnoredArgumentWarning)
        create_calendar_features(
            s,
            features=["month"],
            encoding="cyclical",
            max_values={"month": 6},
            keep_original_columns=False,
        )

