# Unit test create_datetime_features
# ==============================================================================
import pytest
import re
import pandas as pd
import numpy as np
from skforecast.preprocessing import create_datetime_features

# Fixtures
from .fixtures_preprocessing import features_all_onehot

if pd.__version__ < '2.2.0':
    freq_h = "H"
else:
    freq_h = "h"


def test_create_datetime_features_invalid_input_type():
    """
    Test that create_datetime_features raises a ValueError when input is not 
    a pandas DataFrame or Series.
    """
    with pytest.raises(TypeError, match="Input `X` must be a pandas Series or DataFrame"):
        create_datetime_features([1, 2, 3])


def test_create_datetime_features_no_datetime_index():
    """
    Test that create_datetime_features raises a ValueError when input does not 
    have a pandas DatetimeIndex index.
    """
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(TypeError, match="Input `X` must have a pandas DatetimeIndex"):
        create_datetime_features(df, keep_original_columns=False)


def test_create_datetime_features_invalid_encoding():
    """
    Test that create_datetime_features raises a ValueError when encoding is not 
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
        create_datetime_features(df, encoding="invalid encoding")


def test_create_datetime_features_invalid_feature_name():
    """
    Test that create_datetime_features raises a ValueError when a feature name is not valid.
    """
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.date_range(start="1/1/2022", end="1/5/2022", freq="D"),
    )

    err_msg = re.escape(
        "Features {'invalid_feature'} are not supported. Supported features are "
        "['year', 'month', 'week', 'day_of_week', 'day_of_year', 'day_of_month', "
        "'weekend', 'hour', 'minute', 'second', 'quarter']."
    )
    with pytest.raises(ValueError, match=err_msg):
        create_datetime_features(df, features=["invalid_feature"])


def test_create_datetime_features_output_columns_when_cyclical_encoding():
    """
    Test that create_datetime_features returns the expected columns when encoding is 'cyclical'.
    """
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.date_range(start="1/1/2022", end="1/5/2022", freq="D"),
    )

    results = create_datetime_features(df, encoding="cyclical", keep_original_columns=False)
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


def test_create_datetime_features_output_columns_when_onehot_encoding():
    """
    Test that create_datetime_features returns the expected columns when encoding is 'onehot'.
    """
    index = pd.date_range(start="2021-01-01", end="2023-01-01", freq=freq_h)
    df = pd.DataFrame(
            np.random.rand(len(index), 3),
            columns=["col_1", "col_2", "col_3"],
            index=index,
        )

    results = create_datetime_features(df, encoding="onehot", keep_original_columns=False)

    assert all([feature in features_all_onehot for feature in results.columns])
    assert len(results) == len(df)


def test_create_datetime_features_output_columns_when_None_encoding():
    """
    Test that create_datetime_features returns the expected columns when encoding is 'None'.
    """
    index = pd.date_range(start="2021-01-01", end="2023-01-01", freq=freq_h)
    df = pd.DataFrame(
        np.random.rand(len(index), 3),
        columns=["col_1", "col_2", "col_3"],
        index=index,
    )
    results = create_datetime_features(df, encoding=None, keep_original_columns=False)
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


def test_create_datetime_features_output_when_features_year_month_encoding_cyclical():
    """
    Test that create_datetime_features returns the expected columns when features
     is ['year', 'month'] and encoding is 'cyclical'.
    """
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.date_range(start="1/1/2022", end="1/5/2022", freq="D"),
    )

    results = create_datetime_features(
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


def test_create_datetime_features_output_when_features_year_month_encoding_onehot():
    """
    Test that create_datetime_features returns the expected columns when features
    is ['year', 'month', 'weekend'] and encoding is 'onehot'. All predefined
    categories must be present even when only January dates are in the data.
    """
    index = pd.date_range(start="1/1/2022", end="1/5/2022", freq="D")
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=index,
    )

    results = create_datetime_features(
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


def test_create_datetime_features_output_when_features_year_month_encoding_None():
    """
    Test that create_datetime_features returns the expected columns when features
     is ['year', 'month'] and encoding is None.
    """
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.date_range(start="1/1/2022", end="1/5/2022", freq="D"),
    )

    results = create_datetime_features(
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


def test_create_datetime_features_output_when_features_year_month_encoding_cyclical_and_custom_max_values():
    """
    Test that create_datetime_features returns the expected columns when features
    is ['year', 'month'] and encoding is 'cyclical' with custom max values.
    """

    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.DatetimeIndex(
            ["2022-01-31", "2022-02-28", "2022-03-31", "2022-04-30", "2022-05-31"]
        ),
    )

    results = create_datetime_features(
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


def test_create_datetime_features_invalid_features_to_encode():
    """
    Test that create_datetime_features raises ValueError when features_to_encode
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
        create_datetime_features(
            df,
            features=["year", "month"],
            features_to_encode=["month", "invalid_feature"]
        )


def test_create_datetime_features_features_to_encode_cyclical():
    """
    Test that create_datetime_features encodes only features in features_to_encode
    when using cyclical encoding.
    """
    df = pd.DataFrame(
        np.random.rand(2, 1),
        index=pd.DatetimeIndex(["2022-01-31", "2022-02-28"])
    )

    results = create_datetime_features(
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


def test_create_datetime_features_features_to_encode_onehot():
    """
    Test that create_datetime_features encodes only features in features_to_encode
    when using onehot encoding. All 24 hour columns are always generated even
    though only hours 1 and 2 appear in the data.
    """
    df = pd.DataFrame(
        np.random.rand(2, 1),
        index=pd.DatetimeIndex(["2022-01-01 01:00:00", "2022-02-01 02:00:00"])
    )

    results = create_datetime_features(
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


def test_create_datetime_features_features_to_encode_spline():
    """
    Test that create_datetime_features encodes only features in features_to_encode
    when using spline encoding.
    """
    df = pd.DataFrame(
        np.random.rand(2, 1),
        index=pd.DatetimeIndex(["2022-01-01 01:00:00", "2022-02-01 02:00:00"])
    )

    results = create_datetime_features(
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


def test_create_datetime_features_keep_original_columns_True_dataframe():
    """
    Test that create_datetime_features returns original columns when 
    keep_original_columns=True for a DataFrame.
    """
    df = pd.DataFrame(
        {"exog_1": [1, 2], "exog_2": [3, 4]},
        index=pd.DatetimeIndex(["2022-01-01", "2022-02-01"])
    )

    results = create_datetime_features(
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


def test_create_datetime_features_keep_original_columns_True_series():
    """
    Test that create_datetime_features returns original series as a column when 
    keep_original_columns=True for a Series.
    """
    series = pd.Series(
        [1, 2],
        name="target",
        index=pd.DatetimeIndex(["2022-01-01", "2022-02-01"])
    )

    results = create_datetime_features(
        series,
        features=["month"],
        encoding=None,
        keep_original_columns=True
    )

    assert "target" in results.columns
    assert "month" in results.columns
    assert list(results.columns) == ["target", "month"]
    assert results["target"].tolist() == [1, 2]


def test_create_datetime_features_keep_original_columns_True_overlap_error():
    """
    Test that create_datetime_features raises ValueError when keep_original_columns=True
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
        create_datetime_features(
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
        create_datetime_features(
            series,
            features=["month"],
            encoding=None,
            keep_original_columns=True
        )


def test_create_datetime_features_onehot_single_row_generates_all_columns():
    """
    Test that onehot encoding always generates all expected columns even when
    only a single row (or a subset of categories) is present in the input.
    This guards against pd.get_dummies silently dropping unobserved categories,
    which would cause downstream model failures at inference time.
    """
    # A single Tuesday row: day_of_week == 1, should still produce columns 0-6
    index = pd.DatetimeIndex(["2022-01-04"])  # Tuesday
    df = pd.DataFrame({"value": [1.0]}, index=index)

    result = create_datetime_features(
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


def test_create_datetime_features_onehot_all_columns_present_regardless_of_data():
    """
    Test that onehot encoding produces the full set of columns for known-bounded
    features (month, week, day_of_week, day_of_month, day_of_year, hour, minute,
    second, quarter, weekend) regardless of which values appear in the data.
    A January-only dataset should still produce 12 month columns.
    """
    index = pd.date_range(start="2022-01-01", end="2022-01-31", freq="D")
    df = pd.DataFrame({"value": range(len(index))}, index=index)

    result = create_datetime_features(
        df,
        features=["month", "weekend"],
        encoding="onehot",
        keep_original_columns=False,
    )

    month_cols = [c for c in result.columns if c.startswith("month_")]
    assert len(month_cols) == 12
    assert "weekend" in result.columns
    assert result["weekend"].dtype == np.dtype("int64")


def test_create_datetime_features_onehot_year_and_weekend_never_encoded():
    """
    Test that year and weekend are never one-hot encoded when encoding='onehot',
    regardless of whether they appear in features_to_encode.
    """
    index = pd.date_range(start="2022-01-01", periods=7, freq="D")
    df = pd.DataFrame({"value": range(7)}, index=index)

    result = create_datetime_features(
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

