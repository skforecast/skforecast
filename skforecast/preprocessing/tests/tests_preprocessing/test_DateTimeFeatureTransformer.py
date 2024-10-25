# Unit test DateTimeFeatureTransformer
# ==============================================================================
import pytest
import re
import pandas as pd
import numpy as np
from skforecast.preprocessing import DateTimeFeatureTransformer

# Fixtures
from .fixtures_preprocessing import features_all_onehot

if pd.__version__ < '2.2.0':
    freq_h = "H"
else:
    freq_h = "h"


def test_create_datetime_features_invalid_input_type():
    """
    Test that DateTimeFeatureTransformer raises a ValueError when input is not 
    a pandas DataFrame or Series.
    """
    with pytest.raises(TypeError, match="Input `X` must be a pandas Series or DataFrame"):
        DateTimeFeatureTransformer().fit_transform([1, 2, 3])


def test_create_datetime_features_no_datetime_index():
    """
    Test that DateTimeFeatureTransformer raises a ValueError when input does not 
    have a pandas DatetimeIndex index.
    """
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(TypeError, match="Input `X` must have a pandas DatetimeIndex"):
        DateTimeFeatureTransformer().fit_transform(df)


def test_create_datetime_features_invalid_encoding():
    """
    Test that DateTimeFeatureTransformer raises a ValueError when encoding is not 
    one of 'cyclical', 'onehot' or None.
    """
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.date_range(start="1/1/2022", end="1/5/2022", freq="D"),
    )

    with pytest.raises(
        ValueError, match="Encoding must be one of 'cyclical', 'onehot' or None"
    ):
        DateTimeFeatureTransformer(encoding="invalid encoding").fit_transform(df)


def test_create_datetime_features_invalid_feature_name():
    """
    Test that DateTimeFeatureTransformer raises a ValueError when a feature name is
    not valid.
    """
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.date_range(start="1/1/2022", end="1/5/2022", freq="D"),
    )

    err_msg = re.escape(
        "Features {'invalid_feature'} are not supported. Supported features are "
        "['year', 'month', 'week', 'day_of_week', 'day_of_year', 'day_of_month', "
        "'weekend', 'hour', 'minute', 'second']."
    )
    with pytest.raises(ValueError, match=err_msg):
        DateTimeFeatureTransformer(features=["invalid_feature"]).fit_transform(df)


def test_create_datetime_features_output_columns_when_cyclical_encoding():
    """
    Test that DateTimeFeatureTransformer returns the expected columns when encoding is 'cyclical'.
    """
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.date_range(start="1/1/2022", end="1/5/2022", freq="D"),
    )

    results = DateTimeFeatureTransformer(encoding="cyclical").fit_transform(df)
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
    Test that DateTimeFeatureTransformer returns the expected columns when encoding is 'onehot'.
    """
    index = pd.date_range(start="2021-01-01", end="2023-01-01", freq=freq_h)
    df = pd.DataFrame(
        np.random.rand(len(index), 3),
        columns=["col_1", "col_2", "col_3"],
        index=index,
    )

    results = DateTimeFeatureTransformer(encoding="onehot").fit_transform(df)

    assert all([feature in features_all_onehot for feature in results.columns])
    assert len(results) == len(df)


def test_create_datetime_features_output_columns_when_None_encoding():
    """
    Test that DateTimeFeatureTransformer returns the expected columns when encoding is 'None'.
    """
    index = pd.date_range(start="2021-01-01", end="2023-01-01", freq=freq_h)
    df = pd.DataFrame(
        np.random.rand(len(index), 3),
        columns=["col_1", "col_2", "col_3"],
        index=index,
    )
    results = DateTimeFeatureTransformer(encoding=None).fit_transform(df)
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
    ]
    assert all(results.columns == expected_features)
    assert len(results) == len(df)


def test_create_datetime_features_output_when_features_year_month_encoding_cyclical():
    """
    Test that DateTimeFeatureTransformer returns the expected columns when features
     is ['year', 'month'] and encoding is 'cyclical'.
    """
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.date_range(start="1/1/2022", end="1/5/2022", freq="D"),
    )

    results = DateTimeFeatureTransformer(
        features=["year", "month", "weekend"], encoding="cyclical"
    ).fit_transform(df)
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
    Test that DateTimeFeatureTransformer returns the expected columns when features
     is ['year', 'month'] and encoding is 'onehot'.
    """
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.date_range(start="1/1/2022", end="1/5/2022", freq="D"),
    )

    results = DateTimeFeatureTransformer(
        features=["year", "month", "weekend"], encoding="onehot"
    ).fit_transform(
        df,
    )
    expected = pd.DataFrame(
        {
            "year_2022": {
                pd.Timestamp("2022-01-01 00:00:00"): 1,
                pd.Timestamp("2022-01-02 00:00:00"): 1,
                pd.Timestamp("2022-01-03 00:00:00"): 1,
                pd.Timestamp("2022-01-04 00:00:00"): 1,
                pd.Timestamp("2022-01-05 00:00:00"): 1,
            },
            "month_1": {
                pd.Timestamp("2022-01-01 00:00:00"): 1,
                pd.Timestamp("2022-01-02 00:00:00"): 1,
                pd.Timestamp("2022-01-03 00:00:00"): 1,
                pd.Timestamp("2022-01-04 00:00:00"): 1,
                pd.Timestamp("2022-01-05 00:00:00"): 1,
            },
            "weekend_0": {
                pd.Timestamp("2022-01-01 00:00:00"): 0,
                pd.Timestamp("2022-01-02 00:00:00"): 0,
                pd.Timestamp("2022-01-03 00:00:00"): 1,
                pd.Timestamp("2022-01-04 00:00:00"): 1,
                pd.Timestamp("2022-01-05 00:00:00"): 1,
            },
            "weekend_1": {
                pd.Timestamp("2022-01-01 00:00:00"): 1,
                pd.Timestamp("2022-01-02 00:00:00"): 1,
                pd.Timestamp("2022-01-03 00:00:00"): 0,
                pd.Timestamp("2022-01-04 00:00:00"): 0,
                pd.Timestamp("2022-01-05 00:00:00"): 0,
            },
        }
    ).asfreq("D").astype(
        {'year_2022': int, 'month_1': int, 'weekend_0': int, 'weekend_1': int}
    )

    pd.testing.assert_frame_equal(results, expected)


def test_create_datetime_features_output_when_features_year_month_encoding_None():
    """
    Test that DateTimeFeatureTransformer returns the expected columns when features
     is ['year', 'month'] and encoding is None.
    """
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.date_range(start="1/1/2022", end="1/5/2022", freq="D"),
    )

    results = DateTimeFeatureTransformer(
        features=["year", "month", "weekend"], encoding=None
    ).fit_transform(
        df,
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
    Test that DateTimeFeatureTransformer returns the expected columns when features
    is ['year', 'month'] and encoding is 'cyclical' with custom max values.
    """

    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.DatetimeIndex(
            ["2022-01-31", "2022-02-28", "2022-03-31", "2022-04-30", "2022-05-31"]
        ),
    )

    results = DateTimeFeatureTransformer(
        features=["year", "month", "weekend"],
        encoding="cyclical",
        max_values={"month": 6},
    ).fit_transform(df)

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
