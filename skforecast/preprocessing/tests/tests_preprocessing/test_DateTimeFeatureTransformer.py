# Unit test DateTimeFeatureTransformer
# ==============================================================================
import pytest
import re
import pandas as pd
import numpy as np
from sklearn.base import clone
from skforecast.preprocessing import DateTimeFeatureTransformer, create_datetime_features

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
        ValueError, match="Encoding must be one of 'cyclical', 'onehot', 'spline' or None"
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
        "'weekend', 'hour', 'minute', 'second', 'quarter']."
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

    results = DateTimeFeatureTransformer(encoding="cyclical", keep_original_columns=False).fit_transform(df)
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
        "quarter_sin",
        "quarter_cos",
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

    results = DateTimeFeatureTransformer(encoding="onehot", keep_original_columns=False).fit_transform(df)

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
    results = DateTimeFeatureTransformer(encoding=None, keep_original_columns=False).fit_transform(df)
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
    Test that DateTimeFeatureTransformer returns the expected columns when features
     is ['year', 'month'] and encoding is 'cyclical'.
    """
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.date_range(start="1/1/2022", end="1/5/2022", freq="D"),
    )

    results = DateTimeFeatureTransformer(
        features=["year", "month", "weekend"], encoding="cyclical", keep_original_columns=False
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
    is ['year', 'month', 'weekend'] and encoding is 'onehot'. All predefined
    categories must be present even when only January dates are in the data.
    """
    index = pd.date_range(start="1/1/2022", end="1/5/2022", freq="D")
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=index,
    )

    results = DateTimeFeatureTransformer(
        features=["year", "month", "weekend"], encoding="onehot", keep_original_columns=False
    ).fit_transform(df)

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
    Test that DateTimeFeatureTransformer returns the expected columns when features
     is ['year', 'month'] and encoding is None.
    """
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.date_range(start="1/1/2022", end="1/5/2022", freq="D"),
    )

    results = DateTimeFeatureTransformer(
        features=["year", "month", "weekend"], encoding=None, keep_original_columns=False
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
        keep_original_columns=False
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


def test_DateTimeFeatureTransformer_get_params_returns_constructor_values():
    """
    Test that get_params returns the exact values passed to __init__, including
    None for defaulted parameters (sklearn BaseEstimator contract).
    """
    transformer = DateTimeFeatureTransformer()
    params = transformer.get_params()

    assert params == {"features": None, "features_to_encode": None, "encoding": "cyclical", "max_values": None, "spline_kwargs": None, "keep_original_columns": True}


def test_DateTimeFeatureTransformer_get_params_returns_custom_values():
    """
    Test that get_params returns the custom values passed to __init__.
    """
    transformer = DateTimeFeatureTransformer(
        features=["year", "month"],
        encoding="onehot",
        max_values={"month": 6},
        keep_original_columns=False
    )
    params = transformer.get_params()

    assert params == {
        "features": ["year", "month"],
        "features_to_encode": None,
        "encoding": "onehot",
        "max_values": {"month": 6},
        "spline_kwargs": None,
        "keep_original_columns": False,
    }


def test_DateTimeFeatureTransformer_clone_preserves_none_defaults():
    """
    Test that sklearn clone() round-trips correctly when default (None) params
    are used. The cloned transformer must produce identical output.
    """
    df = pd.DataFrame(
        np.random.rand(5, 2),
        columns=["a", "b"],
        index=pd.date_range(start="2022-01-01", periods=5, freq="D"),
    )
    transformer = DateTimeFeatureTransformer()
    cloned = clone(transformer)

    assert cloned.features is None
    assert cloned.encoding == "cyclical"
    assert cloned.max_values is None

    pd.testing.assert_frame_equal(
        transformer.fit_transform(df), cloned.fit_transform(df)
    )


def test_DateTimeFeatureTransformer_set_params_updates_values():
    """
    Test that set_params correctly updates transformer parameters.
    """
    transformer = DateTimeFeatureTransformer()
    transformer.set_params(features=["year", "month"], encoding="onehot")

    assert transformer.features == ["year", "month"]
    assert transformer.encoding == "onehot"
    assert transformer.max_values is None


def test_DateTimeFeatureTransformer_week_feature_dtype_is_int():
    """
    Test that the 'week' feature is returned as int64, consistent with all
    other extracted features (isocalendar().week returns UInt32 by default).
    """
    df = pd.DataFrame(
        np.random.rand(5, 1),
        columns=["value"],
        index=pd.date_range(start="2022-01-01", periods=5, freq="D"),
    )
    result = DateTimeFeatureTransformer(
        features=["week"], encoding=None
    ).fit_transform(df)

    assert result["week"].dtype == np.dtype("int64")


def test_DateTimeFeatureTransformer_get_feature_names_out_raises_before_transform():
    """
    Test that get_feature_names_out raises NotFittedError if transform has not
    been called yet.
    """
    from sklearn.exceptions import NotFittedError

    transformer = DateTimeFeatureTransformer()
    with pytest.raises(NotFittedError):
        transformer.get_feature_names_out()


@pytest.mark.parametrize(
    "encoding, features, expected",
    [
        (
            None,
            ["year", "month", "weekend"],
            ["year", "month", "weekend"],
        ),
        (
            "cyclical",
            ["year", "month", "weekend"],
            ["year", "weekend", "month_sin", "month_cos"],
        ),
        (
            "onehot",
            ["year", "month"],
            ["year", "month_1"],
        ),
        (
            "spline",
            ["year", "month"],
            ["year", "month_sp_0"],
        ),
    ],
    ids=["encoding_None", "encoding_cyclical", "encoding_onehot", "encoding_spline"],
)
def test_DateTimeFeatureTransformer_get_feature_names_out(encoding, features, expected):
    """
    Test that get_feature_names_out returns the correct column names for each
    encoding mode and matches the columns of the transform output.
    """
    df = pd.DataFrame(
        np.random.rand(5, 1),
        columns=["value"],
        index=pd.date_range(start="2022-01-01", periods=5, freq="D"),
    )
    transformer = DateTimeFeatureTransformer(features=features, encoding=encoding)
    result = transformer.fit_transform(df)
    names_out = transformer.get_feature_names_out()

    assert names_out == list(result.columns)
    assert all(e in names_out for e in expected)


def test_DateTimeFeatureTransformer_quarter_feature():
    """
    Test that 'quarter' is correctly extracted and cyclically encoded.
    """
    df = pd.DataFrame(
        np.random.rand(12, 1),
        columns=["value"],
        index=pd.date_range(start="2022-01-01", periods=12, freq="MS"),
    )
    result_none = DateTimeFeatureTransformer(
        features=["quarter"], encoding=None, keep_original_columns=False
    ).fit_transform(df)

    assert list(result_none.columns) == ["quarter"]
    assert result_none["quarter"].dtype == np.dtype("int64")
    assert set(result_none["quarter"].unique()).issubset({1, 2, 3, 4})

    result_cyclical = DateTimeFeatureTransformer(
        features=["quarter"], encoding="cyclical", keep_original_columns=False
    ).fit_transform(df)

    assert list(result_cyclical.columns) == ["quarter_sin", "quarter_cos"]


def test_create_datetime_features_output_columns_when_spline_encoding():
    """
    Test that DateTimeFeatureTransformer returns the expected columns when encoding
    is 'spline'. Features with a max_values entry are replaced by spline columns;
    features without one (year, weekend) are kept as raw integers.
    With default n_knots=max_val+1=13, include_bias=True and periodic extrapolation,
    month produces n_knots - 1 = 12 spline columns.
    """
    df = pd.DataFrame(
        np.random.rand(5, 1),
        columns=["value"],
        index=pd.date_range(start="2022-01-01", periods=5, freq="D"),
    )
    results = DateTimeFeatureTransformer(
        features=["year", "month", "weekend"], encoding="spline"
    ).fit_transform(df)

    # year and weekend have no max_values entry -> kept as integers
    assert "year" in results.columns
    assert "weekend" in results.columns
    # month: n_knots=13, include_bias=True, periodic -> 12 spline columns
    assert "month" not in results.columns
    month_sp_cols = [c for c in results.columns if c.startswith("month_sp_")]
    assert len(month_sp_cols) == 12
    assert len(results) == len(df)


def test_create_datetime_features_output_shape_with_custom_spline_kwargs():
    """
    Test that the number of spline output columns respects a custom n_knots.
    With n_knots=4, include_bias=True (default), periodic extrapolation the formula
    yields n_knots - 1 = 3 columns per feature.
    """
    df = pd.DataFrame(
        np.random.rand(5, 1),
        columns=["value"],
        index=pd.date_range(start="2022-01-01", periods=5, freq="D"),
    )
    results = DateTimeFeatureTransformer(
        features=["month"],
        encoding="spline",
        spline_kwargs={"n_knots": 4},
        keep_original_columns=False
    ).fit_transform(df)

    assert "month" not in results.columns
    # n_knots=4, include_bias=True, periodic -> 3 spline columns
    assert list(results.columns) == ["month_sp_0", "month_sp_1", "month_sp_2"]
    assert len(results) == len(df)


def test_create_datetime_features_spline_encoding_expected_values():
    """
    Test that the spline encoding produces the expected numerical values.
    Uses 4 dates spaced one quarter apart (Jan, Apr, Jul, Oct) with the 'month'
    feature only. With default settings (n_knots=13, degree=3, include_bias=True,
    periodic) and the corrected knot range [1, 13], the 12 output columns sum
    to 1.0 per row and each row activates exactly 3 splines with the canonical
    cubic-B-spline-at-knot values (1/6, 4/6, 1/6).

    Expected values pre-computed with:
        knots = np.linspace(1, 13, 13).reshape(-1, 1)
        SplineTransformer(degree=3, knots=knots, extrapolation='periodic',
                          include_bias=True).fit_transform([[1],[4],[7],[10]])
    """
    df = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0, 4.0]},
        index=pd.DatetimeIndex(
            ["2022-01-15", "2022-04-15", "2022-07-15", "2022-10-15"]
        ),
    )
    result = DateTimeFeatureTransformer(
        features=["month"], encoding="spline", keep_original_columns=False
    ).fit_transform(df)

    one_sixth = 1 / 6
    four_sixths = 4 / 6
    expected = pd.DataFrame(
        {
            "month_sp_0":  [one_sixth,   0.0,         0.0,         0.0        ],
            "month_sp_1":  [four_sixths, 0.0,         0.0,         0.0        ],
            "month_sp_2":  [one_sixth,   0.0,         0.0,         0.0        ],
            "month_sp_3":  [0.0,         one_sixth,   0.0,         0.0        ],
            "month_sp_4":  [0.0,         four_sixths, 0.0,         0.0        ],
            "month_sp_5":  [0.0,         one_sixth,   0.0,         0.0        ],
            "month_sp_6":  [0.0,         0.0,         one_sixth,   0.0        ],
            "month_sp_7":  [0.0,         0.0,         four_sixths, 0.0        ],
            "month_sp_8":  [0.0,         0.0,         one_sixth,   0.0        ],
            "month_sp_9":  [0.0,         0.0,         0.0,         one_sixth  ],
            "month_sp_10": [0.0,         0.0,         0.0,         four_sixths],
            "month_sp_11": [0.0,         0.0,         0.0,         one_sixth  ],
        },
        index=pd.DatetimeIndex(
            ["2022-01-15", "2022-04-15", "2022-07-15", "2022-10-15"]
        ),
    )

    # Each row must sum to 1.0 (partition of unity with include_bias=True)
    np.testing.assert_allclose(result.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
    pd.testing.assert_frame_equal(result, expected, atol=1e-6, check_dtype=False)


def test_create_datetime_features_accepts_series_input():
    """
    Test that create_datetime_features accepts a pandas Series with a
    DatetimeIndex, identical to a DataFrame input.
    """
    index = pd.date_range(start="2022-01-01", periods=5, freq="D")
    series = pd.Series(np.random.rand(5), index=index, name="target")
    df = pd.DataFrame({"value": series.values}, index=index)

    result_series = create_datetime_features(series, features=["year", "month"], encoding=None, keep_original_columns=False)
    result_df = create_datetime_features(df, features=["year", "month"], encoding=None, keep_original_columns=False)

    pd.testing.assert_frame_equal(result_series, result_df)


def test_create_datetime_features_standalone_invalid_encoding():
    """
    Test that create_datetime_features raises ValueError for an invalid encoding
    when called directly (not via DateTimeFeatureTransformer).
    """
    series = pd.Series(
        np.random.rand(5),
        index=pd.date_range(start="2022-01-01", periods=5, freq="D"),
    )
    with pytest.raises(
        ValueError, match="Encoding must be one of 'cyclical', 'onehot', 'spline' or None"
    ):
        create_datetime_features(series, encoding="invalid")


def test_DateTimeFeatureTransformer_invalid_features_to_encode():
    """
    Test that DateTimeFeatureTransformer raises ValueError when features_to_encode
    contains features not present in features list.
    """
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=["col_1", "col_2", "col_3"],
        index=pd.DatetimeIndex(
            ["2022-01-31", "2022-02-28", "2022-03-31", "2022-04-30", "2022-05-31"]
        ),
    )

    transformer = DateTimeFeatureTransformer(
        features=["year", "month"],
        features_to_encode=["month", "invalid_feature"]
    )
    
    err_msg = re.escape("Features {'invalid_feature'} are not present in `features`.")
    with pytest.raises(ValueError, match=err_msg):
        transformer.fit_transform(df)


def test_DateTimeFeatureTransformer_features_to_encode_cyclical():
    """
    Test that DateTimeFeatureTransformer encodes only features in features_to_encode
    when using cyclical encoding.
    """
    df = pd.DataFrame(
        np.random.rand(2, 1),
        index=pd.DatetimeIndex(["2022-01-31", "2022-02-28"])
    )

    transformer = DateTimeFeatureTransformer(
        features=["month", "hour"],
        features_to_encode=["hour"],
        encoding="cyclical",
        keep_original_columns=False
    )
    results = transformer.fit_transform(df)
    
    expected = pd.DataFrame({
        "month": [1, 2],
        "hour_sin": [0.0, 0.0],
        "hour_cos": [1.0, 1.0]
    }, index=df.index).astype({"month": int})

    pd.testing.assert_frame_equal(results, expected)


def test_DateTimeFeatureTransformer_features_to_encode_onehot():
    """
    Test that DateTimeFeatureTransformer encodes only features in features_to_encode
    when using onehot encoding. All 24 hour columns are always generated even
    though only hours 1 and 2 appear in the data.
    """
    df = pd.DataFrame(
        np.random.rand(2, 1),
        index=pd.DatetimeIndex(["2022-01-01 01:00:00", "2022-02-01 02:00:00"])
    )

    transformer = DateTimeFeatureTransformer(
        features=["month", "hour"],
        features_to_encode=["hour"],
        encoding="onehot",
        keep_original_columns=False
    )
    results = transformer.fit_transform(df)

    # month is not encoded — kept as raw integer
    assert results["month"].tolist() == [1, 2]

    # All 24 hour columns are generated regardless of which hours appear
    hour_cols = [c for c in results.columns if c.startswith("hour_")]
    assert len(hour_cols) == 24
    assert results["hour_1"].tolist() == [1, 0]
    assert results["hour_2"].tolist() == [0, 1]
    assert results["hour_0"].tolist() == [0, 0]

    assert results.shape == (2, 25)


def test_DateTimeFeatureTransformer_features_to_encode_spline():
    """
    Test that DateTimeFeatureTransformer encodes only features in features_to_encode
    when using spline encoding.
    """
    df = pd.DataFrame(
        np.random.rand(2, 1),
        index=pd.DatetimeIndex(["2022-01-01 01:00:00", "2022-02-01 02:00:00"])
    )

    transformer = DateTimeFeatureTransformer(
        features=["month", "hour"],
        features_to_encode=["hour"],
        encoding="spline",
        keep_original_columns=False
    )
    results = transformer.fit_transform(df)

    assert "month" in results.columns
    assert "hour" not in results.columns
    
    spline_cols = [c for c in results.columns if "hour_sp_" in c or "hour" in c]
    assert len(spline_cols) > 0
    assert len(results.columns) > 1


def test_DateTimeFeatureTransformer_keep_original_columns_True_dataframe():
    """
    Test that DateTimeFeatureTransformer returns original columns when 
    keep_original_columns=True for a DataFrame.
    """
    df = pd.DataFrame(
        {"exog_1": [1, 2], "exog_2": [3, 4]},
        index=pd.DatetimeIndex(["2022-01-01", "2022-02-01"])
    )

    transformer = DateTimeFeatureTransformer(
        features=["month"],
        encoding=None,
        keep_original_columns=True
    )
    results = transformer.fit_transform(df)

    assert "exog_1" in results.columns
    assert "exog_2" in results.columns
    assert "month" in results.columns
    assert list(results.columns) == ["exog_1", "exog_2", "month"]
    assert results["exog_1"].tolist() == [1, 2]


def test_DateTimeFeatureTransformer_keep_original_columns_True_series():
    """
    Test that DateTimeFeatureTransformer returns original series as a column when 
    keep_original_columns=True for a Series.
    """
    series = pd.Series(
        [1, 2],
        name="target",
        index=pd.DatetimeIndex(["2022-01-01", "2022-02-01"])
    )

    transformer = DateTimeFeatureTransformer(
        features=["month"],
        encoding=None,
        keep_original_columns=True
    )
    results = transformer.fit_transform(series)

    assert "target" in results.columns
    assert "month" in results.columns
    assert list(results.columns) == ["target", "month"]
    assert results["target"].tolist() == [1, 2]


def test_DateTimeFeatureTransformer_keep_original_columns_True_overlap_error():
    """
    Test that DateTimeFeatureTransformer raises ValueError when keep_original_columns=True
    and there is a column name overlap with extracted features.
    """
    df = pd.DataFrame(
        {"month": [1, 2], "exog_1": [3, 4]},
        index=pd.DatetimeIndex(["2022-01-01", "2022-02-01"])
    )

    transformer = DateTimeFeatureTransformer(
        features=["month"],
        encoding=None,
        keep_original_columns=True
    )
    
    err_msg = re.escape(
        "The following extracted feature names already exist in the input DataFrame: "
        "['month']. To avoid duplicate columns, rename the original columns or "
        "avoid extracting these features."
    )
    with pytest.raises(ValueError, match=err_msg):
        transformer.fit_transform(df)

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
        transformer.fit_transform(series)


def test_DateTimeFeatureTransformer_get_feature_names_out_after_fit_before_transform():
    """
    Test that get_feature_names_out() returns the correct column names after
    fit() is called but before transform() is called. This verifies the sklearn
    API contract: attributes ending with '_' must be set during fit().
    """
    df = pd.DataFrame(
        np.random.rand(5, 1),
        columns=["value"],
        index=pd.date_range(start="2022-01-01", periods=5, freq="D"),
    )
    transformer = DateTimeFeatureTransformer(
        features=["year", "month"], encoding="cyclical", keep_original_columns=False
    )
    transformer.fit(df)
    names_after_fit = transformer.get_feature_names_out()

    result = transformer.transform(df)
    names_after_transform = transformer.get_feature_names_out()

    assert names_after_fit == names_after_transform
    assert names_after_fit == list(result.columns)


def test_DateTimeFeatureTransformer_onehot_single_row_generates_all_columns():
    """
    Test that onehot encoding generates all expected columns even when a single
    row is passed to fit_transform. Guards against pd.get_dummies silently
    omitting categories absent from the input, which would cause model crashes
    at inference time.
    """
    # A single Wednesday: day_of_week == 2, should still produce columns 0-6
    index = pd.DatetimeIndex(["2022-01-05"])  # Wednesday
    df = pd.DataFrame({"value": [1.0]}, index=index)

    result = DateTimeFeatureTransformer(
        features=["day_of_week"],
        encoding="onehot",
        keep_original_columns=False,
    ).fit_transform(df)

    expected_cols = [f"day_of_week_{i}" for i in range(7)]
    assert list(result.columns) == expected_cols
    assert result["day_of_week_2"].iloc[0] == 1  # Wednesday
    assert result["day_of_week_0"].iloc[0] == 0  # not Monday


def test_DateTimeFeatureTransformer_onehot_year_and_weekend_never_encoded():
    """
    Test that year and weekend are never one-hot encoded when encoding='onehot',
    regardless of whether they appear in features_to_encode.
    """
    index = pd.date_range(start="2022-01-01", periods=7, freq="D")
    df = pd.DataFrame({"value": range(7)}, index=index)

    result = DateTimeFeatureTransformer(
        features=["year", "weekend", "month"],
        features_to_encode=["year", "weekend", "month"],
        encoding="onehot",
        keep_original_columns=False,
    ).fit_transform(df)

    assert "year" in result.columns
    assert result["year"].dtype == np.dtype("int64")
    assert not any(c.startswith("year_") for c in result.columns)

    assert "weekend" in result.columns
    assert result["weekend"].dtype == np.dtype("int64")
    assert "weekend_0" not in result.columns
    assert "weekend_1" not in result.columns

    month_cols = [c for c in result.columns if c.startswith("month_")]
    assert len(month_cols) == 12


def test_DateTimeFeatureTransformer_fit_empty_input_raises():
    """
    Test that calling fit on an empty DataFrame or Series raises ValueError
    with a clear message, instead of letting the error surface from inside
    SplineTransformer (or producing a degenerate transformer).
    """
    empty_df = pd.DataFrame(
        columns=["value"], index=pd.DatetimeIndex([], name="datetime")
    )
    transformer = DateTimeFeatureTransformer(features=["month"], encoding="cyclical")
    with pytest.raises(ValueError, match=r"Cannot fit on empty input\."):
        transformer.fit(empty_df)

    empty_series = pd.Series([], dtype=float, index=pd.DatetimeIndex([]), name="y")
    with pytest.raises(ValueError, match=r"Cannot fit on empty input\."):
        DateTimeFeatureTransformer(features=["month"], encoding="cyclical").fit(empty_series)

