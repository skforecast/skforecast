# Unit test reshape_series_long_to_dict
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from ...preprocessing import reshape_series_exog_dict_to_long

def test_reshape_series_exog_dict_to_long_raises_ValueError_when_both_None():
    """
    Test that ValueError is raised when both series and exog are None.
    """
    with pytest.raises(ValueError, match="Both series and exog cannot be None."):
        reshape_series_exog_dict_to_long(series=None, exog=None)


def test_reshape_series_exog_dict_to_long_raises_TypeError_when_series_not_dict():
    """
    Test that TypeError is raised when series is not a dictionary.
    """
    with pytest.raises(TypeError, match="`series` must be a dictionary"):
        reshape_series_exog_dict_to_long(series=[1, 2, 3], exog=None)


def test_reshape_series_exog_dict_to_long_raises_TypeError_when_series_value_not_Series():
    """
    Test that TypeError is raised when a value in series dict is not a pandas Series.
    """
    series = {
        'series_1': pd.Series([1, 2, 3]),
        'series_2': [4, 5, 6]  # Not a Series
    }
    with pytest.raises(TypeError, match="series\\['series_2'\\] must be pd.Series"):
        reshape_series_exog_dict_to_long(series=series, exog=None)


def test_reshape_series_exog_dict_to_long_raises_TypeError_when_exog_not_dict():
    """
    Test that TypeError is raised when exog is not a dictionary.
    """
    with pytest.raises(TypeError, match="`exog` must be a dictionary"):
        reshape_series_exog_dict_to_long(series=None, exog=[1, 2, 3])


def test_reshape_series_exog_dict_to_long_raises_TypeError_when_exog_value_invalid():
    """
    Test that TypeError is raised when a value in exog dict is not a Series or DataFrame.
    """
    exog = {
        'series_1': pd.Series([1, 2, 3]),
        'series_2': [4, 5, 6]  # Not a Series or DataFrame
    }
    with pytest.raises(TypeError, match="exog\\['series_2'\\] must be pd.Series or pd.DataFrame"):
        reshape_series_exog_dict_to_long(series=None, exog=exog)


def test_reshape_series_exog_dict_to_long_only_series():
    """
    Test reshaping when only series dict is provided.
    """
    series = {
        'series_1': pd.Series([1, 2, 3], index=pd.date_range('2020-01-01', periods=3)),
        'series_2': pd.Series([4, 5, 6], index=pd.date_range('2020-01-01', periods=3))
    }
    
    result = reshape_series_exog_dict_to_long(series=series, exog=None)
    expected_result = pd.DataFrame(
        data={"series_value": [1, 2, 3, 4, 5, 6]},
        index=pd.MultiIndex.from_product(
            [["series_1", "series_2"], pd.date_range("2020-01-01", periods=3)],
            names=["series_id", "datetime"],
        ),
    )

    pd.testing.assert_frame_equal(result, expected_result)


def test_reshape_series_exog_dict_to_long_only_exog_Series():
    """
    Test reshaping when only exog dict with Series is provided.
    """
    exog = {
        'series_1': pd.Series([1, 2, 3], index=pd.date_range('2020-01-01', periods=3), name='exog1'),
        'series_2': pd.Series([4, 5, 6], index=pd.date_range('2020-01-01', periods=3), name='exog1')
    }
    
    result = reshape_series_exog_dict_to_long(series=None, exog=exog)
    expected_result = pd.DataFrame(
        data={"exog1": [1, 2, 3, 4, 5, 6]},
        index=pd.MultiIndex.from_product(
            [["series_1", "series_2"], pd.date_range("2020-01-01", periods=3)],
            names=["series_id", "datetime"],
        ),
    )
    pd.testing.assert_frame_equal(result, expected_result)


def test_reshape_series_exog_dict_to_long_only_exog_DataFrame():
    """
    Test reshaping when only exog dict with DataFrames is provided.
    """
    exog = {
        'series_1': pd.DataFrame(
            {'exog1': [1, 2, 3], 'exog2': [7, 8, 9]},
            index=pd.date_range('2020-01-01', periods=3)
        ),
        'series_2': pd.DataFrame(
            {'exog1': [4, 5, 6], 'exog2': [10, 11, 12]},
            index=pd.date_range('2020-01-01', periods=3)
        )
    }
    
    result = reshape_series_exog_dict_to_long(series=None, exog=exog)
    expected_result = pd.DataFrame(
        data={"exog1": [1, 2, 3, 4, 5, 6], "exog2": [7, 8, 9, 10, 11, 12]},
        index=pd.MultiIndex.from_product(
            [["series_1", "series_2"], pd.date_range("2020-01-01", periods=3)],
            names=["series_id", "datetime"],
        ),
    )
    pd.testing.assert_frame_equal(result, expected_result)


def test_reshape_series_exog_dict_to_long_series_and_exog():
    """
    Test reshaping when both series and exog dicts are provided.
    """
    series_dict = {
        "series_1": pd.Series(
            data=np.arange(3), index=pd.date_range(start="2020-01-01", periods=3, freq="D")
        ),
        "series_2": pd.Series(
            data=np.arange(3, 6),
            index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
        ),
    }

    exog_dict = {
        "series_1": pd.DataFrame(
            data={
                "exog_1": np.arange(3),
                "exog_2": np.arange(10, 13),
                "exog_3": np.arange(20, 23),
            },
            index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
        ),
        "series_2": pd.DataFrame(
            data={"exog_1": np.arange(3, 6), "exog_2": np.arange(13, 16)},
            index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
        ),
    }
    result = reshape_series_exog_dict_to_long(
        series=series_dict,
        exog=exog_dict
    )
    expected_result = pd.DataFrame(
        data={
            "series_value": [0, 1, 2, 3, 4, 5],
            "exog_1": [0, 1, 2, 3, 4, 5],
            "exog_2": [10, 11, 12, 13, 14, 15],
            "exog_3": [20, 21, 22, np.nan, np.nan, np.nan],
        },
        index=pd.MultiIndex.from_product(
            [["series_1", "series_2"], pd.date_range("2020-01-01", periods=3)],
            names=["series_id", "datetime"],
        ),
    )
    pd.testing.assert_frame_equal(result, expected_result)