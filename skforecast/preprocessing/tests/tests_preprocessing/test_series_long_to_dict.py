# Unit test series_long_to_dict
# ==============================================================================
import pytest
import pandas as pd
from ...preprocessing import series_long_to_dict
from ....exceptions import MissingValuesWarning

# Fixtures
from .fixtures_preprocessing import values_A, values_B, values_C
from .fixtures_preprocessing import index_A, index_B, index_C
from .fixtures_preprocessing import series_long


def test_check_output_series_long_to_dict():
    """
    Check output of series_long_to_dict.
    """

    expected = {
        "A": pd.Series(values_A, index=index_A, name="A"),
        "B": pd.Series(values_B, index=index_B, name="B"),
        "C": pd.Series(values_C, index=index_C, name="C"),
    }

    results = series_long_to_dict(
        data=series_long,
        series_id="series_id",
        index="datetime",
        values="values",
        freq="D",
    )

    for k in expected.keys():
        pd.testing.assert_series_equal(results[k], expected[k])


def test_TypeError_when_data_is_not_dataframe():
    """
    Raise TypeError if data is not a pandas DataFrame.
    """
    err_msg = "`data` must be a pandas DataFrame."
    with pytest.raises(TypeError, match=err_msg):
        series_long_to_dict(
            data='not_a_dataframe',
            series_id="series_id",
            index="datetime",
            values="values",
            freq="D",
        )


def test_ValueError_when_series_id_not_in_data():
    """
    Raise ValueError if series_id is not in data.
    """
    series_id = "series_id_not_in_data"
    err_msg = f"Column '{series_id}' not found in `data`."
    with pytest.raises(ValueError, match=err_msg):
        series_long_to_dict(
            data=series_long,
            series_id=series_id,
            index="datetime",
            values="values",
            freq="D",
        )


def test_ValueError_when_index_not_in_data():
    """
    Raise ValueError if index is not in data.
    """
    index = "series_id_not_in_data"
    err_msg = f"Column '{index}' not found in `data`."
    with pytest.raises(ValueError, match=err_msg):
        series_long_to_dict(
            data=series_long,
            series_id="series_id",
            index=index,
            values="values",
            freq="D",
        )


def test_ValueError_when_values_not_in_data():
    """
    Raise ValueError if values is not in data.
    """
    values = "values_not_in_data"
    err_msg = f"Column '{values}' not found in `data`."
    with pytest.raises(ValueError, match=err_msg):
        series_long_to_dict(
            data=series_long,
            series_id="series_id",
            index="datetime",
            values=values,
            freq="D",
        )


def test_warning_when_series_is_incomplete():
    """
    Raise warning if series is incomplete and NaN values are introduced
    after setting the index frequency.
    """
    data = pd.DataFrame({
        "series_id": ["A"] * 4 + ["B"] * 4,
        "index": pd.date_range("2020-01-01", periods=4, freq="D").tolist() * 2,
        "values": [1, 2, 3, 4] * 2,
    })
    data = data.iloc[[0, 1, 2, 3, 4, 5, 7]]
    msg = (
        "Series 'B' is incomplete. NaNs have been introduced after setting the frequency."
    )
    with pytest.warns(MissingValuesWarning, match=msg):
        series_long_to_dict(
            data=data,
            series_id="series_id",
            index="index",
            values="values",
            freq="D",
        )