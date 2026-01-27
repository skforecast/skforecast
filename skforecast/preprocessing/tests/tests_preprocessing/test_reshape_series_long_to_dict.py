# Unit test reshape_series_long_to_dict
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from ...preprocessing import reshape_series_long_to_dict
from ....exceptions import MissingValuesWarning

# Fixtures
from .fixtures_preprocessing import values_A, values_B, values_C
from .fixtures_preprocessing import index_A, index_B, index_C
from .fixtures_preprocessing import series_long
series_long_multiindex = series_long.set_index(["series_id", "datetime"])


def test_TypeError_when_data_is_not_dataframe():
    """
    Raise TypeError if data is not a pandas DataFrame.
    """
    err_msg = "`data` must be a pandas DataFrame."
    with pytest.raises(TypeError, match=err_msg):
        reshape_series_long_to_dict(
            data='not_a_dataframe',
            series_id="series_id",
            index="datetime",
            values="values",
            freq="D",
        )


def test_reshape_series_long_to_dict_raise_value_error_when_arguments_series_id_index_values_not_provided():
    """
    Check that ValueError is raised when the input dataframe does not have MultiIndex and the
    arguments `series_id`, `index` and `values` are not provided
    """
    err_msg = (
        "Arguments `series_id`, `index`, and `values` must be "
        "specified when the input DataFrame does not have a MultiIndex. "
        "Please provide a value for each of these arguments."
    )
    with pytest.raises(ValueError, match=err_msg):
        reshape_series_long_to_dict(data=series_long, freq="D")


def test_ValueError_when_series_id_not_in_data():
    """
    Raise ValueError if series_id is not in data.
    """
    series_id = "series_id_not_in_data"

    err_msg = f"Column '{series_id}' not found in `data`."
    with pytest.raises(ValueError, match=err_msg):
        reshape_series_long_to_dict(
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
        reshape_series_long_to_dict(
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
        reshape_series_long_to_dict(
            data=series_long,
            series_id="series_id",
            index="datetime",
            values=values,
            freq="D",
        )


def test_MissingValuesWarning_when_series_is_incomplete():
    """
    Raise MissingValuesWarning if series is incomplete and NaN values are introduced
    after setting the index frequency.
    """
    data = pd.DataFrame({
        "series_id": ["A"] * 4 + ["B"] * 4,
        "index": pd.date_range("2020-01-01", periods=4, freq="D").tolist() * 2,
        "values": [1., 2., 3., 4.] * 2,
    })
    data = data.iloc[[0, 1, 2, 3, 4, 5, 7]]

    expected = {
        "A": pd.Series(
                np.array([1., 2., 3., 4.]), 
                index= pd.date_range("2020-01-01", periods=4, freq="D"), 
                name="A"
            ),
        "B": pd.Series(
                np.array([1., 2., np.nan, 4.]), 
                index= pd.date_range("2020-01-01", periods=4, freq="D"), 
                name="B"
            )
    }

    msg = (
        "Series 'B' is incomplete. NaNs have been introduced after setting the frequency."
    )
    with pytest.warns(MissingValuesWarning, match=msg):
        results = reshape_series_long_to_dict(
                      data=data,
                      series_id="series_id",
                      index="index",
                      values="values",
                      freq="D",
                  )

    for k in expected.keys():
        pd.testing.assert_series_equal(results[k], expected[k])


def test_check_output_reshape_series_long_to_dict():
    """
    Check output of reshape_series_long_to_dict.
    """

    expected = {
        "A": pd.Series(values_A, index=index_A, name="A"),
        "B": pd.Series(values_B, index=index_B, name="B"),
        "C": pd.Series(values_C, index=index_C, name="C"),
    }

    results = reshape_series_long_to_dict(
        data=series_long,
        series_id="series_id",
        index="datetime",
        values="values",
        freq="D",
    )

    for k in expected.keys():
        pd.testing.assert_series_equal(results[k], expected[k])


def test_check_output_reshape_series_long_to_dict_when_multiindex():
    """
    Check output of reshape_series_long_to_dict when data is a MultiIndex DataFrame.
    """

    expected = {
        "A": pd.Series(values_A, index=index_A, name="A"),
        "B": pd.Series(values_B, index=index_B, name="B"),
        "C": pd.Series(values_C, index=index_C, name="C"),
    }

    results = reshape_series_long_to_dict(
        data=series_long_multiindex,
        freq="D",
    )

    for k in expected.keys():
        pd.testing.assert_series_equal(results[k], expected[k])


@pytest.mark.parametrize("fill_value, expected_fill", 
                         [(None, np.nan), (-999., -999.)], 
                         ids=lambda x: f'fill_value: {x}')
def test_check_output_reshape_series_long_to_dict_with_fill_value(fill_value, expected_fill):
    """
    Check output of reshape_series_long_to_dict with fill_value parameter
    when gaps are created by setting the frequency.
    """
    data = pd.DataFrame({
        "series_id": ["A"] * 4 + ["B"] * 4,
        "index": pd.date_range("2020-01-01", periods=4, freq="D").tolist() * 2,
        "values": [1., 2., 3., 4.] * 2,
    })
    # Remove one row to create a gap in series B
    data = data.iloc[[0, 1, 2, 3, 4, 5, 7]]

    expected = {
        "A": pd.Series(
                np.array([1., 2., 3., 4.]), 
                index=pd.date_range("2020-01-01", periods=4, freq="D"), 
                name="A"
            ),
        "B": pd.Series(
                np.array([1., 2., expected_fill, 4.]), 
                index=pd.date_range("2020-01-01", periods=4, freq="D"), 
                name="B"
            )
    }

    results = reshape_series_long_to_dict(
        data=data,
        series_id="series_id",
        index="index",
        values="values",
        freq="D",
        fill_value=fill_value,
        suppress_warnings=True
    )

    for k in expected.keys():
        pd.testing.assert_series_equal(results[k], expected[k])


@pytest.mark.parametrize("fill_value, expected_fill", 
                         [(None, np.nan), (0., 0.)], 
                         ids=lambda x: f'fill_value: {x}')
def test_check_output_reshape_series_long_to_dict_with_fill_value_when_multiindex(fill_value, expected_fill):
    """
    Check output of reshape_series_long_to_dict with fill_value parameter
    when data is a MultiIndex DataFrame and gaps are created by setting the frequency.
    """
    data = pd.DataFrame({
        "series_id": ["A"] * 4 + ["B"] * 4,
        "datetime": pd.date_range("2020-01-01", periods=4, freq="D").tolist() * 2,
        "values": [1., 2., 3., 4.] * 2,
    })
    # Remove one row to create a gap in series B
    data = data.iloc[[0, 1, 2, 3, 4, 5, 7]]
    data = data.set_index(["series_id", "datetime"])

    expected = {
        "A": pd.Series(
                np.array([1., 2., 3., 4.]), 
                index=pd.date_range("2020-01-01", periods=4, freq="D"), 
                name="A"
            ),
        "B": pd.Series(
                np.array([1., 2., expected_fill, 4.]), 
                index=pd.date_range("2020-01-01", periods=4, freq="D"), 
                name="B"
            )
    }

    results = reshape_series_long_to_dict(
        data=data,
        freq="D",
        fill_value=fill_value
    )

    for k in expected.keys():
        pd.testing.assert_series_equal(results[k], expected[k])
