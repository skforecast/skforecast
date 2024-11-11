# Unit test exog_long_to_dict
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from ...preprocessing import exog_long_to_dict
from ....exceptions import MissingValuesWarning

# Fixtures
from .fixtures_preprocessing import exog_A, exog_B, exog_C, n_exog_A, n_exog_B, n_exog_C
from .fixtures_preprocessing import exog_long


def test_check_output_series_long_to_dict_dropna_False():
    """
    Check output of exog_long_to_dict with dropna=False.
    """
    expected = {
        'A': pd.DataFrame(
            {
                'exog_1': np.arange(n_exog_A),
                'exog_2': np.nan,
                'exog_3': np.nan
            },
            index=pd.date_range("2020-01-01", periods=n_exog_A, freq="D")
        ),
        'B': pd.DataFrame(
            {
                'exog_1': np.arange(n_exog_B),
                'exog_2': 'b',
                'exog_3': np.nan
            },
            index=pd.date_range("2020-01-01", periods=n_exog_B, freq="D")
        ),
        'C': pd.DataFrame(
            {
                'exog_1': np.arange(n_exog_C),
                'exog_2': np.nan,
                'exog_3': 1.0
            },
            index=pd.date_range("2020-01-01", periods=n_exog_C, freq="D")
        )
    }

    for k in expected.keys():
        expected[k]['exog_1'] = expected[k]['exog_1'].astype(int)
        expected[k]['exog_2'] = expected[k]['exog_2'].astype(str)
        expected[k]['exog_3'] = expected[k]['exog_3'].astype(float)

    results = exog_long_to_dict(
        data=exog_long,
        series_id="series_id",
        index="datetime",
        freq="D",
        dropna=False,
    )

    for k in expected.keys():
        results[k].equals(expected[k])


def test_check_output_series_long_to_dict_dropna_True():
    """
    Check output of series_long_to_dict with dropna=True.
    """

    expected = {
        "A": exog_A.set_index("datetime").asfreq("D").drop(columns="series_id"),
        "B": exog_B.set_index("datetime").asfreq("D").drop(columns="series_id"),
        "C": exog_C.set_index("datetime").asfreq("D").drop(columns="series_id"),
    }

    for k in expected.keys():
        expected[k].index.name = None

    results = exog_long_to_dict(
        data=exog_long,
        series_id="series_id",
        index="datetime",
        freq="D",
        dropna=True,
    )

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k], check_dtype=False)


def test_TypeError_when_data_is_not_dataframe():
    """
    Raise TypeError if data is not a pandas DataFrame.
    """
    err_msg = "`data` must be a pandas DataFrame."
    with pytest.raises(TypeError, match=err_msg):
        exog_long_to_dict(
            data="not_a_dataframe",
            series_id="series_id",
            index="datetime",
            freq="D",
        )


def test_ValueError_when_series_id_not_in_data():
    """
    Raise ValueError if series_id is not in data.
    """
    series_id = "series_id_not_in_data"
    err_msg = f"Column '{series_id}' not found in `data`."
    with pytest.raises(ValueError, match=err_msg):
        exog_long_to_dict(
            data=exog_long,
            series_id=series_id,
            index="datetime",
            freq="D",
        )


def test_ValueError_when_index_not_in_data():
    """
    Raise ValueError if index is not in data.
    """
    index = "series_id_not_in_data"
    err_msg = f"Column '{index}' not found in `data`."
    with pytest.raises(ValueError, match=err_msg):
        exog_long_to_dict(
            data=exog_long,
            series_id="series_id",
            index=index,
            freq="D",
        )


def test_warning_when_exog_are_incomplete_and_dropna_False():
    """
    Raise warning if exogenous variables are incomplete and NaN values are introduced
    after setting the index frequency.
    """
    data = exog_long.copy().reset_index(drop=True)
    data = data.loc[[0, 1] + list(range(3, 30))]
    msg = (
        "Exogenous variables for series 'A' are incomplete. NaNs have been introduced "
        "after setting the frequency."
    )
    with pytest.warns(MissingValuesWarning, match=msg):
        exog_long_to_dict(
            data=data,
            series_id='series_id',
            index='datetime',
            freq='D',
            dropna=False,
        )
