# Unit test reshape_exog_long_to_dict
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
import platform
from ...preprocessing import reshape_exog_long_to_dict
from ....exceptions import MissingValuesWarning

# Fixtures
from .fixtures_preprocessing import exog_A, exog_B, exog_C, n_exog_A, n_exog_B, n_exog_C
from .fixtures_preprocessing import exog_long
exog_long_multiindex = exog_long.set_index(["series_id", "datetime"])


def test_TypeError_when_data_is_not_dataframe():
    """
    Raise TypeError if data is not a pandas DataFrame.
    """
    err_msg = re.escape("`data` must be a pandas DataFrame.")
    with pytest.raises(TypeError, match=err_msg):
        reshape_exog_long_to_dict(
            data="not_a_dataframe",
            series_id="series_id",
            index="datetime",
            freq="D",
            drop_all_nan_cols=True,
            consolidate_dtypes=True,
        )


def test_ValueError_reshape_exog_long_to_dict_when_arguments_series_id_index_not_provided():
    """
    Check that ValueError is raised when the input dataframe does not have MultiIndex and the
    arguments `series_id`, `index` and `values` are not provided
    """
    err_msg = re.escape(
        "Arguments `series_id`, and `index` must be "
        "specified when the input DataFrame does not have a MultiIndex. "
        "Please provide a value for each of these arguments."
    )
    with pytest.raises(ValueError, match=err_msg):
        reshape_exog_long_to_dict(data=exog_long, freq="D")


def test_ValueError_when_series_id_not_in_data():
    """
    Raise ValueError if series_id is not in data.
    """
    series_id = "series_id_not_in_data"

    err_msg = re.escape(f"Column '{series_id}' not found in `data`.")
    with pytest.raises(ValueError, match=err_msg):
        reshape_exog_long_to_dict(
            data=exog_long,
            series_id=series_id,
            index="datetime",
            freq="D",
            drop_all_nan_cols=True,
            consolidate_dtypes=True,
        )


def test_ValueError_when_index_not_in_data():
    """
    Raise ValueError if index is not in data.
    """
    index = "series_id_not_in_data"
    
    err_msg = re.escape(f"Column '{index}' not found in `data`.")
    with pytest.raises(ValueError, match=err_msg):
        reshape_exog_long_to_dict(
            data=exog_long,
            series_id="series_id",
            index=index,
            freq="D",
            drop_all_nan_cols=True,
            consolidate_dtypes=True,
        )


def test_check_output_reshape_exog_long_to_dict_dropna_False():
    """
    Check output of reshape_exog_long_to_dict with dropna=False.
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
        expected[k]['exog_2'] = expected[k]['exog_2'].astype(object)
        expected[k]['exog_3'] = expected[k]['exog_3'].astype(float)

    results = reshape_exog_long_to_dict(
        data=exog_long,
        series_id="series_id",
        index="datetime",
        freq="D",
        drop_all_nan_cols=False,
        consolidate_dtypes=True,
    )

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


def test_check_output_reshape_exog_long_to_dict_dropna_False_when_multiindex():
    """
    Check output of reshape_exog_long_to_dict with dropna=False when input data is a MultiIndex DataFrame.
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
        expected[k]['exog_2'] = expected[k]['exog_2'].astype(object)
        expected[k]['exog_3'] = expected[k]['exog_3'].astype(float)

    results = reshape_exog_long_to_dict(
        data=exog_long_multiindex,
        freq="D",
        drop_all_nan_cols=False,
        consolidate_dtypes=True,
    )

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


def test_check_output_reshape_exog_long_to_dict_dropna_True():
    """
    Check output of reshape_exog_long_to_dict with dropna=True.
    """
    expected = {
        "A": exog_A.set_index("datetime").asfreq("D").drop(columns="series_id"),
        "B": exog_B.set_index("datetime").asfreq("D").drop(columns="series_id"),
        "C": exog_C.set_index("datetime").asfreq("D").drop(columns="series_id"),
    }

    for k in expected.keys():
        expected[k].index.name = None

    results = reshape_exog_long_to_dict(
        data=exog_long,
        series_id="series_id",
        index="datetime",
        freq="D",
        drop_all_nan_cols=True,
        consolidate_dtypes=True,
    )

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k], check_dtype=False)


def test_check_output_reshape_exog_long_to_dict_dropna_True_when_multiindex():
    """
    Check output of reshape_exog_long_to_dict with dropna=True when input data is a MultiIndex DataFrame.
    """
    expected = {
        "A": exog_A.set_index("datetime").asfreq("D").drop(columns="series_id"),
        "B": exog_B.set_index("datetime").asfreq("D").drop(columns="series_id"),
        "C": exog_C.set_index("datetime").asfreq("D").drop(columns="series_id"),
    }

    for k in expected.keys():
        expected[k].index.name = None

    results = reshape_exog_long_to_dict(
        data=exog_long_multiindex,
        freq="D",
        drop_all_nan_cols=True,
        consolidate_dtypes=True,
    )

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k], check_dtype=False)


def test_MissingValuesWarning_when_exog_are_incomplete_and_dropna_False():
    """
    Raise MissingValuesWarning if exogenous variables are incomplete and NaN values are introduced
    after setting the index frequency.
    """
    data = exog_long.copy().reset_index(drop=True)
    data = data.loc[[0, 1] + list(range(3, 30))]

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
    expected['A'].loc['2020-01-03', 'exog_1'] = np.nan

    for k in expected.keys():
        expected[k]['exog_1'] = expected[k]['exog_1'].astype(float)
        expected[k]['exog_2'] = expected[k]['exog_2'].astype(object)
        expected[k]['exog_3'] = expected[k]['exog_3'].astype(float)
    
    msg = (
        "Exogenous variables for series 'A' are incomplete. NaNs have been introduced "
        "after setting the frequency."
    )
    with pytest.warns(MissingValuesWarning, match=msg):
        results = reshape_exog_long_to_dict(
            data=data,
            series_id='series_id',
            index='datetime',
            freq='D',
            drop_all_nan_cols=False,
            consolidate_dtypes=True,
        )

    for k in expected.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


def test_reshape_exog_long_to_dict_output_when_npnan_are_added_in_integer_columns_and_consolidate_true():
    """
    Test the output of the function reshape_exog_long_to_dict when np.nan are added in integer columns
    these columns should be converted to float
    """
    exog_series_1 = pd.DataFrame({
        'exog_1': np.random.normal(0, 1, 10),
        'exog_2': np.random.choice(["A", "B", "C"], 10),
        'exog_3': np.random.randint(0, 10, 10),
        'series': 'series_1',
    }, index = pd.date_range(start='1-1-2000', periods=10, freq='D'))
    exog_series_2 = pd.DataFrame({
        'exog_1': np.random.normal(0, 1, 10),
        'exog_2': np.random.choice(["A", "B"], 10),
        'exog_3': np.random.randint(0, 10, 10),
        'series': 'series_2',
    }, index = pd.date_range(start='1-1-2000', periods=10, freq='D'))
    
    exog_long = (
        pd.concat([exog_series_1, exog_series_2], axis=0)
        .reset_index()
        .rename(columns={"index": "datetime"})
    )
    exog_long = exog_long.loc[
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], :
    ].copy()
    exog_long["exog_2"] = exog_long["exog_2"].astype("category")
    exog_long["series"] = exog_long["series"].astype("category")

    exog_dict = reshape_exog_long_to_dict(
        data=exog_long,
        series_id="series",
        index="datetime",
        freq="D",
        consolidate_dtypes=True,
        suppress_warnings=True,
    )

    pd.testing.assert_series_equal(exog_dict['series_1'].dtypes, exog_dict['series_2'].dtypes)


def test_reshape_exog_long_to_dict_output_when_npnan_are_added_in_integer_columns_and_consolidate_false():
    """
    Test the output of the function reshape_exog_long_to_dict when np.nan are added in integer columns
    these columns should be converted to float
    """
    exog_series_1 = pd.DataFrame({
        'exog_1': np.random.normal(0, 1, 10),
        'exog_2': np.random.choice(["A", "B", "C"], 10),
        'exog_3': np.random.randint(0, 10, 10),
        'series': 'series_1',
    }, index = pd.date_range(start='1-1-2000', periods=10, freq='D'))
    exog_series_2 = pd.DataFrame({
        'exog_1': np.random.normal(0, 1, 10),
        'exog_2': np.random.choice(["A", "B"], 10),
        'exog_3': np.random.randint(0, 10, 10),
        'series': 'series_2',
    }, index = pd.date_range(start='1-1-2000', periods=10, freq='D'))
    
    exog_long = (
        pd.concat([exog_series_1, exog_series_2], axis=0)
        .reset_index()
        .rename(columns={"index": "datetime"})
    )
    exog_long = exog_long.loc[
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], :
    ].copy()
    exog_long["exog_2"] = exog_long["exog_2"].astype("category")
    exog_long["series"] = exog_long["series"].astype("category")
    
    exog_dict = reshape_exog_long_to_dict(
        data=exog_long,
        series_id="series",
        index="datetime",
        freq="D",
        consolidate_dtypes=False,
        suppress_warnings=True,
    )
   
    assert exog_dict['series_1'].dtypes.astype(str).to_list() == ['float64', 'category', 'float64']

    if platform.system() == 'Windows':
        assert exog_dict['series_2'].dtypes.astype(str).to_list() == ['float64', 'category', 'int32']
    else:
        assert exog_dict['series_2'].dtypes.astype(str).to_list() == ['float64', 'category', 'int64']


@pytest.mark.parametrize(
    "fill_value, expected_fill",
    [(None, np.nan), (-999., -999.)],
    ids=lambda x: f"fill_value: {x}",
)
def test_check_output_reshape_exog_long_to_dict_with_fill_value(fill_value, expected_fill):
    """
    Check output of reshape_exog_long_to_dict with fill_value parameter.
    """
    data = pd.DataFrame({
        'series_id': ['A'] * 3 + ['B'] * 3,
        'datetime': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-04'] * 2),
        'exog_1': [1.0, 2.0, 4.0, 10.0, 20.0, 40.0],
        'exog_2': [0.1, 0.2, 0.4, 1.0, 2.0, 4.0]
    })

    warn_match = (
        "NaNs have been introduced"
        if fill_value is None
        else f"Missing values have been filled with {fill_value}"
    )
    with pytest.warns(MissingValuesWarning, match=re.escape(warn_match)):
        results = reshape_exog_long_to_dict(
            data=data,
            series_id='series_id',
            index='datetime',
            freq='D',
            fill_value=fill_value,
            suppress_warnings=False,
        )

    expected_A = pd.DataFrame(
        {
            'exog_1': [1.0, 2.0, expected_fill, 4.0],
            'exog_2': [0.1, 0.2, expected_fill, 0.4]
        },
        index=pd.date_range('2020-01-01', periods=4, freq='D')
    )
    expected_B = pd.DataFrame(
        {
            'exog_1': [10.0, 20.0, expected_fill, 40.0],
            'exog_2': [1.0, 2.0, expected_fill, 4.0]
        },
        index=pd.date_range('2020-01-01', periods=4, freq='D')
    )

    pd.testing.assert_frame_equal(results['A'], expected_A)
    pd.testing.assert_frame_equal(results['B'], expected_B)


@pytest.mark.parametrize(
    "fill_value, expected_fill",
    [(None, np.nan), (0., 0.)],
    ids=lambda x: f"fill_value: {x}",
)
def test_check_output_reshape_exog_long_to_dict_with_fill_value_when_multiindex(fill_value, expected_fill):
    """
    Check output of reshape_exog_long_to_dict with fill_value parameter when
    input data is a MultiIndex DataFrame.
    """
    data = pd.DataFrame({
        'series_id': ['A'] * 3 + ['B'] * 3,
        'datetime': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-04'] * 2),
        'exog_1': [1.0, 2.0, 4.0, 10.0, 20.0, 40.0],
        'exog_2': [0.1, 0.2, 0.4, 1.0, 2.0, 4.0]
    })
    data = data.set_index(['series_id', 'datetime'])

    warn_match = (
        "NaNs have been introduced"
        if fill_value is None
        else f"Missing values have been filled with {fill_value}"
    )
    with pytest.warns(MissingValuesWarning, match=re.escape(warn_match)):
        results = reshape_exog_long_to_dict(
            data=data,
            freq='D',
            fill_value=fill_value,
        )

    expected_A = pd.DataFrame(
        {
            'exog_1': [1.0, 2.0, expected_fill, 4.0],
            'exog_2': [0.1, 0.2, expected_fill, 0.4]
        },
        index=pd.date_range('2020-01-01', periods=4, freq='D')
    )
    expected_B = pd.DataFrame(
        {
            'exog_1': [10.0, 20.0, expected_fill, 40.0],
            'exog_2': [1.0, 2.0, expected_fill, 4.0]
        },
        index=pd.date_range('2020-01-01', periods=4, freq='D')
    )

    pd.testing.assert_frame_equal(results['A'], expected_A)
    pd.testing.assert_frame_equal(results['B'], expected_B)


@pytest.mark.parametrize(
    "fill_value",
    [None, 0.],
    ids=lambda x: f"fill_value: {x}",
)
def test_check_output_reshape_exog_long_to_dict_with_fill_value_string_and_categorical_columns(
    fill_value,
):
    """
    Check output of reshape_exog_long_to_dict with fill_value parameter when
    exogenous variables include string and categorical columns. When fill_value
    is not None, it is applied only to numeric columns and the warning message
    includes information about non-numeric columns still containing NaN.
    """
    data = pd.DataFrame({
        'series_id': ['A'] * 3 + ['B'] * 3,
        'datetime': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-04'] * 2),
        'exog_num': [1.0, 2.0, 4.0, 10.0, 20.0, 40.0],
        'exog_str': ['x', 'y', 'z', 'a', 'b', 'c'],
        'exog_cat': pd.Categorical(['low', 'mid', 'high'] * 2),
    })

    if fill_value is not None:
        msg = re.escape(
            "in numeric columns only. Non-numeric columns"
        )
        with pytest.warns(MissingValuesWarning, match=msg):
            results = reshape_exog_long_to_dict(
                data=data,
                series_id='series_id',
                index='datetime',
                freq='D',
                fill_value=fill_value,
                suppress_warnings=False,
            )

        expected_A = pd.DataFrame(
            {
                'exog_num': [1.0, 2.0, fill_value, 4.0],
                'exog_str': ['x', 'y', np.nan, 'z'],
                'exog_cat': pd.Categorical(
                    ['low', 'mid', np.nan, 'high'],
                    categories=['high', 'low', 'mid']
                ),
            },
            index=pd.date_range('2020-01-01', periods=4, freq='D')
        )
        expected_B = pd.DataFrame(
            {
                'exog_num': [10.0, 20.0, fill_value, 40.0],
                'exog_str': ['a', 'b', np.nan, 'c'],
                'exog_cat': pd.Categorical(
                    ['low', 'mid', np.nan, 'high'],
                    categories=['high', 'low', 'mid']
                ),
            },
            index=pd.date_range('2020-01-01', periods=4, freq='D')
        )

        pd.testing.assert_frame_equal(results['A'], expected_A)
        pd.testing.assert_frame_equal(results['B'], expected_B)
    else:
        warn_match = "NaNs have been introduced"
        with pytest.warns(MissingValuesWarning, match=re.escape(warn_match)):
            results = reshape_exog_long_to_dict(
                data=data,
                series_id='series_id',
                index='datetime',
                freq='D',
                fill_value=fill_value,
                suppress_warnings=False,
            )

        expected_A = pd.DataFrame(
            {
                'exog_num': [1.0, 2.0, np.nan, 4.0],
                'exog_str': ['x', 'y', np.nan, 'z'],
                'exog_cat': pd.Categorical(
                    ['low', 'mid', np.nan, 'high'],
                    categories=['high', 'low', 'mid']
                ),
            },
            index=pd.date_range('2020-01-01', periods=4, freq='D')
        )
        expected_B = pd.DataFrame(
            {
                'exog_num': [10.0, 20.0, np.nan, 40.0],
                'exog_str': ['a', 'b', np.nan, 'c'],
                'exog_cat': pd.Categorical(
                    ['low', 'mid', np.nan, 'high'],
                    categories=['high', 'low', 'mid']
                ),
            },
            index=pd.date_range('2020-01-01', periods=4, freq='D')
        )

        pd.testing.assert_frame_equal(results['A'], expected_A)
        pd.testing.assert_frame_equal(results['B'], expected_B)


@pytest.mark.parametrize(
    "fill_value",
    [None, 0.],
    ids=lambda x: f"fill_value: {x}",
)
def test_check_output_reshape_exog_long_to_dict_with_fill_value_string_and_categorical_columns_when_multiindex(
    fill_value,
):
    """
    Check output of reshape_exog_long_to_dict with fill_value parameter when
    input data is a MultiIndex DataFrame with string and categorical columns.
    When fill_value is not None, it is applied only to numeric columns and
    the warning message includes information about non-numeric columns.
    """
    data = pd.DataFrame({
        'series_id': ['A'] * 3 + ['B'] * 3,
        'datetime': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-04'] * 2),
        'exog_num': [1.0, 2.0, 4.0, 10.0, 20.0, 40.0],
        'exog_str': ['x', 'y', 'z', 'a', 'b', 'c'],
        'exog_cat': pd.Categorical(['low', 'mid', 'high'] * 2),
    })
    data = data.set_index(['series_id', 'datetime'])

    if fill_value is not None:
        msg = re.escape(
            "in numeric columns only. Non-numeric columns"
        )
        with pytest.warns(MissingValuesWarning, match=msg):
            results = reshape_exog_long_to_dict(
                data=data,
                freq='D',
                fill_value=fill_value,
                suppress_warnings=False,
            )

        expected_A = pd.DataFrame(
            {
                'exog_num': [1.0, 2.0, fill_value, 4.0],
                'exog_str': ['x', 'y', np.nan, 'z'],
                'exog_cat': pd.Categorical(
                    ['low', 'mid', np.nan, 'high'],
                    categories=['high', 'low', 'mid']
                ),
            },
            index=pd.date_range('2020-01-01', periods=4, freq='D')
        )
        expected_B = pd.DataFrame(
            {
                'exog_num': [10.0, 20.0, fill_value, 40.0],
                'exog_str': ['a', 'b', np.nan, 'c'],
                'exog_cat': pd.Categorical(
                    ['low', 'mid', np.nan, 'high'],
                    categories=['high', 'low', 'mid']
                ),
            },
            index=pd.date_range('2020-01-01', periods=4, freq='D')
        )

        pd.testing.assert_frame_equal(results['A'], expected_A)
        pd.testing.assert_frame_equal(results['B'], expected_B)
    else:
        warn_match = "NaNs have been introduced"
        with pytest.warns(MissingValuesWarning, match=re.escape(warn_match)):
            results = reshape_exog_long_to_dict(
                data=data,
                freq='D',
                fill_value=fill_value,
                suppress_warnings=False,
            )

        expected_A = pd.DataFrame(
            {
                'exog_num': [1.0, 2.0, np.nan, 4.0],
                'exog_str': ['x', 'y', np.nan, 'z'],
                'exog_cat': pd.Categorical(
                    ['low', 'mid', np.nan, 'high'],
                    categories=['high', 'low', 'mid']
                ),
            },
            index=pd.date_range('2020-01-01', periods=4, freq='D')
        )
        expected_B = pd.DataFrame(
            {
                'exog_num': [10.0, 20.0, np.nan, 40.0],
                'exog_str': ['a', 'b', np.nan, 'c'],
                'exog_cat': pd.Categorical(
                    ['low', 'mid', np.nan, 'high'],
                    categories=['high', 'low', 'mid']
                ),
            },
            index=pd.date_range('2020-01-01', periods=4, freq='D')
        )

        pd.testing.assert_frame_equal(results['A'], expected_A)
        pd.testing.assert_frame_equal(results['B'], expected_B)
