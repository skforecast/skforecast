# Unit test _split_long_format
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import _split_long_format

# Fixtures
data_long = pd.DataFrame(
    {
        'value': [1., 2., 3., 4., 5., 6.],
        'other': [10, 20, 30, 40, 50, 60]
    },
    index=pd.MultiIndex.from_arrays(
        [
            np.array(['b', 'b', 1, 1, None, 'b'], dtype=object),
            pd.DatetimeIndex(
                ['2020-01-01', '2020-01-02', '2020-01-01', 'NaT', '2020-01-01', '2020-01-03']
            )
        ],
        names=['series_id', 'datetime']
    )
)


def test_split_long_format_ValueError_when_MultiIndex_has_not_two_levels():
    """
    Test ValueError is raised when the MultiIndex of the data has more than
    two levels.
    """
    data = data_long.set_index('other', append=True)

    err_msg = re.escape(
        "Long-format data must have a MultiIndex with two levels: the "
        "series IDs and the dates. Got 3 levels."
    )
    with pytest.raises(ValueError, match=err_msg):
        _split_long_format(data)


@pytest.mark.parametrize(
    "as_series", [False, True], ids=lambda as_series: f'as_series: {as_series}'
)
def test_split_long_format_output_when_ids_of_mixed_types_missing_ids_and_missing_dates(
    as_series
):
    """
    Test _split_long_format with IDs of mixed types (numbers are sorted before
    text, as groupby does), a row with a missing ID, which is dropped, and a
    missing date, which is kept as NaT. A DataFrame gives a DataFrame for each
    series and a Series gives a Series named after its ID.
    """
    data = data_long['value'] if as_series else data_long
    data_copy = data.copy()

    results = _split_long_format(data)

    expected = {
        1: pd.DataFrame(
               {'value': [3., 4.], 'other': [30, 40]},
               index = pd.DatetimeIndex(['2020-01-01', 'NaT'])
           ),
        'b': pd.DataFrame(
                 {'value': [1., 2., 6.], 'other': [10, 20, 60]},
                 index = pd.DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03'])
             ),
    }

    assert list(results.keys()) == [1, 'b']
    for k in expected:
        if as_series:
            pd.testing.assert_series_equal(results[k], expected[k]['value'].rename(k))
        else:
            pd.testing.assert_frame_equal(results[k], expected[k])
    if as_series:
        pd.testing.assert_series_equal(data, data_copy)
    else:
        pd.testing.assert_frame_equal(data, data_copy)


@pytest.mark.parametrize(
    "data",
    [
        data_long.iloc[:0],
        data_long.iloc[[4]],
    ],
    ids=['no rows', 'only rows with a missing ID']
)
def test_split_long_format_output_when_no_rows_with_an_ID(data):
    """
    Test _split_long_format returns an empty dictionary when the data has no
    rows, or when all of them have a missing ID.
    """
    results = _split_long_format(data)

    assert results == {}
