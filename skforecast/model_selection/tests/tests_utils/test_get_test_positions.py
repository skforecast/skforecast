# Unit test _get_test_positions
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.model_selection._utils import _get_test_positions


def test_get_test_positions_output():
    """
    Test that the positions in `y` of the index values of the test set are
    returned, also when the test set skips some dates (rows left out).
    """
    y = pd.Series(
        np.arange(10.), index=pd.date_range('2020-01-01', periods=10, freq='D'), name='y'
    )
    test_index = y.index[[5, 6, 8, 9]]

    results = _get_test_positions(y=y, test_index=test_index)

    np.testing.assert_array_equal(results, np.array([5, 6, 8, 9]))


def test_get_test_positions_ValueError_when_date_not_in_series():
    """
    Test that a ValueError is raised when a date of the test set is not in the
    index of the series, instead of reading another value of the series.
    """
    y = pd.Series(
        np.arange(10.), index=pd.date_range('2020-01-01', periods=10, freq='D'), name='y'
    )
    test_index = pd.date_range('2020-01-09', periods=3, freq='D')

    err_msg = re.escape(
        "1 dates of the test set are not in the index of the series 'y', so its "
        "true values cannot be read from it. First missing date: 2020-01-11 00:00:00."
    )
    with pytest.raises(ValueError, match=err_msg):
        _get_test_positions(y=y, test_index=test_index)
