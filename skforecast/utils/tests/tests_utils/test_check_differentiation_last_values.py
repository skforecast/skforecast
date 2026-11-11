# Unit test _check_differentiation_last_values
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import _check_differentiation_last_values

# Fixtures
index = pd.date_range(start='2020-01-01', periods=5, freq='D')


@pytest.mark.parametrize(
    "values, order, series, dates, of_series, last_values",
    [([1., 2., 3., 4., np.nan], 1, None, ['2020-01-05 00:00:00'], "",
      "last value"),
     ([1., 2., 3., np.nan, 5.], 2, 'l1', ['2020-01-04 00:00:00'], " 'l1'",
      "last 2 values"),
     ([1., 2., np.nan, np.nan, 5.], 3, None,
      ['2020-01-03 00:00:00', '2020-01-04 00:00:00'], "", "last 3 values")],
    ids=["order_1", "order_2_series", "order_3_two_dates"]
)
def test_check_differentiation_last_values_ValueError_when_NaN_in_last_values(
    values, order, series, dates, of_series, last_values
):
    """
    Test ValueError is raised, with the series, the dates and the remedy, when
    any of the last `order` values of the last window is NaN.
    """
    err_msg = re.escape(
        f"The series{of_series} (or `last_window`) has missing values (NaN) at "
        f"{dates}. With `differentiation`, the predictions are reverted to the "
        f"original scale from the {last_values} of the series, so all of them "
        f"would be NaN, with any estimator. Fill the missing values of the "
        f"series (or `last_window`)."
    )
    with pytest.raises(ValueError, match=err_msg):
        _check_differentiation_last_values(
            last_window_values = np.array(values),
            last_window_index  = index,
            order              = order,
            series             = series
        )


def test_check_differentiation_last_values_no_error_when_NaN_before_last_values():
    """
    Test no error is raised when the last window has NaN values only before
    its last `order` values, which the differentiation does not use to revert
    the predictions.
    """
    _check_differentiation_last_values(
        last_window_values = np.array([np.nan, 2., np.nan, 4., 5.]),
        last_window_index  = index,
        order              = 2
    )
