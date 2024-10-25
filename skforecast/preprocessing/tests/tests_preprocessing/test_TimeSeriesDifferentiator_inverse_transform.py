# Unit test TimeSeriesDifferentiator inverse_transform
# ==============================================================================
import re
import pytest
import numpy as np
from skforecast.preprocessing import TimeSeriesDifferentiator

# Fixtures
y = np.array([1, 4, 8, 10, 13, 22, 40, 46], dtype=float)
# y_diff_1 = np.array([np.nan, 3., 4., 2., 3., 9., 18., 6.])
# y_diff_2 = np.array([np.nan, np.nan, 1., -2., 1., 6., 9., -12.])
# y_diff_3 = np.array([np.nan, np.nan, np.nan, -3., 3., 5., 3., -21.])


@pytest.mark.parametrize("order",
                         [1, 2, 3],
                         ids = lambda values: f'order: {values}')
def test_TimeSeriesDifferentiator_inverse_transform(order):
    """
    Test TimeSeriesDifferentiator.inverse_transform method.
    """
    transformer = TimeSeriesDifferentiator(order=order)
    y_diff = transformer.fit_transform(y)
    results = transformer.inverse_transform(y_diff)

    np.testing.assert_array_equal(results, y)