# Unit test TimeSeriesDifferentiator inverse_transform_next_window
# ==============================================================================
import re
import pytest
import numpy as np
from skforecast.preprocessing import TimeSeriesDifferentiator

# Fixtures
y = np.array([1, 4, 8, 10, 13, 22, 40, 46], dtype=float)
next_window = np.array([55, 70, 71], dtype=float)
next_window_diff_1 = np.array([9., 15., 1.], dtype=float)
next_window_diff_2 = np.array([3., 6., -14.], dtype=float)
next_window_diff_3 = np.array([15., 3., -20.], dtype=float)
next_window_series = np.concatenate([next_window, next_window], axis=0).reshape(2, -1).T
next_window_diff_1_2d = np.array([[9., 9.], [15., 15.], [1., 1.]], dtype=float)
next_window_diff_2_2d = np.array([[3., 3.], [6., 6.], [-14., -14.]], dtype=float)
next_window_diff_3_2d = np.array([[15., 15.], [3., 3.], [-20., -20.]], dtype=float)


@pytest.mark.parametrize("order, next_window_diff, expected",
                         [(1, next_window_diff_1, next_window),
                          (2, next_window_diff_2, next_window),
                          (3, next_window_diff_3, next_window)],
                         ids = lambda values: f'order: {values}')
def test_TimeSeriesDifferentiator_inverse_transform_next_window(order, next_window_diff, expected):
    """
    Test TimeSeriesDifferentiator.inverse_transform_next_window method.
    """
    transformer = TimeSeriesDifferentiator(order=order)
    transformer.fit_transform(y)
    results = transformer.inverse_transform_next_window(next_window_diff)
    
    np.testing.assert_array_equal(results, expected)


@pytest.mark.parametrize("order, next_window_diff, expected",
                         [(1, next_window_diff_1_2d, next_window_series),
                          (2, next_window_diff_2_2d, next_window_series),
                          (3, next_window_diff_3_2d, next_window_series)],
                         ids = lambda values: f'order: {values}')
def test_TimeSeriesDifferentiator_inverse_transform_next_window_2d(order, next_window_diff, expected):
    """
    Test TimeSeriesDifferentiator.inverse_transform_next_window method with 2d series.
    """
    transformer = TimeSeriesDifferentiator(order=order)
    transformer.fit_transform(y)
    results = transformer.inverse_transform_next_window(next_window_diff)
    
    np.testing.assert_array_equal(results, expected)
