# Unit test TimeSeriesDifferentiator init
# ==============================================================================
import re
import pytest
import numpy as np
from skforecast.preprocessing import TimeSeriesDifferentiator

# Fixtures
y = np.array([1, 4, 8, 10, 13, 22, 40, 46], dtype=float)
y_diff_1 = np.array([np.nan, 3., 4., 2., 3., 9., 18., 6.])
y_diff_2 = np.array([np.nan, np.nan, 1., -2., 1., 6., 9., -12.])
y_diff_3 = np.array([np.nan, np.nan, np.nan, -3., 3., 5., 3., -21.])

next_window = np.array([55, 70, 71], dtype=float)
next_window_diff_1 = np.array([9., 15., 1.], dtype=float)
next_window_diff_2 = np.array([3., 6., -14.], dtype=float)
next_window_diff_3 = np.array([15., 3., -20.], dtype=float)
next_window_series = np.concatenate([next_window, next_window], axis=0).reshape(2, -1).T
next_window_diff_1_2d = np.array([[9., 9.], [15., 15.], [1., 1.]], dtype=float)
next_window_diff_2_2d = np.array([[3., 3.], [6., 6.], [-14., -14.]], dtype=float)
next_window_diff_3_2d = np.array([[15., 15.], [3., 3.], [-20., -20.]], dtype=float)


def test_TimeSeriesDifferentiator_validate_params():
    """
    TimeSeriesDifferentiator validate params.
    """

    err_msg = re.escape(
        f"Parameter `order` must be an integer greater than 0. "
        f"Found {type(1.5)}."
    ) 
    with pytest.raises(TypeError, match = err_msg):
        TimeSeriesDifferentiator(order = 1.5)

    err_msg = re.escape(
        "Parameter `order` must be an integer greater than 0. "
        "Found 0."
    ) 
    with pytest.raises(ValueError, match = err_msg):
        TimeSeriesDifferentiator(order = 0)

    err_msg = re.escape(
        f"Parameter `window_size` must be an integer greater than 0. "
        f"Found {type(1.5)}."
    ) 
    with pytest.raises(TypeError, match = err_msg):
        TimeSeriesDifferentiator(order = 1, window_size=1.5)

    err_msg = re.escape(
        "Parameter `window_size` must be an integer greater than 0. "
        "Found 0."
    ) 
    with pytest.raises(ValueError, match = err_msg):
        TimeSeriesDifferentiator(order = 1, window_size=0)


def test_TimeSeriesDifferentiator_fit_order_1():
    """
    Test that TimeSeriesDifferentiator fit method with order 1.
    """
    X = np.arange(10)
    tsd = TimeSeriesDifferentiator(order=1, window_size=3)

    assert tsd.fit(X) == tsd
    assert tsd.initial_values == [0.]
    assert tsd.pre_train_values == [2.]
    assert tsd.last_values == [9.]


def test_TimeSeriesDifferentiator_fit_order_2():
    """
    Test that TimeSeriesDifferentiator fit method with order 2.
    """
    tsd = TimeSeriesDifferentiator(order=2, window_size=3)
    tsd.fit(y)

    assert tsd.fit(y) == tsd
    assert tsd.initial_values == [1., 3.]
    assert tsd.pre_train_values == [4., 4.]
    assert tsd.last_values == [46., 6.]


@pytest.mark.parametrize("order, expected",
                         [(1, y_diff_1),
                          (2, y_diff_2),
                          (3, y_diff_3)],
                         ids = lambda values: f'order, expected: {values}')
def test_TimeSeriesDifferentiator_transform(order, expected):
    """
    Test TimeSeriesDifferentiator transform method.
    """
    transformer = TimeSeriesDifferentiator(order=order)
    _ = transformer.fit(y)
    results = transformer.transform(y)

    np.testing.assert_array_equal(results, expected)


@pytest.mark.parametrize("order, expected",
                         [(1, y_diff_1),
                          (2, y_diff_2),
                          (3, y_diff_3)],
                         ids = lambda values: f'order, expected: {values}')
def test_TimeSeriesDifferentiator_fit_transform(order, expected):
    """
    Test TimeSeriesDifferentiator fit_transform method.
    """
    transformer = TimeSeriesDifferentiator(order=order)
    results = transformer.fit_transform(y)

    np.testing.assert_array_equal(results, expected)


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


def test_TimeSeriesDifferentiator_inverse_transform_ValueError_no_window_size():
    """
    Test TimeSeriesDifferentiator inverse_transform_training ValueError when
    window_size is not provided.
    """
    transformer = TimeSeriesDifferentiator(order = 1)

    err_msg = re.escape(
        "The `window_size` parameter must be set before fitting the "
        "transformer to revert the differentiation of the training "
        "time series."
    ) 
    with pytest.raises(ValueError, match = err_msg):
        transformer.inverse_transform_training(y_diff_1)        


@pytest.mark.parametrize("order, y_diff",
                         [(1, y_diff_1),
                          (2, y_diff_2),
                          (3, y_diff_3)],
                         ids = lambda values: f'order, y_diff: {values}')
def test_TimeSeriesDifferentiator_inverse_transform_training(order, y_diff):
    """
    Test TimeSeriesDifferentiator inverse_transform_training method.
    """
    window_size = 2 + order  # 2 lags + differentiation
    y_diff_train = y_diff[window_size:]  # Same as y_train predicted

    transformer = TimeSeriesDifferentiator(order=order, window_size=window_size)
    transformer.fit_transform(y)
    results = transformer.inverse_transform_training(y_diff_train)
    
    y_train_expected = y[window_size:]  # No differentiated

    assert len(y_train_expected) == len(y_diff_train)
    np.testing.assert_array_equal(results, y_train_expected)


@pytest.mark.parametrize("order, next_window_diff, expected",
                         [(1, next_window_diff_1, next_window),
                          (2, next_window_diff_2, next_window),
                          (3, next_window_diff_3, next_window)],
                         ids = lambda values: f'order, next_window_diff, expected: {values}')
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
                         ids = lambda values: f'order, next_window_diff, expected: {values}')
def test_TimeSeriesDifferentiator_inverse_transform_next_window_2d(order, next_window_diff, expected):
    """
    Test TimeSeriesDifferentiator.inverse_transform_next_window method with 2d series.
    """
    transformer = TimeSeriesDifferentiator(order=order)
    transformer.fit_transform(y)
    results = transformer.inverse_transform_next_window(next_window_diff)
    
    np.testing.assert_array_equal(results, expected)
