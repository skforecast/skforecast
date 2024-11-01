# Unit test add_y_train_argument
# ==============================================================================
import inspect
import numpy as np
from skforecast.metrics import add_y_train_argument


def test_add_y_train_argument_func_no_y_train():
    """
    Test add_y_train_argument when the function does not have the 'y_train' argument.
    """
    def test_func(x):
        return x

    wrapped_func = add_y_train_argument(test_func)

    # Check if the wrapped function has the 'y_train' argument
    assert 'y_train' in inspect.signature(wrapped_func).parameters

    # Test the wrapped function with 'y_train' argument
    result = wrapped_func(10, y_train=[1, 2, 3])
    assert result == 10

    # Test the wrapped function without 'y_train' argument
    result = wrapped_func(10)
    assert result == 10


def test_add_y_train_argument_func_with_y_train():
    """ 
    Test add_y_train_argument when the function already has the 'y_train' argument.
    """
    def test_func(x, y_train):
        return x + y_train
    
    wrapped_func = add_y_train_argument(test_func)

    # Check if the wrapped function has the 'y_train' argument
    assert 'y_train' in inspect.signature(wrapped_func).parameters

    # Test the wrapped function with 'y_train' argument
    result = wrapped_func(10, y_train=np.array([1, 2, 3]))
    np.testing.assert_array_equal(result, np.array([11, 12, 13]))