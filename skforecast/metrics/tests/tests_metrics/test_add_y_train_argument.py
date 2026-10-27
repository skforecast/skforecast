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
    np.testing.assert_array_almost_equal(result, np.array([11, 12, 13]))

def test_add_y_train_argument_returns_same_function_when_already_processed():
    """
    Test that a function already processed by add_y_train_argument is returned
    as it is, so that `_needs_y_train` is not overwritten when it is processed
    again (for example, by a search and then by the backtesting).
    """
    def test_func(x):
        return x

    def test_func_y_train(x, y_train):
        return x + y_train

    wrapped_func = add_y_train_argument(test_func)
    wrapped_func_y_train = add_y_train_argument(test_func_y_train)

    assert add_y_train_argument(wrapped_func) is wrapped_func
    assert wrapped_func._needs_y_train is False
    assert add_y_train_argument(wrapped_func_y_train) is wrapped_func_y_train
    assert wrapped_func_y_train._needs_y_train is True
