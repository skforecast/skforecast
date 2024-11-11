# Unit test _check_X_numpy_ndarray_1d
# ==============================================================================
import re
import numpy as np
import pandas as pd
import pytest
from skforecast.preprocessing import TimeSeriesDifferentiator


def test_TypeError_decorator_check_X_numpy_ndarray_1d_when_X_not_numpy_ndarray():
    """
    Test TypeError raised when X is not a numpy ndarray.
    """
    y = [1, 2, 3, 4, 5]
    differentiator = TimeSeriesDifferentiator(order=1)    

    err_msg = re.escape(
        ("'X' must be a numpy ndarray. Found <class 'list'>.")
    )
    with pytest.raises(TypeError, match = err_msg): 
        differentiator.fit(y)
    with pytest.raises(TypeError, match = err_msg): 
        differentiator.transform(X=y)
    with pytest.raises(TypeError, match = err_msg): 
        differentiator.fit_transform(y)
    with pytest.raises(TypeError, match = err_msg): 
        differentiator.inverse_transform(y)
    with pytest.raises(TypeError, match = err_msg): 
        differentiator.inverse_transform_next_window(y)


def test_ValueError_decorator_check_X_numpy_ndarray_1d_when_X_not_1D_numpy_ndarray():
    """
    Test ValueError raised when X is not a 1D numpy ndarray.
    """
    y = pd.Series(np.array([1, 2, 3, 4, 5])).to_frame().to_numpy()
    differentiator = TimeSeriesDifferentiator(order=1)    

    err_msg = re.escape(
        ("'X' must be a 1D array. Found 2 dimensions.")
    )
    with pytest.raises(ValueError, match = err_msg): 
        differentiator.fit(y)
    with pytest.raises(ValueError, match = err_msg): 
        differentiator.transform(X=y)
    with pytest.raises(ValueError, match = err_msg): 
        differentiator.fit_transform(y)
    with pytest.raises(ValueError, match = err_msg): 
        differentiator.inverse_transform(y)
