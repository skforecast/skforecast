# Unit test TimeSeriesDifferentiator init
# ==============================================================================
import re
import pytest
from skforecast.preprocessing import TimeSeriesDifferentiator


def test_TypeError_when_order_is_not_int_when_initialization():
    """
    Raise TypeError if order is not an integer when initializing the differentiator.
    """
    order = 1.5

    err_msg = re.escape(
        (f"Parameter 'order' must be an integer greater than 0. "
         f"Found {type(order)}.")
    ) 
    with pytest.raises(TypeError, match = err_msg):
        TimeSeriesDifferentiator(order = order)


def test_ValueError_when_order_is_less_than_1_when_initialization():
    """
    Raise ValueError if order is less than 1 when initializing the differentiator.
    """
    order = 0

    err_msg = re.escape(
        (f"Parameter 'order' must be an integer greater than 0. "
         f"Found {order}.")
    ) 
    with pytest.raises(ValueError, match = err_msg):
        TimeSeriesDifferentiator(order = order)

