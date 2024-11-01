# Unit test _check_interval
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.utils.utils import check_interval


def test_check_interval_TypeError_when_interval_is_not_a_list():
    """
    Check `TypeError` is raised when `interval` is not a `list`.
    """
    err_msg = re.escape(
        ('`interval` must be a `list`. For example, interval of 95% '
         'should be as `interval = [2.5, 97.5]`.')
    )
    with pytest.raises(TypeError, match = err_msg):
        check_interval(interval = 'not_a_list')


def test_check_interval_ValueError_when_interval_len_is_not_2():
    """
    Check `ValueError` is raised when `interval` len is not 2.
    """
    err_msg = re.escape(
                ('`interval` must contain exactly 2 values, respectively the '
                 'lower and upper interval bounds. For example, interval of 95% '
                 'should be as `interval = [2.5, 97.5]`.')
              )
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = [2.5, 50.0, 97.5])


def test_check_interval_ValueError_when_interval_lower_bound_less_than_0():
    """
    Check `ValueError` is raised when lower bound is less than 0.
    """
    err_msg = re.escape(
                ('Lower interval bound (-1.0) must be >= 0 and < 100.')
              )
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = [-1.0, 97.5])


@pytest.mark.parametrize("interval", 
                         [[100.0, 97.5], [101.0, 97.5]], 
                         ids = lambda value : f'interval: {value}' )
def test_check_interval_ValueError_when_interval_lower_bound_greater_than_or_equal_to_100(interval):
    """
    Check `ValueError` is raised when lower bound is greater than or equal to 100.
    """
    err_msg = re.escape(
                (f'Lower interval bound ({interval[0]}) must be >= 0 and < 100.')
              )
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = interval)


@pytest.mark.parametrize("interval", 
                         [[2.5, 0.0], [2.5, -1.0]], 
                         ids = lambda value : f'interval: {value}' )
def test_check_interval_ValueError_when_interval_upper_bound_less_than_or_equal_to_0(interval):
    """
    Check `ValueError` is raised when upper bound is less than or equal to 0.
    """
    err_msg = re.escape(
                (f'Upper interval bound ({interval[1]}) must be > 0 and <= 100.')
              )
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = interval)


def test_check_interval_ValueError_when_interval_upper_bound_greater_than_100():
    """
    Check `ValueError` is raised when upper bound is greater than 100.
    """
    err_msg = re.escape(
                ('Upper interval bound (101.0) must be > 0 and <= 100.')
              )
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = [2.5, 101.0])


@pytest.mark.parametrize("interval", 
                         [[2.5, 2.5], [2.5, 2.0]], 
                         ids = lambda value : f'interval: {value}' )
def test_check_interval_ValueError_when_interval_lower_bound_greater_than_or_equal_to_upper_bound(interval):
    """
    Check `ValueError` is raised when lower bound is greater than or equal to
    upper bound.
    """
    err_msg = re.escape(
                (f'Lower interval bound ({interval[0]}) must be less than the '
                 f'upper interval bound ({interval[1]}).')
              )
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = interval)


def test_check_interval_TypeError_when_quantiles_is_not_a_list():
    """
    Check `TypeError` is raised when `quantiles` is not a `list`.
    """
    err_msg = re.escape(
                ("`quantiles` must be a `list`. For example, quantiles 0.05, "
                 "0.5, and 0.95 should be as `quantiles = [0.05, 0.5, 0.95]`.")
            )
    with pytest.raises(TypeError, match = err_msg):
        check_interval(quantiles = 'not_a_list')


@pytest.mark.parametrize("quantiles", 
                         [[-0.01, 0.01, 0.5], [0., 1., 1.1], [-2], [-2, 2]], 
                         ids = lambda value : f'quantiles: {value}' )
def test_check_interval_ValueError_when_elements_in_quantiles_are_out_of_bounds(quantiles):
    """
    Check `ValueError` is raised when any element in `quantiles` is 
    not between 0 and 100.
    """
    err_msg = re.escape(
                ("All elements in `quantiles` must be >= 0 and <= 1.")
            )
    with pytest.raises(ValueError, match = err_msg):
        check_interval(quantiles=quantiles)


def test_check_interval_TypeError_when_alpha_is_not_float():
    """
    Check `TypeError` is raised when `alpha` is not a `float`.
    """
    err_msg = re.escape(
                ('`alpha` must be a `float`. For example, interval of 95% '
                 'should be as `alpha = 0.05`.')
            )
    with pytest.raises(TypeError, match = err_msg):
        check_interval(alpha = 'not_a_float')


@pytest.mark.parametrize("alpha", 
                         [1., 0.], 
                         ids = lambda value : f'alpha: {value}' )
def test_check_interval_ValueError_when_alpha_is_out_of_bounds(alpha):
    """
    Check `ValueError` is raised when alpha is not between 0 and 1.
    """
    err_msg = re.escape(
                f'`alpha` must have a value between 0 and 1. Got {alpha}.'
            )
    with pytest.raises(ValueError, match = err_msg):
        check_interval(alpha=alpha)
