# Unit test _check_interval
# ==============================================================================
import re
import pytest
from skforecast.utils import check_interval
from skforecast.utils.utils import _normalize_interval_scale


def test_check_interval_TypeError_when_interval_is_not_a_list():
    """
    Check `TypeError` is raised when `interval` is not a `list`.
    """
    err_msg = re.escape(
        "`interval` must be a `list` or `tuple`. For example, interval of 95% "
        "should be as `interval = [0.025, 0.975]`."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_interval(interval = 'not_a_list')


def test_check_interval_ValueError_when_interval_len_is_not_2():
    """
    Check `ValueError` is raised when `interval` len is not 2.
    """
    err_msg = re.escape(
        "`interval` must contain exactly 2 values, respectively the "
        "lower and upper interval bounds. For example, interval of 95% "
        "should be as `interval = [0.025, 0.975]`."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = [0.025, 0.5, 0.975])


def test_check_interval_ValueError_when_interval_lower_bound_less_than_0():
    """
    Check `ValueError` is raised when lower bound is less than 0.
    """
    err_msg = re.escape("Lower interval bound (-1.0) must be >= 0 and < 1.")
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = [-1.0, 0.975])


@pytest.mark.parametrize("interval", 
                         [[1.0, 0.975], (1.5, 0.975)], 
                         ids = lambda value: f'interval: {value}')
def test_check_interval_ValueError_when_interval_lower_bound_greater_than_or_equal_to_1(interval):
    """
    Check `ValueError` is raised when lower bound is greater than or equal to 1.
    """
    err_msg = re.escape(f"Lower interval bound ({interval[0]}) must be >= 0 and < 1.")
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = interval)


@pytest.mark.parametrize("interval", 
                         [[0.025, 0.0], (0.025, -1.0)], 
                         ids = lambda value: f'interval: {value}')
def test_check_interval_ValueError_when_interval_upper_bound_less_than_or_equal_to_0(interval):
    """
    Check `ValueError` is raised when upper bound is less than or equal to 0.
    """
    err_msg = re.escape(f"Upper interval bound ({interval[1]}) must be > 0 and <= 1.")
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = interval)


def test_check_interval_ValueError_when_interval_upper_bound_greater_than_1():
    """
    Check `ValueError` is raised when upper bound is greater than 1.
    """
    err_msg = re.escape('Upper interval bound (1.5) must be > 0 and <= 1.')
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = [0.025, 1.5])


@pytest.mark.parametrize("interval", 
                         [[0.025, 0.025], (0.5, 0.2)], 
                         ids = lambda value: f'interval: {value}')
def test_check_interval_ValueError_when_interval_lower_bound_greater_than_or_equal_to_upper_bound(interval):
    """
    Check `ValueError` is raised when lower bound is greater than or equal to
    upper bound.
    """
    err_msg = re.escape(
        f"Lower interval bound ({interval[0]}) must be less than the "
        f"upper interval bound ({interval[1]})."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = interval)


@pytest.mark.parametrize("interval", 
                         [[0.0, 0.95], (0.2, 0.81)], 
                         ids = lambda value: f'interval: {value}')
def test_check_interval_ValueError_when_interval_is_not_symmetric(interval):
    """
    Check `ValueError` is raised when interval is not symmetric.
    """
    err_msg = re.escape(
        f"Interval must be symmetric, the sum of the lower, ({interval[0]}), "
        f"and upper, ({interval[1]}), interval bounds must be equal to "
        f"1. Got {interval[0] + interval[1]}."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval = interval, ensure_symmetric_intervals = True)


def test_check_interval_TypeError_when_quantiles_is_not_a_list():
    """
    Check `TypeError` is raised when `quantiles` is not a `list`.
    """
    err_msg = re.escape(
        "`quantiles` must be a `list` or `tuple`. For example, quantiles "
        "0.05, 0.5, and 0.95 should be as `quantiles = [0.05, 0.5, 0.95]`."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_interval(quantiles = 'not_a_list')


@pytest.mark.parametrize("quantiles", 
                         [[-0.01, 0.01, 0.5], [0., 1., 1.1], [-2], (-2, 2)], 
                         ids = lambda value: f'quantiles: {value}')
def test_check_interval_ValueError_when_elements_in_quantiles_are_out_of_bounds(quantiles):
    """
    Check `ValueError` is raised when any element in `quantiles` is 
    not between 0 and 100.
    """
    err_msg = re.escape("All elements in `quantiles` must be >= 0 and <= 1.")
    with pytest.raises(ValueError, match = err_msg):
        check_interval(quantiles=quantiles)


def test_check_interval_TypeError_when_alpha_is_not_float():
    """
    Check `TypeError` is raised when `alpha` is not a `float`.
    """
    err_msg = re.escape(
        "`alpha` must be a `float`. For example, interval of 95% "
        "should be as `alpha = 0.05`."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_interval(alpha = 'not_a_float')


@pytest.mark.parametrize("alpha", 
                         [1., 0.], 
                         ids = lambda value: f'alpha: {value}')
def test_check_interval_ValueError_when_alpha_is_out_of_bounds(alpha):
    """
    Check `ValueError` is raised when alpha is not between 0 and 1.
    """
    err_msg = re.escape(f'`interval` must have a value between 0 and 1. Got {alpha}.')
    with pytest.raises(ValueError, match = err_msg):
        check_interval(alpha=alpha, alpha_literal='interval')


@pytest.mark.parametrize("interval",
                         [[0.025, 0.975], (0.05, 0.95), [0.0, 1.0]],
                         ids = lambda value: f'interval: {value}')
def test_check_interval_quantile_scale_valid_intervals(interval):
    """
    Check no error is raised for valid intervals in the quantile (0-1) scale.
    """
    check_interval(interval=interval)


def test_check_interval_quantile_scale_ValueError_when_lower_bound_out_of_range():
    """
    Check `ValueError` is raised when lower bound is >= 1 in quantile scale.
    """
    err_msg = re.escape("Lower interval bound (1.0) must be >= 0 and < 1.")
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval=[1.0, 0.95])


def test_check_interval_quantile_scale_ValueError_when_upper_bound_out_of_range():
    """
    Check `ValueError` is raised when upper bound is > 1 in quantile scale.
    """
    err_msg = re.escape("Upper interval bound (1.5) must be > 0 and <= 1.")
    with pytest.raises(ValueError, match = err_msg):
        check_interval(interval=[0.05, 1.5])


def test_check_interval_quantile_scale_ValueError_when_not_symmetric():
    """
    Check `ValueError` is raised when interval is not symmetric in quantile scale.
    """
    err_msg = re.escape(
        "Interval must be symmetric, the sum of the lower, (0.1), "
        "and upper, (0.95), interval bounds must be equal to 1. Got 1.05."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_interval(
            interval=[0.1, 0.95],
            ensure_symmetric_intervals=True
        )


@pytest.mark.parametrize("interval, expected",
                         [([0.05, 0.95], [0.05, 0.95]),
                          ((0.025, 0.975), [0.025, 0.975]),
                          ([0.0, 1.0], [0.0, 1.0])],
                         ids = lambda value: f'{value}')
def test_normalize_interval_scale_quantiles_unchanged(interval, expected):
    """
    Check values already in the 0-1 scale are returned unchanged.
    """
    results = _normalize_interval_scale(interval)
    assert results == expected


def test_normalize_interval_scale_percentiles_converted_with_warning():
    """
    Check legacy percentiles (all > 1) are divided by 100 and a `FutureWarning`
    is emitted.
    """
    err_msg = re.escape(
        "Passing `interval` as percentiles (0-100) is deprecated. Use "
        "quantiles (0-1) instead. For example, use `interval=[0.05, 0.95]` "
        "instead of `interval=[5, 95]`. Percentile support will be removed "
        "in skforecast 0.24.0."
    )
    with pytest.warns(FutureWarning, match = err_msg):
        results = _normalize_interval_scale([5, 95])

    assert results == [0.05, 0.95]


@pytest.mark.parametrize("interval",
                         [[1, 50], [0.5, 95], (1.0, 97.5)],
                         ids = lambda value: f'interval: {value}')
def test_normalize_interval_scale_ValueError_when_mixed(interval):
    """
    Check `ValueError` is raised when interval mixes values <= 1 and > 1.
    """
    err_msg = re.escape(
        "`interval` mixes values <= 1 and > 1, so the scale is ambiguous. "
        "Use quantiles in the [0, 1] range, e.g. `interval=[0.05, 0.95]`."
    )
    with pytest.raises(ValueError, match = err_msg):
        _normalize_interval_scale(interval)
