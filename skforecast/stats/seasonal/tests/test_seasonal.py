# Unit tests for skforecast.stats.seasonal
# ==============================================================================
import re
import numpy as np
import pytest
from .._seasonal_strength import seas_heuristic
from .._differencing import (
    is_constant,
    diff,
    ndiffs,
    nsdiffs,
)


# Fixtures
# ------------------------------------------------------------------------------
@pytest.fixture
def seasonal_series():
    """Generate a seasonal time series."""
    rng = np.random.default_rng(42)
    t = np.arange(120)
    seasonal = 5 * np.sin(2 * np.pi * t / 12)
    noise = rng.normal(0, 0.5, 120)
    return seasonal + noise + 10


@pytest.fixture
def random_walk():
    """Generate a random walk (non-stationary)."""
    rng = np.random.default_rng(42)
    return np.cumsum(rng.normal(0, 1, 100))


@pytest.fixture
def stationary_series():
    """Generate a stationary series."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, 100)


# Tests seas_heuristic
# ------------------------------------------------------------------------------
def test_seas_heuristic_returns_zero_for_short_series():
    """Test returns 0 for series shorter than 2*period."""
    x = np.arange(10, dtype=float)
    assert seas_heuristic(x, period=12) == 0.0


def test_seas_heuristic_returns_value_in_valid_range(seasonal_series):
    """Test returns value in [0, 1]."""
    strength = seas_heuristic(seasonal_series, period=12)
    assert 0.0 <= strength <= 1.0


def test_seas_heuristic_high_strength_for_seasonal_data(seasonal_series):
    """Test high strength for clearly seasonal data."""
    strength = seas_heuristic(seasonal_series, period=12)
    assert strength > 0.5


def test_seas_heuristic_low_strength_for_random_data(stationary_series):
    """Test low strength for non-seasonal data."""
    strength = seas_heuristic(stationary_series, period=12)
    assert strength < 0.5


def test_seas_heuristic_handles_nan_values(seasonal_series):
    """Test handles NaN values in input."""
    x_with_nan = seasonal_series.copy()
    x_with_nan[10:15] = np.nan
    strength = seas_heuristic(x_with_nan, period=12)
    assert np.isfinite(strength)


def test_seas_heuristic_returns_zero_for_constant_variance():
    """Test returns 0 when var_seasonal_remainder is near zero."""
    x = np.ones(50)
    strength = seas_heuristic(x, period=12)
    assert strength == 0.0


# Tests is_constant
# ------------------------------------------------------------------------------
def test_is_constant_constant_array_returns_true():
    """Test constant array returns True."""
    x = np.array([1.0, 1.0, 1.0, 1.0])
    assert is_constant(x) is True


def test_is_constant_non_constant_array_returns_false():
    """Test non-constant array returns False."""
    x = np.array([1.0, 2.0, 3.0])
    assert is_constant(x) is False


def test_is_constant_empty_array_returns_true():
    """Test empty array returns True."""
    x = np.array([])
    assert is_constant(x) is True


def test_is_constant_all_nan_returns_true():
    """Test all-NaN array returns True."""
    x = np.array([np.nan, np.nan, np.nan])
    assert is_constant(x) is True


def test_is_constant_respects_tolerance():
    """Test tolerance parameter is respected."""
    x = np.array([1.0, 1.0 + 1e-12, 1.0 - 1e-12])
    assert is_constant(x, tol=1e-10) is True
    x2 = np.array([1.0, 1.1, 1.0])
    assert is_constant(x2, tol=1e-10) is False


# Tests diff
# ------------------------------------------------------------------------------
def test_diff_lag_1():
    """Test first-order differencing."""
    x = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
    result = diff(x, lag=1, differences=1)
    expected = np.array([2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(result, expected)


def test_diff_lag_2():
    """Test differencing with lag 2."""
    x = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
    result = diff(x, lag=2, differences=1)
    expected = np.array([3.0, 5.0, 7.0])
    np.testing.assert_array_equal(result, expected)


def test_diff_order_2():
    """Test second-order differencing."""
    x = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
    result = diff(x, lag=1, differences=2)
    expected = np.array([1.0, 1.0, 1.0])
    np.testing.assert_array_equal(result, expected)


# Tests ndiffs
# ------------------------------------------------------------------------------
def test_ndiffs_stationary_series_returns_zero(stationary_series):
    """Test stationary series needs no differencing."""
    d = ndiffs(stationary_series, test="kpss")
    assert d == 0


def test_ndiffs_random_walk_needs_differencing(random_walk):
    """Test random walk needs differencing."""
    d = ndiffs(random_walk, test="kpss")
    assert d >= 1


def test_ndiffs_constant_series_returns_zero():
    """Test constant series needs no differencing."""
    x = np.ones(100)
    d = ndiffs(x)
    assert d == 0


def test_ndiffs_respects_max_d(random_walk):
    """Test max_d parameter is respected."""
    d = ndiffs(random_walk, max_d=1)
    assert d <= 1


def test_ndiffs_warns_alpha_too_small(stationary_series):
    """Test warns when alpha < 0.01."""
    msg = re.escape("Specified alpha value is less than the minimum")
    with pytest.warns(UserWarning, match=msg):
        ndiffs(stationary_series, alpha=0.001)


def test_ndiffs_warns_alpha_too_large(stationary_series):
    """Test warns when alpha > 0.1."""
    msg = re.escape("Specified alpha value is larger than the maximum")
    with pytest.warns(UserWarning, match=msg):
        ndiffs(stationary_series, alpha=0.5)


def test_ndiffs_adf_test(random_walk):
    """Test with ADF test."""
    d = ndiffs(random_walk, test="adf")
    assert 0 <= d <= 2


def test_ndiffs_handles_nan_values(random_walk):
    """Test handles NaN values in input."""
    x = random_walk.copy()
    x[10:15] = np.nan
    d = ndiffs(x)
    assert 0 <= d <= 2


# Tests nsdiffs
# ------------------------------------------------------------------------------
def test_nsdiffs_seasonal_series_may_need_differencing(seasonal_series):
    """Test seasonal series detection."""
    D = nsdiffs(seasonal_series, period=12, test="seas")
    assert D in (0, 1)


def test_nsdiffs_stationary_series_no_seasonal_diff(stationary_series):
    """Test non-seasonal series needs no seasonal differencing."""
    D = nsdiffs(stationary_series, period=12, test="seas")
    assert D == 0


def test_nsdiffs_constant_series_returns_zero():
    """Test constant series needs no seasonal differencing."""
    x = np.ones(120)
    D = nsdiffs(x, period=12)
    assert D == 0


def test_nsdiffs_raises_for_period_one(seasonal_series):
    """Test raises ValueError for period=1."""
    msg = re.escape("Non-seasonal data (period=1)")
    with pytest.raises(ValueError, match=msg):
        nsdiffs(seasonal_series, period=1)


def test_nsdiffs_warns_period_less_than_one(seasonal_series):
    """Test warns for period < 1."""
    msg = re.escape("Cannot handle data with frequency less than 1")
    with pytest.warns(UserWarning, match=msg):
        D = nsdiffs(seasonal_series, period=0)
    assert D == 0


def test_nsdiffs_returns_zero_period_exceeds_length():
    """Test returns 0 when period >= len(x)."""
    x = np.arange(10, dtype=float)
    D = nsdiffs(x, period=12)
    assert D == 0


def test_nsdiffs_respects_max_D(seasonal_series):
    """Test max_D parameter is respected."""
    D = nsdiffs(seasonal_series, period=12, max_D=0)
    assert D == 0


def test_nsdiffs_warns_alpha_too_small(seasonal_series):
    """Test warns when alpha < 0.01."""
    msg = re.escape("Specified alpha value is less than the minimum")
    with pytest.warns(UserWarning, match=msg):
        nsdiffs(seasonal_series, period=12, alpha=0.001)


def test_nsdiffs_warns_alpha_too_large(seasonal_series):
    """Test warns when alpha > 0.1."""
    msg = re.escape("Specified alpha value is larger than the maximum")
    with pytest.warns(UserWarning, match=msg):
        nsdiffs(seasonal_series, period=12, alpha=0.5)


def test_nsdiffs_warns_ocsb_alpha(seasonal_series):
    """Test warns for ocsb test with non-0.05 alpha."""
    msg = re.escape("Significance levels other than 5%")
    with pytest.warns(UserWarning, match=msg):
        nsdiffs(seasonal_series, period=12, test="ocsb", alpha=0.01)


def test_nsdiffs_raises_for_unimplemented_tests(seasonal_series):
    """Test raises NotImplementedError for hegy and ch tests."""
    with pytest.raises(NotImplementedError, match="'hegy'"):
        nsdiffs(seasonal_series, period=12, test="hegy")
    with pytest.raises(NotImplementedError, match="'ch'"):
        nsdiffs(seasonal_series, period=12, test="ch")


# Integration tests
# ------------------------------------------------------------------------------
def test_integration_strong_seasonal_detected():
    """Test strong seasonal pattern is detected."""
    rng = np.random.default_rng(42)
    t = np.arange(120)
    # Very strong seasonal component
    y = 20 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.1, 120)
    
    strength = seas_heuristic(y, period=12)
    assert strength > 0.9


def test_integration_pipeline_ndiffs_then_nsdiffs(seasonal_series):
    """Test typical workflow: first seasonal then regular differencing."""
    # First determine seasonal differences
    D = nsdiffs(seasonal_series, period=12)
    
    # Apply seasonal differencing if needed
    y = seasonal_series.copy()
    if D > 0:
        y = diff(y, lag=12, differences=D)
    
    # Then determine regular differences
    d = ndiffs(y)
    
    assert D >= 0 and d >= 0
    assert D + d <= 3  # Sanity check
