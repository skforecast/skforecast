# Unit tests for skforecast.stats.transformations._box_cox
# ==============================================================================
import re
import numpy as np
import pytest
from .._box_cox import (
    _guerrero_cv,
    guerrero,
    bcloglik,
    box_cox_lambda,
    box_cox,
    inv_box_cox,
    box_cox_biasadj,
)


# Fixtures
# ------------------------------------------------------------------------------
@pytest.fixture
def positive_series():
    """Generate a positive time series for testing."""
    rng = np.random.default_rng(42)
    return np.abs(rng.normal(10, 2, 100)) + 1


@pytest.fixture
def seasonal_series():
    """Generate a seasonal positive time series."""
    rng = np.random.default_rng(42)
    t = np.arange(120)
    trend = 0.1 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 12)
    noise = rng.normal(0, 0.5, 120)
    return np.abs(trend + seasonal + noise) + 10


# Tests for _guerrero_cv
# ------------------------------------------------------------------------------
def test_guerrero_cv_returns_finite_value(positive_series):
    """Test that _guerrero_cv returns a finite coefficient of variation."""
    cv = _guerrero_cv(0.5, positive_series, m=12)
    assert np.isfinite(cv)
    assert cv >= 0


def test_guerrero_cv_empty_or_short_returns_inf():
    """Test _guerrero_cv returns inf for problematic inputs."""
    # Very short series where nobst=0
    assert _guerrero_cv(0.5, np.array([1.0]), m=12) == np.inf
    # Series with zero mean in some periods
    x_zeros = np.array([0.0] * 24)
    assert _guerrero_cv(0.5, x_zeros, m=12) == np.inf


# Tests for guerrero
# ------------------------------------------------------------------------------
def test_guerrero_returns_lambda_in_bounds(positive_series):
    """Test guerrero returns lambda within specified bounds."""
    lam = guerrero(positive_series, m=12, lower=-1.0, upper=2.0)
    assert -1.0 <= lam <= 2.0


def test_guerrero_warns_for_non_positive_data():
    """Test guerrero warns when data contains non-positive values."""
    x_with_negative = np.array([1, 2, -1, 3, 4, 5] * 10)
    msg = re.escape("Guerrero's method for selecting a Box-Cox parameter")
    with pytest.warns(UserWarning, match=msg):
        guerrero(x_with_negative, m=2)


def test_guerrero_handles_nan_values(positive_series):
    """Test guerrero handles NaN values in input."""
    x_with_nan = positive_series.copy()
    x_with_nan[10:15] = np.nan
    lam = guerrero(x_with_nan, m=12)
    assert np.isfinite(lam)


# Tests for bcloglik
# ------------------------------------------------------------------------------
def test_bcloglik_returns_lambda_in_bounds(positive_series):
    """Test bcloglik returns lambda within specified bounds."""
    lam = bcloglik(positive_series, m=12, lower=-1.0, upper=2.0)
    assert -1.0 <= lam <= 2.0


def test_bcloglik_raises_for_non_positive_data():
    """Test bcloglik raises error for non-positive data."""
    x_with_negative = np.array([1, 2, -1, 3, 4, 5] * 10)
    msg = re.escape("x must be positive for log-likelihood method")
    with pytest.raises(ValueError, match=msg):
        bcloglik(x_with_negative, m=2)


@pytest.mark.parametrize("m,is_ts", [(1, True), (1, False), (12, True)])
def test_bcloglik_different_configurations(positive_series, m, is_ts):
    """Test bcloglik with different seasonal periods and is_ts settings."""
    lam = bcloglik(positive_series, m=m, is_ts=is_ts)
    assert np.isfinite(lam)


# Tests for box_cox_lambda
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["guerrero", "loglik"])
def test_box_cox_lambda_methods(positive_series, method):
    """Test box_cox_lambda with different methods."""
    lam = box_cox_lambda(positive_series, m=12, method=method)
    assert 0.0 <= lam <= 1.0  # default bounds


def test_box_cox_lambda_unknown_method_raises(positive_series):
    """Test box_cox_lambda raises error for unknown method."""
    msg = re.escape("Unknown method: invalid. Choose 'guerrero' or 'loglik'.")
    with pytest.raises(ValueError, match=msg):
        box_cox_lambda(positive_series, method="invalid")


def test_box_cox_lambda_short_series_returns_one():
    """Test box_cox_lambda returns 1.0 for very short series."""
    x_short = np.array([1.0, 2.0])
    lam = box_cox_lambda(x_short, m=12)
    assert lam == 1.0


def test_box_cox_lambda_adjusts_lower_for_non_positive():
    """Test box_cox_lambda adjusts lower bound for non-positive data."""
    x_with_zero = np.array([0.0, 1, 2, 3, 4, 5] * 10)
    # Should not raise even with default lower=0
    lam = box_cox_lambda(x_with_zero, m=2, lower=-1.0)
    assert lam >= 0.0


# Tests for box_cox
# ------------------------------------------------------------------------------
def test_box_cox_auto_lambda(positive_series):
    """Test box_cox with automatic lambda selection."""
    transformed, lam = box_cox(positive_series, m=12, lambda_bc="auto")
    assert transformed.shape == positive_series.shape
    assert np.isfinite(lam)
    assert np.all(np.isfinite(transformed))


def test_box_cox_fixed_lambda(positive_series):
    """Test box_cox with fixed lambda values."""
    # Lambda = 0 (log transform)
    transformed_log, lam_log = box_cox(positive_series, lambda_bc=0.0)
    assert lam_log == 0.0
    np.testing.assert_allclose(transformed_log, np.log(positive_series), rtol=1e-10)
    
    # Lambda = 1 (linear transform)
    transformed_lin, lam_lin = box_cox(positive_series, lambda_bc=1.0)
    assert lam_lin == 1.0
    np.testing.assert_allclose(transformed_lin, positive_series - 1, rtol=1e-10)
    
    # Lambda = 0.5 (square root transform)
    transformed_sqrt, lam_sqrt = box_cox(positive_series, lambda_bc=0.5)
    assert lam_sqrt == 0.5


def test_box_cox_none_lambda(positive_series):
    """Test box_cox with lambda=None returns unchanged data."""
    transformed, lam = box_cox(positive_series, lambda_bc=None)
    np.testing.assert_array_equal(transformed, positive_series)
    assert lam == 1.0


def test_box_cox_negative_lambda_handles_negative_values():
    """Test box_cox with negative lambda sets negative values to NaN."""
    x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    transformed, lam = box_cox(x, lambda_bc=-0.5)
    assert np.isnan(transformed[0])  # negative value becomes NaN


# Tests for inv_box_cox
# ------------------------------------------------------------------------------
def test_inv_box_cox_roundtrip(positive_series):
    """Test that inv_box_cox reverses box_cox transformation."""
    for lam in [0.0, 0.5, 1.0, 2.0]:
        transformed, _ = box_cox(positive_series, lambda_bc=lam)
        recovered = inv_box_cox(transformed, lam)
        np.testing.assert_allclose(recovered, positive_series, rtol=1e-6)


def test_inv_box_cox_negative_lambda():
    """Test inv_box_cox with negative lambda clips values correctly."""
    x_transformed = np.array([-0.5, 0.0, 0.5, 1.0])
    lam = -0.5  # threshold = 2.0
    result = inv_box_cox(x_transformed, lam)
    assert np.all(np.isfinite(result))


def test_inv_box_cox_biasadj_with_float_fvar(positive_series):
    """Test inv_box_cox with bias adjustment using scalar fvar."""
    transformed, lam = box_cox(positive_series, lambda_bc=0.5)
    result = inv_box_cox(transformed, lam, biasadj=True, fvar=0.1)
    assert result.shape == positive_series.shape
    assert np.all(np.isfinite(result))


def test_inv_box_cox_biasadj_with_array_fvar(positive_series):
    """Test inv_box_cox with bias adjustment using array fvar."""
    transformed, lam = box_cox(positive_series, lambda_bc=0.5)
    fvar_array = np.full(len(transformed), 0.1)
    result = inv_box_cox(transformed, lam, biasadj=True, fvar=fvar_array)
    assert result.shape == positive_series.shape


def test_inv_box_cox_biasadj_with_dict_fvar():
    """Test inv_box_cox with bias adjustment using dict fvar (prediction intervals)."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    transformed, lam = box_cox(x, lambda_bc=0.5)
    
    # Create mock prediction intervals
    fvar_dict = {
        'level': [95],
        'upper': transformed + 0.5,
        'lower': transformed - 0.5
    }
    result = inv_box_cox(transformed, lam, biasadj=True, fvar=fvar_dict)
    assert result.shape == x.shape
    assert np.all(np.isfinite(result))


def test_inv_box_cox_biasadj_with_dict_fvar_2d():
    """Test inv_box_cox with dict fvar containing 2D arrays (multiple levels)."""
    x = np.array([1.0, 2.0, 3.0])
    transformed, lam = box_cox(x, lambda_bc=0.5)
    
    # 2D arrays for multiple confidence levels
    fvar_dict = {
        'level': [80, 95],
        'upper': np.column_stack([transformed + 0.3, transformed + 0.5]),
        'lower': np.column_stack([transformed - 0.3, transformed - 0.5])
    }
    result = inv_box_cox(transformed, lam, biasadj=True, fvar=fvar_dict)
    assert result.shape == x.shape


def test_inv_box_cox_biasadj_missing_fvar_raises():
    """Test inv_box_cox raises error when biasadj=True but fvar is None."""
    x = np.array([1.0, 2.0, 3.0])
    msg = re.escape("fvar must be provided when biasadj=True")
    with pytest.raises(ValueError, match=msg):
        inv_box_cox(x, lambda_bc=0.5, biasadj=True, fvar=None)


def test_inv_box_cox_biasadj_fvar_shape_broadcast():
    """Test inv_box_cox broadcasts fvar when shape doesn't match."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    transformed, lam = box_cox(x, lambda_bc=0.5)
    fvar_short = np.array([0.1, 0.2])  # shorter than x
    result = inv_box_cox(transformed, lam, biasadj=True, fvar=fvar_short)
    assert result.shape == x.shape


# Tests for box_cox_biasadj
# ------------------------------------------------------------------------------
def test_box_cox_biasadj_uses_residual_variance(positive_series):
    """Test box_cox_biasadj computes variance when fvar is None."""
    transformed, lam = box_cox(positive_series, lambda_bc=0.5)
    result = box_cox_biasadj(transformed, lam, fvar=None)
    assert result.shape == positive_series.shape
    assert np.all(np.isfinite(result))


def test_box_cox_biasadj_with_explicit_fvar(positive_series):
    """Test box_cox_biasadj with explicit fvar."""
    transformed, lam = box_cox(positive_series, lambda_bc=0.5)
    result = box_cox_biasadj(transformed, lam, fvar=0.1)
    assert result.shape == positive_series.shape


# Integration tests
# ------------------------------------------------------------------------------
def test_full_pipeline_seasonal_data(seasonal_series):
    """Test complete transformation pipeline with seasonal data."""
    # Transform
    transformed, lam = box_cox(seasonal_series, m=12, lambda_bc="auto")
    assert np.all(np.isfinite(transformed))
    
    # Inverse transform
    recovered = inv_box_cox(transformed, lam)
    np.testing.assert_allclose(recovered, seasonal_series, rtol=1e-5)
    
    # Bias-adjusted inverse
    recovered_adj = inv_box_cox(transformed, lam, biasadj=True, fvar=0.01)
    assert np.all(np.isfinite(recovered_adj))


@pytest.mark.parametrize("lam", [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
def test_roundtrip_various_lambdas(positive_series, lam):
    """Test roundtrip transformation for various lambda values."""
    transformed, _ = box_cox(positive_series, lambda_bc=lam)
    recovered = inv_box_cox(transformed, lam)
    np.testing.assert_allclose(recovered, positive_series, rtol=1e-6)
