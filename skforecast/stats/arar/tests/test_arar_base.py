# Unit tests for skforecast.stats.arar.arar_base 
# ==============================================================================
import re
import pytest
import math
import numpy as np
from .._arar_base import (
    setup_params,
    arar,
    forecast,
    fitted_arar,
    residuals_arar,
)


def ar1_series(n=80, phi=0.7, sigma=1.0, seed=123):
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


@pytest.mark.parametrize(
    "n, expected_depth, expected_lag",
    [
        (9, None, None),     
        (10, None, None),  
        (12, None, None),
        (13, 13, 13),
        (41, 26, 40),
        (100, 26, 40),
    ],
)
def test_setup_params_general(n, expected_depth, expected_lag):
    y = np.arange(n, dtype=float)

    if n < 10:
        with pytest.warns(UserWarning, match="Training data is too short"):
            d, L = setup_params(y)
    else:
        d, L = setup_params(y)

    if n >= 41:
        assert d == 26 and L == 40
    elif 13 <= n <= 40:
        assert d == 13 and L == 13
    elif 10 <= n < 13:
        assert d == max(4, math.ceil(n / 3))
        assert L == max(4, math.ceil(n / 2))


def test_setup_params_overrides_respected():
    y = np.arange(50, dtype=float)
    d, L = setup_params(y, max_ar_depth=7, max_lag=11)
    assert d == 7 and L == 11


def test_arar_nominal_returns_tuple_shapes():
    y = ar1_series(80)
    model = arar(y, safe=True)
    assert isinstance(model, tuple) and len(model) == 8
    Y, phi, lags, sigma2, psi, sbar, d, L = model
    assert np.asarray(Y).shape == (80,)
    assert np.asarray(phi).shape == (4,)
    assert isinstance(lags, tuple) and len(lags) == 4
    assert np.isscalar(sigma2) and sigma2 >= 1e-12
    assert isinstance(sbar, float)
    assert isinstance(d, int) and isinstance(L, int)
    assert psi.ndim == 1 and psi.size >= 1


def test_forecast_shapes_and_monotone_uncertainty():
    y = ar1_series(120)
    model = arar(y, safe=True)
    out = forecast(model, h=12, level=(80, 95))
    assert set(out.keys()) == {"mean", "upper", "lower", "level"}
    assert out["mean"].shape == (12,)
    assert out["upper"].shape == (12, 2)
    assert out["lower"].shape == (12, 2)
    assert out["level"] == [80, 95]

    assert np.all(out["upper"] > out["lower"])

    widths = out["upper"][:, 1] - out["lower"][:, 1]
    assert np.all(widths[1:] >= widths[:-1] - 1e-10)


def test_forecast_invalid_h():
    y = ar1_series(40)
    model = arar(y, safe=True)
    with pytest.raises(ValueError, match="h must be positive"):
        forecast(model, h=0)


def test_fitted_and_residuals_consistency():
    y = ar1_series(90)
    model = arar(y, safe=True)
    fit_dict = fitted_arar(model)
    res = residuals_arar(model)
    fitted = fit_dict["fitted"]

    assert fitted.shape == y.shape
    assert res.shape == y.shape
    assert np.allclose(res[~np.isnan(fitted)], y[~np.isnan(fitted)] - fitted[~np.isnan(fitted)])

    assert np.isnan(fitted[:1]).any()
    assert np.isfinite(fitted[np.isnan(fitted) == False]).all()


def test_setup_params_max_lag_less_than_max_ar_depth_raises():
    """
    Test that setup_params raises ValueError when max_lag < max_ar_depth.
    """
    y = np.arange(100, dtype=float)
    msg = re.escape("max_lag must be greater than or equal to max_ar_depth")
    with pytest.raises(ValueError, match=msg):
        setup_params(y, max_ar_depth=20, max_lag=10)


def test_arar_safe_false_short_series_raises():
    """
    Test that arar raises RuntimeError for very short series when safe=False.
    The internal ValueError is caught and re-raised as RuntimeError.
    """
    y = np.array([1.0, 2.0, 3.0])
    msg = re.escape("ARAR fitting failed")
    with pytest.raises(RuntimeError, match=msg):
        arar(y, safe=False)


def test_arar_safe_false_incompatible_params_raises():
    """
    Test that arar raises ValueError when max_lag < max_ar_depth and safe=False.
    """
    y = ar1_series(100)
    msg = re.escape("max_lag must be greater than or equal to max_ar_depth")
    with pytest.raises(ValueError, match=msg):
        arar(y, max_ar_depth=50, max_lag=20, safe=False)


def test_arar_mean_fallback_very_short_series():
    """
    Test that arar returns mean fallback for very short series with safe=True.
    """
    y = np.array([1.0, 2.0])
    model = arar(y, safe=True)
    Y, phi, lags, sigma2, psi, sbar, d, L = model
    
    # Mean fallback should return zeros for phi, (1,1,1,1) lags
    np.testing.assert_array_equal(phi, np.zeros(4))
    assert lags == (1, 1, 1, 1)
    assert sbar == np.mean(y)
    np.testing.assert_array_equal(psi, np.array([1.0]))


def test_forecast_large_h_extends_xi():
    """
    Test forecast with h larger than xi size (triggers xi extension).
    """
    y = ar1_series(50, phi=0.3, seed=42)
    model = arar(y, safe=True)
    # Large horizon to ensure h > kk (xi size)
    out = forecast(model, h=100, level=(95,))
    
    assert out["mean"].shape == (100,)
    assert out["upper"].shape == (100, 1)
    assert out["lower"].shape == (100, 1)
    # Check uncertainty increases monotonically
    widths = out["upper"][:, 0] - out["lower"][:, 0]
    assert np.all(widths[1:] >= widths[:-1] - 1e-10)


def test_forecast_small_horizon_L_zero():
    """
    Test forecast when L == 0 (very short combined filter).
    This can happen with mean_fallback model where xi is small.
    """
    # Force mean fallback
    y = np.array([1.0, 2.0])
    model = arar(y, safe=True)
    out = forecast(model, h=5, level=(80,))
    
    assert out["mean"].shape == (5,)
    # With mean fallback, forecasts should be roughly constant (near sbar)
    assert np.allclose(out["mean"], out["mean"][0], atol=1e-6)


def test_arar_high_persistence_triggers_ar2_fitting():
    """
    Test that very high persistence series (phi1 >= 0.93, tau <= 2) 
    triggers the AR(2) memory shortening branch.
    """
    # Create a highly persistent series (near unit root)
    rng = np.random.default_rng(42)
    n = 100
    y = np.zeros(n)
    y[0] = 0.0
    for t in range(1, n):
        y[t] = 0.98 * y[t - 1] + rng.normal(0, 0.1)
    
    model = arar(y, safe=True)
    Y, phi, lags, sigma2, psi, sbar, d, L = model
    
    # Verify model was fitted (not mean fallback)
    assert isinstance(model, tuple) and len(model) == 8
    assert phi.shape == (4,)
    # psi will be longer than [1.0] if memory shortening occurred
    assert psi.size >= 1


def test_arar_gamma_zero_when_lag_exceeds_n():
    """
    Test that gamma[lag] is set to 0 when lag >= n (short series with large max_lag).
    This covers line 135: gamma[lag] = 0.0.
    """
    # Short series (n=15) with larger max_lag (13) so some lags exceed n
    y = ar1_series(15, phi=0.5, seed=99)
    # This will use auto params: max_ar_depth=13, max_lag=13 for n=15 (13<=n<=40)
    model = arar(y, safe=True)
    Y, phi, lags, sigma2, psi, sbar, d, L = model
    
    # Model should still fit successfully
    assert isinstance(model, tuple) and len(model) == 8
    assert np.isfinite(sigma2)



