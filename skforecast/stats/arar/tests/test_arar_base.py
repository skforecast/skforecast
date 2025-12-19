# Unit tests for skforecast.stats.arar.arar_base 
# ==============================================================================
import math
import numpy as np
import pytest
from .._arar_base import (
    setup_params,
    arar,
    forecast,
    fitted_arar,
    residuals_arar,
    summary_arar,
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


def test_summary_prints(capsys):
    y = ar1_series(60)
    model = arar(y, safe=True)
    summary_arar(model)
    captured = capsys.readouterr().out
    assert "ARAR Model Summary" in captured
    assert "Number of observations:" in captured
    assert "Selected AR lags:" in captured

def test_invalid_parameter_ordering():
    with pytest.raises(ValueError, match="max_lag must be greater than or equal to max_ar_depth. Got max_lag=12, max_ar_depth=13"):
        arar(ar1_series(60), max_ar_depth=13, max_lag=12)

