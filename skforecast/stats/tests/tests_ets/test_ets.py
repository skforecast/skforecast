import math
import numpy as np
import pandas as pd
import pytest

from skforecast.stats._ets import (
    ets,
    forecast_ets,
    auto_ets,
    residual_diagnostics,
    Ets,
    ETSConfig,
    ETSParams,
    init_states,
    get_bounds,
    admissible,
    check_param,
)


def ar1_series(n=80, phi=0.7, sigma=1.0, seed=123):
    """Generate AR(1) series for testing"""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


def seasonal_series(n=120, m=12, seed=42):
    """Generate series with seasonal pattern"""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 10 + 0.1 * t
    seasonal = 3 * np.sin(2 * np.pi * t / m)
    noise = rng.normal(0, 0.5, n)
    return trend + seasonal + noise


@pytest.mark.parametrize(
    "model_spec, m",
    [
        ("ANN", 1),
        ("AAN", 1),
        ("AAdN", 1),
        ("ANA", 12),
        ("AAA", 12),
    ],
)
def test_ets_nominal_returns_model(model_spec, m):
    """Test that ets() returns ETSModel with correct structure"""
    y = seasonal_series(80, m=m) if m > 1 else ar1_series(80)

    damped = "d" in model_spec
    model_clean = model_spec.replace("d", "")

    model = ets(y, m=m, model=model_clean, damped=damped)

    assert hasattr(model, "config")
    assert hasattr(model, "params")
    assert hasattr(model, "fitted")
    assert hasattr(model, "residuals")
    assert hasattr(model, "states")
    assert hasattr(model, "loglik")
    assert hasattr(model, "aic")
    assert hasattr(model, "bic")
    assert hasattr(model, "sigma2")

    assert model.fitted.shape == y.shape
    assert model.residuals.shape == y.shape
    assert np.isfinite(model.aic)
    assert np.isfinite(model.bic)
    assert model.sigma2 >= 0


def test_ets_config_properties():
    """Test ETSConfig dataclass properties"""
    config = ETSConfig(error="A", trend="A", season="A", damped=True, m=12)

    assert config.error_code == 1  # A = Additive = 1
    assert config.trend_code == 1  # A = 1
    assert config.season_code == 1
    assert config.n_states == 1 + 1 + (12 - 1)  # level + trend + seasonal


def test_ets_params_to_from_vector():
    """Test parameter vectorization and devectorization"""
    config = ETSConfig(error="A", trend="A", season="N", damped=False, m=1)
    params = ETSParams(
        alpha=0.2,
        beta=0.1,
        gamma=0.05,
        phi=0.95,
        init_states=np.array([10.0, 0.5])
    )

    vec = params.to_vector(config)
    assert len(vec) == 2 + 2  # alpha, beta + 2 init states (level, trend)

    params_back = ETSParams.from_vector(vec, config)
    assert np.isclose(params_back.alpha, 0.2)
    assert np.isclose(params_back.beta, 0.1)
    assert np.allclose(params_back.init_states, params.init_states)


def test_forecast_ets_shapes_and_uncertainty():
    """Test forecast shapes and monotone uncertainty"""
    y = ar1_series(120)
    model = ets(y, m=1, model="AAN")

    out = forecast_ets(model, h=12, level=[80, 95])

    assert "mean" in out
    assert out["mean"].shape == (12,)
    assert "lower_80" in out
    assert "upper_80" in out
    assert "lower_95" in out
    assert "upper_95" in out

    # Check that upper > lower
    assert np.all(out["upper_80"] > out["lower_80"])
    assert np.all(out["upper_95"] > out["lower_95"])

    # 95% interval should be wider than 80%
    width_80 = out["upper_80"] - out["lower_80"]
    width_95 = out["upper_95"] - out["lower_95"]
    assert np.all(width_95 >= width_80)

    # Uncertainty should generally increase with horizon
    assert width_95[-1] >= width_95[0] - 1e-6


def test_forecast_ets_without_intervals():
    """Test forecast without prediction intervals"""
    y = ar1_series(80)
    model = ets(y, m=1, model="ANN")

    out = forecast_ets(model, h=10, level=None)

    assert "mean" in out
    assert out["mean"].shape == (10,)
    assert "lower_80" not in out
    assert "upper_80" not in out


def test_init_states_various_configs():
    """Test initial state computation for various configurations"""
    y = seasonal_series(80, m=12)

    # No trend, no season
    config_nn = ETSConfig(error="A", trend="N", season="N", m=1)
    states_nn = init_states(y, config_nn)
    assert len(states_nn) == 1  # Only level

    # Trend, no season
    config_an = ETSConfig(error="A", trend="A", season="N", m=1)
    states_an = init_states(y, config_an)
    assert len(states_an) == 2  # Level + trend

    # No trend, with season
    config_na = ETSConfig(error="A", trend="N", season="A", m=12)
    states_na = init_states(y, config_na)
    assert len(states_na) == 1 + 11  # Level + m-1 seasonal

    # Trend + season
    config_aa = ETSConfig(error="A", trend="A", season="A", m=12)
    states_aa = init_states(y, config_aa)
    assert len(states_aa) == 2 + 11  # Level + trend + m-1 seasonal


def test_get_bounds():
    """Test parameter bounds generation"""
    config = ETSConfig(error="A", trend="A", season="A", damped=True, m=4)
    lower, upper = get_bounds(config)

    # Should have bounds for alpha, beta, gamma, phi, and init states
    n_states = config.n_states  # 1 + 1 + 3 = 5
    expected_len = 4 + n_states  # smoothing params + states

    assert len(lower) == expected_len
    assert len(upper) == expected_len

    # Check smoothing parameter bounds
    assert lower[0] > 0  # alpha
    assert upper[0] < 1
    assert lower[1] > 0  # beta
    assert upper[1] < 1


def test_admissible_parameter_checks():
    """Test admissibility constraints"""
    # Simple exponential smoothing (always admissible for valid alpha)
    assert admissible(alpha=0.3, beta=None, gamma=None, phi=None, m=1)

    # Invalid phi (out of admissible range)
    assert not admissible(alpha=0.3, beta=None, gamma=None, phi=1.5, m=1)

    # Holt's method - check that it doesn't raise errors
    # Note: admissible() checks stability, not just bounds
    result = admissible(alpha=0.2, beta=0.1, gamma=None, phi=None, m=1)
    assert isinstance(result, (bool, np.bool_))


def test_auto_ets_model_selection():
    """Test automatic model selection"""
    y = seasonal_series(120, m=12)

    model = auto_ets(y, m=12, seasonal=True, verbose=False)

    assert hasattr(model, "config")
    assert hasattr(model, "params")
    assert model.fitted.shape == y.shape
    assert np.isfinite(model.aic)
    assert np.isfinite(model.bic)


def test_auto_ets_without_seasonality():
    """Test auto_ets with seasonal=False"""
    y = ar1_series(80)

    model = auto_ets(y, m=1, seasonal=False, verbose=False)

    assert model.config.season == "N"
    assert model.config.m == 1


def test_auto_ets_trend_detection():
    """Test automatic trend detection"""
    # Series with strong trend
    t = np.arange(100)
    y_trend = 10 + 0.5 * t + np.random.normal(0, 0.1, 100)

    model = auto_ets(y_trend, m=1, trend=None, verbose=False)

    # Should detect trend
    assert model.config.trend in ["A", "M"]


def test_residual_diagnostics():
    """Test residual diagnostics computation"""
    y = ar1_series(100)
    model = ets(y, m=1, model="AAN")

    diag = residual_diagnostics(model)

    assert "mean" in diag
    assert "std" in diag
    assert "mae" in diag
    assert "rmse" in diag
    assert "ljung_box_stat" in diag
    assert "ljung_box_p" in diag
    assert "jarque_bera_p" in diag
    assert "shapiro_p" in diag
    assert "acf" in diag

    # Mean should be close to zero
    assert abs(diag["mean"]) < 0.5

    # Stats should be finite
    assert np.isfinite(diag["std"])
    assert np.isfinite(diag["mae"])
    assert np.isfinite(diag["rmse"])


def test_estimator_fit_and_attributes():
    """Test Ets estimator fit and attributes"""
    y = ar1_series(100)
    est = Ets(m=1, model="ANN")
    est.fit(y)

    assert hasattr(est, "model_")
    assert hasattr(est, "y_")
    assert hasattr(est, "config_")
    assert hasattr(est, "params_")
    assert hasattr(est, "fitted_values_")
    assert hasattr(est, "residuals_in_")
    assert hasattr(est, "n_features_in_")

    assert est.y_.shape == y.shape
    assert est.fitted_values_.shape == y.shape
    assert est.residuals_in_.shape == y.shape
    assert est.n_features_in_ == 1


def test_estimator_predict():
    """Test Ets estimator predict method"""
    y = ar1_series(120)
    est = Ets(m=1, model="AAN")
    est.fit(y)

    mean = est.predict(steps=8)
    assert mean.shape == (8,)
    assert np.all(np.isfinite(mean))


def test_estimator_predict_interval():
    """Test Ets estimator predict_interval method"""
    y = ar1_series(120)
    est = Ets(m=1, model="AAN")
    est.fit(y)

    # Test with as_frame=True
    df = est.predict_interval(steps=5, level=(80, 95), as_frame=True)
    assert isinstance(df, pd.DataFrame)
    assert "mean" in df.columns
    assert "lower_80" in df.columns
    assert "upper_80" in df.columns
    assert "lower_95" in df.columns
    assert "upper_95" in df.columns
    assert len(df) == 5

    # Test with as_frame=False
    raw = est.predict_interval(steps=3, level=(90,), as_frame=False)
    assert isinstance(raw, dict)
    assert "mean" in raw
    assert "lower_90" in raw
    assert "upper_90" in raw
    assert raw["mean"].shape == (3,)


def test_estimator_invalid_steps():
    """Test Ets estimator with invalid steps parameter"""
    y = ar1_series(50)
    est = Ets(m=1, model="ANN")
    est.fit(y)

    with pytest.raises(ValueError, match="positive integer"):
        est.predict(steps=0)

    with pytest.raises(ValueError, match="positive integer"):
        est.predict(steps=-2)

    with pytest.raises(ValueError, match="positive integer"):
        est.predict(steps=1.5)


def test_estimator_unfitted():
    """Test Ets estimator before fitting"""
    est = Ets(m=1, model="ANN")

    with pytest.raises(Exception):
        est.predict(steps=1)


def test_estimator_residuals_and_fitted_helpers():
    """Test residuals_() and fitted_() helper methods"""
    y = ar1_series(70)
    est = Ets(m=1, model="AAN").fit(y)

    r = est.residuals_()
    f = est.fitted_()

    assert r.shape == y.shape
    assert f.shape == y.shape

    # Residuals should equal y - fitted (approximately)
    assert np.allclose(r, y - f, atol=1e-10)


def test_estimator_summary(capsys):
    """Test Ets estimator summary output"""
    y = ar1_series(100)
    est = Ets(m=1, model="AAN").fit(y)

    est.summary()
    captured = capsys.readouterr().out

    assert "ETS Model Summary" in captured
    assert "Number of observations:" in captured
    assert "Smoothing parameters:" in captured
    assert "alpha (level):" in captured


def test_estimator_score():
    """Test Ets estimator score method"""
    y = ar1_series(100)
    est = Ets(m=1, model="AAN").fit(y)

    score = est.score()

    # R^2 should be between -inf and 1
    assert score <= 1.0
    assert np.isfinite(score) or np.isnan(score)


def test_estimator_auto_selection():
    """Test Ets estimator with automatic model selection"""
    y = seasonal_series(120, m=12)
    est = Ets(m=12, model="ZZZ")
    est.fit(y)

    assert hasattr(est, "model_")
    assert hasattr(est, "config_")

    # Should have selected a model
    assert est.config_.error in ["A", "M"]
    assert est.config_.trend in ["N", "A", "M"]
    assert est.config_.season in ["N", "A", "M"]


def test_estimator_with_pandas_series():
    """Test Ets estimator with pandas Series input"""
    y_array = ar1_series(80)
    y_series = pd.Series(y_array)

    est = Ets(m=1, model="ANN")
    est.fit(y_series)

    assert est.y_.shape == (80,)
    assert isinstance(est.y_, np.ndarray)


def test_ets_too_short_series():
    """Test ETS with series too short for the model"""
    y = np.array([1.0, 2.0, 3.0])

    # Should raise about insufficient data for seasonal model
    with pytest.raises(ValueError, match="Cannot fit seasonal model"):
        ets(y, m=12, model="AAA")


def test_ets_constant_series():
    """Test ETS with constant series"""
    y = np.ones(50)

    # Should handle constant series gracefully
    model = ets(y, m=1, model="ZZZ")

    assert model.config.error == "A"
    assert model.config.trend == "N"
    assert model.config.season == "N"


def test_ets_with_box_cox_transform():
    """Test ETS with Box-Cox transformation"""
    y = np.exp(ar1_series(80))  # Positive series

    model = ets(y, m=1, model="ANN", lambda_auto=True)

    assert model.transform is not None
    assert hasattr(model.transform, "lambda_param")

    # Forecast should also work with transform
    forecasts = forecast_ets(model, h=10)
    assert forecasts["mean"].shape == (10,)
    assert np.all(forecasts["mean"] > 0)


def test_ets_seasonal_too_high_frequency():
    """Test ETS with m > 24 (should raise error)"""
    y = ar1_series(100)

    # m > 24 should raise an error for seasonal models
    with pytest.raises(ValueError, match="Frequency too high"):
        model = ets(y, m=48, model="AAA")


def test_estimator_get_set_params():
    """Test get_params and set_params methods"""
    est = Ets(m=12, model="AAA", damped=True)

    params = est.get_params()
    assert params["m"] == 12
    assert params["model"] == "AAA"
    assert params["damped"] == True

    est.set_params(m=4, model="ANN")
    assert est.m == 4
    assert est.model == "ANN"


def test_forecast_ets_seasonal_model():
    """Test forecasting with seasonal model"""
    y = seasonal_series(120, m=12)
    model = ets(y, m=12, model="AAA")

    forecasts = forecast_ets(model, h=24, level=[95])

    assert forecasts["mean"].shape == (24,)
    assert "lower_95" in forecasts
    assert "upper_95" in forecasts


def test_check_param_bounds():
    """Test parameter bounds checking"""
    lower = np.array([1e-4, 1e-4, 1e-4, 0.8])
    upper = np.array([0.9999, 0.9999, 0.9999, 0.98])

    # Test that check_param returns a boolean
    result = check_param(0.3, 0.1, 0.1, 0.95, lower, upper, "both", 12)
    assert isinstance(result, (bool, np.bool_))

    # Invalid alpha (too high)
    assert not check_param(1.5, 0.1, 0.1, 0.95, lower, upper, "usual", 12)

    # Invalid beta (greater than alpha) - should fail usual bounds check
    assert not check_param(0.3, 0.5, 0.1, 0.95, lower, upper, "usual", 12)


def test_ets_invalid_model_string():
    """Test ETS with invalid model string"""
    y = ar1_series(80)

    with pytest.raises(ValueError, match="Model must be 3 characters"):
        ets(y, m=1, model="AN")  # Too short

    with pytest.raises(ValueError, match="Model must be 3 characters"):
        ets(y, m=1, model="AANN")  # Too long


def test_estimator_with_fixed_parameters():
    """Test Ets estimator with fixed smoothing parameters"""
    y = ar1_series(80)
    est = Ets(m=1, model="ANN", alpha=0.3)
    est.fit(y)

    # Alpha should be close to the fixed value
    assert abs(est.params_.alpha - 0.3) < 0.1


def test_auto_ets_ic_selection():
    """Test auto_ets with different information criteria"""
    y = ar1_series(100)

    model_aic = auto_ets(y, m=1, ic="aic", verbose=False)
    model_bic = auto_ets(y, m=1, ic="bic", verbose=False)
    model_aicc = auto_ets(y, m=1, ic="aicc", verbose=False)

    # All should return valid models
    assert hasattr(model_aic, "config")
    assert hasattr(model_bic, "config")
    assert hasattr(model_aicc, "config")


def test_ets_multiplicative_error():
    """Test ETS with multiplicative error"""
    y = np.exp(ar1_series(80))  # Positive series
    y = y + 10  # Make sure all positive

    model = ets(y, m=1, model="MNN")

    assert model.config.error == "M"
    assert model.fitted.shape == y.shape


def test_ets_damped_trend():
    """Test ETS with damped trend"""
    y = ar1_series(100)
    model = ets(y, m=1, model="AAN", damped=True)

    assert model.config.damped == True
    assert 0 < model.params.phi < 1


def test_estimator_seasonal_model():
    """Test Ets estimator with seasonal model"""
    y = seasonal_series(120, m=12)
    est = Ets(m=12, model="AAA")
    est.fit(y)

    assert est.config_.season == "A"
    assert est.config_.m == 12

    # Forecast should capture seasonality
    forecasts = est.predict(steps=24)
    assert len(forecasts) == 24


def test_residual_diagnostics_acf():
    """Test that ACF is computed correctly in diagnostics"""
    y = ar1_series(100)
    model = ets(y, m=1, model="AAN")
    diag = residual_diagnostics(model)

    acf = diag["acf"]
    assert len(acf) > 0
    assert acf[0] == 1.0  # ACF at lag 0 is always 1

    # ACF values should be in [-1, 1]
    assert np.all(np.abs(acf) <= 1.0 + 1e-10)

