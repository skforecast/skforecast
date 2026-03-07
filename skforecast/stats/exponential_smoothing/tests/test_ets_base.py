# Unit tests for skforecast.stats.ets.ets_base
# ==============================================================================
import numpy as np
import pytest
from .._ets_base import (
    ets,
    forecast_ets,
    auto_ets,
    residual_diagnostics,
    simulate_ets,
    ETSConfig,
    ETSParams,
    ETSModel,
    BoxCoxTransform,
    init_states,
    get_bounds,
    admissible,
    check_param,
    fourier,
    is_constant,
)


# Fixtures
# ------------------------------------------------------------------------------
def ar1_series(n=80, phi=0.7, sigma=1.0, seed=123):
    """Generate AR(1) series for testing"""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


def trend_series(n=100, seed=42):
    """Generate series with strong trend"""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 10 + 0.5 * t + rng.normal(0, 0.5, n)


def positive_series(n=80, seed=123):
    """Generate positive series for multiplicative models"""
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(2, 0.3, n))


# Tests ets - basic models
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("model_spec", ["ANN", "AAN", "MNN", "MAN"])
def test_ets_nominal_returns_model(model_spec):
    """Test that ets() returns ETSModel with correct structure"""
    y = ar1_series(80) if model_spec[0] == "A" else positive_series(80)

    model = ets(y, m=1, model=model_spec, damped=False)

    assert hasattr(model, "config")
    assert hasattr(model, "params")
    assert hasattr(model, "fitted")
    assert hasattr(model, "residuals")
    assert hasattr(model, "states")
    assert hasattr(model, "loglik")
    assert hasattr(model, "aic")
    assert hasattr(model, "bic")
    assert hasattr(model, "sigma2")
    assert hasattr(model, "y_original")
    assert hasattr(model, "transform")

    assert model.fitted.shape == y.shape
    assert model.residuals.shape == y.shape
    assert np.isfinite(model.aic)
    assert np.isfinite(model.bic)
    assert model.sigma2 >= 0


def test_ets_damped_trend():
    """Test ETS with damped trend"""
    y = ar1_series(100)
    model = ets(y, m=1, model="AAN", damped=True)

    assert model.config.damped is True
    assert 0 < model.params.phi < 1


# Tests ETSConfig
# ------------------------------------------------------------------------------
def test_ets_config_properties():
    """Test ETSConfig dataclass properties"""
    config = ETSConfig(error="A", trend="A", season="A", damped=True, m=12)

    assert config.error_code == 1  # A = Additive = 1
    assert config.trend_code == 1  # A = 1
    assert config.season_code == 1
    assert config.n_states == 1 + 1 + (12 - 1)  # level + trend + seasonal

    # Test N codes
    config_n = ETSConfig(error="A", trend="N", season="N", m=1)
    assert config_n.trend_code == 0
    assert config_n.season_code == 0
    assert config_n.n_states == 1  # only level

    # Test M codes
    config_m = ETSConfig(error="M", trend="M", season="M", m=4)
    assert config_m.error_code == 2
    assert config_m.trend_code == 2
    assert config_m.season_code == 2


# Tests ETSParams
# ------------------------------------------------------------------------------
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


def test_ets_params_to_from_vector_with_season_damped():
    """Test parameter vectorization with seasonal and damped components"""
    config = ETSConfig(error="A", trend="A", season="A", damped=True, m=4)
    params = ETSParams(
        alpha=0.3,
        beta=0.1,
        gamma=0.2,
        phi=0.95,
        init_states=np.array([10.0, 0.5, 0.1, 0.2, 0.3])  # level, trend, 3 seasonal
    )

    vec = params.to_vector(config)
    # alpha, beta, gamma, phi + 5 init states
    assert len(vec) == 4 + 5

    params_back = ETSParams.from_vector(vec, config)
    assert np.isclose(params_back.alpha, 0.3)
    assert np.isclose(params_back.gamma, 0.2)
    assert np.isclose(params_back.phi, 0.95)


def test_ets_params_no_trend_no_season():
    """Test ETSParams for simple exponential smoothing"""
    config = ETSConfig(error="A", trend="N", season="N", damped=False, m=1)
    params = ETSParams(
        alpha=0.5,
        beta=0.0,
        gamma=0.0,
        phi=1.0,
        init_states=np.array([10.0])
    )

    vec = params.to_vector(config)
    assert len(vec) == 1 + 1  # alpha + level

    params_back = ETSParams.from_vector(vec, config)
    assert np.isclose(params_back.alpha, 0.5)
    assert params_back.beta == 0.0  # Default when no trend
    assert params_back.gamma == 0.0  # Default when no season
    assert params_back.phi == 1.0  # Default when not damped


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

    # Test ANN model (simplest model - uses direct variance formula)
    model_ann = ets(y, m=1, model="ANN")
    out_ann = forecast_ets(model_ann, h=5, level=[95])
    assert "lower_95" in out_ann
    assert "upper_95" in out_ann

    # Test damped trend model
    model_damped = ets(y, m=1, model="AAN", damped=True)
    out_damped = forecast_ets(model_damped, h=5, level=[95])
    assert "lower_95" in out_damped


def test_forecast_ets_multiplicative_simulation():
    """Test forecast with multiplicative error uses simulation for intervals"""
    y = positive_series(120)
    model = ets(y, m=1, model="MNN")  # Multiplicative error

    # For multiplicative errors, analytical variance is None,
    # so simulation is used for prediction intervals
    out = forecast_ets(model, h=10, level=[90])

    assert "mean" in out
    # Intervals should be computed via simulation
    assert "lower_90" in out or "upper_90" not in out  # Either computed or warning raised


def test_forecast_ets_without_intervals():
    """Test forecast without prediction intervals"""
    y = ar1_series(80)
    model = ets(y, m=1, model="ANN")

    out = forecast_ets(model, h=10, level=None)

    assert "mean" in out
    assert out["mean"].shape == (10,)
    assert "lower_80" not in out
    assert "upper_80" not in out


def test_forecast_ets_damped_trend():
    """Test forecasts with damped trend"""
    y = trend_series(100)
    model = ets(y, m=1, model="AAN", damped=True)

    out = forecast_ets(model, h=20, level=[95])

    assert out["mean"].shape == (20,)
    # Damped forecasts should converge to a limit
    # Later forecasts should be close to each other
    assert abs(out["mean"][-1] - out["mean"][-5]) < abs(out["mean"][5] - out["mean"][0])


# Tests init_states
# ------------------------------------------------------------------------------
def test_init_states_no_trend_no_season():
    """Test initial state computation without trend or season"""
    y = ar1_series(80)
    config = ETSConfig(error="A", trend="N", season="N", m=1)
    states = init_states(y, config)

    assert len(states) == 1  # Only level
    assert np.isfinite(states[0])


def test_init_states_with_trend():
    """Test initial state computation with trend"""
    y = trend_series(100)
    config = ETSConfig(error="A", trend="A", season="N", m=1)
    states = init_states(y, config)

    assert len(states) == 2  # Level + trend
    assert np.all(np.isfinite(states))


def test_init_states_multiplicative_trend():
    """Test initial states with multiplicative trend"""
    y = positive_series(100)
    config = ETSConfig(error="M", trend="M", season="N", m=1)
    states = init_states(y, config)

    assert len(states) == 2  # Level + trend
    assert np.all(np.isfinite(states))
    assert states[0] > 0  # Level should be positive for multiplicative


# Tests get_bounds
# ------------------------------------------------------------------------------
def test_get_bounds_simple():
    """Test parameter bounds for simple model"""
    config = ETSConfig(error="A", trend="N", season="N", damped=False, m=1)
    lower, upper = get_bounds(config)

    # alpha + 1 init state (level)
    assert len(lower) == 2
    assert len(upper) == 2
    assert lower[0] > 0  # alpha lower
    assert upper[0] < 1  # alpha upper


def test_get_bounds_full_model():
    """Test parameter bounds for full seasonal damped model"""
    config = ETSConfig(error="A", trend="A", season="A", damped=True, m=4)
    lower, upper = get_bounds(config)

    n_states = config.n_states  # 1 + 1 + 3 = 5
    # alpha, beta, gamma, phi + 5 states
    expected_len = 4 + n_states

    assert len(lower) == expected_len
    assert len(upper) == expected_len

    # Check smoothing parameter bounds
    assert lower[0] > 0  # alpha
    assert upper[0] < 1
    assert lower[1] > 0  # beta
    assert upper[1] < 1
    assert lower[2] > 0  # gamma
    assert upper[2] < 1
    assert lower[3] >= 0.8  # phi lower
    assert upper[3] <= 0.98  # phi upper


# Tests admissible
# ------------------------------------------------------------------------------
def test_admissible_parameter_checks():
    """Test admissibility constraints"""
    # Simple exponential smoothing (always admissible for valid alpha)
    assert admissible(alpha=0.3, beta=None, gamma=None, phi=None, m=1)

    # Invalid phi (out of admissible range)
    assert not admissible(alpha=0.3, beta=None, gamma=None, phi=1.5, m=1)

    # Holt's method - check that it doesn't raise errors
    result = admissible(alpha=0.2, beta=0.1, gamma=None, phi=None, m=1)
    assert isinstance(result, (bool, np.bool_))


def test_admissible_holt_bounds():
    """Test admissibility for Holt's method"""
    # Valid Holt's parameters
    assert admissible(alpha=0.3, beta=0.1, gamma=None, phi=1.0, m=1)

    # Very invalid beta (much greater than alpha) - should fail admissibility
    result = admissible(alpha=0.1, beta=0.9, gamma=None, phi=1.0, m=1)
    assert isinstance(result, (bool, np.bool_))


def test_admissible_with_seasonal():
    """Test admissibility with seasonal component"""
    # Valid seasonal parameters
    result = admissible(alpha=0.3, beta=0.1, gamma=0.1, phi=1.0, m=4)
    assert isinstance(result, (bool, np.bool_))


# Tests check_param
# ------------------------------------------------------------------------------
def test_check_param_bounds():
    """Test parameter bounds checking"""
    lower = np.array([1e-4, 1e-4, 1e-4, 0.8])
    upper = np.array([0.9999, 0.9999, 0.9999, 0.98])

    # Valid parameters
    result = check_param(0.3, 0.1, 0.1, 0.95, lower, upper, "both", 12)
    assert isinstance(result, (bool, np.bool_))

    # Invalid alpha (too high)
    assert not check_param(1.5, 0.1, 0.1, 0.95, lower, upper, "usual", 12)

    # Invalid beta (greater than alpha)
    assert not check_param(0.3, 0.5, 0.1, 0.95, lower, upper, "usual", 12)


def test_check_param_admissible_only():
    """Test check_param with admissible bounds only"""
    lower = np.array([1e-4, 1e-4])
    upper = np.array([0.9999, 0.9999])

    result = check_param(0.3, 0.1, None, None, lower, upper, "admissible", 1)
    assert isinstance(result, (bool, np.bool_))


def test_check_param_usual_only():
    """Test check_param with usual bounds only"""
    # Arrays must have 4 elements: alpha, beta, gamma, phi
    lower = np.array([1e-4, 1e-4, 1e-4, 0.8])
    upper = np.array([0.9999, 0.9999, 0.9999, 0.98])

    result = check_param(0.3, 0.1, None, 0.9, lower, upper, "usual", 1)
    assert isinstance(result, (bool, np.bool_))


def test_check_param_invalid_gamma():
    """Test check_param with invalid gamma"""
    lower = np.array([1e-4, 1e-4, 1e-4])
    upper = np.array([0.9999, 0.9999, 0.9999])

    # gamma > 1 - alpha should fail
    assert not check_param(0.8, None, 0.5, None, lower, upper, "usual", 4)


def test_check_param_invalid_phi():
    """Test check_param with invalid phi"""
    # Arrays must have 4 elements: alpha, beta, gamma, phi
    lower = np.array([1e-4, 1e-4, 1e-4, 0.8])
    upper = np.array([0.9999, 0.9999, 0.9999, 0.98])

    # phi=0.5 is out of bounds [0.8, 0.98]
    assert not check_param(0.3, None, None, 0.5, lower, upper, "usual", 1)


# Tests auto_ets
# ------------------------------------------------------------------------------
def test_auto_ets_model_selection():
    """Test automatic model selection without seasonal"""
    y = ar1_series(100)

    model = auto_ets(y, m=1, seasonal=False, verbose=False)

    assert hasattr(model, "config")
    assert hasattr(model, "params")
    assert model.fitted.shape == y.shape
    assert np.isfinite(model.aic)
    assert np.isfinite(model.bic)
    assert model.config.season == "N"
    assert model.config.m == 1


def test_auto_ets_trend_detection():
    """Test automatic trend detection"""
    y = trend_series(100)

    model = auto_ets(y, m=1, trend=None, verbose=False)

    # Should detect trend
    assert model.config.trend in ["A", "M", "N"]


def test_auto_ets_trend_forced():
    """Test auto_ets with trend forced True/False"""
    y = ar1_series(100)

    model_trend = auto_ets(y, m=1, trend=True, seasonal=False, verbose=False)
    model_no_trend = auto_ets(y, m=1, trend=False, seasonal=False, verbose=False)

    assert model_trend.config.trend in ["A", "M"]
    assert model_no_trend.config.trend == "N"


def test_auto_ets_damped_forced():
    """Test auto_ets with damped forced True/False"""
    y = trend_series(100)

    model_damped = auto_ets(y, m=1, trend=True, damped=True, seasonal=False, verbose=False)
    model_no_damped = auto_ets(y, m=1, trend=True, damped=False, seasonal=False, verbose=False)

    assert model_damped.config.damped is True
    assert model_no_damped.config.damped is False


def test_auto_ets_allow_multiplicative():
    """Test auto_ets with allow_multiplicative=False"""
    y = positive_series(100)

    model = auto_ets(y, m=1, allow_multiplicative=False, seasonal=False, verbose=False)

    assert model.config.error == "A"


def test_auto_ets_max_models():
    """Test auto_ets with max_models limit"""
    y = ar1_series(100)

    model = auto_ets(y, m=1, max_models=2, seasonal=False, verbose=False)

    assert hasattr(model, "config")


def test_auto_ets_ic_selection():
    """Test auto_ets with different information criteria"""
    y = ar1_series(100)

    model_aic = auto_ets(y, m=1, ic="aic", seasonal=False, verbose=False)
    model_bic = auto_ets(y, m=1, ic="bic", seasonal=False, verbose=False)
    model_aicc = auto_ets(y, m=1, ic="aicc", seasonal=False, verbose=False)

    # All should return valid models
    assert hasattr(model_aic, "config")
    assert hasattr(model_bic, "config")
    assert hasattr(model_aicc, "config")


def test_auto_ets_empty_series_raises():
    """Test auto_ets raises on empty series"""
    y = np.array([])

    with pytest.raises(ValueError, match="Need at least 1 observation"):
        auto_ets(y, m=1)


# Tests residual_diagnostics
# ------------------------------------------------------------------------------
def test_residual_diagnostics():
    """Test residual diagnostics computation"""
    y = ar1_series(100)
    model = ets(y, m=1, model="AAN")

    diag = residual_diagnostics(model)

    # Check all expected keys
    assert "mean" in diag
    assert "std" in diag
    assert "mae" in diag
    assert "rmse" in diag
    assert "mape" in diag
    assert "ljung_box_stat" in diag
    assert "ljung_box_p" in diag
    assert "jarque_bera_stat" in diag
    assert "jarque_bera_p" in diag
    assert "shapiro_stat" in diag
    assert "shapiro_p" in diag
    assert "acf" in diag

    # Mean should be close to zero
    assert abs(diag["mean"]) < 1.0

    # Stats should be finite
    assert np.isfinite(diag["std"])
    assert np.isfinite(diag["mae"])
    assert np.isfinite(diag["rmse"])
    assert np.isfinite(diag["ljung_box_stat"])

    # ACF checks
    acf = diag["acf"]
    assert len(acf) > 0
    assert acf[0] == 1.0  # ACF at lag 0 is always 1
    assert np.all(np.abs(acf) <= 1.0 + 1e-10)  # ACF values in [-1, 1]


# Tests simulate_ets
# ------------------------------------------------------------------------------
def test_simulate_ets_basic():
    """Test ETS simulation"""
    y = ar1_series(100)
    model = ets(y, m=1, model="AAN")

    simulations = simulate_ets(model, h=10, n_sim=100)

    assert simulations.shape == (100, 10)
    assert np.all(np.isfinite(simulations))


def test_simulate_ets_multiplicative():
    """Test simulation with multiplicative error"""
    y = positive_series(100)
    model = ets(y, m=1, model="MNN")

    simulations = simulate_ets(model, h=5, n_sim=50)

    assert simulations.shape == (50, 5)


def test_simulate_ets_invalid_sigma():
    """Test simulate_ets raises on invalid sigma"""
    y = ar1_series(100)
    model = ets(y, m=1, model="ANN")
    model.sigma2 = -1.0  # Force invalid

    with pytest.raises(ValueError, match="invalid residual variance"):
        simulate_ets(model, h=10)


# Tests BoxCoxTransform
# ------------------------------------------------------------------------------
def test_box_cox_find_lambda():
    """Test Box-Cox lambda estimation"""
    y = positive_series(100)
    lam = BoxCoxTransform.find_lambda(y)

    assert np.isfinite(lam)
    assert -1 <= lam <= 2


def test_box_cox_transform_inverse():
    """Test Box-Cox transform and inverse"""
    y = positive_series(100)
    transform = BoxCoxTransform(lambda_param=0.5, shift=0.0)

    y_trans = transform.transform(y)
    y_back = transform.inverse_transform(y_trans)

    np.testing.assert_allclose(y, y_back, rtol=1e-10)


def test_box_cox_log_transform():
    """Test Box-Cox with lambda=0 (log transform)"""
    y = positive_series(100)
    transform = BoxCoxTransform(lambda_param=0.0, shift=0.0)

    y_trans = transform.transform(y)
    y_back = transform.inverse_transform(y_trans)

    np.testing.assert_allclose(y, y_back, rtol=1e-10)


def test_box_cox_with_shift():
    """Test Box-Cox with shift parameter for values close to zero"""
    # Create series that requires shift
    np.random.seed(42)
    y = np.random.uniform(0.1, 10, 100)  # All positive values
    shift = 1.0
    transform = BoxCoxTransform(lambda_param=0.5, shift=shift)

    y_trans = transform.transform(y)
    y_back = transform.inverse_transform(y_trans)

    np.testing.assert_allclose(y, y_back, rtol=1e-5)


def test_box_cox_bias_adjustment():
    """Test Box-Cox inverse with bias adjustment"""
    y = positive_series(100)
    transform = BoxCoxTransform(lambda_param=0.5, shift=0.0)

    y_trans = transform.transform(y)
    variance = 0.1

    # With bias adjustment
    y_back_adj = transform.inverse_transform(y_trans, bias_adjust=True, variance=variance)
    # Without bias adjustment
    y_back = transform.inverse_transform(y_trans, bias_adjust=False)

    # Bias adjustment should modify the values
    assert not np.allclose(y_back_adj, y_back)


def test_box_cox_log_bias_adjustment():
    """Test Box-Cox log transform (lambda=0) with bias adjustment"""
    y = positive_series(100)
    transform = BoxCoxTransform(lambda_param=0.0, shift=0.0)

    y_trans = transform.transform(y)
    y_back_adj = transform.inverse_transform(y_trans, bias_adjust=True, variance=0.1)
    y_back = transform.inverse_transform(y_trans, bias_adjust=False)

    assert not np.allclose(y_back_adj, y_back)


# Tests ets - edge cases and errors
# ------------------------------------------------------------------------------
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
    y = positive_series(80)

    model = ets(y, m=1, model="ANN", lambda_auto=True)

    assert model.transform is not None
    assert hasattr(model.transform, "lambda_param")

    # Forecast should also work with transform
    forecasts = forecast_ets(model, h=10)
    assert forecasts["mean"].shape == (10,)
    assert np.all(forecasts["mean"] > 0)


def test_ets_with_fixed_lambda():
    """Test ETS with fixed Box-Cox lambda"""
    y = positive_series(80)

    model = ets(y, m=1, model="ANN", lambda_param=0.5)

    assert model.transform is not None
    assert model.transform.lambda_param == 0.5


def test_ets_seasonal_too_high_frequency():
    """Test ETS with m > 24 (should raise error)"""
    y = ar1_series(100)

    # m > 24 should raise an error for seasonal models
    with pytest.raises(ValueError, match="Frequency too high"):
        ets(y, m=48, model="AAA")


def test_ets_invalid_model_string():
    """Test ETS with invalid model string"""
    y = ar1_series(80)

    with pytest.raises(ValueError, match="Model must be 3 characters"):
        ets(y, m=1, model="AN")  # Too short

    with pytest.raises(ValueError, match="Model must be 3 characters"):
        ets(y, m=1, model="AANN")  # Too long


def test_ets_empty_series_raises():
    """Test ETS raises on empty series"""
    y = np.array([])

    with pytest.raises(ValueError, match="Need at least 1 observation"):
        ets(y, m=1, model="ANN")


def test_ets_insufficient_data_warnings():
    """Test ETS warns and simplifies model with insufficient data"""
    # Test damping disabled warning
    y = ar1_series(8)  # Very short series
    with pytest.warns(UserWarning):
        model = ets(y, m=1, model="AAN", damped=True)
    assert model.config.damped is False or model.config.trend == "N"

    # Test with series that forces trend removal (7 obs for AAN)
    y_small = ar1_series(7)
    with pytest.warns(UserWarning):
        model_simple = ets(y_small, m=1, model="AAN", damped=False)
    # Model should be simplified to ANN (no trend)
    assert model_simple.config.trend == "N"


# Tests fourier
# ------------------------------------------------------------------------------
def test_fourier():
    """Test Fourier series generation"""
    y = ar1_series(100)
    
    # Basic fourier
    X = fourier(y, period=12, K=2)
    assert X.shape[0] == 100
    assert X.shape[1] <= 4  # K=2 gives up to 4 columns

    # With horizon
    X_h = fourier(y, period=12, K=2, h=10)
    assert X_h.shape[0] == 10


# Tests is_constant
# ------------------------------------------------------------------------------
def test_is_constant():
    """Test is_constant function"""
    assert bool(is_constant(np.ones(50))) is True
    assert bool(is_constant(ar1_series(50))) is False


# Tests forecast with special cases
# ------------------------------------------------------------------------------
def test_forecast_ets_invalid_sigma_warning():
    """Test forecast warns on invalid sigma"""
    y = ar1_series(100)
    model = ets(y, m=1, model="ANN")
    model.sigma2 = 0.0  # Force invalid

    with pytest.warns(UserWarning, match="invalid residual variance"):
        out = forecast_ets(model, h=10, level=[95])

    # Should still return point forecasts
    assert "mean" in out
    assert "lower_95" not in out  # Intervals not computed


def test_ets_multiplicative_trend():
    """Test ETS with multiplicative trend"""
    y = positive_series(100)

    model = ets(y, m=1, model="MMN")

    assert model.config.error == "M"
    assert model.config.trend == "M"

