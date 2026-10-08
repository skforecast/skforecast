# Unit tests for skforecast.stats.ets.ets_base
# ==============================================================================
import re
import numpy as np
import pytest
from .._ets_base import (
    PARAM_LOWER,
    PARAM_UPPER,
    _compute_prediction_variance,
    _forecast_ets,
    initial_smoothing_params,
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


def test_auto_ets_aicc_counts_the_variance_as_a_parameter(capsys):
    """
    Test that the AICc used by auto_ets to compare models counts the same
    parameters as the AIC it corrects (smoothing parameters, initial states
    and the variance), as in R's forecast::ets. The value is read from the
    verbose output, before the trend penalty is added.
    """
    y = np.array([
        10.5335, 7.5413, 11.1268, 9.2641, 8.7228, 6.3361, 12.537, 8.2767,
        10.8999, 8.3079, 11.4537, 10.9171, 8.5956, 6.7733, 11.2666, 8.953,
        7.377, 6.5915, 11.2618, 4.3945, 7.2241, 8.0699
    ])
    n = len(y)
    auto_ets(
        y, m=4, damped=False, allow_multiplicative=False, ic="aicc", verbose=True
    )
    out = capsys.readouterr().out
    printed = dict(re.findall(r"^\s+(\w+)\s*: AICC=(-?\d+\.\d+)", out, flags=re.M))

    assert set(printed) == {"ANA", "AAA"}
    for name in ("ANA", "AAA"):
        model = ets(y, m=4, model=name, damped=False)
        # Parameters counted by the AIC: aic = -2 * loglik + 2 * k
        k = (model.aic + 2 * model.loglik) / 2
        expected = model.aic + 2 * k * (k + 1) / (n - k - 1)
        assert float(printed[name]) == pytest.approx(expected, abs=0.006)


def test_ets_aicc_uses_the_parameters_of_the_aic():
    """
    Test that the AICc stored in the model is the AIC plus the small-sample
    correction, computed with the number of parameters of the AIC.
    """
    np.random.seed(42)
    y = np.cumsum(np.random.randn(40)) + 50
    n = len(y)
    for model_spec, damped, k in [("ANN", False, 3), ("AAN", True, 6), ("AAA", False, 9)]:
        model = ets(y, m=4, model=model_spec, damped=damped)
        # aic = -2 * loglik + 2 * k
        assert (model.aic + 2 * model.loglik) / 2 == pytest.approx(k)
        expected = model.aic + 2 * k * (k + 1) / (n - k - 1)
        assert model.aicc == pytest.approx(expected)


def test_ets_aicc_when_damping_is_disabled_for_short_series():
    """
    Test that the AICc counts the parameters of the fitted model when the
    series is too short for the requested one. With 9 observations the damped
    trend is dropped, so the AICc is the same as that of the model without it.
    """
    y = np.array([50.1, 52.3, 53.9, 56.2, 58.4, 59.7, 62.1, 64.0, 66.3])
    with pytest.warns(UserWarning, match="Disabling damping"):
        model_damped = ets(y, m=1, model="AAN", damped=True)
    model = ets(y, m=1, model="AAN", damped=False)

    assert model_damped.config.damped is False
    assert model_damped.aicc == pytest.approx(model.aicc)
    assert np.isfinite(model.aicc)


def test_ets_aicc_is_infinite_when_correction_is_not_defined():
    """
    Test that the AICc is infinite, instead of negative or a division by
    zero, when n <= k + 1. Only a constant series can reach this case, since
    the rest of the models need more than k + 3 observations.
    """
    for n in (2, 3, 4):
        with pytest.warns(UserWarning, match="Series is constant"):
            model = ets(np.full(n, 5.0), m=1, model="ZZZ")
        assert model.aicc == np.inf

    with pytest.warns(UserWarning, match="Series is constant"):
        model = ets(np.full(10, 5.0), m=1, model="ZZZ")
    assert model.aicc == pytest.approx(model.aic + 2 * 3 * 4 / (10 - 3 - 1))


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


@pytest.mark.parametrize(
    "model",
    ["XAN", "NAN", "AXN", "AAX", "aan"],
    ids=lambda model: f'model: {model}'
)
def test_ets_ValueError_when_model_has_invalid_components(model):
    """
    Test that ets raises a ValueError when a component of the model string is
    not valid. Before the fix, a KeyError was raised.
    """
    y = ar1_series(80)

    err_msg = re.escape(
        f"Invalid model '{model}'. The error component must be 'A' or 'M', "
        f"and the trend and seasonal components must be 'N', 'A' or 'M' "
        f"(uppercase), or use model='ZZZ' for automatic selection."
    )
    with pytest.raises(ValueError, match=err_msg):
        ets(y, m=1, model=model)


@pytest.mark.parametrize(
    "model",
    ["ZZN", "AZN", "ANZ", "ZAA"],
    ids=lambda model: f'model: {model}'
)
def test_ets_ValueError_when_model_is_partial_automatic(model):
    """
    Test that ets raises a ValueError when the model string mixes automatic
    ('Z') and fixed components. Before the fix, a KeyError: 'Z' was raised.
    """
    y = ar1_series(80)

    err_msg = re.escape(
        f"Partial automatic model specifications such as '{model}' are not supported."
    )
    with pytest.raises(ValueError, match=err_msg):
        ets(y, m=1, model=model)


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


# Regression tests of the estimation
# ------------------------------------------------------------------------------
def r_admissible(alpha, beta, gamma, phi, m):
    """
    Reference implementation of `admissible` of R's forecast::ets (seasonal
    case), with the roots computed by numpy.
    """
    if gamma < max(1 - 1 / phi - alpha, 0) or gamma > 1 + 1 / phi - alpha:
        return False
    if alpha < 1 - 1 / phi - gamma * (1 - m + phi + phi * m) / (2 * phi * m):
        return False
    if beta < -(1 - phi) * (gamma / m + alpha):
        return False
    P = np.r_[
        phi * (1 - alpha - gamma),
        alpha + beta - alpha * phi + gamma - 1,
        np.repeat(alpha + beta - alpha * phi, m - 2),
        alpha + beta - phi,
        1.0,
    ]
    return np.max(np.abs(np.roots(P[::-1]))) <= 1 + 1e-10


@pytest.mark.parametrize("m", [2, 4, 7, 12])
def test_admissible_seasonal_matches_R_characteristic_polynomial(m):
    """
    Test that the admissibility of seasonal models matches the characteristic
    polynomial of R's forecast::ets (the polynomial used before had the wrong
    degree and coefficients, and its root finding failed on complex roots).
    """
    rng = np.random.default_rng(m)
    n_admissible = 0
    for _ in range(500):
        alpha, beta, gamma = rng.uniform(0, 1.2, 3)
        phi = rng.uniform(0.75, 1.0)
        expected = r_admissible(alpha, beta, gamma, phi, m)
        assert admissible(alpha, beta, gamma, phi, m) == expected
        n_admissible += expected

    assert 0 < n_admissible < 500


def test_check_param_trend_and_season_without_damping():
    """
    Test that the usual bounds of beta and gamma are read from their own
    positions, and that a model without damping is checked with phi = 1.
    """
    assert check_param(0.5, 0.3, None, None, PARAM_LOWER, PARAM_UPPER, "both", 1)
    assert check_param(0.3, 0.05, 0.1, None, PARAM_LOWER, PARAM_UPPER, "both", 12)
    assert not check_param(0.5, 0.6, None, None, PARAM_LOWER, PARAM_UPPER, "both", 1)
    assert not check_param(0.5, 0.3, 0.6, None, PARAM_LOWER, PARAM_UPPER, "both", 12)


def test_initial_smoothing_params_inside_usual_bounds():
    """
    Test that the starting values follow R's initparam and are strictly inside
    the usual bounds (phi used to start at its upper bound).
    """
    config = ETSConfig("A", "A", "A", True, 12)
    alpha, beta, gamma, phi = initial_smoothing_params(config)

    np.testing.assert_allclose(alpha, 1e-4 + 0.2 * (0.9999 - 1e-4) / 12)
    np.testing.assert_allclose(beta, 1e-4 + 0.1 * (alpha - 1e-4))
    np.testing.assert_allclose(gamma, 1e-4 + 0.05 * (1 - alpha - 1e-4))
    np.testing.assert_allclose(phi, 0.8 + 0.99 * 0.18)
    assert phi < PARAM_UPPER[3]


@pytest.mark.parametrize(
    "model, damped",
    [("ANN", False), ("AAN", False), ("AAN", True), ("ANA", False), ("AAA", False)],
    ids=lambda x: str(x),
)
def test_ets_estimates_do_not_stay_at_starting_values(model, damped):
    """
    Test that the smoothing parameters are estimated (they used to stay at
    their starting values because the parameter checks rejected every
    candidate) and that the likelihood improves on the starting values.
    """
    rng = np.random.default_rng(0)
    t = np.arange(96)
    y = 50 + 0.2 * t + 5 * np.sin(2 * np.pi * t / 4) + np.cumsum(rng.normal(0, 1, 96))
    m = 4 if model[2] != "N" else 1
    config = ETSConfig(model[0], model[1], model[2], damped, m)
    start = initial_smoothing_params(config)

    fitted = ets(y, m=m, model=model, damped=damped)
    at_start = ets(
        y, m=m, model=model, damped=damped,
        alpha=start[0],
        beta=start[1] if model[1] != "N" else None,
        gamma=start[2] if model[2] != "N" else None,
        phi=start[3] if damped else None,
    )

    assert fitted.params.alpha != pytest.approx(start[0], abs=1e-3)
    assert fitted.loglik > at_start.loglik + 0.1


def test_ets_fixed_parameters_are_not_estimated():
    """
    Test that fixed smoothing parameters keep their values and constrain the
    estimated ones (beta <= alpha <= 1 - gamma).
    """
    y = trend_series(100) + 3 * np.sin(2 * np.pi * np.arange(100) / 4)

    model = ets(y, m=4, model="AAA", beta=0.3, gamma=0.5)

    assert model.params.beta == 0.3
    assert model.params.gamma == 0.5
    assert 0.3 <= model.params.alpha <= 0.5

    model = ets(y, m=1, model="AAN", alpha=0.4, damped=True, phi=0.9)
    assert model.params.alpha == 0.4
    assert model.params.phi == 0.9
    assert model.params.beta <= 0.4


def test_ets_fixed_parameters_out_of_bounds_raises():
    """
    Test that fixed smoothing parameters outside the usual bounds raise an
    error instead of being silently replaced.
    """
    y = trend_series(100)
    err_msg = re.escape(
        "The fixed smoothing parameters are out of range for the 'both' bounds"
    )
    with pytest.raises(ValueError, match=err_msg):
        ets(y, m=1, model="AAN", alpha=0.2, beta=0.5)


@pytest.mark.parametrize(
    "model_spec, fixed, fixed_msg",
    [
        ("AAA", {"alpha": 0.9999}, "alpha=0.9999"),
        ("ANA", {"gamma": 0.9999}, "gamma=0.9999"),
        ("AAA", {"beta": 0.6, "gamma": 0.5}, "beta=0.6, gamma=0.5"),
    ],
    ids=lambda x: str(x),
)
def test_ets_ValueError_when_fixed_parameters_leave_empty_range(model_spec, fixed, fixed_msg):
    """
    Test that fixed smoothing parameters that leave no value within the usual
    bounds for an estimated one raise a ValueError that names only the fixed
    parameters (a fixed alpha raised the error of scipy, and the message
    named the starting values of the estimated beta and gamma).
    """
    y = trend_series(100) + 3 * np.sin(2 * np.pi * np.arange(100) / 4)
    err_msg = re.escape(
        f"No value of the estimated smoothing parameters satisfies the usual "
        f"bounds with the fixed {fixed_msg} (1e-4 <= beta <= alpha and "
        f"1e-4 <= gamma <= 1 - alpha)."
    )
    with pytest.raises(ValueError, match=err_msg):
        ets(y, m=4, model=model_spec, **fixed)


@pytest.mark.parametrize(
    "fixed, fixed_msg",
    [
        ({"beta": 0.3, "gamma": 0.5}, "beta=0.3, gamma=0.5"),
        ({"beta": 0.9999}, "beta=0.9999"),
    ],
    ids=lambda x: str(x),
)
def test_ets_ValueError_when_fixed_parameters_leave_no_admissible_value(fixed, fixed_msg):
    """
    Test that fixed smoothing parameters that leave no admissible value of the
    estimated ones raise a ValueError, as R's check.param, instead of
    returning the penalized starting point as a fitted model. With m=12,
    beta=0.3 and gamma=0.5 no alpha in [0.3, 0.5] is admissible; with
    beta=0.9999, alpha=0.9999 leaves no gamma >= 1e-4 below 1 - alpha.
    """
    t = np.arange(120)
    y = trend_series(120) + 3 * np.sin(2 * np.pi * t / 12)
    err_msg = re.escape(
        f"No value of the estimated smoothing parameters satisfies the 'both' "
        f"bounds with the fixed {fixed_msg}."
    )
    with pytest.raises(ValueError, match=err_msg):
        ets(y, m=12, model="AAA", **fixed)


@pytest.mark.parametrize("model_spec", ["ANN", "AAN"])
@pytest.mark.parametrize("shift", [-2e5, 5e6], ids=lambda x: f"shift: {x}")
def test_ets_additive_model_invariant_to_level_shift(shift, model_spec):
    """
    Test that an additive model of a shifted series gives the same estimates
    and likelihood. Series around -200000 collided with the -99999 sentinel
    of the likelihood, and series above 1e6 with the bounds of the initial
    states.
    """
    rng = np.random.default_rng(1)
    y = (
        50 + 0.3 * np.arange(100) + np.cumsum(rng.normal(0, 1, 100))
        + rng.normal(0, 1, 100)
    )

    model = ets(y, m=1, model=model_spec)
    model_shift = ets(y + shift, m=1, model=model_spec)

    np.testing.assert_allclose(model_shift.params.alpha, model.params.alpha, atol=1e-5)
    np.testing.assert_allclose(model_shift.params.beta, model.params.beta, atol=1e-5)
    np.testing.assert_allclose(model_shift.loglik, model.loglik, rtol=1e-6)
    assert np.isfinite(model_shift.aic)


def test_ets_multiplicative_model_non_positive_series_raises():
    """
    Test that multiplicative models are rejected for series with non-positive
    values, and that the automatic selection only considers additive models.
    """
    y = ar1_series(100)
    err_msg = re.escape("Inappropriate model 'MNN' for data with negative or zero values")
    with pytest.raises(ValueError, match=err_msg):
        ets(y, m=1, model="MNN")

    model = auto_ets(y - y.mean(), m=1)
    assert model.config.error == "A"
    assert model.config.trend != "M"


def test_forecast_ets_multiplicative_trend_non_positive_state_is_nan():
    """
    Test that a multiplicative trend with a non-positive state forecasts NaN
    (it used to return the -99999 sentinel).
    """
    forecasts = _forecast_ets(10.0, -0.5, np.zeros(1), 3, 1, 2, 0, 1.0)
    assert np.all(np.isnan(forecasts))


def test_simulate_ets_reproducible():
    """
    Test that simulated paths, and the intervals of models without an
    analytical variance, are reproducible.
    """
    y = positive_series(100)
    model = ets(y, m=1, model="MAN")

    sim_1 = simulate_ets(model, h=6, n_sim=200)
    sim_2 = simulate_ets(model, h=6, n_sim=200)
    sim_3 = simulate_ets(model, h=6, n_sim=200, random_state=456)

    assert sim_1.shape == (200, 6)
    np.testing.assert_array_equal(sim_1, sim_2)
    assert not np.allclose(sim_1, sim_3)

    out_1 = forecast_ets(model, h=6, level=[80, 95])
    out_2 = forecast_ets(model, h=6, level=[80, 95])
    for key in out_1:
        np.testing.assert_array_equal(out_1[key], out_2[key])


def test_box_cox_find_lambda_guerrero():
    """
    Test that the automatic Box-Cox lambda uses Guerrero's method as R's
    forecast::BoxCox.lambda (it always returned the lower bound -1).
    BoxCox.lambda(AirPassengers) = -0.2947156 in R (whose optimizer stops at
    a tolerance of about 1.2e-4).
    """
    from ...tests.tests_arima.fixtures_arima import air_passengers

    lam = BoxCoxTransform.find_lambda(air_passengers.to_numpy(dtype=float), m=12)

    np.testing.assert_allclose(lam, -0.2947156, atol=1e-4)


def test_box_cox_bias_adjustment_formula():
    """
    Test the bias-adjusted back-transformation of R's forecast::InvBoxCox:
    y * (1 + 0.5 * variance * (1 - lambda) / y^(2 * lambda)).
    """
    transform = BoxCoxTransform(lambda_param=0.5, shift=0.0)
    y = np.array([4.0, 9.0, 16.0])
    variance = 0.3

    y_back = transform.inverse_transform(transform.transform(y), True, variance)

    np.testing.assert_allclose(y_back, y * (1 + 0.5 * variance * 0.5 / y))


def test_forecast_ets_box_cox_intervals_on_original_scale():
    """
    Test that the prediction intervals of a Box-Cox model are back-transformed
    (they mixed the forecast on the original scale with the standard
    deviation on the transformed scale).
    """
    y = positive_series(100)

    model_log = ets(y, m=1, model="ANN", lambda_param=0.0)
    model_on_log = ets(np.log(y), m=1, model="ANN")
    out_log = forecast_ets(model_log, h=5, bias_adjust=False, level=[80, 95])
    out_on_log = forecast_ets(model_on_log, h=5, level=[80, 95])

    for key in out_log:
        np.testing.assert_allclose(out_log[key], np.exp(out_on_log[key]), rtol=1e-10)


@pytest.mark.parametrize(
    "lambda_param, y_trans, expected",
    [
        (0.5, [-1.0, -2.0, -3.0, -4.0], [0.25, 0.0, -0.25, -1.0]),
        (-0.5, [1.0, 1.9, 2.1, 3.0], [4.0, 400.0, np.nan, np.nan]),
    ],
    ids=lambda x: str(x),
)
def test_box_cox_inverse_transform_out_of_range_as_R(lambda_param, y_trans, expected):
    """
    Test that values outside the range of the Box-Cox transformation
    (lambda * y + 1 < 0) are back-transformed as R's forecast::InvBoxCox:
    NaN when lambda < 0, and with their sign otherwise, so the inverse is
    monotonic. With even powers they were positive (0.25 for y = -3 and
    lambda = 0.5, 4 for y = 3 and lambda = -0.5).
    """
    transform = BoxCoxTransform(lambda_param=lambda_param, shift=0.0)

    y_back = transform.inverse_transform(np.array(y_trans))

    np.testing.assert_allclose(y_back, expected)


def test_forecast_ets_box_cox_negative_lambda_upper_bounds_out_of_range():
    """
    Test that the upper bounds of a Box-Cox model with lambda < 0 that fall
    outside the range of the transformation are NaN, as in R, instead of
    finite values below the mean (the 95% upper bound was 0.56, below the
    mean, 1.24, and below the 80% upper bound, 11.8).
    """
    rng = np.random.default_rng(7)
    y = rng.exponential(1.0, 120) + 0.01
    model = ets(y, m=1, model="ANN", lambda_param=-0.5)

    out = forecast_ets(model, h=3, level=[80, 95])

    assert np.all(np.isnan(out["upper_80"]))
    assert np.all(np.isnan(out["upper_95"]))
    assert np.all(out["lower_95"] < out["lower_80"])
    assert np.all(out["lower_80"] < out["mean"])


def test_ets_multistart_finds_global_optimum_air_passengers():
    """
    Test that the starting points of the optimizer find the best optimum of
    an ANA model of AirPassengers (m=12). From R's starting values alone,
    L-BFGS-B and Nelder-Mead stop at a local optimum with a -2 log-likelihood
    51.8 higher (815.20 instead of 763.42). The value changes by about 1e-8
    under 1-ULP perturbations of the series.
    """
    from ...tests.tests_arima.fixtures_arima import air_passengers

    y = air_passengers.to_numpy(dtype=float)
    model = ets(y, m=12, model="ANA")

    np.testing.assert_allclose(-2 * model.loglik, 763.4188655657, atol=1e-2)


def test_compute_prediction_variance_damped_additive_seasonal_matches_R():
    """
    Test that the forecast variance of an ETS(A,Ad,A) model is analytical and
    matches R's forecast.ets (class 1 state-space formula). Parameters and
    variances from forecast 9.0.2 for a quarterly series.
    """
    y = 50 + 0.3 * np.arange(60) + 5 * np.sin(2 * np.pi * np.arange(60) / 4)
    model = ets(y, m=4, model="AAA", damped=True)
    model.params = ETSParams(
        alpha=0.02314127, beta=0.02313899, gamma=0.0001478949, phi=0.9799998,
        init_states=model.params.init_states
    )
    model.sigma2 = 2.773066

    var = _compute_prediction_variance(model, h=8)
    expected = np.array([
        2.773066, 2.778888, 2.791725, 2.814097, 2.848454, 2.896828, 2.961382,
        3.044053
    ])
    np.testing.assert_allclose(var, expected, rtol=1e-6)


@pytest.mark.parametrize("lambda_param", [0.0, 0.5], ids=lambda x: f"lambda: {x}")
def test_forecast_ets_box_cox_bias_adjustment_uses_forecast_variance(lambda_param):
    """
    Test that the bias-adjusted point forecasts of a Box-Cox model use the
    forecast variance of each horizon, as R's forecast.ets (InvBoxCox with
    the forecast variance), and R's formula y * (1 + variance / 2) when
    lambda = 0. They used the one-step variance for every horizon, and
    exp(variance / 2) when lambda = 0.
    """
    y = positive_series(100) + 0.5 * np.arange(100)
    model = ets(y, m=1, model="AAN", lambda_param=lambda_param)
    h = 12

    mean_adjusted = forecast_ets(model, h=h, bias_adjust=True)["mean"]
    mean = forecast_ets(model, h=h, bias_adjust=False)["mean"]

    var = _compute_prediction_variance(model, h)
    if lambda_param == 0.0:
        expected = mean * (1 + var / 2)
    else:
        expected = mean * (1 + (1 - lambda_param) * var / (2 * mean ** (2 * lambda_param)))
    assert np.all(np.diff(var) > 0)
    np.testing.assert_allclose(mean_adjusted, expected, rtol=1e-12)


def test_ets_ZZZ_runs_automatic_selection():
    """
    Test that ets() with model='ZZZ' runs the automatic selection for any
    seasonal period, passing the Box-Cox options (it raised KeyError: 'Z'
    when m <= 24).
    """
    from ...tests.tests_arima.fixtures_arima import air_passengers

    y = air_passengers.to_numpy(dtype=float)

    model = ets(y, m=12, model="ZZZ", lambda_param=0.0)
    expected = auto_ets(y, m=12, damped=False, lambda_param=0.0)

    assert model.config == expected.config
    assert model.config.error == "A"
    assert model.transform.lambda_param == 0.0
    np.testing.assert_allclose(model.loglik, expected.loglik)
