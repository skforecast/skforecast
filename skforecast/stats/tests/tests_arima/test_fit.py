# Unit test fit method - Arima
# ==============================================================================
import platform
import numpy as np
import pandas as pd
import pytest
from ..._arima import Arima
from .fixtures_arima import air_passengers


# Fixture functions
# ------------------------------------------------------------------------------
def ar1_series(n=100, phi=0.7, sigma=1.0, seed=123):
    """Helper function to generate AR(1) series for testing."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


# Input and Parameter Validation
# ------------------------------------------------------------------------------
def test_arima_fit_invalid_y_type_raises():
    """
    Test that fit raises TypeError for invalid y type.
    """
    y = [1, 2, 3, 4, 5]  # List, not Series or ndarray
    model = Arima(order=(1, 0, 0))
    msg = "`y` must be a pandas Series or numpy array."
    with pytest.raises(TypeError, match=msg):
        model.fit(y)


def test_arima_fit_invalid_exog_type_raises():
    """
    Test that fit raises TypeError for invalid exog type.
    """
    y = np.random.randn(50)
    exog = [1, 2, 3]  # List, not valid type
    model = Arima(order=(1, 0, 0))
    msg = "`exog` must be a pandas Series, DataFrame, numpy array, or None."
    with pytest.raises(TypeError, match=msg):
        model.fit(y, exog=exog)


def test_arima_fit_multidimensional_y_raises():
    """
    Test that fit raises error for multidimensional y input.
    """
    y = np.random.randn(50, 2)
    model = Arima(order=(1, 0, 0))
    msg = "`y` must be 1-dimensional."
    with pytest.raises(ValueError, match=msg):
        model.fit(y)


def test_arima_fit_with_exog_length_mismatch():
    """
    Test that fit raises error when exog length doesn't match y length.
    """
    y = ar1_series(100)
    exog_wrong = np.random.randn(80, 2)  # Wrong length
    model = Arima(order=(1, 0, 0))
    msg = r"Length of `exog` \(80\) does not match length of `y` \(100\)"
    with pytest.raises(ValueError, match=msg):
        model.fit(y, exog=exog_wrong)


def test_arima_fit_with_exog_3d_raises():
    """
    Test that fit raises error for 3D exog input.
    """
    y = ar1_series(50)
    exog_3d = np.random.randn(50, 2, 3)  # 3D array
    model = Arima(order=(1, 0, 0))
    msg = "`exog` must be 1- or 2-dimensional."
    with pytest.raises(ValueError, match=msg):
        model.fit(y, exog=exog_3d)


# Basic Fit Functionality Tests
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "y_input_type",
    ["numpy_array", "pandas_series"],
    ids=lambda x: f"y_as_{x}"
)
def test_arima_fit_with_default_parameters(y_input_type):
    """
    Test fit with default parameters and verify all attributes with exact values.
    Tests both numpy array and pandas Series inputs.
    """
    y = ar1_series(100, seed=123)
    
    # Convert to appropriate type
    if y_input_type == "pandas_series":
        y = pd.Series(y, index=pd.date_range(start='2020-01-01', periods=100, freq='D'))
    
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    result = model.fit(y)
    
    # Check return value
    assert result is model
    
    # Check exact coefficient values
    expected_coef = np.array([0.58779763, 0.06262394, 0.24901211])
    np.testing.assert_array_almost_equal(model.coef_, expected_coef, decimal=6)
    
    # Check exact fit statistics
    assert model.sigma2_ > 0
    np.testing.assert_almost_equal(model.sigma2_, 0.7900849128777775, decimal=6)
    np.testing.assert_almost_equal(model.loglik_, -130.57765786963125, decimal=5)
    
    # Check that all fitted attributes are set with correct types/values
    assert model.model_ is not None
    assert len(model.y_train_) == 100
    assert isinstance(model.coef_, np.ndarray)
    assert isinstance(model.coef_names_, list)
    assert len(model.coef_names_) == len(model.coef_)
    assert model.aic_ > 0
    # BIC can be None in some cases
    if model.bic_ is not None:
        assert model.bic_ > 0
        assert model.bic_ > model.aic_  # BIC penalizes complexity more
    assert isinstance(model.arma_, list)
    assert model.converged_ is True
    assert model.n_features_in_ == 1
    assert model.n_exog_features_in_ == 0
    assert len(model.fitted_values_) == 100
    assert len(model.in_sample_residuals_) == 100
    assert model.var_coef_ is not None
    assert model.is_memory_reduced is False


def test_arima_fit_ar_model():
    """
    Test fitting a pure AR model (p, 0, 0) with exact coefficient values.
    """
    y = ar1_series(100, phi=0.7, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    # Check exact AR coefficient (should be close to true value 0.7)
    expected_coef = np.array([0.71227673, -0.13859403])
    np.testing.assert_array_almost_equal(model.coef_, expected_coef, decimal=6)
    
    # Check exact sigma2
    np.testing.assert_almost_equal(model.sigma2_, 0.5916008281275975, decimal=6)
    
    assert model.converged_ is True
    assert len(model.coef_) == 2  # AR coef + intercept
    assert model.n_exog_features_in_ == 0


def test_arima_fit_ma_model():
    """
    Test fitting a pure MA model (0, 1, 1).
    """
    np.random.seed(123)
    y = np.cumsum(np.random.randn(50))
    model = Arima(order=(0, 1, 1), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    # Check exact MA coefficient (values verified against corrected Kalman filter)
    expected_coef = np.array([0.13056289])
    np.testing.assert_array_almost_equal(model.coef_, expected_coef, decimal=6)
    expected_sigma2 = 1.37647445
    np.testing.assert_almost_equal(model.sigma2_, expected_sigma2, decimal=6)
    assert model.converged_ is True
    assert len(model.coef_) >= 1


def test_arima_fit_seasonal_model():
    """
    Test fitting a seasonal ARIMA model with exact coefficient values.
    """
    np.random.seed(123)
    y = np.cumsum(np.random.randn(100))
    model = Arima(order=(1, 0, 0), seasonal_order=(1, 0, 0), m=12)
    model.fit(y)
    
    # Check exact coefficients
    expected_coef = np.array([0.94557857, -0.00845976, 2.49797772])
    np.testing.assert_array_almost_equal(model.coef_, expected_coef, decimal=4)
    
    # Check exact sigma2
    np.testing.assert_almost_equal(model.sigma2_, 1.2449496729615976, decimal=6)
    
    # Check ARMA specification
    assert model.arma_[4] == 12  # Check m is stored correctly
    assert model.arma_ == [1, 0, 1, 0, 12, 0, 0]  # [p, q, P, Q, m, d, D]
    
    assert model.converged_ in [True, False]  # May not converge depending on data
    assert model.n_features_in_ == 1
    assert model.n_exog_features_in_ == 0
    assert len(model.coef_) == 3  # AR + SAR + intercept


def test_arima_fit_with_exog_numpy_array():
    """
    Test fit with exogenous variables as numpy array with exact coefficient values.
    """
    rng = np.random.default_rng(42)
    y = np.cumsum(rng.standard_normal(80))
    exog = rng.standard_normal((80, 2))
    
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y, exog=exog)
    
    # Check exact coefficients (R-based implementation values)
    expected_coef = np.array([0.91005932,  0.11445059, -0.69434627,  0.8389574 ,  0.80990506])
    np.testing.assert_array_almost_equal(model.coef_, expected_coef, decimal=5)

    assert model.n_exog_features_in_ == 2
    assert len(model.coef_) == 5  # AR + MA + 2 exog + intercept
    assert model.converged_ in [True, False]
    assert model.n_features_in_ == 1


def test_arima_fit_with_exog_pandas_series():
    """
    Test fit with exogenous variable as pandas Series with exact values.
    """
    np.random.seed(42)
    y = np.cumsum(np.random.randn(80))
    exog = pd.Series(np.random.randn(80))
    
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y, exog=exog)
    
    # Check exact coefficients (R-based implementation values)
    expected_coef = np.array([0.97514494,  5.17778438, -0.30422759])
    np.testing.assert_array_almost_equal(model.coef_, expected_coef, decimal=4)

    # Check exact sigma2 and aic
    np.testing.assert_almost_equal(model.sigma2_, 0.9225263409649781, decimal=5)
    np.testing.assert_almost_equal(model.aic_, 234.6069627223404, decimal=4)
    
    assert model.n_exog_features_in_ == 1
    assert len(model.coef_) == 3  # AR + exog + intercept
    assert model.converged_ is True


def test_arima_fit_with_exog_pandas_dataframe():
    """
    Test fit with exogenous variables as pandas DataFrame with exact values.
    """
    np.random.seed(42)
    y = np.cumsum(np.random.randn(80))
    exog = pd.DataFrame({
        'x1': np.random.randn(80),
        'x2': np.random.randn(80),
        'x3': np.random.randn(80)
    })
    
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y, exog=exog)
    
    # Check exact coefficients
    expected_coef = np.array([0.97567034, 2.526397, -1.83108388, 4.16904743, -0.96568156])
    np.testing.assert_array_almost_equal(model.coef_, expected_coef, decimal=4)
    
    assert model.n_exog_features_in_ == 3
    assert len(model.coef_) == 5  # AR + 3 exog + intercept
    assert model.converged_ is True


def test_arima_fit_stores_arma_specification():
    """
    Test that fit correctly stores ARMA specification.
    """
    y = ar1_series(100)
    model = Arima(order=(2, 1, 1), seasonal_order=(1, 0, 1), m=4)
    model.fit(y)
    
    # arma_ should be [p, q, P, Q, m, d, D]
    assert model.arma_ == [2, 1, 1, 1, 4, 1, 0]


def test_arima_fit_2d_y_with_single_column():
    """
    Test that fit handles 2D y with single column correctly and checks exact values.
    """
    y = ar1_series(50).reshape(-1, 1)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)
    
    # Check exact coefficients
    expected_coef = np.array([0.52901588, 0.47293639])
    np.testing.assert_array_almost_equal(model.coef_, expected_coef, decimal=6)
    
    # Check exact sigma2 and aic
    np.testing.assert_almost_equal(model.sigma2_, 0.9363021000223529, decimal=6)
    np.testing.assert_almost_equal(model.aic_, 145.2596115789171, decimal=5)
    
    assert model.y_train_.ndim == 1
    assert len(model.y_train_) == 50
    assert len(model.coef_) == 2  # AR + intercept
    assert model.converged_ is True


@pytest.mark.skipif(
    platform.system() == 'Darwin',
    reason="ARIMA optimizer converges to different values on macOS"
)
def test_arima_fit_method_css():
    """
    Test fitting with CSS method and verify exact values.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0), method="CSS")
    model.fit(y)
    
    # Check exact coefficients
    expected_coef = np.array([0.66519088, 0.10578618, -0.17749673])
    np.testing.assert_array_almost_equal(model.coef_, expected_coef, decimal=5)
    
    # Check exact sigma2 (aic is nan for CSS method)
    np.testing.assert_almost_equal(model.sigma2_, 0.5914853493450223, decimal=5)
    
    assert "CSS" in model.model_['method'] or "ARIMA" in model.model_['method']
    assert model.converged_ is True
    assert len(model.coef_) == 3  # AR + MA + intercept
    assert model.n_features_in_ == 1


def test_arima_fit_method_ml():
    """
    Test fitting with ML method and verify exact values.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0), method="ML")
    model.fit(y)
    
    # Check exact coefficients
    expected_coef = np.array([0.65964959, 0.1100205, -0.13214965])
    np.testing.assert_array_almost_equal(model.coef_, expected_coef, decimal=6)
    
    # Check exact sigma2 and aic
    np.testing.assert_almost_equal(model.sigma2_, 0.5875533283678644, decimal=6)
    np.testing.assert_almost_equal(model.aic_, 239.91965018574223, decimal=5)
    
    assert "ML" in model.model_['method'] or "ARIMA" in model.model_['method']
    assert len(model.coef_) == 3  # AR + MA + intercept
    assert model.loglik_ is not None
    assert model.converged_ in [True, False]


def test_arima_fit_without_mean():
    """
    Test fitting with include_mean=False and verify exact coefficient values.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0), include_mean=False)
    model.fit(y)
    
    # Check exact coefficient (only AR, no intercept)
    expected_coef = np.array([0.71614504])
    np.testing.assert_array_almost_equal(model.coef_, expected_coef, decimal=6)
    
    # Should have only 1 coefficient (AR term, no intercept)
    assert len(model.coef_) == 1
    assert model.converged_ is True
    assert model.sigma2_ > 0
    assert model.n_exog_features_in_ == 0


def test_arima_fit_returns_self():
    """
    Test that fit returns self for method chaining and model is properly fitted.
    """
    y = ar1_series(50)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    result = model.fit(y)
    
    assert result is model
    assert model.converged_ is True
    assert len(model.coef_) >= 1
    assert np.all(np.isfinite(model.coef_))
    assert model.sigma2_ > 0
    assert model.model_ is not None


def test_arima_fit_auto_arima_air_passengers_data():
    """
    Test fit works correctly with auto ARIMA mode when applied to
    Air Passengers dataset.
    """

    model = Arima(
        order=None,
        seasonal_order=None,
        start_p=0,
        start_q=0,
        max_p=5,
        max_q=5,
        max_P=2,
        max_Q=2,
        max_order=5,
        max_d=2,
        max_D=1,
        ic="aic",
        seasonal=True,
        test="kpss",
        nmodels=94,
        optim_method="BFGS",
        m=12,
        trace=False,
        stepwise=True,
    )
    model.fit(air_passengers, suppress_warnings=True)
    
    assert model.is_auto is True
    assert model.best_params_['order'] == (2, 1, 1)
    assert model.best_params_['seasonal_order'] == (0, 1, 0)
    assert model.best_params_['m'] == 12
    assert model.estimator_name_ == "AutoArima(2,1,1)(0,1,0)[12]"
