# Unit test fit method - Arar
# ==============================================================================
import numpy as np
import pandas as pd
import pytest
from ..._arar import Arar


def ar1_series(n=80, phi=0.7, sigma=1.0, seed=123):
    """Helper function to generate AR(1) series for testing."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


def test_estimator_fit_and_attributes():
    """
    Test basic fit functionality and resulting attributes.
    """
    y = ar1_series(100)
    est = Arar()
    est.fit(y)

    assert hasattr(est, "model_")
    assert est.y_.shape == y.shape
    assert est.coef_.shape == (4,)
    assert isinstance(est.lags_, tuple) and len(est.lags_) == 4
    assert np.isscalar(est.sigma2_) and est.sigma2_ >= 1e-12
    assert est.psi_.ndim == 1 and est.psi_.size >= 1
    assert isinstance(est.sbar_, float)
    assert est.n_features_in_ == 1

    assert est.fitted_values_.shape == y.shape
    assert est.residuals_in_.shape == y.shape


def test_estimator_safe_false_too_short_raises():
    """
    Test that fit raises error for very short series when safe=False.
    """
    y = np.array([1.0]) 
    est = Arar(safe=False)
    
    with pytest.raises(ValueError, match="Series too short"):
        est.fit(y)


def test_arar_fit_with_exog():
    """
    Test Arar fit with exogenous variables.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog = np.column_stack([
        np.random.randn(n),
        np.sin(np.linspace(0, 4*np.pi, n))
    ])
    y = y + 0.5 * exog[:, 0] + 2.0 * exog[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog)
    
    assert model.exog_model_ is not None
    assert model.coef_exog_ is not None
    assert len(model.coef_exog_) == 2
    assert model.n_features_in_ == 1  # Always 1 for time series
    assert model.n_exog_features_in_ == 2
    assert model.y_.shape == y.shape
    assert np.allclose(model.y_, y)  # Should store original y, not residuals


def test_arar_fit_without_exog():
    """
    Test Arar fit without exogenous variables.
    """
    np.random.seed(42)
    y = np.random.randn(100).cumsum()
    
    model = Arar()
    model.fit(y)
    
    assert model.exog_model_ is None
    assert model.coef_exog_ is None
    assert model.n_features_in_ == 1  # Always 1 for time series
    assert model.n_exog_features_in_ == 0
    assert model.y_.shape == y.shape


def test_arar_fit_with_exog_length_mismatch():
    """
    Test that fit raises error when exog length doesn't match y length.
    """
    np.random.seed(42)
    y = np.random.randn(100).cumsum()
    exog_wrong = np.random.randn(80, 2)  # Wrong length
    
    model = Arar()
    
    with pytest.raises(ValueError, match="Length of exog"):
        model.fit(y, exog=exog_wrong)


def test_arar_fit_with_pandas_series_and_dataframe():
    """
    Test Arar with pandas Series for y and DataFrame for exog.
    """
    np.random.seed(42)
    n = 100
    y = pd.Series(np.random.randn(n).cumsum())
    exog = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.sin(np.linspace(0, 4*np.pi, n))
    })
    
    model = Arar()
    model.fit(y, exog=exog)
    
    assert model.n_features_in_ == 1  # Always 1 for time series
    assert model.n_exog_features_in_ == 2
    assert model.exog_model_ is not None


def test_arar_1d_exog():
    """
    Test Arar with 1D exogenous variable (should be reshaped to 2D).
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_1d = np.random.randn(n)
    
    model = Arar()
    model.fit(y, exog=exog_1d)
    
    assert model.n_features_in_ == 1  # Always 1 for time series
    assert model.n_exog_features_in_ == 1
    assert model.exog_model_ is not None


def test_arar_multiple_features_exog():
    """
    Test Arar with multiple exogenous features.
    """
    np.random.seed(42)
    n = 150
    y = np.random.randn(n).cumsum()
    exog = np.column_stack([
        np.random.randn(n),
        np.sin(np.linspace(0, 4*np.pi, n)),
        np.cos(np.linspace(0, 4*np.pi, n)),
        np.arange(n) / n  # trend
    ])
    y = y + 0.5 * exog[:, 0] + 2.0 * exog[:, 1] + 1.5 * exog[:, 2] + 10.0 * exog[:, 3]
    
    model = Arar()
    model.fit(y, exog=exog)
    
    assert model.n_features_in_ == 1  # Always 1 for time series
    assert model.n_exog_features_in_ == 4
    assert len(model.coef_exog_) == 4


def test_arar_exog_with_custom_params():
    """
    Test Arar with exogenous variables and custom max_ar_depth and max_lag.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog = np.random.randn(n, 2)
    y = y + 0.5 * exog[:, 0] + 2.0 * exog[:, 1]
    
    model = Arar(max_ar_depth=10, max_lag=20)
    model.fit(y, exog=exog)
    
    assert model.max_ar_depth == 10
    assert model.max_lag == 20
    assert model.exog_model_ is not None


def test_arar_exog_zero_variance_feature():
    """
    Test Arar handles exogenous features with zero variance.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    
    # One feature with variance, one constant
    exog = np.column_stack([
        np.random.randn(n),
        np.ones(n) * 5.0  # constant feature
    ])
    y = y + 0.5 * exog[:, 0]
    
    model = Arar()
    # Should not crash even with constant feature
    model.fit(y, exog=exog)
    
    assert model.n_features_in_ == 1  # Always 1 for time series
    assert model.n_exog_features_in_ == 2


def test_arar_exog_coefficient_recovery():
    """
    Test that exogenous coefficients are reasonably recovered.
    """
    np.random.seed(42)
    n = 500  # Larger sample for better coefficient recovery
    
    # Create data with known coefficients
    true_coef = [0.5, 2.0, -1.5]
    exog = np.column_stack([
        np.random.randn(n),
        np.sin(np.linspace(0, 10*np.pi, n)),
        np.cos(np.linspace(0, 10*np.pi, n))
    ])
    
    # Pure linear relationship (no ARAR component)
    y = 10.0 + exog @ true_coef + 0.1 * np.random.randn(n)
    
    model = Arar()
    model.fit(y, exog=exog)
    
    # Coefficients should be close to true values
    assert len(model.coef_exog_) == 3
    # Check they're in the right ballpark (not exact due to ARAR modeling)
    for i, true_c in enumerate(true_coef):
        # Allow for some deviation since ARAR will model residuals
        assert abs(model.coef_exog_[i] - true_c) < 1.0


def test_refit_resets_memory_reduced_flag():
    """
    Test that refitting resets the memory_reduced flag.
    """
    y = ar1_series(100)
    est = Arar()
    est.fit(y)
    est.reduce_memory()
    
    assert est.memory_reduced_ is True
    
    # Refit
    est.fit(y)
    
    assert est.memory_reduced_ is False
    assert est.fitted_values_ is not None
    assert est.residuals_in_ is not None
