# Unit test fit method - Ets
# ==============================================================================
import numpy as np
import pandas as pd
import pytest
from ..._ets import Ets


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


def test_estimator_with_pandas_series():
    """Test Ets estimator with pandas Series input"""
    y_array = ar1_series(80)
    y_series = pd.Series(y_array)

    est = Ets(m=1, model="ANN")
    est.fit(y_series)

    assert est.y_.shape == (80,)
    assert isinstance(est.y_, np.ndarray)


def test_estimator_with_fixed_parameters():
    """Test Ets estimator with fixed smoothing parameters"""
    y = ar1_series(80)
    est = Ets(m=1, model="ANN", alpha=0.3)
    est.fit(y)

    # Alpha should be close to the fixed value
    assert abs(est.params_.alpha - 0.3) < 0.1


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


def test_estimator_seasonal_model():
    """Test Ets estimator with seasonal model"""
    y = seasonal_series(120, m=12)
    est = Ets(m=12, model="AAA")
    est.fit(y)

    assert est.config_.season == "A"
    assert est.config_.m == 12
