# Unit test get_feature_importances method - Arima
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from ..._arima import Arima


def ar1_series(n=100, phi=0.7, sigma=1.0, seed=123):
    """Helper function to generate AR(1) series for testing."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


def test_get_feature_importances_raises_NotFittedError_when_not_fitted():
    """
    Test that get_feature_importances raises NotFittedError when the model
    has not been fitted.
    """
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))

    msg = re.escape(
        "This Arima instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=msg):
        model.get_feature_importances()


def test_get_feature_importances_returns_dataframe():
    """
    Test that get_feature_importances returns a pandas DataFrame.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)

    result = model.get_feature_importances()

    assert isinstance(result, pd.DataFrame)


def test_get_feature_importances_dataframe_columns():
    """
    Test that the returned DataFrame has 'feature' and 'importance' columns.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)

    result = model.get_feature_importances()

    assert list(result.columns) == ['feature', 'importance']


def test_get_feature_importances_no_exog_feature_names():
    """
    Test that get_feature_importances returns the expected feature names
    for a model fitted without exogenous variables (AR coefficients + intercept).
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)

    result = model.get_feature_importances()

    # AR(1) model with intercept should have: ar1, intercept
    assert 'ar1' in result['feature'].values
    assert 'intercept' in result['feature'].values
    assert len(result) == 2


def test_get_feature_importances_no_exog_exact_values():
    """
    Test that get_feature_importances returns exact coefficient values
    for a model fitted without exogenous variables.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)

    result = model.get_feature_importances()

    expected = pd.DataFrame({
        'feature': ['ar1', 'intercept'],
        'importance': [0.71161924, -0.15533905]
    })

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        atol=1e-4
    )


def test_get_feature_importances_feature_column_equals_coef_names():
    """
    Test that the 'feature' column equals coef_names_ attribute.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)

    result = model.get_feature_importances()

    assert list(result['feature']) == model.coef_names_


def test_get_feature_importances_importance_column_equals_coef():
    """
    Test that the 'importance' column equals the coef_ attribute.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)

    result = model.get_feature_importances()

    np.testing.assert_array_almost_equal(
        result['importance'].values, model.coef_, decimal=10
    )


def test_get_feature_importances_with_exog_feature_names():
    """
    Test that get_feature_importances includes exogenous variable names
    when the model was fitted with exogenous variables.
    """
    np.random.seed(111)
    y = ar1_series(100, seed=42)
    exog = pd.DataFrame({
        'f1': np.random.randn(100),
        'f2': np.random.randn(100)
    })
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y, exog=exog)

    result = model.get_feature_importances()

    assert 'f1' in result['feature'].values
    assert 'f2' in result['feature'].values
    assert 'ar1' in result['feature'].values
    assert len(result) == 4  # ar1 + intercept + f1 + f2


def test_get_feature_importances_with_exog_exact_values():
    """
    Test exact feature importances values when fitted with exogenous variables.
    """
    np.random.seed(111)
    y = ar1_series(100, seed=42)
    exog = pd.DataFrame({
        'f1': np.random.randn(100),
        'f2': np.random.randn(100)
    })
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y, exog=exog)

    result = model.get_feature_importances()

    expected = pd.DataFrame({
        'feature': ['ar1', 'intercept', 'f1', 'f2'],
        'importance': [0.73768179, 0.05372286, 0.05749417, 0.24147824]
    })

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        atol=1e-4
    )


def test_get_feature_importances_remains_available_after_reduce_memory():
    """
    Test that get_feature_importances remains available after reduce_memory()
    since coef_ is not deleted by reduce_memory().
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)
    model.reduce_memory()

    # Should not raise any error
    result = model.get_feature_importances()

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['feature', 'importance']
    assert len(result) > 0


def test_get_feature_importances_arma_model_feature_names():
    """
    Test feature names for an ARMA(1,1) model (AR + MA + intercept terms).
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    model.fit(y)

    result = model.get_feature_importances()

    assert 'ar1' in result['feature'].values
    assert 'ma1' in result['feature'].values
    assert len(result) >= 2
