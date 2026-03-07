# Unit test predict method - Arar
# ==============================================================================
import re
import pytest
import numpy as np
from sklearn.exceptions import NotFittedError
from ..._arar import Arar


def ar1_series(n=80, phi=0.7, sigma=1.0, seed=123):
    """Helper function to generate AR(1) series for testing."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


def test_predict_raises_errors_for_invalid_steps_and_unfitted_model():
    """
    Test that predict raises errors for invalid steps and unfitted model.
    """
    est = Arar()

    error_msg = re.escape(
        f"This {type(est).__name__} instance is not fitted yet. Call "
        f"'fit' with appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=error_msg):
        est.predict(steps=1)

    y = ar1_series(50)
    est.fit(y)
    
    with pytest.raises(ValueError, match="`steps` must be a positive integer."):
        est.predict(steps=0)
    with pytest.raises(ValueError, match="`steps` must be a positive integer."):
        est.predict(steps=-2)
    with pytest.raises(ValueError, match="`steps` must be a positive integer."):
        est.predict(steps=1.5)


def test_arar_predict_without_exog_raises_error_when_fitted_with_exog():
    """
    Test that predict raises error when exog is missing but model was fitted with exog.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog = np.random.randn(n, 2)
    model = Arar()
    model.fit(y, exog=exog)
    
    with pytest.raises(ValueError, match="Model was fitted with exog, so `exog` is required for prediction."):
        model.predict(steps=5)


def test_arar_predict_with_exog_raises_error_when_fitted_without_exog():
    """
    Test that predict raises error when exog is provided but model was fitted without exog.
    """
    np.random.seed(42)
    y = np.random.randn(100).cumsum()
    model = Arar()
    model.fit(y)
    exog_pred = np.random.randn(5, 2)

    error_msg = re.escape(
        "Model was fitted without exog, but `exog` was provided for prediction. "
        "Please refit the model with exogenous variables."
    )
    with pytest.raises(ValueError, match=error_msg):
        model.predict(steps=5, exog=exog_pred)


def test_arar_exog_feature_count_mismatch():
    """
    Test that predict raises error when exog has wrong number of features.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 2)
    model = Arar()
    model.fit(y, exog=exog_train)
    exog_pred = np.random.randn(5, 3)  # Wrong number of features

    error_msg = re.escape("Mismatch in exogenous features: fitted with 2, got 3.")
    with pytest.raises(ValueError, match=error_msg):
        model.predict(steps=5, exog=exog_pred)


def test_arar_exog_length_mismatch():
    """
    Test that predict raises error when exog length doesn't match steps.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 2)
    model = Arar()
    model.fit(y, exog=exog_train)
    exog_pred = np.random.randn(3, 2)  # Wrong length
    
    error_msg = re.escape("Length of exog (3) must match steps (5).")
    with pytest.raises(ValueError, match=error_msg):
        model.predict(steps=5, exog=exog_pred)


def test_arar_predict_exog_3d_raises():
    """
    Test that predict raises error for 3D exog input.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n)
    exog_train = np.random.randn(n, 2)
    model = Arar()
    model.fit(y, exog=exog_train)
    exog_3d = np.random.randn(5, 2, 3)  # 3D array
    
    with pytest.raises(ValueError, match="`exog` must be 1D or 2D."):
        model.predict(steps=5, exog=exog_3d)


def test_predict_output():
    """
    Test basic predict functionality.
    """
    y = ar1_series(120)
    est = Arar()
    est.fit(y)
    mean = est.predict(steps=8)
    expected = np.array([
        0.58329928, 0.92906587, 0.90186554, 1.00217975,
        0.72872694, 0.57427369, 0.38811633, -0.16481327
    ])
    
    assert mean.shape == (8,)
    np.testing.assert_array_almost_equal(mean, expected, decimal=8)


def test_reduce_memory_preserves_predictions():
    """
    Test that predictions are the same after reduce_memory().
    """
    y = ar1_series(100)
    est = Arar()
    est.fit(y)
    expected = np.array([
        0.71180394, 0.63207346, 0.39776969, 0.41563613, 0.50341699,
        0.4689158, 0.57454116, 0.38022698, 0.30215485, 0.14378399
    ])
    
    pred_before = est.predict(steps=10)
    est.reduce_memory()
    pred_after = est.predict(steps=10)

    np.testing.assert_array_almost_equal(pred_before, expected, decimal=8)
    np.testing.assert_array_equal(pred_before, pred_after)
    np.testing.assert_array_almost_equal(pred_after, expected, decimal=8)


def test_arar_predict_with_multiple_exog_features():
    """
    Test predict with multiple exogenous features.
    """
    np.random.seed(42)
    n = 150
    y = np.random.randn(n).cumsum()
    exog = np.column_stack([
        np.random.randn(n),
        np.sin(np.linspace(0, 4 * np.pi, n)),
        np.cos(np.linspace(0, 4 * np.pi, n)),
        np.arange(n) / n  # trend
    ])
    y = y + 0.5 * exog[:, 0] + 2.0 * exog[:, 1] + 1.5 * exog[:, 2] + 10.0 * exog[:, 3]
    
    model = Arar()
    model.fit(y, exog=exog, suppress_warnings=True)
    exog_future = np.column_stack([
        np.random.randn(10),
        np.sin(np.linspace(4 * np.pi, 5 * np.pi, 10)),
        np.cos(np.linspace(4 * np.pi, 5 * np.pi, 10)),
        np.arange(10) / n + 1.0
    ])
    
    pred = model.predict(steps=10, exog=exog_future)
    expected = np.array([
        -1.17765989, 0.05214722, 0.77938843, 0.79109465, 0.04091216,
        -1.00788048, -1.94952114, -4.02466845, -5.16595928, -6.79566385
    ])
    
    assert pred.shape == (10,)
    np.testing.assert_array_almost_equal(pred, expected, decimal=6)
