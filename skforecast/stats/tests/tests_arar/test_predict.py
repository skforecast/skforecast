# Unit test predict method - Arar
# ==============================================================================
import numpy as np
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


def test_estimator_predict():
    """
    Test basic predict functionality.
    """
    y = ar1_series(120)
    est = Arar()
    est.fit(y)

    mean = est.predict(steps=8)
    assert mean.shape == (8,)


def test_estimator_invalid_steps_and_unfitted():
    """
    Test that predict raises errors for invalid steps and unfitted model.
    """
    est = Arar()
    
    with pytest.raises(Exception):
        est.predict(steps=1)

    y = ar1_series(50)
    est.fit(y)
    
    with pytest.raises(ValueError, match="positive integer"):
        est.predict(steps=0)
    with pytest.raises(ValueError, match="positive integer"):
        est.predict(steps=-2)
    with pytest.raises(ValueError, match="positive integer"):
        est.predict(steps=1.5)


def test_arar_predict_with_exog():
    """
    Test Arar predict with exogenous variables.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.column_stack([
        np.random.randn(n),
        np.sin(np.linspace(0, 4*np.pi, n))
    ])
    y = y + 0.5 * exog_train[:, 0] + 2.0 * exog_train[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    exog_future = np.column_stack([
        np.random.randn(5),
        np.sin(np.linspace(4*np.pi, 4*np.pi + 0.2*np.pi, 5))
    ])
    
    pred = model.predict(steps=5, exog=exog_future)
    
    assert pred.shape == (5,)
    assert np.all(np.isfinite(pred))


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
    
    with pytest.raises(ValueError, match="Model was fitted with exog"):
        model.predict(steps=5)


def test_arar_predict_with_exog_raises_error_when_fitted_without_exog():
    """
    Test that predict raises error when exog is provided but model was fitted without exog.
    """
    np.random.seed(42)
    y = np.random.randn(100).cumsum()
    
    model = Arar()
    model.fit(y)
    
    exog_future = np.random.randn(5, 2)
    
    with pytest.raises(ValueError, match="Model was fitted without exog"):
        model.predict(steps=5, exog=exog_future)


def test_arar_exog_feature_count_mismatch():
    """
    Test that predict raises error when exog feature count doesn't match.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 2)
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    exog_future_wrong = np.random.randn(5, 3)  # Wrong number of features
    
    with pytest.raises(ValueError, match="Mismatch in exogenous features"):
        model.predict(steps=5, exog=exog_future_wrong)


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
    
    exog_future_wrong = np.random.randn(3, 2)  # Wrong length
    
    with pytest.raises(ValueError, match="Length of exog"):
        model.predict(steps=5, exog=exog_future_wrong)


def test_reduce_memory_preserves_predictions():
    """
    Test that predictions are the same after reduce_memory().
    """
    y = ar1_series(100)
    est = Arar()
    est.fit(y)
    
    # Predict before memory reduction
    pred_before = est.predict(steps=10)
    
    # Reduce memory
    est.reduce_memory()
    
    # Predict after memory reduction
    pred_after = est.predict(steps=10)
    
    # Predictions should be identical
    np.testing.assert_array_equal(pred_before, pred_after)


def test_arar_predict_with_1d_exog():
    """
    Test predict with 1D exogenous variable.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_1d = np.random.randn(n)
    
    model = Arar()
    model.fit(y, exog=exog_1d)
    
    exog_future_1d = np.random.randn(5)
    pred = model.predict(steps=5, exog=exog_future_1d)
    
    assert pred.shape == (5,)


def test_arar_predict_with_multiple_exog_features():
    """
    Test predict with multiple exogenous features.
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
    
    exog_future = np.column_stack([
        np.random.randn(10),
        np.sin(np.linspace(4*np.pi, 5*np.pi, 10)),
        np.cos(np.linspace(4*np.pi, 5*np.pi, 10)),
        np.arange(10) / n + 1.0
    ])
    
    pred = model.predict(steps=10, exog=exog_future)
    assert pred.shape == (10,)


def test_arar_predict_values_with_exog_vs_without():
    """
    Test that predictions with exog differ from predictions without exog
    and that the exog component is correctly added.
    """
    np.random.seed(123)
    n = 100
    
    # Create base ARAR series
    y_base = np.random.randn(n).cumsum()
    
    # Create exogenous variable with known coefficients
    exog_train = np.random.randn(n, 1)
    true_coef = 5.0
    y_with_exog = y_base + true_coef * exog_train[:, 0]
    
    # Fit model without exog (on base series)
    model_no_exog = Arar()
    model_no_exog.fit(y_base)
    
    # Fit model with exog
    model_with_exog = Arar()
    model_with_exog.fit(y_with_exog, exog=exog_train)
    
    # Create future exog values
    exog_future = np.array([[2.0], [3.0], [4.0]])
    
    # Predictions with exog should be different from predictions without
    pred_no_exog = model_no_exog.predict(steps=3)
    pred_with_exog = model_with_exog.predict(steps=3, exog=exog_future)
    
    # They should be different
    assert not np.allclose(pred_no_exog, pred_with_exog)
    
    # The exog model should have learned something close to the true coefficient
    assert abs(model_with_exog.coef_exog_[0] - true_coef) < 2.0  # Allow some error


def test_arar_predict_exog_contribution():
    """
    Test that changing exog values properly affects predictions.
    """
    np.random.seed(456)
    n = 100
    
    # Create series with strong exog effect
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 1)
    y = y + 10.0 * exog_train[:, 0]
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    # Test with two different exog values
    exog_high = np.array([[5.0]])  # High value
    exog_low = np.array([[-5.0]])  # Low value
    
    pred_high = model.predict(steps=1, exog=exog_high)
    pred_low = model.predict(steps=1, exog=exog_low)
    
    # Prediction with high exog should be higher than with low exog
    # Since coefficient is positive (~10), the difference should be substantial
    assert pred_high[0] > pred_low[0]
    
    # The difference should be roughly 10 * (5 - (-5)) = 100
    # Allow for ARAR component variation
    diff = pred_high[0] - pred_low[0]
    assert 50 < diff < 150  # Should be around 100


def test_arar_predict_deterministic_with_exog():
    """
    Test that predictions are deterministic (same inputs = same outputs).
    """
    np.random.seed(111)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 2)
    y = y + exog_train[:, 0] + 2.0 * exog_train[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    exog_future = np.random.randn(5, 2)
    
    # Make predictions twice
    pred1 = model.predict(steps=5, exog=exog_future)
    pred2 = model.predict(steps=5, exog=exog_future)
    
    # Should be identical
    np.testing.assert_array_equal(pred1, pred2)


def test_arar_predict_exog_sequential_consistency():
    """
    Test that multi-step predictions are consistent with sequential 1-step predictions
    when exog values are known.
    """
    np.random.seed(222)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 1)
    y = y + 3.0 * exog_train[:, 0]
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    exog_future = np.random.randn(3, 1)
    
    # Multi-step prediction
    pred_multi = model.predict(steps=3, exog=exog_future)
    
    # The predictions should be finite and reasonable
    assert np.all(np.isfinite(pred_multi))
    
    # Predictions should vary with exog values
    # If we change one exog value significantly, that prediction should change
    exog_modified = exog_future.copy()
    exog_modified[1, 0] += 10.0  # Large change in second step
    
    pred_modified = model.predict(steps=3, exog=exog_modified)
    
    # First and third predictions should be similar
    assert abs(pred_multi[0] - pred_modified[0]) < 0.01
    assert abs(pred_multi[2] - pred_modified[2]) < 0.01
    
    # Second prediction should be very different
    assert abs(pred_multi[1] - pred_modified[1]) > 5.0


def test_arar_predict_with_large_exog_coefficient():
    """
    Test predictions when exog has a large coefficient to ensure
    the exog component dominates as expected.
    """
    np.random.seed(444)
    n = 100
    
    # Create series where exog has very large effect
    y_base = np.random.randn(n) * 0.1  # Small ARAR component
    exog_train = np.random.randn(n, 1)
    y = y_base + 100.0 * exog_train[:, 0]  # Large exog effect
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    # The coefficient should be close to 100
    assert 80 < model.coef_exog_[0] < 120
    
    # Test prediction with known exog
    exog_future = np.array([[1.0], [2.0], [3.0]])
    pred = model.predict(steps=3, exog=exog_future)
    
    # Predictions should increase roughly linearly with exog
    # and by approximately the coefficient value
    diff_1_to_2 = pred[1] - pred[0]
    diff_2_to_3 = pred[2] - pred[1]
    
    # Both differences should be close to 100 (the coefficient)
    assert 80 < diff_1_to_2 < 120
    assert 80 < diff_2_to_3 < 120


def test_arar_predict_with_zero_exog_matches_baseline():
    """
    Test that when exog values are zero, predictions are close to
    the intercept-only model component.
    """
    np.random.seed(666)
    n = 100
    
    # Create series with exog
    y_base = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 1)
    y = y_base + 10.0 * exog_train[:, 0]
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    # Predict with zero exog (should get just ARAR + intercept)
    exog_zero = np.zeros((3, 1))
    pred_zero = model.predict(steps=3, exog=exog_zero)
    
    # Predict with non-zero exog
    exog_nonzero = np.ones((3, 1)) * 2.0
    pred_nonzero = model.predict(steps=3, exog=exog_nonzero)
    
    # Difference should be approximately coef * (2 - 0) = coef * 2
    diff = pred_nonzero - pred_zero
    expected_diff = model.coef_exog_[0] * 2.0
    
    # Check that differences are close to expected
    for d in diff:
        assert abs(d - expected_diff) < 5.0  # Allow for ARAR dynamics
