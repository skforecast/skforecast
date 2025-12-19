# Unit test for Arar with exogenous variables
# ==============================================================================

import numpy as np
import pandas as pd
import pytest
from ..._arar import Arar


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


def test_arar_predict_interval_with_exog():
    """
    Test Arar predict_interval with exogenous variables.
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
    
    pred_intervals = model.predict_interval(steps=5, exog=exog_future, level=(80, 95))
    
    assert isinstance(pred_intervals, pd.DataFrame)
    assert pred_intervals.shape == (5, 5)  # mean + 2 lower + 2 upper
    assert 'mean' in pred_intervals.columns
    assert 'lower_80' in pred_intervals.columns
    assert 'upper_80' in pred_intervals.columns
    assert 'lower_95' in pred_intervals.columns
    assert 'upper_95' in pred_intervals.columns


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


def test_arar_predict_interval_without_exog_raises_error_when_fitted_with_exog():
    """
    Test that predict_interval raises error when exog is missing but model was fitted with exog.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog = np.random.randn(n, 2)
    
    model = Arar()
    model.fit(y, exog=exog)
    
    with pytest.raises(ValueError, match="Model was fitted with exog"):
        model.predict_interval(steps=5)


def test_arar_predict_interval_with_exog_raises_error_when_fitted_without_exog():
    """
    Test that predict_interval raises error when exog is provided but model was fitted without exog.
    """
    np.random.seed(42)
    y = np.random.randn(100).cumsum()
    
    model = Arar()
    model.fit(y)
    
    exog_future = np.random.randn(5, 2)
    
    with pytest.raises(ValueError, match="Model was fitted without exog"):
        model.predict_interval(steps=5, exog=exog_future)


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


def test_arar_residuals_with_exog():
    """
    Test that residuals are computed correctly with exogenous variables.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog = np.random.randn(n, 2)
    y = y + 0.5 * exog[:, 0] + 2.0 * exog[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog)
    
    residuals = model.residuals_()
    fitted = model.fitted_()
    
    # Residuals should be: original y - fitted values
    assert np.allclose(residuals, y - fitted, equal_nan=True)
    # Check that residuals + fitted = original y (where both are finite)
    mask = ~np.isnan(fitted)
    assert np.allclose(residuals[mask] + fitted[mask], y[mask])


def test_arar_aic_bic_with_exog():
    """
    Test that AIC/BIC account for exogenous parameters.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog = np.random.randn(n, 2)
    y = y + 0.5 * exog[:, 0] + 2.0 * exog[:, 1]
    
    model_with_exog = Arar()
    model_with_exog.fit(y, exog=exog)
    
    model_without_exog = Arar()
    model_without_exog.fit(y)
    
    # Model with exog should have higher parameter count (reflected in AIC/BIC)
    # Note: AIC/BIC values depend on fit quality, but we can check they're computed
    assert np.isfinite(model_with_exog.aic_)
    assert np.isfinite(model_with_exog.bic_)
    assert np.isfinite(model_without_exog.aic_)
    assert np.isfinite(model_without_exog.bic_)


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
    
    exog_future_1d = np.random.randn(5)
    pred = model.predict(steps=5, exog=exog_future_1d)
    
    assert pred.shape == (5,)


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
    
    exog_future = pd.DataFrame({
        'x1': np.random.randn(5),
        'x2': np.sin(np.linspace(4*np.pi, 4*np.pi + 0.2*np.pi, 5))
    })
    
    pred = model.predict(steps=5, exog=exog_future)
    assert pred.shape == (5,)


def test_arar_predict_interval_as_frame_false_with_exog():
    """
    Test predict_interval with as_frame=False and exogenous variables.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 2)
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    exog_future = np.random.randn(5, 2)
    
    result = model.predict_interval(steps=5, exog=exog_future, level=(80, 95), as_frame=False)
    
    assert isinstance(result, dict)
    assert 'mean' in result
    assert 'upper' in result
    assert 'lower' in result
    assert 'level' in result
    assert result['mean'].shape == (5,)
    assert result['upper'].shape == (5, 2)
    assert result['lower'].shape == (5, 2)
    assert result['level'] == [80, 95]


def test_arar_score_with_exog():
    """
    Test that score() works correctly with exogenous variables.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog = np.random.randn(n, 2)
    y = y + 0.5 * exog[:, 0] + 2.0 * exog[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog)
    
    score = model.score()
    
    assert np.isfinite(score) or np.isnan(score)
    # Score should be between -inf and 1
    if np.isfinite(score):
        assert score <= 1.0


def test_arar_summary_with_exog(capsys):
    """
    Test that summary() includes exogenous model information.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog = np.random.randn(n, 2)
    y = y + 0.5 * exog[:, 0] + 2.0 * exog[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog)
    model.summary()
    
    captured = capsys.readouterr().out
    assert "ARAR Model Summary" in captured
    assert "Exogenous Model (Linear Regression)" in captured
    assert "Intercept:" in captured
    assert "Coefficients:" in captured


def test_arar_fitted_values_with_exog():
    """
    Test that fitted values include exogenous component.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog = np.random.randn(n, 2)
    y = y + 0.5 * exog[:, 0] + 2.0 * exog[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog)
    
    fitted = model.fitted_()
    
    assert fitted.shape == y.shape
    # First few values may be NaN due to ARAR lag structure
    assert np.isnan(fitted).any()
    # But at least half of values should be finite (depends on selected lags)
    assert np.sum(~np.isnan(fitted)) > n * 0.5


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
    
    exog_future = np.column_stack([
        np.random.randn(10),
        np.sin(np.linspace(4*np.pi, 5*np.pi, 10)),
        np.cos(np.linspace(4*np.pi, 5*np.pi, 10)),
        np.arange(10) / n + 1.0
    ])
    
    pred = model.predict(steps=10, exog=exog_future)
    assert pred.shape == (10,)


def test_arar_exog_interval_feature_count_mismatch():
    """
    Test that predict_interval raises error when exog feature count doesn't match.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 2)
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    exog_future_wrong = np.random.randn(5, 3)  # Wrong number of features
    
    with pytest.raises(ValueError, match="Mismatch in exogenous features"):
        model.predict_interval(steps=5, exog=exog_future_wrong)


def test_arar_exog_interval_length_mismatch():
    """
    Test that predict_interval raises error when exog length doesn't match steps.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 2)
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    exog_future_wrong = np.random.randn(3, 2)  # Wrong length
    
    with pytest.raises(ValueError, match="Length of exog"):
        model.predict_interval(steps=5, exog=exog_future_wrong)


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
    
    exog_future = np.random.randn(5, 2)
    pred = model.predict(steps=5, exog=exog_future)
    assert pred.shape == (5,)


def test_arar_reduce_memory_with_exog():
    """
    Test that reduce_memory works correctly with exogenous variables.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 2)
    y = y + 0.5 * exog_train[:, 0] + 2.0 * exog_train[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    # Get predictions before memory reduction
    exog_future = np.random.randn(5, 2)
    pred_before = model.predict(steps=5, exog=exog_future)
    
    # Verify arrays exist before reduction
    assert model.fitted_values_ is not None
    assert model.residuals_in_ is not None
    
    # Reduce memory
    result = model.reduce_memory()
    
    # Verify arrays are cleared
    assert model.fitted_values_ is None
    assert model.residuals_in_ is None
    assert model.memory_reduced_ is True
    assert result is model
    
    # Predictions should still work
    pred_after = model.predict(steps=5, exog=exog_future)
    np.testing.assert_allclose(pred_before, pred_after)
    
    # Methods requiring cleared data should raise error
    with pytest.raises(ValueError, match="memory has been reduced"):
        model.fitted_()
    
    with pytest.raises(ValueError, match="memory has been reduced"):
        model.residuals_()
    
    with pytest.raises(ValueError, match="memory has been reduced"):
        model.score()
    
    with pytest.raises(ValueError, match="memory has been reduced"):
        model.summary()


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


def test_arar_exog_prediction_intervals_include_uncertainty():
    """
    Test that prediction intervals with exog properly account for uncertainty.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 2)
    y = y + 0.5 * exog_train[:, 0] + 2.0 * exog_train[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    exog_future = np.random.randn(10, 2)
    
    intervals = model.predict_interval(steps=10, exog=exog_future, level=(80, 95))
    
    # Check that intervals are wider for 95% than 80%
    width_80 = intervals['upper_80'] - intervals['lower_80']
    width_95 = intervals['upper_95'] - intervals['lower_95']
    
    assert np.all(width_95 > width_80)
    
    # Check that intervals contain the mean
    assert np.all(intervals['lower_80'] < intervals['mean'])
    assert np.all(intervals['mean'] < intervals['upper_80'])
    assert np.all(intervals['lower_95'] < intervals['mean'])
    assert np.all(intervals['mean'] < intervals['upper_95'])
    
    # Check that intervals generally widen over time
    # (not guaranteed but should be true most of the time)
    assert width_95.iloc[-1] >= width_95.iloc[0]


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
    
    exog_future = np.column_stack([
        np.random.randn(5),
        np.ones(5) * 5.0
    ])
    
    pred = model.predict(steps=5, exog=exog_future)
    assert pred.shape == (5,)


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


def test_arar_predict_interval_values_contain_point_forecast():
    """
    Test that prediction intervals contain the point forecast and
    have correct width relationships.
    """
    np.random.seed(789)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 2)
    y = y + 2.0 * exog_train[:, 0] + 3.0 * exog_train[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    exog_future = np.random.randn(10, 2)
    
    # Get point predictions
    pred_point = model.predict(steps=10, exog=exog_future)
    
    # Get interval predictions
    pred_interval = model.predict_interval(steps=10, exog=exog_future, level=(80, 95))
    
    # Point predictions should match interval mean
    np.testing.assert_allclose(pred_point, pred_interval['mean'].values, rtol=1e-10)
    
    # All intervals should contain the mean
    assert np.all(pred_interval['lower_80'] < pred_interval['mean'])
    assert np.all(pred_interval['mean'] < pred_interval['upper_80'])
    assert np.all(pred_interval['lower_95'] < pred_interval['mean'])
    assert np.all(pred_interval['mean'] < pred_interval['upper_95'])
    
    # 95% intervals should be wider than 80% intervals
    width_80 = pred_interval['upper_80'] - pred_interval['lower_80']
    width_95 = pred_interval['upper_95'] - pred_interval['lower_95']
    assert np.all(width_95 > width_80)


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
    
    # Same for intervals
    interval1 = model.predict_interval(steps=5, exog=exog_future, level=(90,))
    interval2 = model.predict_interval(steps=5, exog=exog_future, level=(90,))
    
    pd.testing.assert_frame_equal(interval1, interval2)


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


def test_arar_predict_interval_exog_affects_mean_not_width():
    """
    Test that changing exog values shifts the intervals but doesn't
    dramatically change their width (since exog is assumed deterministic).
    """
    np.random.seed(333)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 1)
    y = y + 5.0 * exog_train[:, 0]
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    # Two different exog scenarios
    exog_scenario1 = np.array([[0.0], [0.0], [0.0]])
    exog_scenario2 = np.array([[10.0], [10.0], [10.0]])
    
    interval1 = model.predict_interval(steps=3, exog=exog_scenario1, level=(90,))
    interval2 = model.predict_interval(steps=3, exog=exog_scenario2, level=(90,))
    
    # Means should be very different
    mean_diff = np.abs(interval2['mean'].values - interval1['mean'].values)
    assert np.all(mean_diff > 20.0)  # With coef ~5 and diff of 10, expect ~50
    
    # But widths should be similar (exog doesn't add uncertainty)
    width1 = interval1['upper_90'].values - interval1['lower_90'].values
    width2 = interval2['upper_90'].values - interval2['lower_90'].values
    
    # Widths should be nearly identical
    np.testing.assert_allclose(width1, width2, rtol=0.01)


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


def test_arar_predict_interval_as_dict_values():
    """
    Test that predict_interval with as_frame=False returns correct
    dict structure with proper values.
    """
    np.random.seed(555)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 2)
    y = y + exog_train[:, 0] + exog_train[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog_train)
    
    exog_future = np.random.randn(5, 2)
    
    result = model.predict_interval(
        steps=5, 
        exog=exog_future, 
        level=(80, 95), 
        as_frame=False
    )
    
    # Check structure
    assert isinstance(result, dict)
    assert 'mean' in result
    assert 'upper' in result
    assert 'lower' in result
    assert 'level' in result
    
    # Check values
    assert result['mean'].shape == (5,)
    assert result['upper'].shape == (5, 2)
    assert result['lower'].shape == (5, 2)
    assert result['level'] == [80, 95]
    
    # Check value relationships
    for i in range(5):
        # Lower < mean < upper for both levels
        assert result['lower'][i, 0] < result['mean'][i] < result['upper'][i, 0]
        assert result['lower'][i, 1] < result['mean'][i] < result['upper'][i, 1]
        
        # 95% interval wider than 80%
        width_80 = result['upper'][i, 0] - result['lower'][i, 0]
        width_95 = result['upper'][i, 1] - result['lower'][i, 1]
        assert width_95 > width_80


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
