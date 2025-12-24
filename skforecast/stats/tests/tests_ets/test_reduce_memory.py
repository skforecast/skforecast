# Unit test reduce_memory method - Ets
# ==============================================================================

import pytest
import numpy as np
import pandas as pd
from ..._ets import Ets


def test_reduce_memory_clears_arrays():
    """
    Test that reduce_memory() correctly clears all memory-heavy arrays.
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Ets(m=12, model="AAN")
    model.fit(y)
    
    # Verify arrays exist before reduction
    assert model.y_train_ is not None
    assert model.fitted_values_ is not None
    assert model.in_sample_residuals_ is not None
    assert model.model_.fitted is not None
    assert model.model_.residuals is not None
    assert model.model_.y_original is not None
    
    # Call reduce_memory
    result = model.reduce_memory()
    
    # Verify arrays are cleared
    assert model.y_train_ is None
    assert model.fitted_values_ is None
    assert model.in_sample_residuals_ is None
    assert model.model_.fitted is None
    assert model.model_.residuals is None
    assert model.model_.y_original is None
    
    # Verify method returns self
    assert result is model


def test_reduce_memory_sets_flag():
    """
    Test that reduce_memory() sets memory_reduced_ flag to True.
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Ets(m=12, model="AAN")
    model.fit(y)
    
    # Flag should be False after fit
    assert model.is_memory_reduced is False
    
    # Call reduce_memory
    model.reduce_memory()
    
    # Flag should be True after reduction
    assert model.is_memory_reduced is True


def test_reduce_memory_preserves_predictions():
    """
    Test that predictions are identical before and after reduce_memory().
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Ets(m=12, model="AAN")
    model.fit(y)
    
    # Get predictions before reduction
    pred_before = model.predict(steps=24)
    pred_interval_before = model.predict_interval(steps=24, level=[80, 95])
    
    # Reduce memory
    model.reduce_memory()
    
    # Get predictions after reduction
    pred_after = model.predict(steps=24)
    pred_interval_after = model.predict_interval(steps=24, level=[80, 95])
    
    # Verify predictions are identical
    np.testing.assert_allclose(pred_before, pred_after)
    pd.testing.assert_frame_equal(pred_interval_before, pred_interval_after)


def test_get_fitted_values_raises_error_after_reduce_memory():
    """
    Test that get_fitted_values() raises ValueError after reduce_memory().
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Ets(m=12, model="AAN")
    model.fit(y)
    
    # Should work before reduction
    fitted = model.get_fitted_values()
    assert len(fitted) == len(y)
    
    # Reduce memory
    model.reduce_memory()
    
    # Should raise ValueError after reduction
    with pytest.raises(ValueError, match="memory has been reduced"):
        model.get_fitted_values()


def test_get_residuals_raises_error_after_reduce_memory():
    """
    Test that get_residuals() raises ValueError after reduce_memory().
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Ets(m=12, model="AAN")
    model.fit(y)
    
    # Should work before reduction
    residuals = model.get_residuals()
    assert len(residuals) == len(y)
    
    # Reduce memory
    model.reduce_memory()
    
    # Should raise ValueError after reduction
    with pytest.raises(ValueError, match="memory has been reduced"):
        model.get_residuals()


def test_score_raises_error_after_reduce_memory():
    """
    Test that get_score() raises ValueError after reduce_memory().
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Ets(m=12, model="AAN")
    model.fit(y)
    
    # Should work before reduction
    score = model.get_score()
    assert isinstance(score, float)
    
    # Reduce memory
    model.reduce_memory()
    
    # Should raise ValueError after reduction
    with pytest.raises(ValueError, match="memory has been reduced"):
        model.get_score()


def test_summary_raises_error_after_reduce_memory(capsys):
    """
    Test that summary() raises ValueError after reduce_memory().
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Ets(m=12, model="AAN")
    model.fit(y)
    
    # Should work before reduction
    model.summary()
    captured = capsys.readouterr()
    assert "ETS Model Summary" in captured.out
    
    # Reduce memory
    model.reduce_memory()
    
    # Should raise ValueError after reduction
    with pytest.raises(ValueError, match="memory has been reduced"):
        model.summary()


def test_refit_resets_memory_reduced_flag():
    """
    Test that refitting resets memory_reduced_ flag to False.
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Ets(m=12, model="AAN")
    model.fit(y)
    model.reduce_memory()
    
    # Flag should be True after reduction
    assert model.is_memory_reduced is True
    
    # Refit the model
    model.fit(y)
    
    # Flag should be reset to False
    assert model.is_memory_reduced is False
    
    # Diagnostic methods should work again
    fitted = model.get_fitted_values()
    assert len(fitted) == len(y)


def test_reduce_memory_with_different_models():
    """
    Test reduce_memory() works with different ETS model types.
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    models = ["ANN", "AAN", "ANA", "AAA", "MAN"]
    
    for model_type in models:
        model = Ets(m=12, model=model_type)
        model.fit(y)
        
        pred_before = model.predict(steps=12)
        model.reduce_memory()
        pred_after = model.predict(steps=12)
        
        np.testing.assert_allclose(pred_before, pred_after)
        assert model.is_memory_reduced is True


def test_reduce_memory_with_auto_selection():
    """
    Test reduce_memory() works with automatic model selection (ZZZ).
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Ets(m=12, model="ZZZ")
    model.fit(y)
    
    pred_before = model.predict(steps=12)
    model.reduce_memory()
    pred_after = model.predict(steps=12)
    
    np.testing.assert_allclose(pred_before, pred_after)
    assert model.is_memory_reduced is True


def test_reduce_memory_with_damped_trend():
    """
    Test reduce_memory() works with damped trend models.
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Ets(m=12, model="AAN", damped=True)
    model.fit(y)
    
    pred_before = model.predict(steps=12)
    model.reduce_memory()
    pred_after = model.predict(steps=12)
    
    np.testing.assert_allclose(pred_before, pred_after)
    assert model.is_memory_reduced is True


def test_reduce_memory_with_box_cox():
    """
    Test reduce_memory() works with Box-Cox transformation.
    """
    np.random.seed(42)
    y = np.abs(np.random.randn(1000).cumsum()) + 100
    
    model = Ets(m=12, model="AAN", lambda_auto=True)
    model.fit(y)
    
    pred_before = model.predict(steps=12)
    model.reduce_memory()
    pred_after = model.predict(steps=12)
    
    np.testing.assert_allclose(pred_before, pred_after)
    assert model.is_memory_reduced is True


def test_reduce_memory_chaining():
    """
    Test that reduce_memory() returns self for method chaining.
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Ets(m=12, model="AAN")
    result = model.fit(y).reduce_memory()
    
    assert result is model
    assert model.is_memory_reduced is True
    
    # Predictions should still work
    pred = result.predict(steps=12)
    assert len(pred) == 12


def test_reduce_memory_with_pandas_series():
    """
    Test reduce_memory() works when fitted with pandas Series.
    """
    np.random.seed(42)
    y = pd.Series(np.random.randn(1000).cumsum() + 100)
    
    model = Ets(m=12, model="AAN")
    model.fit(y)
    
    pred_before = model.predict(steps=12)
    model.reduce_memory()
    pred_after = model.predict(steps=12)
    
    np.testing.assert_allclose(pred_before, pred_after)
    assert model.is_memory_reduced is True


def test_reduce_memory_error_message_content():
    """
    Test that error messages contain expected content.
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Ets(m=12, model="AAN")
    model.fit(y)
    model.reduce_memory()
    
    # Test get_fitted_values() error message
    with pytest.raises(ValueError) as exc_info:
        model.get_fitted_values()
    assert "reduce_memory()" in str(exc_info.value)
    
    # Test get_residuals() error message
    with pytest.raises(ValueError) as exc_info:
        model.get_residuals()
    assert "reduce_memory()" in str(exc_info.value)
    
    # Test get_score() error message
    with pytest.raises(ValueError) as exc_info:
        model.get_score()
    assert "reduce_memory()" in str(exc_info.value)
    
    # Test summary() error message
    with pytest.raises(ValueError) as exc_info:
        model.summary()
    assert "reduce_memory()" in str(exc_info.value)
