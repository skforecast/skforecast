################################################################################
#                     tests_arar_reduce_memory.py                              #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

import pytest
import numpy as np
import pandas as pd
from skforecast.stats import Arar


def test_reduce_memory_clears_arrays():
    """
    Test that reduce_memory() correctly clears memory-heavy arrays.
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Arar()
    model.fit(y)
    
    # Verify arrays exist before reduction
    assert model.fitted_values_ is not None
    assert model.residuals_in_ is not None
    
    # Call reduce_memory
    result = model.reduce_memory()
    
    # Verify arrays are cleared
    assert model.fitted_values_ is None
    assert model.residuals_in_ is None
    
    # Verify method returns self
    assert result is model


def test_reduce_memory_sets_flag():
    """
    Test that reduce_memory() sets memory_reduced_ flag to True.
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Arar()
    model.fit(y)
    
    # Flag should be False after fit
    assert model.memory_reduced_ is False
    
    # Call reduce_memory
    model.reduce_memory()
    
    # Flag should be True after reduction
    assert model.memory_reduced_ is True


def test_reduce_memory_preserves_predictions():
    """
    Test that predictions are identical before and after reduce_memory().
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Arar()
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


def test_fitted_raises_error_after_reduce_memory():
    """
    Test that fitted_() raises ValueError after reduce_memory().
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Arar()
    model.fit(y)
    
    # Should work before reduction
    fitted = model.fitted_()
    assert len(fitted) == len(y)
    
    # Reduce memory
    model.reduce_memory()
    
    # Should raise ValueError after reduction
    with pytest.raises(ValueError, match="Cannot call fitted_\\(\\)"):
        model.fitted_()


def test_residuals_raises_error_after_reduce_memory():
    """
    Test that residuals_() raises ValueError after reduce_memory().
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Arar()
    model.fit(y)
    
    # Should work before reduction
    residuals = model.residuals_()
    assert len(residuals) == len(y)
    
    # Reduce memory
    model.reduce_memory()
    
    # Should raise ValueError after reduction
    with pytest.raises(ValueError, match="Cannot call residuals_\\(\\)"):
        model.residuals_()


def test_score_raises_error_after_reduce_memory():
    """
    Test that score() raises ValueError after reduce_memory().
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Arar()
    model.fit(y)
    
    # Should work before reduction
    score = model.score()
    assert isinstance(score, float)
    
    # Reduce memory
    model.reduce_memory()
    
    # Should raise ValueError after reduction
    with pytest.raises(ValueError, match="Cannot call score\\(\\)"):
        model.score()


def test_summary_raises_error_after_reduce_memory(capsys):
    """
    Test that summary() raises ValueError after reduce_memory().
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Arar()
    model.fit(y)
    
    # Should work before reduction
    model.summary()
    captured = capsys.readouterr()
    assert "ARAR" in captured.out
    
    # Reduce memory
    model.reduce_memory()
    
    # Should raise ValueError after reduction
    with pytest.raises(ValueError, match="Cannot call summary\\(\\)"):
        model.summary()


def test_refit_resets_memory_reduced_flag():
    """
    Test that refitting resets memory_reduced_ flag to False.
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Arar()
    model.fit(y)
    model.reduce_memory()
    
    # Flag should be True after reduction
    assert model.memory_reduced_ is True
    
    # Refit the model
    model.fit(y)
    
    # Flag should be reset to False
    assert model.memory_reduced_ is False
    
    # Diagnostic methods should work again
    fitted = model.fitted_()
    assert len(fitted) == len(y)


def test_reduce_memory_with_different_configs():
    """
    Test reduce_memory() works with different max_ar_depth and max_lag configs.
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    configs = [
        {},
        {"max_ar_depth": 13, "max_lag": 20},
        {"max_ar_depth": 26, "max_lag": 40},
    ]
    
    for config in configs:
        model = Arar(**config)
        model.fit(y)
        
        pred_before = model.predict(steps=12)
        model.reduce_memory()
        pred_after = model.predict(steps=12)
        
        np.testing.assert_allclose(pred_before, pred_after)
        assert model.memory_reduced_ is True


def test_reduce_memory_with_safe_mode():
    """
    Test reduce_memory() works with safe=True and safe=False.
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    for safe in [True, False]:
        model = Arar(safe=safe)
        model.fit(y)
        
        pred_before = model.predict(steps=12)
        model.reduce_memory()
        pred_after = model.predict(steps=12)
        
        np.testing.assert_allclose(pred_before, pred_after)
        assert model.memory_reduced_ is True


def test_reduce_memory_chaining():
    """
    Test that reduce_memory() returns self for method chaining.
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Arar()
    result = model.fit(y).reduce_memory()
    
    assert result is model
    assert model.memory_reduced_ is True
    
    # Predictions should still work
    pred = result.predict(steps=12)
    assert len(pred) == 12


def test_reduce_memory_with_pandas_series():
    """
    Test reduce_memory() works when fitted with pandas Series.
    """
    np.random.seed(42)
    y = pd.Series(np.random.randn(1000).cumsum() + 100)
    
    model = Arar()
    model.fit(y)
    
    pred_before = model.predict(steps=12)
    model.reduce_memory()
    pred_after = model.predict(steps=12)
    
    np.testing.assert_allclose(pred_before, pred_after)
    assert model.memory_reduced_ is True


def test_reduce_memory_error_message_content():
    """
    Test that error messages contain expected content.
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Arar()
    model.fit(y)
    model.reduce_memory()
    
    # Test fitted_() error message
    with pytest.raises(ValueError) as exc_info:
        model.fitted_()
    assert "reduce_memory()" in str(exc_info.value)
    assert "Refit the model" in str(exc_info.value)
    
    # Test residuals_() error message
    with pytest.raises(ValueError) as exc_info:
        model.residuals_()
    assert "reduce_memory()" in str(exc_info.value)
    
    # Test score() error message
    with pytest.raises(ValueError) as exc_info:
        model.score()
    assert "reduce_memory()" in str(exc_info.value)
    
    # Test summary() error message
    with pytest.raises(ValueError) as exc_info:
        model.summary()
    assert "reduce_memory()" in str(exc_info.value)


def test_reduce_memory_with_short_series():
    """
    Test reduce_memory() works with short time series.
    """
    np.random.seed(42)
    y = np.random.randn(50).cumsum() + 100
    
    model = Arar(safe=True)
    model.fit(y)
    
    pred_before = model.predict(steps=5)
    model.reduce_memory()
    pred_after = model.predict(steps=5)
    
    np.testing.assert_allclose(pred_before, pred_after)
    assert model.memory_reduced_ is True


def test_reduce_memory_preserves_model_tuple():
    """
    Test that reduce_memory() does not affect the model_ tuple.
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Arar()
    model.fit(y)
    
    # Store model_ tuple reference
    model_tuple_before = model.model_
    
    # Reduce memory
    model.reduce_memory()
    
    # model_ tuple should be unchanged
    assert model.model_ is model_tuple_before
    assert len(model.model_) == 8
    
    # y_ should still reference model_[0]
    assert model.y_ is model.model_[0]
