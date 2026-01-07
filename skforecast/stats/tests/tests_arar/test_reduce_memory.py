# Unit tests for skforecast.stats.Arar.reduce_memory()
# ==============================================================================

import pytest
import numpy as np
import pandas as pd
from ..._arar import Arar


def test_reduce_memory_clears_arrays():
    """
    Test that reduce_memory() correctly clears memory-heavy arrays.
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Arar()
    model.fit(y)
    assert model.is_memory_reduced is False
    assert model.fitted_values_ is not None
    assert model.in_sample_residuals_ is not None
    result = model.reduce_memory()
    assert model.fitted_values_ is None
    assert model.in_sample_residuals_ is None
    assert model.is_memory_reduced is True


def test_reduce_memory_preserves_predictions():
    """
    Test that predictions are identical before and after reduce_memory().
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    model = Arar()
    model.fit(y)
    pred_before = model.predict(steps=24)
    pred_interval_before = model.predict_interval(steps=24, level=[80, 95])
    
    model.reduce_memory()
    pred_after = model.predict(steps=24)
    pred_interval_after = model.predict_interval(steps=24, level=[80, 95])
    
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
    fitted = model.get_fitted_values()
    assert len(fitted) == len(y)
    model.reduce_memory()
    
    with pytest.raises(
        ValueError, 
        match="Cannot call fitted_\\(\\): model memory has been reduced via reduce_memory\\(\\)"
    ):
        model.get_fitted_values()

def test_residuals_raises_error_after_reduce_memory():
    """
    Test that residuals_() raises ValueError after reduce_memory().
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Arar()
    model.fit(y)
    residuals = model.get_residuals()
    assert len(residuals) == len(y)
    model.reduce_memory()
    
    with pytest.raises(
        ValueError, 
        match="Cannot call residuals_\\(\\): model memory has been reduced via reduce_memory\\(\\)"
    ):
        model.get_residuals()

def test_score_raises_error_after_reduce_memory():
    """
    Test that get_score() raises ValueError after reduce_memory().
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Arar()
    model.fit(y)
    score = model.get_score()
    assert isinstance(score, float)
    
    model.reduce_memory()
    
    with pytest.raises(
        ValueError, 
        match="Cannot call score\\(\\): model memory has been reduced via reduce_memory\\(\\)"
    ):
        model.get_score()


def test_summary_raises_error_after_reduce_memory(capsys):
    """
    Test that summary() raises ValueError after reduce_memory().
    """
    np.random.seed(42)
    y = np.random.randn(1000).cumsum() + 100
    
    model = Arar()
    model.fit(y)
    model.summary()
    captured = capsys.readouterr()
    assert "ARAR" in captured.out
    
    model.reduce_memory()
    
    with pytest.raises(
        ValueError, 
        match="Cannot call summary\\(\\): model memory has been reduced via reduce_memory\\(\\)"
    ):
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

    assert model.is_memory_reduced is True

    model.fit(y)
    assert model.is_memory_reduced is False