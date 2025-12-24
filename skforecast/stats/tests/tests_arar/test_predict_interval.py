# Unit test predict_interval method - Arar
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


def test_predict_interval_raises_errors_for_invalid_steps_and_unfitted_model():
    """
    Test that predict_interval raises errors for invalid steps and unfitted model.
    """
    est = Arar()
    msg = (
        "This Arar instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
    )
    with pytest.raises(TypeError, match=msg):
        est.predict_interval(steps=1)

    y = ar1_series(50)
    est.fit(y)
    
    with pytest.raises(ValueError, match="`steps` must be a positive integer."):
        est.predict_interval(steps=0)
    with pytest.raises(ValueError, match="`steps` must be a positive integer."):
        est.predict_interval(steps=-2)
    with pytest.raises(ValueError, match="`steps` must be a positive integer."):
        est.predict_interval(steps=1.5)


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
    
    with pytest.raises(ValueError, match="Model was fitted with exog, so `exog` is required for prediction."):
        model.predict_interval(steps=5)


def test_arar_predict_interval_with_exog_raises_error_when_fitted_without_exog():
    """
    Test that predict_interval raises error when exog is provided but model was fitted without exog.
    """
    np.random.seed(42)
    y = np.random.randn(100).cumsum()
    model = Arar()
    model.fit(y)
    exog_pred = np.random.randn(5, 2)
    msg = (
        "Model was fitted without exog, but `exog` was provided for prediction. "
        "Please refit the model with exogenous variables."
    )
    with pytest.raises(ValueError, match=msg):
        model.predict_interval(steps=5, exog=exog_pred)


def test_arar_predict_interval_exog_feature_count_mismatch():
    """
    Test that predict_interval raises error when exog has wrong number of features.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 2)
    model = Arar()
    model.fit(y, exog=exog_train)
    exog_pred = np.random.randn(5, 3)  # Wrong number of features
    msg = "Mismatch in exogenous features: fitted with 2, got 3."
    with pytest.raises(ValueError, match=msg):
        model.predict_interval(steps=5, exog=exog_pred)


def test_arar_predict_interval_exog_length_mismatch():
    """
    Test that predict_interval raises error when exog length doesn't match steps.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n).cumsum()
    exog_train = np.random.randn(n, 2)
    model = Arar()
    model.fit(y, exog=exog_train)
    exog_pred = np.random.randn(3, 2)  # Wrong length
    
    msg = r"Length of exog \(3\) must match steps \(5\)\."
    with pytest.raises(ValueError, match=msg):
        model.predict_interval(steps=5, exog=exog_pred)


def test_arar_predict_interval_exog_3d_raises():
    """
    Test that predict_interval raises error for 3D exog input.
    """
    np.random.seed(42)
    n = 100
    y = np.random.randn(n)
    exog_train = np.random.randn(n, 2)
    model = Arar()
    model.fit(y, exog=exog_train)
    exog_3d = np.random.randn(5, 2, 3)  # 3D array
    
    with pytest.raises(ValueError, match="`exog` must be 1D or 2D."):
        model.predict_interval(steps=5, exog=exog_3d)


def test_predict_interval_output_as_frame():
    """
    Test basic predict_interval functionality returning DataFrame.
    """
    y = ar1_series(120)
    est = Arar()
    est.fit(y)
    result = est.predict_interval(steps=8, level=(80, 95), as_frame=True)
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (8, 5)
    assert list(result.columns) == ['mean', 'lower_80', 'upper_80', 'lower_95', 'upper_95']
    assert result.index.name == 'step'
    assert list(result.index) == list(range(1, 9))
    
    expected_mean = np.array([
        0.58329928, 0.92906587, 0.90186554, 1.00217975,
        0.72872694, 0.57427369, 0.38811633, -0.16481327
    ])
    np.testing.assert_array_almost_equal(result['mean'].values, expected_mean, decimal=8)
    
    expected_lower_80 = np.array([
        -0.57918853, -0.47313545, -0.5967535, -0.53830729,
        -0.8117624, -0.97436655, -1.17541415, -1.74349659
    ])
    expected_upper_80 = np.array([
        1.7457871, 2.33126719, 2.40048457, 2.54266678,
        2.26921628, 2.12291394, 1.95164682, 1.41387004
    ])
    expected_lower_95 = np.array([
        -1.19457241, -1.21541599, -1.39007449, -1.35379186,
        -1.6272482, -1.79416718, -2.0030972, -2.57920106
    ])
    expected_upper_95 = np.array([
        2.36117097, 3.07354773, 3.19380556, 3.35815135,
        3.08470208, 2.94271456, 2.77932986, 2.24957451
    ])
    
    np.testing.assert_array_almost_equal(result['lower_80'].values, expected_lower_80, decimal=6)
    np.testing.assert_array_almost_equal(result['upper_80'].values, expected_upper_80, decimal=6)
    np.testing.assert_array_almost_equal(result['lower_95'].values, expected_lower_95, decimal=6)
    np.testing.assert_array_almost_equal(result['upper_95'].values, expected_upper_95, decimal=6)
    
    assert np.all(result['lower_80'] < result['mean'])
    assert np.all(result['mean'] < result['upper_80'])
    assert np.all(result['lower_95'] < result['mean'])
    assert np.all(result['mean'] < result['upper_95'])
    
    assert np.all(result['lower_95'] < result['lower_80'])
    assert np.all(result['upper_95'] > result['upper_80'])


def test_predict_interval_output_as_dict():
    """
    Test predict_interval functionality returning dict.
    """
    y = ar1_series(120)
    est = Arar()
    est.fit(y)
    result = est.predict_interval(steps=8, level=(80, 95), as_frame=False)
    
    assert isinstance(result, dict)
    assert 'mean' in result
    assert 'lower' in result
    assert 'upper' in result
    assert 'level' in result
    
    assert result['mean'].shape == (8,)
    assert result['lower'].shape == (8, 2)
    assert result['upper'].shape == (8, 2)
    assert result['level'] == [80, 95]
    
    expected_mean = np.array([
        0.58329928, 0.92906587, 0.90186554, 1.00217975,
        0.72872694, 0.57427369, 0.38811633, -0.16481327
    ])
    np.testing.assert_array_almost_equal(result['mean'], expected_mean, decimal=8)
    
    expected_lower_80 = np.array([
        -0.57918853, -0.47313545, -0.5967535, -0.53830729,
        -0.8117624, -0.97436655, -1.17541415, -1.74349659
    ])
    expected_upper_80 = np.array([
        1.7457871, 2.33126719, 2.40048457, 2.54266678,
        2.26921628, 2.12291394, 1.95164682, 1.41387004
    ])
    np.testing.assert_array_almost_equal(result['lower'][:, 0], expected_lower_80, decimal=6)
    np.testing.assert_array_almost_equal(result['upper'][:, 0], expected_upper_80, decimal=6)
    
    expected_lower_95 = np.array([
        -1.19457241, -1.21541599, -1.39007449, -1.35379186,
        -1.6272482, -1.79416718, -2.0030972, -2.57920106
    ])
    expected_upper_95 = np.array([
        2.36117097, 3.07354773, 3.19380556, 3.35815135,
        3.08470208, 2.94271456, 2.77932986, 2.24957451
    ])
    np.testing.assert_array_almost_equal(result['lower'][:, 1], expected_lower_95, decimal=6)
    np.testing.assert_array_almost_equal(result['upper'][:, 1], expected_upper_95, decimal=6)


def test_reduce_memory_preserves_predict_interval():
    """
    Test that predict_interval results are the same after reduce_memory().
    """
    y = ar1_series(100)
    est = Arar()
    est.fit(y)
    result_before = est.predict_interval(steps=10, level=(80, 95), as_frame=True)
    est.reduce_memory()
    result_after = est.predict_interval(steps=10, level=(80, 95), as_frame=True)
    
    pd.testing.assert_frame_equal(result_before, result_after)


def test_arar_predict_interval_with_multiple_exog_features():
    """
    Test predict_interval with multiple exogenous features.
    """
    np.random.seed(42)
    n = 150
    y = np.random.randn(n).cumsum()
    exog = np.column_stack([
        np.random.randn(n),
        np.sin(np.linspace(0, 4*np.pi, n)),
        np.cos(np.linspace(0, 4*np.pi, n)),
        np.arange(n) / n
    ])
    y = y + 0.5 * exog[:, 0] + 2.0 * exog[:, 1] + 1.5 * exog[:, 2] + 10.0 * exog[:, 3]
    
    model = Arar()
    model.fit(y, exog=exog, suppress_warnings=True)
    exog_future = np.column_stack([
        np.random.randn(10),
        np.sin(np.linspace(4*np.pi, 5*np.pi, 10)),
        np.cos(np.linspace(4*np.pi, 5*np.pi, 10)),
        np.arange(10) / n + 1.0
    ])
    
    result = model.predict_interval(steps=10, exog=exog_future, level=(95,), as_frame=True)
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (10, 3)
    assert list(result.columns) == ['mean', 'lower_95', 'upper_95']
    
    expected_mean = np.array([
        -1.17765989, 0.05214722, 0.77938843, 0.79109465, 0.04091216,
        -1.00788048, -1.94952114, -4.02466845, -5.16595928, -6.79566385
    ])
    np.testing.assert_array_almost_equal(result['mean'].values, expected_mean, decimal=6)
    
    expected_lower_95 = np.array([
        -2.97002526, -2.32986168, -2.02794823, -2.34447531, -3.35877838,
        -4.70710257, -5.89102004, -8.16906431, -9.4819085, -11.25790565
    ])
    expected_upper_95 = np.array([
        0.61470547, 2.43415613, 3.58672509, 3.92666461, 3.44060271,
        2.69134161, 1.99197776, 0.11972741, -0.85001106, -2.33342205
    ])
    
    np.testing.assert_array_almost_equal(result['lower_95'].values, expected_lower_95, decimal=6)
    np.testing.assert_array_almost_equal(result['upper_95'].values, expected_upper_95, decimal=6)
    
    assert np.all(result['lower_95'] < result['mean'])
    assert np.all(result['mean'] < result['upper_95'])
