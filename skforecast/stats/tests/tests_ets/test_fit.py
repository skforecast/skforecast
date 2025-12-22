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

def test_ets_fit_invalid_y_type_raises():
    """
    Test that fit raises TypeError for invalid y type.
    """
    y = [1, 2, 3, 4, 5]  # List, not Series or ndarray
    model = Ets()
    msg = "`y` must be a pandas Series or numpy ndarray."
    with pytest.raises(ValueError, match=msg):
        model.fit(y)

def test_ets_fit_multidimensional_y_raises():
    """
    Test that fit raises error for multidimensional y input.
    """
    y = np.random.randn(50, 2)
    model = Ets()
    msg = "`y` must be a 1D array-like sequence."
    with pytest.raises(ValueError, match=msg):
        model.fit(y)


def test_ets_fit_empty_series_raises():
    """
    Test that fit raises error for empty series.
    """
    y = np.array([])
    model = Ets()
    msg = "`y` is too short to fit ETS model."
    with pytest.raises(ValueError, match=msg):
        model.fit(y)


def test_ets_fit_2d_single_column_array():
    """
    Test that fit accepts 2D array with single column and squeezes it.
    """
    y = ar1_series(50).reshape(-1, 1)
    model = Ets(m=1, model="ANN")
    model.fit(y)
    
    assert model.y_.shape == (50,)
    assert model.y_.ndim == 1

def test_ets_fit_single_column_dataframe():
    """
    Test that fit accepts DataFrame with single column and squeezes it.
    """
    y = pd.DataFrame(ar1_series(50)).squeeze()
    model = Ets(m=1, model="ANN")
    model.fit(y)
    
    assert model.y_.shape == (50,)
    assert model.y_.ndim == 1

def test_ets_fit_and_attributes():
    """Test Ets estimator fit and attributes"""
    y = ar1_series(100)
    model = Ets(m=1, model="ANN")
    model.fit(y)

    assert hasattr(model, "model_")
    assert hasattr(model, "y_")
    assert hasattr(model, "config_")
    assert hasattr(model, "params_")
    assert hasattr(model, "fitted_values_")
    assert hasattr(model, "residuals_in_")
    assert hasattr(model, "n_features_in_")

    assert model.y_.shape == y.shape
    assert model.fitted_values_.shape == y.shape
    assert model.residuals_in_.shape == y.shape
    assert model.n_features_in_ == 1

def test_fit_ets_ann():
    """
    Test that ANN model learns expected parameter values.
    """
    rng = np.random.default_rng(123)
    y = 10 + 0.5 * np.arange(50) + rng.normal(0, 0.5, 50)
    model = Ets(m=1, model='ANN')
    model.fit(y)
    
    expected_config = {'error': 'A', 'trend': 'N', 'season': 'N', 'damped': False, 'm': 1}
    assert model.config_ == expected_config
    assert model.params_['alpha'] == 0.1
    assert model.params_['beta'] == 0.0
    assert model.params_['gamma'] == 0.0
    assert model.params_['phi'] == 1.0
    assert 'init_states' in model.params_
    
    np.testing.assert_almost_equal(model.y_, y, decimal=8)

    # Check the first 10 fitted values
    expected_fitted = np.array([
        12.29444152, 12.0155413, 11.84559784, 11.82543432, 11.80258961,
        11.86834219, 11.96036316, 12.03250366, 12.20635091, 12.36988604
    ])
    np.testing.assert_array_almost_equal(
        model.fitted_values_[:10],
        expected_fitted,
        decimal=8
    )
    np.testing.assert_array_almost_equal(
        model.residuals_in_,
        y - model.fitted_values_,
        decimal=8
    )


def test_fit_ets_aan():
    """
    Test that AAN model learns expected parameter and trend values.
    """
    rng = np.random.default_rng(123)
    y = 10 + 0.5 * np.arange(50) + rng.normal(0, 0.5, 50)
    model = Ets(m=1, model='AAN')
    model.fit(y)
    
    expected_config = {'error': 'A', 'trend': 'A', 'season': 'N', 'damped': False, 'm': 1}
    assert model.config_ == expected_config
    assert model.params_['alpha'] == 0.1
    assert model.params_['beta'] > 0.0  # Beta should be estimated for trend model
    assert model.params_['gamma'] == 0.0
    assert model.params_['phi'] == 1.0
    assert 'init_states' in model.params_
    
    np.testing.assert_almost_equal(model.y_, y, decimal=8)

    # Check the first 10 fitted values
    expected_fitted = np.array([
        10.04724213, 10.48702146, 10.96218045, 11.52942696, 12.03592687,
        12.58233151, 13.10900153, 13.56805384, 14.09215094, 14.56840645
    ])
    np.testing.assert_array_almost_equal(
        model.fitted_values_[:10],
        expected_fitted,
        decimal=8
    )
    np.testing.assert_array_almost_equal(
        model.residuals_in_,
        y - model.fitted_values_,
        decimal=8
    )


def test_fit_ets_aaa():
    """
    Test that AAA model learns expected seasonal parameter values.
    """
    rng = np.random.default_rng(123)
    t = np.arange(60)
    y = 10 + 0.1*t + 2*np.sin(2*np.pi*t/12) + rng.normal(0, 0.3, 60)
    model = Ets(m=12, model='AAA')
    model.fit(y)
    
    expected_config = {'error': 'A', 'trend': 'A', 'season': 'A', 'damped': False, 'm': 12}
    assert model.config_ == expected_config
    assert model.params_['alpha'] == 0.1
    assert model.params_['beta'] > 0.0  # Beta should be estimated for trend model
    assert model.params_['gamma'] > 0.0  # Gamma should be estimated for seasonal model
    assert model.params_['phi'] == 1.0
    assert 'init_states' in model.params_
    
    np.testing.assert_almost_equal(model.y_, y, decimal=8)

    # Check the first 10 fitted values
    expected_fitted = np.array([
        10.12450248, 10.93641956, 12.06930232, 12.26870937, 12.27042302,
        11.36429999, 10.57106752,  9.61916466,  8.92864668,  8.7293051
    ])
    np.testing.assert_array_almost_equal(
        model.fitted_values_[:10],
        expected_fitted,
        decimal=8
    )
    np.testing.assert_array_almost_equal(
        model.residuals_in_,
        y - model.fitted_values_,
        decimal=8
    )


def test_fit_ets_ana():
    """
    Test that ANA model (additive error, no trend, additive seasonal) learns exact values.
    """
    rng = np.random.default_rng(42)
    t = np.arange(48)
    y = 15 + 3*np.sin(2*np.pi*t/12) + rng.normal(0, 0.4, 48)
    model = Ets(m=12, model='ANA')
    model.fit(y)
    
    expected_config = {'error': 'A', 'trend': 'N', 'season': 'A', 'damped': False, 'm': 12}
    assert model.config_ == expected_config
    assert model.params_['alpha'] == 0.1
    assert model.params_['beta'] == 0.0
    assert model.params_['gamma'] > 0.0  # Gamma should be estimated for seasonal model
    assert model.params_['phi'] == 1.0
    assert 'init_states' in model.params_
    
    np.testing.assert_almost_equal(model.y_, y, decimal=8)

    # Check the first 10 fitted values
    expected_fitted = np.array([
        14.95208798, 16.37390343, 17.64039636, 18.07740199, 17.54573306,
        16.28827869, 15.14573767, 13.33657389, 12.24391   , 11.71337189
    ])
    np.testing.assert_array_almost_equal(
        model.fitted_values_[:10],
        expected_fitted,
        decimal=8
    )
    np.testing.assert_array_almost_equal(
        model.residuals_in_,
        y - model.fitted_values_,
        decimal=8
    )


def test_fit_ets_man():
    """
    Test that MAN model (multiplicative error, additive trend, no season) learns exact values.
    """
    rng = np.random.default_rng(456)
    y = 5 + 0.3 * np.arange(50) + rng.normal(0, 0.3, 50)
    # Ensure positive values for multiplicative error
    y = np.abs(y) + 1.0
    
    model = Ets(m=1, model='MAN')
    model.fit(y)
    
    expected_config = {'error': 'M', 'trend': 'A', 'season': 'N', 'damped': False, 'm': 1}
    assert model.config_ == expected_config
    assert model.params_['alpha'] == 0.1
    assert model.params_['beta'] > 0.0  # Beta should be estimated for trend model
    assert model.params_['gamma'] == 0.0
    assert model.params_['phi'] == 1.0
    assert 'init_states' in model.params_
    
    np.testing.assert_almost_equal(model.y_, y, decimal=8)

    # Check the first 10 fitted values
    expected_fitted = np.array([
        5.84235831, 6.21201906, 6.47862205, 6.8676237 , 7.19767769,
        7.43991747, 7.76034491, 8.06291247, 8.37183227, 8.68293215
    ])
    np.testing.assert_array_almost_equal(
        model.fitted_values_[:10],
        expected_fitted,
        decimal=8
    )
    np.testing.assert_array_almost_equal(
        model.residuals_in_,
        y - model.fitted_values_,
        decimal=8
    )

def test_fit_ets_auto_selection():
    """
    Test that automatic model selection (ZZZ) works correctly.
    For this trending data, auto-selection should pick a model with trend.
    """
    rng = np.random.default_rng(456)
    y = 5 + 0.3 * np.arange(50) + rng.normal(0, 0.3, 50)
    # Ensure positive values for multiplicative error
    y = np.abs(y) + 1.0
    
    model = Ets(m=1, model='ZZZ')
    model.fit(y)
    
    expected_config = {'error': 'A', 'trend': 'A', 'season': 'N', 'damped': False, 'm': 1}
    assert model.config_ == expected_config
    assert model.params_['alpha'] == 0.1
    assert model.params_['beta'] == 0.01
    assert model.params_['gamma'] == 0.0
    assert model.params_['phi'] == 1.0
    assert 'init_states' in model.params_
    
    np.testing.assert_almost_equal(model.y_, y, decimal=8)

    # Check the first 10 fitted values
    expected_fitted = np.array([
        5.84235831, 6.21201906, 6.47862205, 6.8676237 , 7.19767769,
        7.43991747, 7.76034491, 8.06291247, 8.37183227, 8.68293215
    ])
    np.testing.assert_array_almost_equal(
        model.fitted_values_[:10],
        expected_fitted,
        decimal=8
    )
    np.testing.assert_array_almost_equal(
        model.residuals_in_,
        y - model.fitted_values_,
        decimal=8
    )


def test_fit_ets_aan_damped_trend():
    """
    Test that damped trend model includes phi parameter.
    """
    rng = np.random.default_rng(789)
    y = 20 + 2 * np.arange(60) * np.exp(-0.05 * np.arange(60)) + rng.normal(0, 0.5, 60)
    
    model = Ets(m=1, model='AAN', damped=True)
    model.fit(y)
    
    expected_config = {'error': 'A', 'trend': 'A', 'season': 'N', 'damped': True, 'm': 1}
    assert model.config_ == expected_config
    assert 0.0 < model.params_['phi'] <= 1.0  # Damping parameter should be estimated
    assert 'alpha' in model.params_
    assert 'beta' in model.params_
    assert model.params_['gamma'] == 0.0
    assert 'init_states' in model.params_
    
    np.testing.assert_almost_equal(model.y_, y, decimal=8)

    # Check the first 10 fitted values
    expected_fitted = np.array([
        19.61183632, 22.0867558 , 23.58093134, 25.19210652, 26.6835265 ,
        27.86936012, 29.32280836, 30.34063813, 31.31531167, 31.94853773
    ])
    np.testing.assert_array_almost_equal(
        model.fitted_values_[:10],
        expected_fitted,
        decimal=8
    )
    np.testing.assert_array_almost_equal(
        model.residuals_in_,
        y - model.fitted_values_,
        decimal=8
    )


def test_fit_ets_fixed_alpha():
    """
    Test that fixed alpha parameter is respected.
    """
    y = ar1_series(80)
    fixed_alpha = 0.3
    model = Ets(m=1, model='ANN', alpha=fixed_alpha)
    model.fit(y)
    
    assert model.params_['alpha'] == fixed_alpha


def test_ets_fit_fixed_beta():
    """
    Test that fixed beta parameter is respected.
    """
    y = ar1_series(80)
    fixed_beta = 0.2
    model = Ets(m=1, model='AAN', beta=fixed_beta)
    model.fit(y)
    
    assert model.params_['beta'] == fixed_beta


def test_ets_fit_fixed_gamma():
    """
    Test that fixed gamma parameter is respected for seasonal model.
    """
    y = seasonal_series(60, m=12)
    fixed_gamma = 0.15
    model = Ets(m=12, model='AAA', gamma=fixed_gamma)
    model.fit(y)
    
    assert model.params_['gamma'] == fixed_gamma

