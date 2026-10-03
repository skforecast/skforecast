# Unit test fit method - Ets
# ==============================================================================
import re
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


def test_ets_fit_ValueError_when_model_is_partial_automatic():
    """
    Test that Ets raises a clear ValueError, instead of a KeyError, when the
    model string mixes automatic ('Z') and fixed components, and suggests how
    to restrict the automatic search.
    """
    y = np.random.default_rng(123).normal(10, 1, 60)
    model = Ets(model='ZZN')

    err_msg = re.escape(
        "Partial automatic model specifications such as 'ZZN' are not supported. "
        "Use model='ZZZ' (or None) for automatic selection and restrict the search "
        "with `seasonal`, `trend`, `damped`, `allow_multiplicative` and "
        "`allow_multiplicative_trend` (for example, model='ZZZ' with seasonal=False "
        "instead of 'ZZN'), or specify all three components."
    )
    with pytest.raises(ValueError, match=err_msg):
        model.fit(y)


def test_ets_fit_2d_single_column_array():
    """
    Test that fit accepts 2D array with single column and squeezes it.
    """
    y = ar1_series(50).reshape(-1, 1)
    model = Ets(m=1, model="ANN")
    model.fit(y)
    
    assert model.y_train_.shape == (50,)
    assert model.y_train_.ndim == 1

def test_ets_fit_single_column_dataframe():
    """
    Test that fit accepts DataFrame with single column and squeezes it.
    """
    y = pd.DataFrame(ar1_series(50)).squeeze()
    model = Ets(m=1, model="ANN")
    model.fit(y)
    
    assert model.y_train_.shape == (50,)
    assert model.y_train_.ndim == 1

def test_ets_fit_and_attributes():
    """Test Ets estimator fit and attributes"""
    y = ar1_series(100)
    model = Ets(m=1, model="ANN")
    model.fit(y)

    assert hasattr(model, "model_")
    assert hasattr(model, "y_train_")
    assert hasattr(model, "model_config_")
    assert hasattr(model, "params_")
    assert hasattr(model, "fitted_values_")
    assert hasattr(model, "in_sample_residuals_")
    assert hasattr(model, "n_features_in_")

    assert model.y_train_.shape == y.shape
    assert model.fitted_values_.shape == y.shape
    assert model.in_sample_residuals_.shape == y.shape
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
    assert model.model_config_ == expected_config
    np.testing.assert_allclose(model.params_['alpha'], 0.9472238981, atol=1e-6)
    assert model.params_['beta'] == 0.0
    assert model.params_['gamma'] == 0.0
    assert model.params_['phi'] == 1.0
    assert 'init_states' in model.params_
    
    np.testing.assert_almost_equal(model.y_train_, y, decimal=8)

    # Check the first 10 fitted values
    expected_fitted = np.array([
        9.5519216568, 9.5078924811, 10.2734522797, 11.5716324368, 11.5956490835,
        12.4144922848, 12.7688104875, 12.6863619307, 13.7137344018, 13.8349486289
    ])
    np.testing.assert_allclose(
        model.fitted_values_[:10], expected_fitted, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_array_almost_equal(
        model.in_sample_residuals_,
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
    assert model.model_config_ == expected_config
    # A straight line with noise: alpha and beta at their lower bound (1e-4)
    np.testing.assert_allclose(model.params_['alpha'], 1e-4, atol=1e-6)
    np.testing.assert_allclose(model.params_['beta'], 1e-4, atol=1e-6)
    assert model.params_['gamma'] == 0.0
    assert model.params_['phi'] == 1.0
    assert 'init_states' in model.params_
    
    np.testing.assert_almost_equal(model.y_train_, y, decimal=8)

    # Check the first 10 fitted values
    expected_fitted = np.array([
        10.0622024871, 10.5627587088, 11.0633212763, 11.5640246374, 12.0646765268,
        12.5654042077, 13.0661369743, 13.5667705524, 14.0674834084, 14.5681306877
    ])
    np.testing.assert_allclose(
        model.fitted_values_[:10], expected_fitted, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_array_almost_equal(
        model.in_sample_residuals_,
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
    assert model.model_config_ == expected_config
    # Deterministic trend and seasonality: smoothing parameters at their
    # lower bound (1e-4)
    np.testing.assert_allclose(model.params_['alpha'], 1e-4, atol=1e-6)
    np.testing.assert_allclose(model.params_['beta'], 1e-4, atol=1e-6)
    np.testing.assert_allclose(model.params_['gamma'], 1e-4, atol=1e-6)
    assert model.params_['phi'] == 1.0
    assert 'init_states' in model.params_
    
    np.testing.assert_almost_equal(model.y_train_, y, decimal=8)

    # Check the first 10 fitted values
    expected_fitted = np.array([
        10.2551109638, 11.1000627964, 12.2162820972, 12.3768765537, 12.3564951345,
        11.4312810673, 10.5866192378, 9.6428324171, 8.9061797058, 8.680621917
    ])
    np.testing.assert_allclose(
        model.fitted_values_[:10], expected_fitted, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_array_almost_equal(
        model.in_sample_residuals_,
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
    assert model.model_config_ == expected_config
    np.testing.assert_allclose(model.params_['alpha'], 1e-4, atol=1e-6)
    assert model.params_['beta'] == 0.0
    np.testing.assert_allclose(model.params_['gamma'], 1e-4, atol=1e-6)
    assert model.params_['phi'] == 1.0
    assert 'init_states' in model.params_
    
    np.testing.assert_almost_equal(model.y_train_, y, decimal=8)

    # Check the first 10 fitted values
    expected_fitted = np.array([
        14.9828470269, 16.3895103848, 17.6906381526, 18.1097347498, 17.5554769344,
        16.3712480123, 15.248160397, 13.4458555259, 12.3420922532, 11.7869900486
    ])
    np.testing.assert_allclose(
        model.fitted_values_[:10], expected_fitted, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_array_almost_equal(
        model.in_sample_residuals_,
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
    assert model.model_config_ == expected_config
    np.testing.assert_allclose(model.params_['alpha'], 1e-4, atol=1e-6)
    np.testing.assert_allclose(model.params_['beta'], 1e-4, atol=1e-6)
    assert model.params_['gamma'] == 0.0
    assert model.params_['phi'] == 1.0
    assert 'init_states' in model.params_
    
    np.testing.assert_almost_equal(model.y_train_, y, decimal=8)

    # Check the first 10 fitted values
    expected_fitted = np.array([
        6.0134677225, 6.3147699642, 6.6159162805, 6.9172230212, 7.2184765844,
        7.5195730679, 7.8207341635, 8.121858925, 8.4229764291, 8.7240864183
    ])
    np.testing.assert_allclose(
        model.fitted_values_[:10], expected_fitted, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_array_almost_equal(
        model.in_sample_residuals_,
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
    assert model.model_config_ == expected_config
    np.testing.assert_allclose(model.params_['alpha'], 1e-4, atol=1e-6)
    np.testing.assert_allclose(model.params_['beta'], 1e-4, atol=1e-6)
    assert model.params_['gamma'] == 0.0
    assert model.params_['phi'] == 1.0
    assert 'init_states' in model.params_
    
    np.testing.assert_almost_equal(model.y_train_, y, decimal=8)

    # Check the first 10 fitted values
    expected_fitted = np.array([
        6.0649963013, 6.3640691507, 6.6629813678, 6.9620495262, 7.2610602482,
        7.5599098565, 7.8588202679, 8.1576907616, 8.4565506403, 8.7553998723
    ])
    np.testing.assert_allclose(
        model.fitted_values_[:10], expected_fitted, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_array_almost_equal(
        model.in_sample_residuals_,
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
    assert model.model_config_ == expected_config
    # Same estimates as statsmodels' ETSModel
    np.testing.assert_allclose(model.params_['alpha'], 0.3255037610, atol=1e-6)
    np.testing.assert_allclose(model.params_['beta'], 0.1285965468, atol=1e-6)
    np.testing.assert_allclose(model.params_['phi'], 0.8753188574, atol=1e-6)
    assert model.params_['gamma'] == 0.0
    assert 'init_states' in model.params_
    
    np.testing.assert_almost_equal(model.y_train_, y, decimal=8)

    # Check the first 10 fitted values
    expected_fitted = np.array([
        19.61183632, 22.0867558 , 23.58093134, 25.19210652, 26.6835265 ,
        27.86936012, 29.32280836, 30.34063813, 31.31531167, 31.94853773
    ])
    np.testing.assert_allclose(
        model.fitted_values_[:10], expected_fitted, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_array_almost_equal(
        model.in_sample_residuals_,
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

