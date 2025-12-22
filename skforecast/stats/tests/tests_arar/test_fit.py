# Unit test fit method - Arar
# ==============================================================================
import numpy as np
import pandas as pd
import pytest
import warnings
from ..._arar import Arar
from ....exceptions import ExogenousInterpretationWarning

# Fixture functions
# ------------------------------------------------------------------------------
def ar1_series(n=80, phi=0.7, sigma=1.0, seed=123):
    """Helper function to generate AR(1) series for testing."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


def ar4_series(n=200, phi=(0.5, 0.3, -0.2, 0.1), sigma=1.0, seed=123):
    """Helper function to generate AR(4) series for testing coefficient recovery."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    y[:4] = e[:4]
    for t in range(4, n):
        y[t] = phi[0] * y[t-1] + phi[1] * y[t-2] + phi[2] * y[t-3] + phi[3] * y[t-4] + e[t]
    return y


# Input and Parameter Validation
# ------------------------------------------------------------------------------

def test_arar_fit_invalid_y_type_raises():
    """
    Test that fit raises TypeError for invalid y type.
    """
    y = [1, 2, 3, 4, 5]  # List, not Series or ndarray
    model = Arar()
    msg = "`y` must be a pandas Series or numpy ndarray."
    with pytest.raises(TypeError, match=msg):
        model.fit(y)


def test_arar_fit_invalid_exog_type_raises():
    """
    Test that fit raises TypeError for invalid exog type.
    """
    y = np.random.randn(50)
    exog = [1, 2, 3]  # List, not valid type
    model = Arar()
    msg = "`exog` must be None, a pandas Series, pandas DataFrame, or numpy ndarray."
    with pytest.raises(TypeError, match=msg):
        model.fit(y, exog=exog)


def test_arar_fit_multidimensional_y_raises():
    """
    Test that fit raises error for multidimensional y input.
    """
    y = np.random.randn(50, 2)
    model = Arar()
    msg = "`y` must be a 1D array-like sequence."
    with pytest.raises(ValueError, match=msg):
        model.fit(y)


def test_arar_fit_with_exog_length_mismatch():
    """
    Test that fit raises error when exog length doesn't match y length.
    """
    y = ar1_series(100)
    exog_wrong = np.random.randn(80, 2)  # Wrong length
    model = Arar()
    msg = r"Length of exog \(80\) must match length of y \(100\)"
    with pytest.raises(ValueError, match=msg):
        model.fit(y, exog=exog_wrong)


def test_arar_fit_with_exog_3d_raises():
    """
    Test that fit raises error for 3D exog input.
    """
    y = ar1_series(50)
    exog_3d = np.random.randn(50, 2, 3)  # 3D array
    model = Arar()
    msg = "`exog` must be 1D or 2D."
    with pytest.raises(ValueError, match=msg):
        model.fit(y, exog=exog_3d)


def test_arar_fit_very_short_series_safe_true():
    """
    Test that very short series works with safe=True.
    """
    y = np.array([1.0])
    model = Arar(safe=True)
    # Should not raise error with safe=True
    model.fit(y)
    assert model.model_ is not None


def test_arar_fit_very_short_series_safe_false_raises():
    """
    Test that fit raises error for very short series when safe=False.
    """
    y = np.array([1.0]) 
    model = Arar(safe=False)
    msg = "Series too short to fit ARAR when safe=False."
    with pytest.raises(ValueError, match=msg):
        model.fit(y)


def test_arar_fit_two_observations_safe_true():
    """
    Test fit with exactly 2 observations and safe=True.
    """
    y = np.array([1.0, 2.0])
    model = Arar(safe=True)
    # Should work with safe=True
    model.fit(y)
    assert model.model_ is not None


# Basic Fit Functionality Tests
# ------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "y_input_type",
    ["numpy_array", "pandas_series"],
    ids=lambda x: f"y_as_{x}"
)
def test_arar_fit_with_default_parameters(y_input_type):
    """
    Test fit results with default parameters and verify exact learned values.
    Tests both numpy array and pandas Series inputs.
    """
    y = ar1_series(100, seed=123)
    
    # Convert to appropriate type
    if y_input_type == "pandas_series":
        y = pd.Series(y, index=pd.date_range(start='2020-01-01', periods=100, freq='D'))
    
    model = Arar()
    result = model.fit(y)

    assert hasattr(model, "model_")
    np.testing.assert_array_almost_equal(model.y_, y, decimal=10)

    expected_coef = np.array([0.6621508, -0.1511758, -0.08748325, -0.09202529])
    np.testing.assert_array_almost_equal(model.coef_, expected_coef, decimal=6)

    expected_lags = (1, 3, 15, 18)
    assert model.lags_ == expected_lags

    expected_sigma2 = 0.7510658616551584
    np.testing.assert_almost_equal(model.sigma2_, expected_sigma2, decimal=6)

    expected_psi = np.array([1.0])
    np.testing.assert_array_almost_equal(model.psi_, expected_psi, decimal=10)

    expected_sbar = 0.2555764169122907
    np.testing.assert_almost_equal(model.sbar_, expected_sbar, decimal=6)


    # First fitted_values values will be NaN due to lag structure
    assert np.isnan(model.fitted_values_[:18]).all()
    # The first non-NaN fitted value should match expected
    first_valid_idx = np.where(~np.isnan(model.fitted_values_))[0][0]
    assert first_valid_idx == np.max(model.lags_)

    # Check exact first 10 non-NaN fitted values
    expected_first_10_fitted = np.array([
        0.5800004993187984, -0.02746188896173543, 0.18645199039610316,
        -1.1949387396599094, -0.4201486297576694, 1.0862510495918631,
        1.5193628857148958, 1.323152726816019, 0.8730881701811084,
        1.3225032039860298
    ])
    np.testing.assert_array_almost_equal(
        model.fitted_values_[first_valid_idx:first_valid_idx+10],
        expected_first_10_fitted,
        decimal=6
    )

    # Verify residuals = y - fitted_values exactly
    np.testing.assert_array_almost_equal(
        model.residuals_in_,
        y - model.fitted_values_,
        decimal=10
    )

    # Check exact AIC/BIC values
    np.testing.assert_almost_equal(model.aic_, 223.6333233886359, decimal=6)
    np.testing.assert_almost_equal(model.bic_, 238.0736388722214, decimal=6)

    assert model.n_features_in_ == 1
    assert model.n_exog_features_in_ == 0
    assert model.exog_model_ is None
    assert model.coef_exog_ is None
    assert model.memory_reduced_ is False


def test_arar_fit_with_given_parameters():
    """
    Test fit results with given parameters and verify exact learned values.
    """
    y = ar1_series(100, seed=123)
    model = Arar(max_ar_depth=10, max_lag=20)
    result = model.fit(y)

    assert hasattr(model, "model_")
    np.testing.assert_array_almost_equal(model.y_, y, decimal=10)

    expected_coef = np.array([0.68051374, -0.14307879, -0.12198576, 0.12992447])
    np.testing.assert_array_almost_equal(model.coef_, expected_coef, decimal=6)

    expected_lags = (1, 3, 7, 8)
    assert model.lags_ == expected_lags

    expected_sigma2 = 0.7582529108145794
    np.testing.assert_almost_equal(model.sigma2_, expected_sigma2, decimal=6)

    expected_psi = np.array([1.0])
    np.testing.assert_array_almost_equal(model.psi_, expected_psi, decimal=10)

    expected_sbar = 0.2555764169122907
    np.testing.assert_almost_equal(model.sbar_, expected_sbar, decimal=6)

    # First fitted_values values will be NaN due to lag structure (lag 8)
    largest_lag = max(model.lags_)
    assert np.isnan(model.fitted_values_[:largest_lag]).all()
    # The first non-NaN fitted value should match expected
    first_valid_idx = np.where(~np.isnan(model.fitted_values_))[0][0]
    assert first_valid_idx == largest_lag

    # Check exact first 10 non-NaN fitted values
    expected_first_10_fitted = np.array([
        0.468857189527469, 0.03378143687427011, -0.09475666325921547,
        -0.0071033831161424615, -0.9132423255960366, 0.34559013140561684,
        -0.11038949683225877, 0.5767412704995151, 0.6255568996187596,
        1.3184603152225796
    ])
    np.testing.assert_array_almost_equal(
        model.fitted_values_[first_valid_idx:first_valid_idx+10],
        expected_first_10_fitted,
        decimal=6
    )

    # Verify residuals = y - fitted_values exactly
    np.testing.assert_array_almost_equal(
        model.residuals_in_,
        y - model.fitted_values_,
        decimal=10
    )

    # Check exact AIC/BIC values
    np.testing.assert_almost_equal(model.aic_, 248.9417152747054, decimal=6)
    np.testing.assert_almost_equal(model.bic_, 264.0724467369996, decimal=6)

    assert model.n_features_in_ == 1
    assert model.n_exog_features_in_ == 0
    assert model.exog_model_ is None
    assert model.coef_exog_ is None
    assert model.memory_reduced_ is False

def test_arar_fit_exact_aic_bic_values():
    """
    Test that AIC and BIC are computed correctly with exact values.
    """
    y = ar1_series(n=100, phi=0.6, seed=123)
    
    model = Arar()
    model.fit(y)
    
    # Manual AIC/BIC calculation
    n = len(y)
    k_arar = 6  # 4 AR coefficients + sbar + sigma2
    residuals = model.residuals_in_
    valid_residuals = residuals[~np.isnan(residuals)]
    n_valid = len(valid_residuals)
    
    # Log-likelihood (assuming normal errors)
    rss = np.sum(valid_residuals**2)
    log_likelihood = -0.5 * n_valid * (np.log(2 * np.pi) + np.log(rss / n_valid) + 1)
    
    # Expected AIC and BIC
    expected_aic = -2 * log_likelihood + 2 * k_arar
    expected_bic = -2 * log_likelihood + k_arar * np.log(n_valid)
    
    # Check they match
    np.testing.assert_almost_equal(model.aic_, expected_aic, decimal=6)
    np.testing.assert_almost_equal(model.bic_, expected_bic, decimal=6)


def test_arar_fit_updates_max_ar_depth_and_max_lag():
    """
    Test that fit updates max_ar_depth and max_lag when None.
    """
    y = ar1_series(100)
    model = Arar(max_ar_depth=None, max_lag=None)
    model.fit(y)
    # For n=100, defaults should be 26 and 40
    assert model.max_ar_depth == 26
    assert model.max_lag == 40


def test_arar_fit_preserves_explicit_params():
    """
    Test that explicit parameters are preserved after fitting.
    """
    y = ar1_series(100)
    model = Arar(max_ar_depth=10, max_lag=20, safe=True)
    model.fit(y)
    
    # These should be preserved
    assert model.max_ar_depth == 10
    assert model.max_lag == 20
    assert model.safe is True


def test_arar_fit_model_tuple_structure():
    """
    Test that model_ tuple has correct structure and values.
    """
    y = ar1_series(100, seed=123)
    model = Arar()
    model.fit(y)
    
    # Check model_ tuple structure
    assert isinstance(model.model_, tuple)
    assert len(model.model_) == 8
    
    # Unpack and verify
    Y, best_phi, best_lag, sigma2, psi, sbar, max_ar_depth, max_lag = model.model_
    
    # Verify each component matches class attributes
    np.testing.assert_array_almost_equal(Y, model.y_, decimal=10)
    np.testing.assert_array_almost_equal(best_phi, model.coef_, decimal=10)
    assert best_lag == model.lags_
    np.testing.assert_almost_equal(sigma2, model.sigma2_, decimal=10)
    np.testing.assert_array_almost_equal(psi, model.psi_, decimal=10)
    np.testing.assert_almost_equal(sbar, model.sbar_, decimal=10)
    assert max_ar_depth == model.max_ar_depth
    assert max_lag == model.max_lag


def test_arar_fit_resets_memory_reduced_flag():
    """
    Test that refitting resets the memory_reduced flag.
    """
    y = ar1_series(100)
    model = Arar()
    model.fit(y)
    model.reduce_memory()
    
    assert model.memory_reduced_ is True
    assert model.residuals_in_ is None
    assert model.fitted_values_ is None


def test_arar_fit_nan_pattern_in_fitted_values():
    """
    Test that NaN pattern in fitted_values matches expected lag structure.
    """
    y = ar1_series(100, seed=123)
    model = Arar()
    model.fit(y)
    
    # Check NaN pattern based on largest lag
    largest_lag = max(model.lags_)
    
    # First largest_lag values should be NaN
    nan_count = np.sum(np.isnan(model.fitted_values_))
    assert nan_count == largest_lag  # 18 NaN values for lag 18
    
    # First nan_count values should be NaN
    assert np.all(np.isnan(model.fitted_values_[:nan_count]))
    
    # Remaining values should be finite
    assert np.all(np.isfinite(model.fitted_values_[nan_count:]))
    
    # Same pattern for residuals
    assert np.sum(np.isnan(model.residuals_in_)) == nan_count
    assert np.all(np.isnan(model.residuals_in_[:nan_count]))
    assert np.all(np.isfinite(model.residuals_in_[nan_count:]))


# Fit with Exogenous Variables Tests
# ------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "exog_input_type",
    ["numpy_array", "pandas_dataframe"],
    ids=lambda x: f"exog_as_{x}"
)
def test_arar_fit_with_exog(exog_input_type):
    """
    Test Arar fit with exogenous variables and verify exact learned values.
    Tests both numpy array and pandas DataFrame inputs for exog.
    """
    np.random.seed(42)
    n = 100
    
    # Create AR(1) series with exogenous effects
    y_ar = ar1_series(n=n, phi=0.6, sigma=0.5, seed=42)
    exog = np.column_stack([
        np.sin(np.linspace(0, 4*np.pi, n)),
        np.cos(np.linspace(0, 4*np.pi, n))
    ])
    
    # Convert to appropriate type
    if exog_input_type == "pandas_dataframe":
        exog = pd.DataFrame(exog, columns=['exog1', 'exog2'])
        y = y_ar + 2.0 * exog.iloc[:, 0] + 1.5 * exog.iloc[:, 1]
    else:
        y = y_ar + 2.0 * exog[:, 0] + 1.5 * exog[:, 1]
    
    model = Arar()
    model.fit(y, exog=exog, suppress_warnings=True)
    

    assert model.n_features_in_ == 1  # Always 1 for time series
    assert model.n_exog_features_in_ == 2
    assert len(model.coef_exog_) == 2
    expected_coef_exog = np.array([1.96805132, 1.36576084])
    np.testing.assert_array_almost_equal(model.coef_exog_, expected_coef_exog, decimal=6)
    assert hasattr(model.exog_model_, 'intercept_')
    expected_intercept = -0.049223690270959776
    np.testing.assert_almost_equal(model.exog_model_.intercept_, expected_intercept, decimal=6)
    
    # Check exact ARAR coefficients on residuals
    expected_lags = (1, 6, 10, 16)
    assert model.lags_ == expected_lags
    expected_coef = np.array([0.5740112, -0.15199523, 0.17431845, 0.1893768])
    np.testing.assert_array_almost_equal(model.coef_, expected_coef, decimal=6)
    np.testing.assert_almost_equal(model.sigma2_, 0.13259783018819143, decimal=6)
    np.testing.assert_almost_equal(model.sbar_, 2.1094237467877975e-17, decimal=6)
    
    # Check exact AIC/BIC with exog
    np.testing.assert_almost_equal(model.aic_, 79.0305393657764, decimal=6)
    np.testing.assert_almost_equal(model.bic_, 100.90789055536621, decimal=6)

def test_arar_fit_aic_bic_with_exog():
    """
    Test that AIC/BIC are computed correctly with exogenous variables.
    """
    y = ar1_series(100, seed=42)
    exog = np.random.randn(100, 2)
    
    model = Arar()
    model.fit(y, exog=exog, suppress_warnings=True)
    
    # AIC and BIC should account for both ARAR and exog parameters
    assert isinstance(model.aic_, float)
    assert isinstance(model.bic_, float)
    assert np.isfinite(model.aic_)
    assert np.isfinite(model.bic_)
    # BIC should be larger than AIC for reasonable sample size
    assert model.bic_ > model.aic_
    
    # With exog, should have k_arar=6 + k_exog=3 (intercept + 2 features)
    # Total = 9 parameters
    residuals = model.residuals_in_
    valid_residuals = residuals[~np.isnan(residuals)]
    n_valid = len(valid_residuals)
    
    # Verify parameter counting
    k_total = 6 + 3  # ARAR + exog regression
    rss = np.sum(valid_residuals**2)
    log_likelihood = -0.5 * n_valid * (np.log(2 * np.pi) + np.log(rss / n_valid) + 1)
    
    expected_aic = -2 * log_likelihood + 2 * k_total
    expected_bic = -2 * log_likelihood + k_total * np.log(n_valid)
    
    np.testing.assert_almost_equal(model.aic_, expected_aic, decimal=6)
    np.testing.assert_almost_equal(model.bic_, expected_bic, decimal=6)

# Warning tests
# ------------------------------------------------------------------------------
def test_arar_fit_exog_emits_warning():
    """
    Test that fit with exog emits ExogenousInterpretationWarning by default.
    """
    y = ar1_series(50, seed=42)
    exog = np.random.randn(50, 1)
    model = Arar()
    
    warn_msg = (
        r"Exogenous variables are being handled using a two-step approach: "
        r"\(1\) linear regression on exog, \(2\) ARAR on residuals. "
        r"This affects model interpretation:\n"
        r"  - ARAR coefficients \(coef_\) describe residual dynamics, not the original series\n"
        r"  - Pred intervals reflect only ARAR uncertainty, not exog regression uncertainty\n"
        r"  - Assumes a linear, time-invariant relationship between exog and target\n"
        r"For more details, see the fit\(\) method's Notes section of ARAR class. "
    )
    with pytest.warns(ExogenousInterpretationWarning, match=warn_msg):
        model.fit(y, exog=exog)


def test_arar_fit_exog_suppress_warnings():
    """
    Test that suppress_warnings=True prevents warning emission.
    """
    y = ar1_series(50, seed=42)
    exog = np.random.randn(50, 1)
    model = Arar()
    
    # Explicitly check that no ExogenousInterpretationWarning is raised
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")  # Capture all warnings
        model.fit(y, exog=exog, suppress_warnings=True)
        
        # Filter for ExogenousInterpretationWarning
        exog_warnings = [w for w in warning_list if issubclass(w.category, ExogenousInterpretationWarning)]
        assert len(exog_warnings) == 0, "ExogenousInterpretationWarning should not be raised when suppress_warnings=True"

