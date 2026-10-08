# Unit test get_info_criteria method - Arima
# ==============================================================================
import re
import pytest
import numpy as np
from sklearn.exceptions import NotFittedError
from ..._arima import Arima
from .fixtures_arima import air_passengers


def ar1_series(n=100, phi=0.7, sigma=1.0, seed=123):
    """Helper function to generate AR(1) series for testing."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


def test_get_info_criteria_raises_NotFittedError_when_not_fitted():
    """
    Test that get_info_criteria raises NotFittedError when the model has not
    been fitted.
    """
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))

    msg = re.escape(
        "This Arima instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=msg):
        model.get_info_criteria(criteria='aic')


def test_get_info_criteria_raises_ValueError_for_invalid_criteria():
    """
    Test that get_info_criteria raises ValueError for an invalid criteria name.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)

    for invalid in ('hqic', 'aicc', 'log_likelihood', ''):
        msg = re.escape(
            f"Invalid value for `criteria`: '{invalid}'. "
            f"Valid options are 'aic' and 'bic'."
        )
        with pytest.raises(ValueError, match=msg):
            model.get_info_criteria(criteria=invalid)


def test_get_info_criteria_aic_returns_float():
    """
    Test that get_info_criteria returns a float when criteria='aic'.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)

    result = model.get_info_criteria(criteria='aic')

    assert isinstance(result, (float, np.floating))


def test_get_info_criteria_aic_matches_aic_attribute():
    """
    Test that get_info_criteria(criteria='aic') equals the aic_ attribute.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)

    aic = model.get_info_criteria(criteria='aic')

    np.testing.assert_almost_equal(aic, model.aic_, decimal=10)


def test_get_info_criteria_aic_exact_value():
    """
    Test that get_info_criteria returns exact AIC value for a known model.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)

    aic = model.get_info_criteria(criteria='aic')

    np.testing.assert_almost_equal(aic, 238.8595629710312, decimal=4)


def test_get_info_criteria_bic_matches_bic_attribute():
    """
    Test that get_info_criteria(criteria='bic') equals the bic_ attribute,
    which is also available for models with a manual order.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)

    bic = model.get_info_criteria(criteria='bic')

    assert np.isclose(model.bic_, 246.67507352901887)
    np.testing.assert_almost_equal(bic, model.bic_, decimal=10)


@pytest.mark.parametrize(
    "order, seasonal_order, m, expected_aic, expected_bic",
    [((1, 0, 0), (0, 0, 0), 1, -228.13092253933831, -219.2214826406103),
     ((1, 1, 1), (0, 0, 0), 1, -242.62622163300256, -233.73768774222285),
     ((0, 1, 1), (0, 1, 1), 12, -483.39906053339917, -474.7734685637957)],
    ids=['AR(1)', 'ARIMA(1,1,1)', 'airline']
)
def test_get_info_criteria_bic_with_manual_order(
    order, seasonal_order, m, expected_aic, expected_bic
):
    """
    Test that BIC is computed for models with a manual order (before the fix it
    was None and get_info_criteria returned NaN). Values match statsmodels
    SARIMAX (BIC = -2 loglik + k log(n - d - D*m)) on log(air_passengers).
    CSS has no likelihood, so AIC is NaN and BIC is None.
    """
    y = np.log(air_passengers.to_numpy())
    model = Arima(order=order, seasonal_order=seasonal_order, m=m)
    model.fit(y)

    assert np.isclose(model.get_info_criteria('aic'), expected_aic)
    assert np.isclose(model.get_info_criteria('bic'), expected_bic)

    model_css = Arima(order=order, seasonal_order=seasonal_order, m=m, method='CSS')
    model_css.fit(y)

    assert model_css.bic_ is None
    assert np.isnan(model_css.get_info_criteria('bic'))


def test_get_info_criteria_default_criteria_is_aic():
    """
    Test that the default criteria parameter is 'aic'.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)

    result_default = model.get_info_criteria()
    result_aic = model.get_info_criteria(criteria='aic')

    np.testing.assert_almost_equal(result_default, result_aic, decimal=10)


def test_get_info_criteria_aic_is_positive():
    """
    Test that AIC is a positive finite number for a standard AR(1) model.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)

    aic = model.get_info_criteria(criteria='aic')

    assert np.isfinite(aic)
    assert aic > 0


def test_get_info_criteria_aic_different_for_different_orders():
    """
    Test that AIC values differ between models of different orders, indicating
    the criterion discriminates between models.
    """
    y = ar1_series(100, seed=42)

    model_ar1 = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model_ar1.fit(y)

    model_ar2 = Arima(order=(2, 0, 0), seasonal_order=(0, 0, 0))
    model_ar2.fit(y)

    aic_ar1 = model_ar1.get_info_criteria(criteria='aic')
    aic_ar2 = model_ar2.get_info_criteria(criteria='aic')

    # They should differ
    assert aic_ar1 != aic_ar2


def test_get_info_criteria_consistent_across_calls():
    """
    Test that repeated calls to get_info_criteria return the same value.
    """
    y = ar1_series(100, seed=42)
    model = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
    model.fit(y)

    aic_first = model.get_info_criteria(criteria='aic')
    aic_second = model.get_info_criteria(criteria='aic')

    np.testing.assert_almost_equal(aic_first, aic_second, decimal=10)
