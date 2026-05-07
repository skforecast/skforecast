# Unit test pacf
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.stats import pacf


# ==============================================================================
# Input validation
# ==============================================================================

def test_pacf_ValueError_when_x_is_not_1d():
    """
    Test that pacf raises ValueError when x is not 1-D.
    """
    x = np.ones((10, 2))
    err_msg = re.escape(f"`x` must be 1-D, got shape {x.shape}.")
    with pytest.raises(ValueError, match=err_msg):
        pacf(x, nlags=3)


def test_pacf_ValueError_when_x_has_fewer_than_2_observations():
    """
    Test that pacf raises ValueError when x has fewer than 2 observations.
    """
    x = np.array([1.0])
    err_msg = re.escape("`x` must have at least 2 observations, got 1.")
    with pytest.raises(ValueError, match=err_msg):
        pacf(x, nlags=1)


@pytest.mark.parametrize(
    "value",
    [np.nan, np.inf, -np.inf],
    ids=lambda v: f"value={v!r}",
)
def test_pacf_ValueError_when_x_contains_non_finite_values(value):
    """
    Test that pacf raises ValueError when x contains non-finite values.
    """
    x = np.array([1.0, 2.0, value, 4.0])
    err_msg = re.escape(
        "`x` contains non-finite values. Remove or impute them before calling `pacf`."
    )
    with pytest.raises(ValueError, match=err_msg):
        pacf(x, nlags=1)


@pytest.mark.parametrize(
    "x",
    [np.arange(2, dtype=float), np.arange(3, dtype=float)],
    ids=lambda x: f"n={len(x)}",
)
def test_pacf_ValueError_when_nlags_is_None_and_x_has_fewer_than_4_observations(x):
    """
    Test that pacf raises ValueError with a clear message when the default
    nlags cannot be computed for short series.
    """
    err_msg = re.escape(
        f"`x` must have at least 4 observations when `nlags` is None, got {len(x)}."
    )
    with pytest.raises(ValueError, match=err_msg):
        pacf(x)


@pytest.mark.parametrize(
    "nlags",
    [0, -1, 1.5, "3"],
    ids=lambda v: f"nlags={v!r}",
)
def test_pacf_ValueError_when_nlags_is_invalid(nlags):
    """
    Test that pacf raises ValueError when nlags is not a positive integer.
    """
    x = np.arange(20, dtype=float)
    err_msg = re.escape(f"`nlags` must be a positive integer, got {nlags!r}.")
    with pytest.raises(ValueError, match=err_msg):
        pacf(x, nlags=nlags)


def test_pacf_ValueError_when_nlags_ge_n_over_2():
    """
    Test that pacf raises ValueError when nlags >= len(x) // 2.
    """
    x = np.arange(10, dtype=float)  # n // 2 = 5
    err_msg = re.escape(
        "`nlags` (5) must be less than len(x) // 2 (5)."
    )
    with pytest.raises(ValueError, match=err_msg):
        pacf(x, nlags=5)


@pytest.mark.parametrize(
    "alpha",
    [0.0, 1.0, -0.1, 1.5],
    ids=lambda v: f"alpha={v}",
)
def test_pacf_ValueError_when_alpha_is_out_of_range(alpha):
    """
    Test that pacf raises ValueError when alpha is not in (0, 1).
    """
    x = np.arange(20, dtype=float)
    err_msg = re.escape(f"`alpha` must be in (0, 1), got {alpha!r}.")
    with pytest.raises(ValueError, match=err_msg):
        pacf(x, nlags=3, alpha=alpha)


# ==============================================================================
# Output correctness
# ==============================================================================

def test_pacf_lag0_is_always_one():
    """
    Test that pacf[0] is always 1.0.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100)
    result = pacf(x, nlags=10)
    assert result[0] == 1.0


def test_pacf_output_length():
    """
    Test that pacf returns an array of length nlags + 1.
    """
    x = np.arange(20, dtype=float)
    nlags = 7
    result = pacf(x, nlags=nlags)
    assert len(result) == nlags + 1


def test_pacf_accepts_pandas_series():
    """
    Test that pacf accepts a pandas Series and returns the same result as
    when a numpy array is passed.
    """
    rng = np.random.default_rng(123)
    arr = rng.standard_normal(100)
    series = pd.Series(arr)

    result_arr = pacf(arr, nlags=10)
    result_series = pacf(series, nlags=10)

    np.testing.assert_array_equal(result_arr, result_series)


def test_pacf_constant_series_returns_lag0_and_lag1_one_rest_zero():
    """
    Test that pacf returns [1.0, 1.0, 0.0, ...] for a constant series.
    The biased ACF is all-ones, so pacf[1] = acf[1] = 1.0. For k >= 2 the
    Levinson-Durbin numerator is 1 - (1)(1) = 0, so kk = 0.
    """
    x = np.ones(20)
    result = pacf(x, nlags=5)
    assert result[0] == 1.0
    assert result[1] == 1.0
    np.testing.assert_array_equal(result[2:], np.zeros(4))


def test_pacf_values_close_to_statsmodels_yw():
    """
    Test that pacf values are close to statsmodels pacf(method='yw').
    Differences up to ~2e-02 are expected because statsmodels yw uses the
    unbiased ACF estimator while our implementation uses the biased one.

    Expected values were generated with:
        from statsmodels.tsa.stattools import pacf as sm_pacf
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        sm_pacf(x, nlags=20, method='yw')
    """
    rng = np.random.default_rng(42)
    x = rng.standard_normal(500)
    nlags = 20

    result = pacf(x, nlags=nlags)
    expected = np.array([
         1.0,                   0.09933436607002508,  -0.009361900174929069,
        -0.03535704386671785,  -0.04762462062035379,  -0.006975727012403515,
        -0.07919461570055794,   0.044448479616062304, -0.0247267572105437,
        -0.06325928442112123,   0.0030302494535385557,-0.06467585842336651,
         0.018479739892256833,  0.0032914855817446657, 0.00930281542139028,
        -0.07781432409815071,   0.011523340638556598, -0.029763550633736178,
        -0.001596959129040229, -0.08814278354061361,  -0.0749529036007812,
    ])

    np.testing.assert_allclose(result, expected, atol=5e-2)


def test_pacf_ar1_process_lag1_close_to_true_coefficient():
    """
    Test that pacf lag-1 value is close to the true AR(1) coefficient and
    all other lags are close to zero for a correctly specified AR(1) process.
    """
    rng = np.random.default_rng(0)
    phi = 0.7
    n = 2000
    x = np.zeros(n)
    noise = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]

    result = pacf(x, nlags=10)

    # Lag 1 should be close to 0.7
    assert abs(result[1] - phi) < 0.05
    # Lags 2+ should be close to 0
    np.testing.assert_allclose(result[2:], 0.0, atol=0.1)


def test_pacf_nlags_default_matches_statsmodels_convention():
    """
    Test that the default nlags equals min(int(10 * log10(n)), n // 2 - 1).
    """
    import math

    n = 500
    x = np.arange(n, dtype=float)
    expected_nlags = min(int(10 * math.log10(n)), n // 2 - 1)
    result = pacf(x)
    assert len(result) == expected_nlags + 1


def test_pacf_alpha_returns_tuple_with_correct_shapes():
    """
    Test that pacf with alpha returns (pacf_vals, confint) with correct shapes.
    """
    rng = np.random.default_rng(99)
    x = rng.standard_normal(100)
    nlags = 10

    result = pacf(x, nlags=nlags, alpha=0.05)
    assert isinstance(result, tuple) and len(result) == 2

    pacf_vals, confint = result
    assert pacf_vals.shape == (nlags + 1,)
    assert confint.shape == (nlags + 1, 2)


def test_pacf_alpha_lag0_confint_is_one_one():
    """
    Test that the lag-0 confidence interval row equals [1.0, 1.0].
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100)
    _, confint = pacf(x, nlags=5, alpha=0.05)
    np.testing.assert_array_equal(confint[0], [1.0, 1.0])


def test_pacf_alpha_confint_lower_le_upper():
    """
    Test that all lower bounds are <= upper bounds in the confidence interval.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(300)
    _, confint = pacf(x, nlags=20, alpha=0.05)
    assert np.all(confint[:, 0] <= confint[:, 1])


def test_pacf_alpha_symmetric_ci_around_pacf_vals():
    """
    Test that the asymptotic CI is symmetric around the PACF value for lags >= 1.
    """
    rng = np.random.default_rng(5)
    x = rng.standard_normal(200)
    n = len(x)
    nlags = 10

    pacf_vals, confint = pacf(x, nlags=nlags, alpha=0.05)

    import scipy.stats

    z = scipy.stats.norm.ppf(0.975)
    se = z / np.sqrt(n)

    np.testing.assert_allclose(
        confint[1:, 1] - pacf_vals[1:], se * np.ones(nlags), atol=1e-12
    )
    np.testing.assert_allclose(
        pacf_vals[1:] - confint[1:, 0], se * np.ones(nlags), atol=1e-12
    )
