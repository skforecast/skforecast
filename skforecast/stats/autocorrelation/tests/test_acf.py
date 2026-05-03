# Unit test acf
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.stats.autocorrelation import acf


# ==============================================================================
# Input validation
# ==============================================================================

def test_acf_ValueError_when_x_is_not_1d():
    """
    Test that acf raises ValueError when x is not 1-D.
    """
    x = np.ones((10, 2))
    err_msg = re.escape(f"`x` must be 1-D, got shape {x.shape}.")
    with pytest.raises(ValueError, match=err_msg):
        acf(x, nlags=3)


def test_acf_ValueError_when_x_has_fewer_than_2_observations():
    """
    Test that acf raises ValueError when x has fewer than 2 observations.
    """
    x = np.array([1.0])
    err_msg = re.escape("`x` must have at least 2 observations, got 1.")
    with pytest.raises(ValueError, match=err_msg):
        acf(x, nlags=1)


def test_acf_ValueError_when_x_contains_nan():
    """
    Test that acf raises ValueError when x contains NaN values.
    """
    x = np.array([1.0, 2.0, np.nan, 4.0])
    err_msg = re.escape(
        "`x` contains NaN values. Remove or impute them before calling `acf`."
    )
    with pytest.raises(ValueError, match=err_msg):
        acf(x, nlags=2)


@pytest.mark.parametrize(
    "nlags",
    [0, -1, 1.5, "3"],
    ids=lambda v: f"nlags={v!r}",
)
def test_acf_ValueError_when_nlags_is_invalid(nlags):
    """
    Test that acf raises ValueError when nlags is not a positive integer.
    """
    x = np.arange(10, dtype=float)
    err_msg = re.escape(f"`nlags` must be a positive integer, got {nlags!r}.")
    with pytest.raises(ValueError, match=err_msg):
        acf(x, nlags=nlags)


def test_acf_ValueError_when_nlags_ge_n():
    """
    Test that acf raises ValueError when nlags >= len(x).
    """
    x = np.arange(5, dtype=float)
    err_msg = re.escape("`nlags` (5) must be less than len(x) (5).")
    with pytest.raises(ValueError, match=err_msg):
        acf(x, nlags=5)


@pytest.mark.parametrize(
    "alpha",
    [0.0, 1.0, -0.1, 1.5],
    ids=lambda v: f"alpha={v}",
)
def test_acf_ValueError_when_alpha_is_out_of_range(alpha):
    """
    Test that acf raises ValueError when alpha is not in (0, 1).
    """
    x = np.arange(10, dtype=float)
    err_msg = re.escape(f"`alpha` must be in (0, 1), got {alpha!r}.")
    with pytest.raises(ValueError, match=err_msg):
        acf(x, nlags=3, alpha=alpha)


# ==============================================================================
# Output correctness
# ==============================================================================

def test_acf_output_matches_statsmodels_to_machine_epsilon():
    """
    Test that acf output matches statsmodels acf(fft=True) to machine-epsilon
    precision.

    Expected values were generated with:
        from statsmodels.tsa.stattools import acf as sm_acf
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        sm_acf(x, nlags=50, fft=True)
    """
    rng = np.random.default_rng(42)
    x = rng.standard_normal(500)
    nlags = 50

    result = acf(x, nlags=nlags)
    expected = np.array([
         1.0,                    0.09913569733788503,   0.0005954017658856895,
        -0.03565986684406397,   -0.053758843843157184, -0.016569449395598186,
        -0.07809710691401324,    0.031229840484001355, -0.01266534246219786,
        -0.05869013482438313,   -0.0035498658127379665,-0.062377877542713775,
         0.016734517523886928,   0.007552473954470205,  0.019307139539630504,
        -0.057016506921635596,  -0.00794528696139045,  -0.016047224282376923,
        -0.0066022064311911195, -0.07393153548602459,  -0.07661819438467597,
         0.03994520337102848,    0.03209133608866717,   0.022311304864475107,
         0.005416878853908288,   0.02312462728513455,   0.010948892640916799,
         0.05072169766976325,   -0.04305819709766599,  -0.007922585627297385,
        -0.04661400048728328,    0.016817984103572648,  0.026512317546483134,
        -0.05994206278193841,    0.04412681318632541,   0.016341502721982967,
         0.0003044030152496711, -0.0014805848721214491, 0.021863087186091514,
         0.048418341870007854,  -0.03413129657632575,   0.012580318874466066,
        -0.015470733582192059,  -0.07159886270778008,  -0.01130857515687623,
        -0.01283814327638385,   -0.05128186010711154,   0.07354721686448462,
         0.050787909573738084,  -0.058226594094729016, -0.02019702384120793,
    ])

    np.testing.assert_allclose(result, expected, atol=1e-14)


def test_acf_lag0_is_always_one():
    """
    Test that acf[0] is always 1.0.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100)
    result = acf(x, nlags=10)
    assert result[0] == 1.0


def test_acf_output_length():
    """
    Test that acf returns an array of length nlags + 1.
    """
    x = np.arange(20, dtype=float)
    nlags = 7
    result = acf(x, nlags=nlags)
    assert len(result) == nlags + 1


def test_acf_accepts_pandas_series():
    """
    Test that acf accepts a pandas Series and returns the same result as
    when a numpy array is passed.
    """
    rng = np.random.default_rng(123)
    arr = rng.standard_normal(100)
    series = pd.Series(arr)

    result_arr = acf(arr, nlags=10)
    result_series = acf(series, nlags=10)

    np.testing.assert_array_equal(result_arr, result_series)


def test_acf_constant_series_returns_all_ones():
    """
    Test that acf returns all-ones for a constant series (zero variance).
    """
    x = np.ones(20)
    result = acf(x, nlags=5)
    np.testing.assert_array_equal(result, np.ones(6))


def test_acf_adjusted_true_matches_statsmodels():
    """
    Test that acf(adjusted=True) matches statsmodels acf(adjusted=True).

    Expected values were generated with:
        from statsmodels.tsa.stattools import acf as sm_acf
        rng = np.random.default_rng(7)
        x = rng.standard_normal(200)
        sm_acf(x, nlags=20, fft=False, adjusted=True)
    """
    rng = np.random.default_rng(7)
    x = rng.standard_normal(200)
    nlags = 20

    result = acf(x, nlags=nlags, adjusted=True)
    expected = np.array([
         1.0,                  -0.06159089674487768,  0.008195758906243725,
         0.08019021783072725,   0.007953197068213859,  0.009323598815243571,
         0.03651930689864072,  -0.0013895242590364106,-0.04273271333073031,
         0.17202409494733087,  -0.027586358767454595, -0.05805903778977217,
         0.014007668025661953,  0.068408666359161,     0.025838673258917474,
         0.06514209735147368,   0.027678724592418946, -0.02396558512901922,
         0.007046672507134828, -0.06900932999904867,  -0.1580630072809842,
    ])

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_acf_nlags_default_matches_statsmodels_convention():
    """
    Test that the default nlags equals min(int(10 * log10(n)), n - 1).
    """
    import math

    n = 500
    x = np.arange(n, dtype=float)
    expected_nlags = min(int(10 * math.log10(n)), n - 1)
    result = acf(x)
    assert len(result) == expected_nlags + 1


def test_acf_alpha_returns_tuple_with_correct_shapes():
    """
    Test that acf with alpha returns (acf_vals, confint) with correct shapes.
    """
    rng = np.random.default_rng(99)
    x = rng.standard_normal(100)
    nlags = 10

    result = acf(x, nlags=nlags, alpha=0.05)
    assert isinstance(result, tuple) and len(result) == 2

    acf_vals, confint = result
    assert acf_vals.shape == (nlags + 1,)
    assert confint.shape == (nlags + 1, 2)


def test_acf_alpha_lag0_confint_is_one_one():
    """
    Test that the lag-0 confidence interval is [1.0, 1.0].
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100)
    _, confint = acf(x, nlags=5, alpha=0.05)
    np.testing.assert_array_equal(confint[0], [1.0, 1.0])


def test_acf_alpha_confint_lower_le_upper():
    """
    Test that all lower bounds are <= upper bounds in the confidence interval.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(300)
    _, confint = acf(x, nlags=20, alpha=0.05)
    assert np.all(confint[:, 0] <= confint[:, 1])
