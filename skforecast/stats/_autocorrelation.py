################################################################################
#                         Autocorrelation functions                            #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
import math
import warnings
import numpy as np
import pandas as pd
from scipy.fft import irfft, next_fast_len, rfft
from scipy.stats import norm
from skforecast.exceptions import MissingValuesWarning


def _fft_acf(x_centered: np.ndarray, n: int, nlags: int) -> np.ndarray:
    """
    Biased ACF via FFT (internal helper, no validation).

    Parameters
    ----------
    x_centered : numpy ndarray
        Mean-centred 1-D time series.
    n : int
        Length of the original series.
    nlags : int
        Number of lags to return (result has length `nlags + 1`).

    Returns
    -------
    acf_vals : numpy ndarray, shape (nlags + 1,)
        Normalised autocorrelations for lags 0 ... nlags using the biased
        estimator (denominator `n`). If the series has zero variance
        (constant values), all entries are set to 1.0 to avoid division
        by zero.

    """

    n_fft = next_fast_len(2 * n - 1)
    fft_x = rfft(x_centered, n=n_fft)
    autocorr = irfft((fft_x * fft_x.conj()).real, n=n_fft)
    var = autocorr[0]

    return np.ones(nlags + 1) if var == 0.0 else autocorr[:nlags + 1] / var


def _pairwise_acf(x: np.ndarray, n_orig: int, nlags: int) -> np.ndarray:
    """
    Pairwise-deletion ACF for series with interleaved NaN/inf (internal helper).

    For each lag k uses only pairs (x[t-k], x[t]) where both values are
    finite, preserving true temporal distances. Lags with fewer than 2 valid
    pairs are set to NaN.

    Parameters
    ----------
    x : numpy ndarray
        1-D series (may contain NaN/inf at interleaved positions).
    n_orig : int
        Length of `x` after leading/trailing strip.
    nlags : int
        Number of lags to compute (result has length `nlags + 1`).

    Returns
    -------
    acf_vals : numpy ndarray, shape (nlags + 1,)

    """

    finite_mask = np.isfinite(x)
    x_valid = x[finite_mask]

    mean_x = x_valid.mean()
    var_x = np.mean((x_valid - mean_x) ** 2)  # biased, denominator n_valid
    acf_vals = np.ones(nlags + 1)

    if var_x == 0.0:
        return acf_vals

    # Centered series: non-finite positions become NaN so nansum skips them.
    # x_c[k:] and x_c[:n-k] are views (no copies); NaN propagates in the product.
    x_c = np.where(finite_mask, x - mean_x, np.nan)

    for k in range(1, nlags + 1):
        prod = x_c[k:] * x_c[:n_orig - k]
        n_k = int(np.isfinite(prod).sum())
        if n_k < 2:
            acf_vals[k] = np.nan
        else:
            acf_vals[k] = float(np.nansum(prod)) / (n_k * var_x)

    return acf_vals


def acf(
    x: pd.Series | np.ndarray,
    nlags: int | None = None,
    adjusted: bool = False,
    alpha: float | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Autocorrelation function (ACF) via FFT.

    Computes the sample ACF using the convolution theorem: the time series is
    transformed with `scipy.fft.rfft`, the power spectrum is computed as
    `|X[k]|²`, and the result is back-transformed with `irfft`. This achieves
    O(N log N) complexity versus the O(N²) time-domain approach used by
    `statsmodels.tsa.stattools.acf(fft=False)`.

    Parameters
    ----------
    x : pandas Series, numpy ndarray
        1-D time series. Must contain at least 2 observations.
    nlags : int, default None
        Number of lags to return. The result always includes lag 0, so the
        output length is `nlags + 1`. Must satisfy `0 < nlags < len(x)`. If
        `None`, defaults to `min(int(10 * log10(n)), n - 1)`, matching the
        statsmodels convention.
    adjusted : bool, default False
        If `True`, the autocovariance at lag k is divided by `n - k` (unbiased
        estimator). If `False`, it is divided by `n` (biased estimator, always
        produces a positive semi-definite sequence).
    alpha : float, default None
        Significance level for Bartlett confidence intervals. If given (e.g.
        `0.05` for 95% intervals), a second array of shape `(nlags + 1, 2)`
        is returned alongside the ACF values. Lag 0 always has interval
        `[1.0, 1.0]`. For lag k ≥ 1, the standard error follows Bartlett's
        formula: `Var(p_k) = (1/n)(1 + 2 * sum_{j=1}^{k-1} p_j²)`.

    Returns
    -------
    acf_vals : numpy ndarray, shape (nlags + 1,)
        Sample autocorrelations for lags 0, 1, ..., nlags. `acf_vals[0]` is
        always 1.0.
    confint : numpy ndarray, shape (nlags + 1, 2)
        Only returned when `alpha` is not `None`. Each row contains the lower
        and upper bounds of the `(1 - alpha)` confidence interval for the
        corresponding lag.

    Notes
    -----
    The biased estimator (`adjusted=False`) divides all autocovariances by `n`
    and guarantees a positive semi-definite Toeplitz matrix, which is required
    for Levinson-Durbin to be numerically stable. The unbiased estimator
    (`adjusted=True`) divides by `n - k` and can produce values outside
    `[-1, 1]` for large lags.

    The FFT padding length is chosen with `scipy.fft.next_fast_len` for a
    length greater than or equal to `2n - 1`, ensuring circular-convolution
    artefacts do not contaminate any of the `nlags` requested lags.

    If `x` contains leading or trailing non-finite values (NaN, ±inf), they
    are silently removed before any computation. If interleaved non-finite
    values remain after stripping, a ``MissingValuesWarning`` is issued and
    the function falls back to pairwise deletion: for each lag *k* only pairs
    *(x[t-k], x[t])* where both values are finite are used. This preserves
    true temporal distances but requires O(N·p) time instead of O(N log N)
    because FFT cannot be applied to irregular observations. Lags with fewer
    than 2 valid pairs are set to NaN.

    References
    ----------
    .. [1] Brockwell, P.J. and Davis, R.A. (1991). *Time Series: Theory and
       Methods*, 2nd ed. Springer.
    .. [2] Box, G.E.P., Jenkins, G.M., Reinsel, G.C. and Ljung, G.M. (2015).
       *Time Series Analysis: Forecasting and Control*, 5th ed. Wiley.

    Examples
    --------
    ```python
    import numpy as np
    from skforecast.stats import acf

    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)

    # ACF values for lags 0-10
    acf_vals = acf(x, nlags=10)

    # ACF with 95% Bartlett confidence intervals
    acf_vals, confint = acf(x, nlags=10, alpha=0.05)
    ```

    """

    if isinstance(x, pd.Series):
        x = x.to_numpy(dtype=float)
    else:
        x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError(f"`x` must be 1-D, got shape {x.shape}.")
    n = len(x)
    if n < 2:
        raise ValueError(
            f"`x` must have at least 2 observations, got {n}."
        )

    # Strip leading/trailing non-finite values; detect interleaved NaNs.
    valid_idx = np.where(np.isfinite(x))[0]
    if len(valid_idx) == 0:
        raise ValueError("`x` has no finite values.")
    x = x[valid_idx[0] : valid_idx[-1] + 1]
    n = len(x)
    n_valid = len(valid_idx)
    if n_valid < 2:
        raise ValueError(
            f"`x` must have at least 2 finite observations, got {n_valid}."
        )
    has_interleaved_nan = n_valid < n
    if has_interleaved_nan:
        warnings.warn(
            "Interleaved NaN/inf detected. Falling back to pairwise deletion "
            "(slower). Lags with fewer than 2 valid pairs will be NaN.",
            MissingValuesWarning,
            stacklevel=2,
        )

    n_eff = n_valid if has_interleaved_nan else n
    if nlags is None:
        nlags = min(int(10 * math.log10(n_eff)), n_eff - 1)
    if not isinstance(nlags, (int, np.integer)) or nlags < 1:
        raise ValueError(f"`nlags` must be a positive integer, got {nlags!r}.")
    if nlags >= n_eff:
        raise ValueError(
            f"`nlags` ({nlags}) must be less than len(x) ({n_eff})."
        )

    if alpha is not None and not (0.0 < alpha < 1.0):
        raise ValueError(f"`alpha` must be in (0, 1), got {alpha!r}.")

    if has_interleaved_nan:
        acf_vals = _pairwise_acf(x, n, nlags)
    else:
        acf_vals = _fft_acf(x - x.mean(), n, nlags)

    if adjusted:
        ks = np.arange(nlags + 1, dtype=float)
        acf_vals *= n_eff / (n_eff - ks)
        acf_vals[0] = 1.0

    if alpha is None:
        return acf_vals

    # Bartlett confidence intervals
    z = norm.ppf(1.0 - alpha / 2.0)
    varacf = np.ones(nlags + 1) / n_eff
    varacf[0] = 0.0
    if nlags > 1:
        varacf[2:] *= 1.0 + 2.0 * np.cumsum(acf_vals[1:-1] ** 2)
    interval = z * np.sqrt(varacf)
    confint = np.column_stack((acf_vals - interval, acf_vals + interval))
    return acf_vals, confint


def pacf(
    x: pd.Series | np.ndarray,
    nlags: int | None = None,
    alpha: float | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Partial autocorrelation function (PACF) via FFT + Levinson-Durbin.

    Computes the sample PACF in two stages:

    1. The biased ACF is estimated in O(N log N) using `scipy.fft.rfft`.
    2. The PACF coefficients (Yule-Walker reflection coefficients) are
       extracted from the ACF in O(p²) time via the Levinson-Durbin
       recursion.

    The implementation uses two rolling 1-D coefficient vectors (O(p) memory)
    instead of the O(p²) matrix allocation used in naive implementations.

    Parameters
    ----------
    x : pandas Series, numpy ndarray
        1-D time series. Must contain at least 2 observations.
    nlags : int, default None
        Number of lags to return. The result always includes lag 0, so the
        output length is `nlags + 1`. Must satisfy `0 < nlags < len(x) // 2`.
        If `None`, defaults to `min(int(10 * log10(n)), n // 2 - 1)`,
        matching the statsmodels convention. In this case, `x` must contain at
        least 4 observations.
    alpha : float, default None
        Significance level for asymptotic confidence intervals under the
        white-noise null hypothesis. If given (e.g. `0.05` for 95%
        intervals), a second array of shape `(nlags + 1, 2)` is returned.
        Under H₀ the standard error is `1 / sqrt(n)` for all lags k ≥ 1.
        Lag 0 always has interval `[1.0, 1.0]`.

    Returns
    -------
    pacf_vals : numpy ndarray, shape (nlags + 1,)
        Sample partial autocorrelations for lags 0, 1, ..., nlags.
        `pacf_vals[0]` is always 1.0.
    confint : numpy ndarray, shape (nlags + 1, 2)
        Only returned when `alpha` is not `None`. Each row contains the lower
        and upper bounds of the `(1 - alpha)` confidence interval for the
        corresponding lag.

    Notes
    -----
    The biased ACF estimator (denominator `n`) is used internally because it
    guarantees a positive semi-definite Toeplitz matrix, which is a
    requirement for Levinson-Durbin to be numerically stable. This differs
    from `statsmodels.tsa.stattools.pacf(method='yw')`, which uses the
    unbiased estimator (denominator `n - k`), leading to small numerical
    differences (~2e-02 for typical series).

    The confidence intervals are asymptotic and assume white noise. They are
    appropriate for testing whether individual PACF values are significantly
    different from zero, not for joint testing.

    If `x` contains leading or trailing non-finite values (NaN, ±inf), they
    are silently removed before any computation. If interleaved non-finite
    values remain after stripping, a ``MissingValuesWarning`` is issued and
    the function falls back to pairwise deletion: for each lag *k* only pairs
    *(x[t-k], x[t])* where both values are finite are used. This preserves
    true temporal distances but requires O(N·p) time instead of O(N log N)
    because FFT cannot be applied to irregular observations. Lags with fewer
    than 2 valid pairs are set to NaN.

    References
    ----------
    .. [1] Brockwell, P.J. and Davis, R.A. (1991). *Time Series: Theory and
       Methods*, 2nd ed. Springer.
    .. [2] Levinson, N. (1947). The Wiener RMS error criterion in filter
       design and prediction. *Journal of Mathematics and Physics*, 25, 261-278.

    Examples
    --------
    ```python
    import numpy as np
    from skforecast.stats import pacf

    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)

    # PACF values for lags 0-10
    pacf_vals = pacf(x, nlags=10)

    # PACF with 95% white-noise confidence intervals
    pacf_vals, confint = pacf(x, nlags=10, alpha=0.05)
    ```

    """

    if isinstance(x, pd.Series):
        x = x.to_numpy(dtype=float)
    else:
        x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError(f"`x` must be 1-D, got shape {x.shape}.")
    n = len(x)
    if n < 2:
        raise ValueError(
            f"`x` must have at least 2 observations, got {n}."
        )

    # Strip leading/trailing non-finite values; detect interleaved NaNs.
    valid_idx = np.where(np.isfinite(x))[0]
    if len(valid_idx) == 0:
        raise ValueError("`x` has no finite values.")
    x = x[valid_idx[0] : valid_idx[-1] + 1]
    n = len(x)
    n_valid = len(valid_idx)
    if n_valid < 2:
        raise ValueError(
            f"`x` must have at least 2 finite observations, got {n_valid}."
        )
    has_interleaved_nan = n_valid < n
    if has_interleaved_nan:
        warnings.warn(
            "Interleaved NaN/inf detected. Falling back to pairwise deletion "
            "(slower). Lags with fewer than 2 valid pairs will be NaN.",
            MissingValuesWarning,
            stacklevel=2,
        )

    n_eff = n_valid if has_interleaved_nan else n
    if nlags is None:
        if n_eff < 4:
            raise ValueError(
                f"`x` must have at least 4 observations when `nlags` is None, "
                f"got {n_eff}."
            )
        nlags = min(int(10 * math.log10(n_eff)), n_eff // 2 - 1)
    if not isinstance(nlags, (int, np.integer)) or nlags < 1:
        raise ValueError(f"`nlags` must be a positive integer, got {nlags!r}.")
    if nlags >= n_eff // 2:
        raise ValueError(
            f"`nlags` ({nlags}) must be less than len(x) // 2 ({n_eff // 2}). "
            "Levinson-Durbin is unreliable when the AR order approaches "
            "half the sample size."
        )

    if alpha is not None and not (0.0 < alpha < 1.0):
        raise ValueError(f"`alpha` must be in (0, 1), got {alpha!r}.")

    if has_interleaved_nan:
        acf_vals = _pairwise_acf(x, n, nlags)
    else:
        acf_vals = _fft_acf(x - x.mean(), n, nlags)

    pacf_vals = np.zeros(nlags + 1)
    pacf_vals[0] = 1.0
    phi = np.zeros(nlags + 1)       # current  AR(k) coefficients (1-indexed)
    phi_prev = np.zeros(nlags + 1)  # previous AR(k-1) coefficients
    phi_prev[1] = acf_vals[1]
    pacf_vals[1] = acf_vals[1]
    den_tol = np.sqrt(np.finfo(float).eps)
    for k in range(2, nlags + 1):
        num = acf_vals[k] - phi_prev[1:k] @ acf_vals[k - 1:0:-1]
        den = 1.0 - phi_prev[1:k] @ acf_vals[1:k]
        kk = num / den if abs(den) > den_tol else 0.0
        phi[1:k] = phi_prev[1:k] - kk * phi_prev[k - 1:0:-1]
        phi[k] = kk
        pacf_vals[k] = kk
        phi, phi_prev = phi_prev, phi  # swap without copy

    if alpha is None:
        return pacf_vals

    # Asymptotic white-noise confidence intervals: ±z_{α/2} / sqrt(n_eff)
    z = norm.ppf(1.0 - alpha / 2.0)
    se = z / np.sqrt(n_eff)
    confint = np.column_stack((pacf_vals - se, pacf_vals + se))
    confint[0] = pacf_vals[0]  # lag 0: degenerate, no uncertainty
    return pacf_vals, confint


def calculate_lag_autocorrelation(
    data: pd.Series | pd.DataFrame,
    n_lags: int = 50,
    last_n_samples: int | None = None,
    sort_by: str = "partial_autocorrelation_abs",
) -> pd.DataFrame:
    """
    Calculate autocorrelation and partial autocorrelation for a time series.

    Parameters
    ----------
    data : pandas Series, pandas DataFrame
        Time series to calculate autocorrelation. If a DataFrame is provided,
        it must have exactly one column. Leading and trailing non-finite values
        (NaN, ±inf) are silently removed. If interleaved non-finite values
        remain after stripping, a ``MissingValuesWarning`` is issued and
        pairwise deletion is used (see Notes in `acf` and `pacf`).
    n_lags : int, default 50
        Number of lags to calculate autocorrelation.
    last_n_samples : int, default None
        Number of most recent samples to use. If `None`, use the entire series.
        Note that partial correlations can only be computed for lags up to
        50% of the sample size. For example, if the series has 10 samples,
        `n_lags` must be less than or equal to 4. This parameter is useful
        to speed up calculations when the series is very long.
    sort_by : str, default 'partial_autocorrelation_abs'
        Sort results by 'lag', 'partial_autocorrelation_abs',
        'partial_autocorrelation', 'autocorrelation_abs' or 'autocorrelation'.

    Returns
    -------
    results : pandas DataFrame
        Autocorrelation and partial autocorrelation values for lags 1 to
        `n_lags`. Columns: 'lag', 'partial_autocorrelation_abs',
        'partial_autocorrelation', 'autocorrelation_abs', 'autocorrelation'.

    Examples
    --------
    ```python
    import pandas as pd
    from skforecast.stats import calculate_lag_autocorrelation

    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    calculate_lag_autocorrelation(data=data, n_lags=4)

    #    lag  partial_autocorrelation_abs  partial_autocorrelation  autocorrelation_abs  autocorrelation
    # 0    1                     0.700000                 0.700000             0.700000         0.700000
    # 1    3                     0.154907                -0.154907             0.148485         0.148485
    # 2    4                     0.154749                -0.154749             0.078788        -0.078788
    # 3    2                     0.152704                -0.152704             0.412121         0.412121
    ```

    """

    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"`data` must be a pandas Series or a DataFrame with a single column. "
            f"Got {type(data)}."
        )
    if isinstance(data, pd.DataFrame) and data.shape[1] != 1:
        raise ValueError(
            f"If `data` is a DataFrame, it must have exactly one column. "
            f"Got {data.shape[1]} columns."
        )
    if not isinstance(n_lags, int) or n_lags <= 0:
        raise ValueError(f"`n_lags` must be a positive integer. Got {n_lags}.")

    if last_n_samples is not None:
        if not isinstance(last_n_samples, int) or last_n_samples <= 0:
            raise ValueError(
                f"`last_n_samples` must be a positive integer. Got {last_n_samples}."
            )
        data = data.iloc[-last_n_samples:]

    if sort_by not in [
        "lag",
        "partial_autocorrelation_abs",
        "partial_autocorrelation",
        "autocorrelation_abs",
        "autocorrelation",
    ]:
        raise ValueError(
            "`sort_by` must be 'lag', 'partial_autocorrelation_abs', "
            "'partial_autocorrelation', 'autocorrelation_abs' or 'autocorrelation'."
        )

    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]

    # Determine the effective n for the n_lags constraint, accounting for
    # leading/trailing NaNs (stripped) and interleaved NaNs (pairwise n_valid).
    x_arr = data.to_numpy(dtype=float)
    valid_idx = np.where(np.isfinite(x_arr))[0]
    if len(valid_idx) == 0:
        n_for_check = 0
    else:
        stripped_n = int(valid_idx[-1]) - int(valid_idx[0]) + 1
        n_valid = len(valid_idx)
        n_for_check = n_valid if n_valid < stripped_n else stripped_n

    if n_lags >= n_for_check // 2:
        raise ValueError(
            f"`n_lags` ({n_lags}) must be less than len(data) // 2 ({n_for_check // 2}). "
            "Partial autocorrelation cannot be computed for more than half the "
            "sample size."
        )

    pacf_values = pacf(data, nlags=n_lags)
    acf_values = acf(data, nlags=n_lags)

    results = pd.DataFrame(
        {
            "lag": range(n_lags + 1),
            "partial_autocorrelation_abs": np.abs(pacf_values),
            "partial_autocorrelation": pacf_values,
            "autocorrelation_abs": np.abs(acf_values),
            "autocorrelation": acf_values,
        }
    ).iloc[1:]

    if sort_by == "lag":
        results = results.sort_values(by=sort_by, ascending=True).reset_index(drop=True)
    else:
        results = results.sort_values(by=sort_by, ascending=False).reset_index(drop=True)

    return results
