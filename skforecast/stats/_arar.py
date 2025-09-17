################################################################################
#                                 ARAR                                         #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from scipy.stats import norm
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

def arar(y_in, max_ar_depth: int = 26, max_lag: int = 40, safe: bool = True):
    """
    Fit ARAR to a 1D series.

    Returns
    -------
    (Y, best_phi, best_lag, sigma2, psi, sbar)
      Y         : original series (np.ndarray, float)
      best_phi  : shape (4,) array for lags (1, i, j, k)
      best_lag  : tuple (1, i, j, k)
      sigma2    : innovation variance (float, >= 1e-12)
      psi       : memory-shortening filter (np.ndarray)
      sbar      : mean of shortened series (float)
    """
    y_in = np.asarray(y_in, dtype=float)
    Y = y_in.copy()

    def mean_fallback(y):
        mu = float(np.nanmean(y))
        var = float(np.nanvar(y, ddof=1)) if y.size > 1 else 0.0
        return (y.copy(),
                np.zeros(4, dtype=float),
                (1, 1, 1, 1),
                max(var, 1e-12),
                np.array([1.0], dtype=float),
                mu)

    # quick guards
    if y_in.size < 5 or max_ar_depth < 4 or max_lag < max_ar_depth:
        return mean_fallback(y_in) if safe else (_ for _ in ()).throw(
            ValueError("Too short series or incompatible max_ar_depth/max_lag")
        )

    try:
        # --- memory shortening (≤ 3 rounds) ---
        y = y_in.copy()
        psi = np.array([1.0], dtype=float)

        for _ in range(3):
            n = y.size
            taus = np.arange(1, min(15, n - 1) + 1, dtype=int)
            if taus.size == 0:
                break

            # one-step regressions for taus
            best_idx = None
            best_err = np.inf
            best_phi1 = 0.0
            for idx, t in enumerate(taus):
                den = float(np.dot(y[:-t], y[:-t])) + np.finfo(float).eps
                phi1 = float(np.dot(y[t:], y[:-t]) / den)
                den_e = float(np.dot(y[t:], y[t:])) + np.finfo(float).eps
                err = float(np.sum((y[t:] - phi1 * y[:-t]) ** 2) / den_e)
                if err < best_err:
                    best_err, best_idx, best_phi1 = err, idx, phi1

            tau = int(taus[best_idx])

            if best_err <= 8.0 / n or (best_phi1 >= 0.93 and tau > 2):
                # first-order reduction
                y = y[tau:] - best_phi1 * y[:-tau]
                psi = np.concatenate([psi, np.zeros(tau)]) - best_phi1 * np.concatenate([np.zeros(tau), psi])
            elif best_phi1 >= 0.93:
                # second-order reduction
                if n < 3:
                    break
                A = np.zeros((2, 2), dtype=float)
                A[0, 0] = float(np.dot(y[1:n - 1], y[1:n - 1]))
                A[0, 1] = A[1, 0] = float(np.dot(y[0:n - 2], y[1:n - 1]))
                A[1, 1] = float(np.dot(y[0:n - 2], y[0:n - 2]))
                b = np.array([float(np.dot(y[2:n], y[1:n - 1])),
                              float(np.dot(y[2:n], y[0:n - 2]))], dtype=float)
                try:
                    phi2 = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    break
                y = y[2:n] - phi2[0] * y[1:n - 1] - phi2[1] * y[0:n - 2]
                psi = (np.concatenate([psi, [0.0, 0.0]])
                       - phi2[0] * np.concatenate([[0.0], psi, [0.0]])
                       - phi2[1] * np.concatenate([[0.0, 0.0], psi]))
            else:
                break

        # shortened series stats
        sbar = float(np.mean(y))
        X = y - sbar
        n = X.size

        # autocovariance (biased) up to max_lag_eff
        max_lag_eff = min(max_lag, max(1, n - 1))
        gamma = np.empty(max_lag_eff + 1, dtype=float)
        xbar = float(np.mean(X))
        for j in range(max_lag_eff + 1):
            gamma[j] = float(np.sum((X[:n - j] - xbar) * (X[j:] - xbar)) / n)

        # select AR(4) with lags L = [1, i, j, k]
        best_sigma2 = np.inf
        best_lag = (1, 0, 0, 0)
        best_phi = np.zeros(4, dtype=float)

        # helper to build A,b using zero-based gamma: A_rs = gamma[|Lr-Ls|], b_r = gamma[Lr]
        def build_system(i, j, k):
            L = [1, i, j, k]
            if max(L) > max_lag_eff:
                return None, None
            A = np.empty((4, 4), dtype=float)
            for r in range(4):
                for s in range(4):
                    d = abs(L[r] - L[s])
                    if d > max_lag_eff:
                        return None, None
                    A[r, s] = gamma[d]
            b = np.array([gamma[Lr] for Lr in L], dtype=float)
            return A, b

        for i in range(2, max_ar_depth - 1):
            for j in range(i + 1, max_ar_depth):
                for k in range(j + 1, max_ar_depth + 1):
                    A, b = build_system(i, j, k)
                    if A is None:
                        continue
                    try:
                        phi = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        continue
                    sigma2 = float(gamma[0] - float(np.dot(phi, b)))
                    if sigma2 < best_sigma2:
                        best_sigma2 = sigma2
                        best_phi = phi.copy()
                        best_lag = (1, i, j, k)

        if not np.isfinite(best_sigma2) or best_sigma2 <= 0.0 or np.allclose(best_phi, 0.0):
            return mean_fallback(Y) if safe else (_ for _ in ()).throw(RuntimeError("AR selection failed"))

        return (Y,
                best_phi.astype(float, copy=False),
                best_lag,
                max(best_sigma2, 1e-12),
                psi.astype(float, copy=False),
                sbar)

    except Exception:
        return mean_fallback(y_in) if safe else (_ for _ in ()).throw

def forecast(model_tuple, h: int, level=(80, 95)):
    """
    Forecast h steps ahead from a tuple returned by `arar`.

    Parameters
    ----------
    model_tuple : (Y, best_phi, best_lag, sigma2, psi, sbar)
    h           : horizon (>0)
    level       : iterable of confidence levels in percent, e.g. (80,95)

    Returns
    -------
    dict with keys: mean, upper, lower, level
      - mean  : (h,) forecasts
      - upper : (h, len(level)) upper bounds
      - lower : (h, len(level)) lower bounds
    """
    if h <= 0:
        raise ValueError("h must be positive")

    Y, best_phi, best_lag, sigma2, psi, sbar = model_tuple
    Y = np.asarray(Y, dtype=float)
    best_phi = np.asarray(best_phi, dtype=float)
    psi = np.asarray(psi, dtype=float)
    sbar = float(sbar)
    sigma2 = float(sigma2)

    n = Y.size
    _, i, j, k = best_lag

    # build xi (combined filter impulse response)
    def z(m):  # zeros helper
        return np.zeros(max(0, m), dtype=float)

    xi = np.concatenate([psi, z(k)])
    xi -= best_phi[0] * np.concatenate([[0.0], psi, z(k - 1)])
    xi -= best_phi[1] * np.concatenate([z(i), psi, z(k - i)])
    xi -= best_phi[2] * np.concatenate([z(j), psi, z(k - j)])
    xi -= best_phi[3] * np.concatenate([z(k), psi])

    # iterative forecasts
    y_ext = np.concatenate([Y, np.zeros(h, dtype=float)])
    kk = xi.size
    c = (1.0 - float(np.sum(best_phi))) * sbar
    for t in range(1, h + 1):
        L = min(kk - 1, n + t - 1)
        y_ext[n + t - 1] = (-np.dot(xi[1:L + 1], y_ext[n + t - 1 - np.arange(1, L + 1)])
                            + c) if L > 0 else c
    mean_fc = y_ext[n:n + h].copy()

    # extend xi for variance recursion
    if h > kk:
        xi = np.concatenate([xi, np.zeros(h - kk, dtype=float)])

    # tau recursion for forecast error
    tau = np.zeros(h, dtype=float)
    tau[0] = 1.0
    for t in range(1, h):
        J = min(t, xi.size - 1)
        tau[t] = -np.dot(tau[:J], xi[1:J + 1][::-1]) if J > 0 else 0.0

    se = np.sqrt(sigma2 * np.array([np.sum(tau[:t + 1] ** 2) for t in range(h)], dtype=float))

    zq = norm.ppf(0.5 + np.asarray(level) / 200.0)
    upper = np.column_stack([mean_fc + q * se for q in zq])
    lower = np.column_stack([mean_fc - q * se for q in zq])

    return {"mean": mean_fc, "upper": upper, "lower": lower, "level": list(level)}

def fitted_arar(model_tuple):
    """
    Compute in-sample fitted values from ARAR model tuple.

    Returns dict with key 'fitted' (np.ndarray, same length as Y),
    with NaN for first k-1 entries (where k = len(xi)).
    """
    Y, best_phi, best_lag, sigma2, psi, sbar = model_tuple
    Y = np.asarray(Y, dtype=float)
    best_phi = np.asarray(best_phi, dtype=float)
    psi = np.asarray(psi, dtype=float)

    _, i, j, k = best_lag

    def z(m): return np.zeros(max(0, m), dtype=float)

    xi = np.concatenate([psi, z(k)])
    xi -= best_phi[0] * np.concatenate([[0.0], psi, z(k - 1)])
    xi -= best_phi[1] * np.concatenate([z(i), psi, z(k - i)])
    xi -= best_phi[2] * np.concatenate([z(j), psi, z(k - j)])
    xi -= best_phi[3] * np.concatenate([z(k), psi])

    kk = xi.size
    c = (1.0 - float(np.sum(best_phi))) * sbar

    fitted = np.full(Y.size, np.nan, dtype=float)
    for t in range(kk - 1, Y.size):
        fitted[t] = -np.dot(xi[1:kk], Y[t - np.arange(1, kk)]) + c
    return {"fitted": fitted}


def residuals_arar(model_tuple):
    """
    Compute residuals (observed - fitted) from ARAR model tuple.
    """
    Y = np.asarray(model_tuple[0], dtype=float)
    fits = fitted_arar(model_tuple)["fitted"]
    return Y - fits

def summary_arar(model_tuple):
    """
    Print a simple summary of ARAR model tuple.
    """
    Y, best_phi, best_lag, sigma2, psi, sbar = model_tuple
    Y = np.asarray(Y, dtype=float)

    print("ARAR Model Summary")
    print("------------------")
    print(f"Number of observations: {len(Y)}")
    print(f"Selected AR lags: {best_lag}")
    print(f"AR coefficients (phi): {np.round(best_phi, 4)}")
    print(f"Residual variance (sigma^2): {sigma2:.4f}")
    print(f"Mean of shortened series (sbar): {sbar:.4f}")
    print(f"Length of memory-shortening filter (psi): {len(psi)}")

    print("\nTime Series Summary Statistics")
    print(f"Mean: {np.mean(Y):.4f}")
    print(f"Std Dev: {np.std(Y, ddof=1):.4f}")
    print(f"Min: {np.min(Y):.4f}")
    print(f"25%: {np.percentile(Y,25):.4f}")
    print(f"Median: {np.median(Y):.4f}")
    print(f"75%: {np.percentile(Y,75):.4f}")
    print(f"Max: {np.max(Y):.4f}")


class Arar(BaseEstimator, RegressorMixin):
    """
    Scikit-learn style wrapper for the ARAR time-series model.

    This estimator treats a univariate sequence as "the feature".
    Call `fit(y)` with a 1D array-like of observations in time order, then
    produce out-of-sample forecasts via `predict(steps)` and prediction intervals
    via `predict_interval(steps, level=...)`. In-sample diagnostics are available
    through `fitted_`, `residuals_()` and `summary()`.

    Parameters
    ----------
    max_ar_depth : int, default=26
        Maximum AR depth considered for the (1, i, j, k) AR selection stage.
    max_lag : int, default=40
        Maximum lag used when estimating autocovariances.
    safe : bool, default=True
        If True, falls back to a mean-only model on numerical issues or very
        short series; otherwise errors are raised.

    Attributes
    ----------
    max_ar_depth : int
        Maximum AR depth considered for the (1, i, j, k) AR selection stage.
    max_lag : int
        Maximum lag used when estimating autocovariances.
    safe : bool
        If True, falls back to a mean-only model on numerical issues or very
        short series; otherwise errors are raised.
    model_ : tuple
        Raw tuple returned by `arar(...)`: (Y, best_phi, best_lag, sigma2, psi, sbar).
    y_ : ndarray of shape (n_samples,)
        Original training series (float).
    coef_ : ndarray of shape (4,)
        Selected AR coefficients for lags (1, i, j, k).
    lags_ : tuple
        Selected lags (1, i, j, k).
    sigma2_ : float
        Innovation variance.
    psi_ : ndarray
        Memory-shortening filter.
    sbar_ : float
        Mean of shortened series.
    n_features_in_ : int
        For sklearn compatibility (always 1).
    fitted_values_ : ndarray of shape (n_samples,)
        In-sample fitted values (NaN for first k-1 terms).
    residuals_in_ : ndarray of shape (n_samples,)
        In-sample residuals (observed - fitted).
    """

    def __init__(self, max_ar_depth: int = 26, max_lag: int = 40, safe: bool = True):
        self.max_ar_depth = max_ar_depth
        self.max_lag = max_lag
        self.safe = safe

    def fit(self, y: pd.Series | np.ndarray, exog: None = NotImplementedError) -> "Arar":
        """
        Fit the ARAR model to a univariate time series.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Time-ordered numeric sequence.
        exog : None
            Exogenous variables. Ignored, present for API compatibility.

        Returns
        -------
        self : Arar
            Fitted estimator.
        """
        y = np.asarray(y, dtype=float).ravel()
        if y.ndim != 1:
            raise ValueError("`y` must be a 1D array-like sequence.")
        if y.size < 2 and not self.safe:
            raise ValueError("Series too short to fit ARAR when safe=False.")

        self.model_ = arar(y, max_ar_depth=self.max_ar_depth, max_lag=self.max_lag, safe=self.safe)

        (Y, best_phi, best_lag, sigma2, psi, sbar) = self.model_

        self.y_ = np.asarray(Y, dtype=float)
        self.coef_ = np.asarray(best_phi, dtype=float)
        self.lags_ = tuple(best_lag)
        self.sigma2_ = float(sigma2)
        self.psi_ = np.asarray(psi, dtype=float)
        self.sbar_ = float(sbar)
        self.n_features_in_ = 1

        self.fitted_values_ = fitted_arar(self.model_)["fitted"]
        self.residuals_in_ = residuals_arar(self.model_)

        return self
    
    def predict(self, steps: int, exog: None = None) -> np.ndarray:
        """
        Generate mean forecasts steps ahead.

        Parameters
        ----------
        steps : int
            Forecast horizon (must be > 0)
        exog : None
            Exogenous variables. Ignored, present for API compatibility.

        Returns
        -------
        mean : ndarray of shape (h,)
            Point forecasts for steps 1..h.
        """
        check_is_fitted(self, "model_")
        if not isinstance(steps, (int, np.integer)) or steps <= 0:
            raise ValueError("`steps` must be a positive integer.")
        return forecast(self.model_, h=steps)["mean"]

    def predict_interval(
        self,
        steps: int = 1,
        level=(80, 95),
        as_frame: bool = True,
        exog: None = None
    ) -> pd.DataFrame | dict:
        """
        Forecast with symmetric normal-theory prediction intervals.

        Parameters
        ----------
        h : int, default=1
            Forecast horizon.
        level : iterable of int, default=(80, 95)
            Confidence levels in percent.
        as_frame : bool, default=True
            If True, return a tidy DataFrame with columns:
            'mean', 'lower_<L>', 'upper_<L>' for each level L.
        exog : None
            Exogenous variables. Ignored, present for API compatibility.

        Returns
        -------
        DataFrame or dict
            If as_frame=True: DataFrame indexed by step (1..h).
            Else: the raw dict from `predict_arar`.
        """
        check_is_fitted(self, "model_")
        out = forecast(self.model_, h=steps, level=level)
        if not as_frame:
            return out

        idx = pd.RangeIndex(1, steps + 1, name="step")
        df = pd.DataFrame({"mean": out["mean"]}, index=idx)
        for i, L in enumerate(out["level"]):
            df[f"lower_{L}"] = out["lower"][:, i]
            df[f"upper_{L}"] = out["upper"][:, i]
        return df

    def residuals_(self) -> np.ndarray:
        """
        In-sample residuals (observed - fitted).

        Returns
        -------
        residuals : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "model_")
        return self.residuals_in_

    def fitted_(self) -> np.ndarray:
        """
        In-sample fitted values.

        Returns
        -------
        fitted : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "model_")
        return self.fitted_values_

    def summary(self) -> None:
        """
        Print a simple textual summary of the fitted ARAR model.
        """
        check_is_fitted(self, "model_")
        return summary_arar(self.model_)

    def score(self, y=None) -> float:
        """
        R^2 using in-sample fitted values (ignores initial NaNs).

        Parameters
        ----------
        y : ignored
            Present for API compatibility.

        Returns
        -------
        score : float
            Coefficient of determination.
        """
        check_is_fitted(self, "model_")
        y = self.y_
        fitted = self.fitted_values_
        mask = ~np.isnan(fitted)
        if mask.sum() < 2:
            return float("nan")
        ss_res = np.sum((y[mask] - fitted[mask]) ** 2)
        ss_tot = np.sum((y[mask] - y[mask].mean()) ** 2) + np.finfo(float).eps
        return 1.0 - ss_res / ss_tot

