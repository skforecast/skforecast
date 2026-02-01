################################################################################
#                               ARIMA base implementation                      #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import norm
import scipy.optimize as opt
from typing import Tuple, Optional, Dict as DictType, Any, Union, List
import warnings


@njit(cache=True)
def state_prediction(a: np.ndarray, p: int, r: int, d: int, rd: int,
                     phi: np.ndarray, delta: np.ndarray) -> np.ndarray:  # pragma: no cover
    """
    Compute the one-step-ahead state prediction for an ARIMA model.

    This is the prediction step of the Kalman filter for the state vector.

    Parameters
    ----------
    a : np.ndarray
        Current state vector of length rd.
    p : int
        Number of AR coefficients.
    r : int
        State dimension for ARMA part (max(p, q+1)).
    d : int
        Number of differencing terms.
    rd : int
        Total state dimension (r + d).
    phi : np.ndarray
        AR coefficients.
    delta : np.ndarray
        Differencing coefficients.

    Returns
    -------
    anew : np.ndarray
        Predicted state vector.
    """
    anew = np.zeros(rd)

    for i in range(r):
        tmp = a[i + 1] if i < r - 1 else 0.0
        if i < p:
            tmp += phi[i] * a[0]
        anew[i] = tmp

    if d > 0:
        for i in range(r + 1, rd):
            anew[i] = a[i - 1]

        tmp = a[0]
        for i in range(d):
            tmp += delta[i] * a[r + i]
        anew[r] = tmp

    return anew


@njit(cache=True)
def predict_covariance_nodiff(P: np.ndarray, r: int, p: int, q: int,
                               phi: np.ndarray, theta: np.ndarray) -> np.ndarray:  # pragma: no cover
    """
    Predict the state covariance matrix when there is no differencing (d=0).

    Parameters
    ----------
    P : np.ndarray
        Current state covariance matrix (r x r).
    r : int
        State dimension.
    p : int
        Number of AR coefficients.
    q : int
        Number of MA coefficients.
    phi : np.ndarray
        AR coefficients.
    theta : np.ndarray
        MA coefficients.

    Returns
    -------
    Pnew : np.ndarray
        Predicted state covariance matrix.
    """
    Pnew = np.zeros((r, r))
    P00 = P[0, 0]

    for i in range(r):
        if i == 0:
            vi = 1.0
        elif i - 1 < q:
            vi = theta[i - 1]
        else:
            vi = 0.0

        for j in range(r):
            if j == 0:
                tmp = vi
            elif j - 1 < q:
                tmp = vi * theta[j - 1]
            else:
                tmp = 0.0

            if i < p and j < p:
                tmp += phi[i] * phi[j] * P00

            if i < r - 1 and j < r - 1:
                tmp += P[i + 1, j + 1]

            if i < p and j < r - 1:
                tmp += phi[i] * P[j + 1, 0]

            if j < p and i < r - 1:
                tmp += phi[j] * P[i + 1, 0]

            Pnew[i, j] = tmp

    return Pnew



@njit(cache=True)
def predict_covariance_with_diff(P: np.ndarray, r: int, d: int, p: int, q: int,
                                  rd: int, phi: np.ndarray, delta: np.ndarray,
                                  theta: np.ndarray) -> np.ndarray:  # pragma: no cover
    """
    Predict the state covariance matrix when there is differencing (d > 0).

    Parameters
    ----------
    P : np.ndarray
        Current state covariance matrix (rd x rd).
    r : int
        State dimension for ARMA part.
    d : int
        Number of differencing terms.
    p : int
        Number of AR coefficients.
    q : int
        Number of MA coefficients.
    rd : int
        Total state dimension (r + d).
    phi : np.ndarray
        AR coefficients.
    delta : np.ndarray
        Differencing coefficients.
    theta : np.ndarray
        MA coefficients.

    Returns
    -------
    Pnew : np.ndarray
        Predicted state covariance matrix.
    """
    mm = np.zeros((rd, rd))
    Pnew = np.zeros((rd, rd))

    for i in range(r):
        for j in range(rd):
            tmp = 0.0
            if i < p:
                tmp += phi[i] * P[0, j]
            if i < r - 1:
                tmp += P[i + 1, j]
            mm[i, j] = tmp

    for j in range(rd):
        tmp = P[0, j]
        for k in range(d):
            tmp += delta[k] * P[r + k, j]
        mm[r, j] = tmp

    for i in range(1, d):
        for j in range(rd):
            mm[r + i, j] = P[r + i - 1, j]

    for i in range(r):
        for j in range(rd):
            tmp = 0.0
            if i < p:
                tmp += phi[i] * mm[j, 0]
            if i < r - 1:
                tmp += mm[j, i + 1]
            Pnew[j, i] = tmp

    for j in range(rd):
        tmp = mm[j, 0]
        for k in range(d):
            tmp += delta[k] * mm[j, r + k]
        Pnew[j, r] = tmp

    for i in range(1, d):
        for j in range(rd):
            Pnew[j, r + i] = mm[j, r + i - 1]

    for i in range(q + 1):
        if i == 0:
            vi = 1.0
        else:
            vi = theta[i - 1]

        for j in range(q + 1):
            if j == 0:
                vj = 1.0
            else:
                vj = theta[j - 1]
            Pnew[i, j] += vi * vj

    return Pnew

@njit(cache=True)
def kalman_update(y_obs: float, anew: np.ndarray, delta: np.ndarray,
                  Pnew: np.ndarray, d: int, r: int, rd: int
                  ) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:  # pragma: no cover
    """
    Perform the Kalman filter update step.

    Parameters
    ----------
    y_obs : float
        Observed value at current time.
    anew : np.ndarray
        Predicted state vector.
    delta : np.ndarray
        Differencing coefficients.
    Pnew : np.ndarray
        Predicted state covariance.
    d : int
        Number of differencing terms.
    r : int
        ARMA state dimension.
    rd : int
        Total state dimension.

    Returns
    -------
    a : np.ndarray
        Updated state vector.
    P : np.ndarray
        Updated state covariance.
    resid : float
        Prediction residual.
    gain : float
        Kalman gain denominator.
    ssq_contrib : float
        Contribution to sum of squares (resid^2 / gain).
    sumlog_contrib : float
        Contribution to log-determinant (log(gain)).
    """
    resid = y_obs - anew[0]
    for i in range(d):
        resid -= delta[i] * anew[r + i]

    M = np.zeros(rd)
    for i in range(rd):
        tmp = Pnew[i, 0]
        for j in range(d):
            tmp += Pnew[i, r + j] * delta[j]
        M[i] = tmp

    gain = M[0]
    for j in range(d):
        gain += delta[j] * M[r + j]

    if gain < 1e4:
        ssq_contrib = resid ** 2 / gain
        sumlog_contrib = np.log(gain)
    else:
        ssq_contrib = 0.0
        sumlog_contrib = 0.0

    std_resid = resid / np.sqrt(gain) if gain > 0 else np.nan

    a = np.zeros(rd)
    for i in range(rd):
        a[i] = anew[i] + M[i] * resid / gain

    P = np.zeros((rd, rd))
    for i in range(rd):
        for j in range(rd):
            P[i, j] = Pnew[i, j] - (M[i] * M[j]) / gain

    return a, P, resid, gain, ssq_contrib, sumlog_contrib


@njit(cache=True)
def compute_arima_likelihood_core(
    y: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    delta: np.ndarray,
    a_init: np.ndarray,
    P_init: np.ndarray,
    Pn_init: np.ndarray,
    update_start: int,
    give_resid: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # pragma: no cover
    """
    Core Kalman filter likelihood computation (Numba-compatible).

    Parameters
    ----------
    y : np.ndarray
        Observed time series.
    phi : np.ndarray
        AR coefficients.
    theta : np.ndarray
        MA coefficients.
    delta : np.ndarray
        Differencing coefficients.
    a_init : np.ndarray
        Initial state vector.
    P_init : np.ndarray
        Initial state covariance.
    Pn_init : np.ndarray
        Prior state covariance.
    update_start : int
        Time index to begin updating likelihood.
    give_resid : bool
        Whether to compute residuals.

    Returns
    -------
    stats : np.ndarray
        Array [ssq, sumlog, nu] - sum of squares, log-determinant sum, count.
    residuals : np.ndarray
        Standardized residuals (if give_resid=True, else empty).
    a_final : np.ndarray
        Final filtered state vector.
    P_final : np.ndarray
        Final filtered state covariance.
    """
    n = len(y)
    rd = len(a_init)
    p = len(phi)
    q = len(theta)
    d = len(delta)
    r = rd - d

    ssq = 0.0
    sumlog = 0.0
    nu = 0

    a = a_init
    P = P_init
    Pnew = Pn_init

    if give_resid:
        rsResid = np.zeros(n)
    else:
        rsResid = np.empty(0)

    for l in range(n):
        anew = state_prediction(a, p, r, d, rd, phi, delta)

        if not np.isnan(y[l]):
            a_upd, P_upd, resid, gain, ssq_c, sumlog_c = kalman_update(
                y[l], anew, delta, Pnew, d, r, rd
            )
            a = a_upd
            P = P_upd

            if gain < 1e4:
                nu += 1
                ssq += ssq_c
                sumlog += sumlog_c

            if give_resid:
                rsResid[l] = resid / np.sqrt(gain) if gain > 0 else np.nan
        else:
            a = anew
            if give_resid:
                rsResid[l] = np.nan

        if l > update_start:
            if d == 0:
                Pnew = predict_covariance_nodiff(P, r, p, q, phi, theta)
            else:
                Pnew = predict_covariance_with_diff(P, r, d, p, q, rd, phi, delta, theta)
            P = Pnew

    stats = np.array([ssq, sumlog, float(nu)])
    return stats, rsResid, a, P


def compute_arima_likelihood(
    y: np.ndarray,
    model: DictType[str, Any],
    update_start: int = 0,
    give_resid: bool = False
) -> DictType[str, Any]:
    """
    Compute the Gaussian log-likelihood for a univariate ARIMA model using the Kalman filter.

    Parameters
    ----------
    y : np.ndarray
        Observed time series (may contain NaN for missing).
    model : dict
        ArimaStateSpace dictionary with keys: phi, theta, Delta, a, P, Pn, etc.
    update_start : int
        Time index at which to begin updating likelihood.
    give_resid : bool
        If True, also return standardized residuals.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'ssq': Sum of squared innovations.
        - 'sumlog': Accumulated log-determinants.
        - 'nu': Number of innovations.
        - 'resid': Standardized residuals (only if give_resid=True).
        - 'a': Final filtered state vector.
        - 'P': Final filtered state covariance.
    """

    # astype creates copies so no further copying is needed inside the Numba function
    phi = model['phi'].astype(np.float64)
    theta = model['theta'].astype(np.float64)
    delta = model['Delta'].astype(np.float64)
    a = model['a'].astype(np.float64)
    P = model['P'].astype(np.float64)
    Pn = model['Pn'].astype(np.float64)

    stats, residuals, a_final, P_final = compute_arima_likelihood_core(
        y.astype(np.float64), phi, theta, delta, a, P, Pn, update_start, give_resid
    )

    result = {
        'ssq': stats[0],
        'sumlog': stats[1],
        'nu': int(stats[2]),
        'a': a_final,
        'P': P_final
    }

    if give_resid:
        result['resid'] = residuals

    return result


@njit(cache=True)
def transform_unconstrained_to_ar_params(p: int, raw: np.ndarray) -> np.ndarray:  # pragma: no cover
    """
    Convert unconstrained parameters to AR coefficients via Durbin-Levinson.

    Maps unconstrained real values to (-1, 1) via tanh (partial autocorrelations),
    then applies Durbin-Levinson recursion to obtain AR coefficients.

    Parameters
    ----------
    p : int
        Number of parameters to transform.
    raw : np.ndarray
        Unconstrained parameters (length >= p).

    Returns
    -------
    new : np.ndarray
        AR coefficients of length p.
    """
    if p > 100:
        raise ValueError("Can only transform up to 100 parameters")

    new = np.tanh(raw[:p])
    work = np.empty(p)

    for j in range(1, p):
        a = new[j]
        for k in range(j):
            work[k] = new[k] - a * new[j - 1 - k]
        new[:j] = work[:j]

    return new


@njit(cache=True)
def compute_arima_transform_gradient(x: np.ndarray, arma: np.ndarray) -> np.ndarray:  # pragma: no cover
    """
    Compute the Jacobian of the ARIMA parameter transformation.

    Numerically approximates the gradient by perturbing each parameter.

    Parameters
    ----------
    x : np.ndarray
        Input parameter vector.
    arma : np.ndarray
        ARIMA order [p, q, P, Q, s, d, D].

    Returns
    -------
    y : np.ndarray
        Jacobian matrix (n x n).
    """
    eps = 1e-3
    mp = int(arma[0])
    mq = int(arma[1])
    msp = int(arma[2])
    n = len(x)

    y = np.eye(n)

    w1 = np.zeros(100)

    if mp > 0:
        for i in range(mp):
            w1[i] = x[i]

        w2 = transform_unconstrained_to_ar_params(mp, w1)

        for i in range(mp):
            w1[i] += eps
            w3 = transform_unconstrained_to_ar_params(mp, w1)
            for j in range(mp):
                y[i, j] = (w3[j] - w2[j]) / eps
            w1[i] -= eps

    if msp > 0:
        v = mp + mq
        for i in range(msp):
            w1[i] = x[v + i]

        w2 = transform_unconstrained_to_ar_params(msp, w1)

        for i in range(msp):
            w1[i] += eps
            w3 = transform_unconstrained_to_ar_params(msp, w1)
            for j in range(msp):
                y[v + i, v + j] = (w3[j] - w2[j]) / eps
            w1[i] -= eps

    return y


@njit(cache=True)
def undo_arima_parameter_transform(x: np.ndarray, arma: np.ndarray) -> np.ndarray:  # pragma: no cover
    """
    Undo the ARIMA parameter transformation (apply transformation to constrained params).

    Parameters
    ----------
    x : np.ndarray
        Transformed parameters.
    arma : np.ndarray
        ARIMA order [p, q, P, Q, s, d, D].

    Returns
    -------
    res : np.ndarray
        Untransformed parameters.
    """
    mp = int(arma[0])
    mq = int(arma[1])
    msp = int(arma[2])

    res = x.copy()

    if mp > 0:
        res[:mp] = transform_unconstrained_to_ar_params(mp, x)

    v = mp + mq
    if msp > 0:
        res[v:v + msp] = transform_unconstrained_to_ar_params(msp, x[v:])

    return res


@njit(cache=True)
def time_series_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # pragma: no cover
    """
    Perform discrete convolution between two sequences.

    Parameters
    ----------
    a : np.ndarray
        First sequence.
    b : np.ndarray
        Second sequence.

    Returns
    -------
    ab : np.ndarray
        Convolution result of length len(a) + len(b) - 1.
    """
    na = len(a)
    nb = len(b)
    nab = na + nb - 1
    ab = np.zeros(nab)

    for i in range(na):
        for j in range(nb):
            ab[i + j] += a[i] * b[j]

    return ab


@njit(cache=True)
def update_least_squares(
    n_parameters: int,
    xnext: np.ndarray,
    ynext: float,
    d: np.ndarray,
    rbar: np.ndarray,
    thetab: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # pragma: no cover
    """
    Update least-squares regression quantities for Q0 computation.

    Parameters
    ----------
    n_parameters : int
        Number of regression parameters.
    xnext : np.ndarray
        New predictor values.
    ynext : float
        New response value.
    d : np.ndarray
        Diagonal of regression matrix.
    rbar : np.ndarray
        Upper triangular portion.
    thetab : np.ndarray
        Regression coefficients.

    Returns
    -------
    d, rbar, thetab : tuple of np.ndarray
        Updated arrays.
    """
    xrow = xnext.copy()

    ithisr = 0
    for i in range(n_parameters):
        if xrow[i] != 0.0:
            xi = xrow[i]
            di = d[i]
            dpi = di + xi * xi
            d[i] = dpi

            if dpi != 0.0:
                cbar = di / dpi
                sbar = xi / dpi
            else:
                cbar = np.inf
                sbar = np.inf

            for k in range(i + 1, n_parameters):
                xk = xrow[k]
                rbthis = rbar[ithisr]
                xrow[k] = xk - xi * rbthis
                rbar[ithisr] = cbar * rbthis + sbar * xk
                ithisr += 1

            xk = ynext
            ynext = xk - xi * thetab[i]
            thetab[i] = cbar * thetab[i] + sbar * xk

            if di == 0.0:
                return d, rbar, thetab
        else:
            ithisr += n_parameters - i - 1

    return d, rbar, thetab


@njit(cache=True)
def inverse_ar_parameter_transform(phi: np.ndarray) -> np.ndarray:  # pragma: no cover
    """
    Compute inverse transformation from AR coefficients to unconstrained parameters.

    Reverses the Durbin-Levinson transformation by running the recursion backwards
    and applying inverse hyperbolic tangent.

    Parameters
    ----------
    phi : np.ndarray
        AR coefficients.

    Returns
    -------
    result : np.ndarray
        Unconstrained parameters.
    """
    p = len(phi)
    new = phi.copy()
    work = np.zeros(p)

    for j in range(p - 1, 0, -1):
        a = new[j]
        denom = 1.0 - a * a
        if denom == 0.0:
            return np.full(p, np.nan)
        for k in range(j):
            work[k] = (new[k] + a * new[j - 1 - k]) / denom
        new[:j] = work[:j].copy()

    result = np.zeros(p)
    for i in range(p):
        if abs(new[i]) <= 1.0:
            result[i] = np.arctanh(new[i])
        else:
            result[i] = np.nan

    return result


@njit(cache=True)
def inverse_arima_parameter_transform(theta: np.ndarray, arma: np.ndarray) -> np.ndarray:  # pragma: no cover
    """
    Apply inverse ARIMA parameter transformation.

    Parameters
    ----------
    theta : np.ndarray
        Transformed parameters.
    arma : np.ndarray
        ARIMA order [p, q, P, ...].

    Returns
    -------
    transformed : np.ndarray
        Unconstrained parameters.
    """
    mp = int(arma[0])
    mq = int(arma[1])
    msp = int(arma[2])
    n = len(theta)
    v = mp + mq

    transformed = theta.copy()

    if mp > 0:
        transformed[:mp] = inverse_ar_parameter_transform(theta[:mp])

    if msp > 0:
        transformed[v:v + msp] = inverse_ar_parameter_transform(theta[v:v + msp])

    return transformed



@njit(cache=True)
def compute_v(phi: np.ndarray, theta: np.ndarray, r: int) -> np.ndarray:  # pragma: no cover
    """
    Compute the V vector for Q0 covariance matrix computation.

    Parameters
    ----------
    phi : np.ndarray
        AR coefficients.
    theta : np.ndarray
        MA coefficients.
    r : int
        State dimension.

    Returns
    -------
    V : np.ndarray
        V vector of length r*(r+1)/2.
    """
    p = len(phi)
    q = len(theta)
    num_params = r * (r + 1) // 2
    V = np.zeros(num_params)

    ind = 0
    for j in range(r):
        if j == 0:
            vj = 1.0
        elif j - 1 < q:
            vj = theta[j - 1]
        else:
            vj = 0.0

        for i in range(j, r):
            if i == 0:
                vi = 1.0
            elif i - 1 < q:
                vi = theta[i - 1]
            else:
                vi = 0.0

            V[ind] = vi * vj
            ind += 1

    return V


@njit(cache=True)
def handle_r_equals_1(p: int, phi: np.ndarray) -> np.ndarray:  # pragma: no cover
    """
    Handle the special case r=1 for Q0 computation.

    Parameters
    ----------
    p : int
        Number of AR coefficients.
    phi : np.ndarray
        AR coefficients.

    Returns
    -------
    res : np.ndarray
        1x1 covariance matrix.
    """
    res = np.zeros((1, 1))
    if p == 0:
        res[0, 0] = 1.0
    else:
        res[0, 0] = 1.0 / (1.0 - phi[0] ** 2)
    return res


@njit(cache=True)
def handle_p_equals_0(V: np.ndarray, r: int) -> np.ndarray:  # pragma: no cover
    """
    Handle the case p=0 (pure MA) for Q0 computation.

    Parameters
    ----------
    V : np.ndarray
        V vector from compute_v.
    r : int
        State dimension.

    Returns
    -------
    res : np.ndarray
        Flat result array of length r*r.
    """
    num_params = r * (r + 1) // 2
    res = np.zeros(r * r)

    ind = num_params - 1
    indn = num_params - 1

    for i in range(r - 1, -1, -1):
        for j in range(i, -1, -1):
            res[ind] = V[ind]

            if j != 0:
                indn -= 1
                res[ind] += res[indn]

            ind -= 1

    return res


@njit(cache=True)
def handle_p_greater_than_0(
    V: np.ndarray,
    phi: np.ndarray,
    p: int,
    r: int,
    num_params: int,
    nrbar: int
) -> np.ndarray:  # pragma: no cover
    """
    Handle the case p>0 (AR present) for Q0 computation.

    Parameters
    ----------
    V : np.ndarray
        V vector.
    phi : np.ndarray
        AR coefficients.
    p : int
        Number of AR coefficients.
    r : int
        State dimension.
    num_params : int
        r*(r+1)/2.
    nrbar : int
        num_params*(num_params-1)/2.

    Returns
    -------
    res : np.ndarray
        Flat result array of length r*r.
    """
    res = np.zeros(r * r)

    rbar = np.zeros(nrbar)
    thetab = np.zeros(num_params)
    xnext = np.zeros(num_params)

    ind = 0
    ind1 = -1
    npr = num_params - r
    npr1 = npr
    indj = npr
    ind2 = npr - 1

    for j in range(r):
        phij = phi[j] if j < p else 0.0

        xnext[indj] = 0.0
        indj += 1

        indi = npr1 + j
        for i in range(j, r):
            ynext = V[ind]
            ind += 1

            phii = phi[i] if i < p else 0.0

            if j != r - 1:
                xnext[indj] = -phii
                if i != r - 1:
                    xnext[indi] -= phij
                    ind1 += 1
                    xnext[ind1] = -1.0

            xnext[npr] = -phii * phij
            ind2 += 1
            if ind2 >= num_params:
                ind2 = 0
            xnext[ind2] += 1.0

            res, rbar, thetab = update_least_squares(
                num_params, xnext, ynext, res, rbar, thetab
            )

            xnext[ind2] = 0.0
            if i != r - 1:
                xnext[indi] = 0.0
                indi += 1
                xnext[ind1] = 0.0

    # Back substitution
    ithisr = nrbar - 1
    im = num_params - 1

    for i in range(num_params):
        bi = thetab[im]
        jm = num_params - 1
        for j in range(i):
            bi -= rbar[ithisr] * res[jm]
            ithisr -= 1
            jm -= 1
        res[im] = bi
        im -= 1

    xcopy = np.zeros(r)
    ind = npr
    for i in range(r):
        xcopy[i] = res[ind]
        ind += 1

    ind = num_params - 1
    ind1 = npr - 1
    for i in range(npr):
        res[ind] = res[ind1]
        ind -= 1
        ind1 -= 1

    for i in range(r):
        res[i] = xcopy[i]

    return res


@njit(cache=True)
def unpack_full_matrix(res_flat: np.ndarray, r: int) -> np.ndarray:  # pragma: no cover
    """
    Unpack flat array into symmetric matrix.

    Parameters
    ----------
    res_flat : np.ndarray
        Flat array from handle_p_equals_0 or handle_p_greater_than_0.
    r : int
        Matrix dimension.

    Returns
    -------
    result : np.ndarray
        Symmetric r x r matrix.
    """
    num_params = r * (r + 1) // 2
    np_idx = num_params

    for i in range(r - 1, 0, -1):
        for j in range(r - 1, i - 1, -1):
            idx = i * r + j
            np_idx -= 1
            res_flat[idx] = res_flat[np_idx]

    for i in range(r):
        for j in range(i + 1, r):
            res_flat[j * r + i] = res_flat[i * r + j]

    return res_flat.reshape((r, r))


@njit(cache=True)
def compute_q0_covariance_matrix(phi: np.ndarray, theta: np.ndarray) -> np.ndarray:  # pragma: no cover
    """
    Compute initial state covariance matrix for ARIMA model (Gardner 1980 method).

    Parameters
    ----------
    phi : np.ndarray
        AR coefficients.
    theta : np.ndarray
        MA coefficients.

    Returns
    -------
    Q0 : np.ndarray
        Symmetric covariance matrix of size r x r.
    """
    p = len(phi)
    q = len(theta)
    r = max(p, q + 1)
    num_params = r * (r + 1) // 2
    nrbar = num_params * (num_params - 1) // 2

    V = compute_v(phi, theta, r)

    if r == 1:
        return handle_r_equals_1(p, phi)

    if p > 0:
        res_flat = handle_p_greater_than_0(V, phi, p, r, num_params, nrbar)
    else:
        res_flat = handle_p_equals_0(V, r)

    return unpack_full_matrix(res_flat, r)


def compute_q0_bis_covariance_matrix(
    phi: np.ndarray,
    theta: np.ndarray,
    tol: float = np.finfo(float).eps
) -> np.ndarray:
    """
    Compute initial covariance matrix using Rossignol (2011) method.

    This is a more numerically stable method that solves Yule-Walker equations.

    Parameters
    ----------
    phi : np.ndarray
        AR coefficients.
    theta : np.ndarray
        MA coefficients.
    tol : float
        Tolerance for numerical stability.

    Returns
    -------
    P : np.ndarray
        Symmetric covariance matrix.
    """
    phi = np.asarray(phi, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)

    p = len(phi)
    q = len(theta)
    r = max(p, q + 1)

    ttheta = np.zeros(r + q)
    ttheta[0] = 1.0
    ttheta[1:q + 1] = theta

    P = np.zeros((r, r))

    if p > 0:
        r2 = max(p + q, p + 1)

        tphi = np.zeros(p + 1)
        tphi[0] = 1.0
        tphi[1:p + 1] = -phi

        # Build Gamma matrix
        Gamma = np.zeros((r2, r2))

        for j0 in range(r2):
            for i0 in range(j0, r2):
                d = i0 - j0
                if d <= p:
                    Gamma[j0, i0] += tphi[d]

        for i0 in range(r2):
            for j0 in range(1, r2):
                s = i0 + j0
                if s <= p:
                    Gamma[j0, i0] += tphi[s]

        # Solve for u
        g = np.zeros(r2)
        g[0] = 1.0

        try:
            kappa = np.linalg.cond(Gamma)
            if np.isfinite(kappa) and kappa < 1.0 / tol:
                u = np.linalg.solve(Gamma, g)
            else:
                u = np.linalg.solve(Gamma + tol * np.eye(r2), g)
        except np.linalg.LinAlgError:
            u = np.linalg.solve(Gamma + tol * np.eye(r2), g)

        # Compute main contribution to P
        for i0 in range(r):
            k_max = p - 1 - i0
            for j0 in range(i0, r):
                m_max = p - 1 - j0
                acc = 0.0

                for k0 in range(max(0, k_max + 1)):
                    phi_ik = phi[i0 + k0]

                    for L0 in range(k0, k0 + q + 1):
                        tLk = ttheta[L0 - k0]
                        phi_ik_tLk = phi_ik * tLk

                        for m0 in range(max(0, m_max + 1)):
                            phi_jm = phi[j0 + m0]
                            phi_product = phi_ik_tLk * phi_jm

                            for n0 in range(m0, m0 + q + 1):
                                tnm = ttheta[n0 - m0]
                                u_idx = abs(L0 - n0)
                                acc += phi_product * tnm * u[u_idx]

                P[i0, j0] += acc

        # Compute rrz for cross-terms
        rrz = np.zeros(q)
        if q > 0:
            for i0 in range(q):
                val = ttheta[i0]
                jstart = max(0, i0 - p)
                for j0 in range(jstart, i0):
                    val -= rrz[j0] * tphi[i0 - j0]
                rrz[i0] = val

        # Add cross-term contributions
        for i0 in range(r):
            k_max_i = p - 1 - i0

            for j0 in range(i0, r):
                k_max_j = p - 1 - j0
                acc = 0.0

                for k0 in range(max(0, k_max_i + 1)):
                    phi_ik = phi[i0 + k0]

                    for L0 in range(k0 + 1, k0 + q + 1):
                        j0_L0 = j0 + L0
                        if j0_L0 < q + 1:
                            acc += phi_ik * ttheta[j0_L0] * rrz[L0 - k0 - 1]

                for k0 in range(max(0, k_max_j + 1)):
                    phi_jk = phi[j0 + k0]

                    for L0 in range(k0 + 1, k0 + q + 1):
                        i0_L0 = i0 + L0
                        if i0_L0 < q + 1:
                            acc += phi_jk * ttheta[i0_L0] * rrz[L0 - k0 - 1]

                P[i0, j0] += acc

    # Add MA contribution
    for i0 in range(r):
        for j0 in range(i0, r):
            k_max = q - j0
            acc = 0.0
            for k0 in range(max(0, k_max + 1)):
                acc += ttheta[i0 + k0] * ttheta[j0 + k0]
            P[i0, j0] += acc

    # Symmetrize
    for i in range(r):
        for j in range(i + 1, r):
            P[j, i] = P[i, j]

    return P


@njit(cache=True)
def transform_arima_parameters(
    params_in: np.ndarray,
    arma: np.ndarray,
    trans: bool
) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    """
    Transform parameter vector to AR and MA coefficient vectors.

    Parameters
    ----------
    params_in : np.ndarray
        Input parameters.
    arma : np.ndarray
        ARIMA order [p, q, P, Q, s, d, D].
    trans : bool
        Whether to apply stability transformation.

    Returns
    -------
    phi : np.ndarray
        AR coefficients (expanded for seasonal).
    theta : np.ndarray
        MA coefficients (expanded for seasonal).
    """
    mp = int(arma[0])
    mq = int(arma[1])
    msp = int(arma[2])
    msq = int(arma[3])
    ns = int(arma[4])

    p = mp + ns * msp
    q = mq + ns * msq

    phi = np.zeros(max(p, 1))
    theta = np.zeros(max(q, 1))
    params = params_in.copy()

    if trans:
        if mp > 0:
            params[:mp] = transform_unconstrained_to_ar_params(mp, params_in)
        v = mp + mq
        if msp > 0:
            params[v:v + msp] = transform_unconstrained_to_ar_params(msp, params_in[v:])

    if ns > 0:
        # Non-seasonal AR and MA
        phi[:mp] = params[:mp]
        theta[:mq] = params[mp:mp + mq]

        # Seasonal AR
        for j in range(msp):
            phi[(j + 1) * ns - 1] += params[mp + mq + j]
            for i in range(mp):
                phi[(j + 1) * ns + i] -= params[i] * params[mp + mq + j]

        # Seasonal MA
        for j in range(msq):
            theta[(j + 1) * ns - 1] += params[mp + mq + msp + j]
            for i in range(mq):
                theta[(j + 1) * ns + i] += params[mp + i] * params[mp + mq + msp + j]
    else:
        phi[:mp] = params[:mp]
        theta[:mq] = params[mp:mp + mq]

    # Trim to actual size
    phi = phi[:p] if p > 0 else np.zeros(0)
    theta = theta[:q] if q > 0 else np.zeros(0)

    return phi, theta

@njit(cache=True)
def compute_css_residuals(
    y: np.ndarray,
    arma: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    ncond: int
) -> Tuple[float, np.ndarray]:  # pragma: no cover
    """
    Compute conditional sum of squares (CSS) and residuals.

    Parameters
    ----------
    y : np.ndarray
        Observed series.
    arma : np.ndarray
        ARIMA order [p, q, P, Q, s, d, D].
    phi : np.ndarray
        AR coefficients.
    theta : np.ndarray
        MA coefficients.
    ncond : int
        Number of conditioning observations.

    Returns
    -------
    sigma2 : float
        Estimated innovation variance.
    resid : np.ndarray
        Residuals.
    """
    n = len(y)
    p = len(phi)
    q = len(theta)

    w = y.copy()

    # Non-seasonal differencing
    d = int(arma[5])
    for _ in range(d):
        for l in range(n - 1, 0, -1):
            w[l] -= w[l - 1]

    # Seasonal differencing
    ns = int(arma[4])
    D = int(arma[6])
    for _ in range(D):
        for l in range(n - 1, ns - 1, -1):
            w[l] -= w[l - ns]

    resid = np.zeros(n)
    ssq = 0.0
    nu = 0

    for l in range(ncond, n):
        tmp = w[l]

        # AR contribution
        for j in range(p):
            if l - j - 1 >= 0:
                tmp -= phi[j] * w[l - j - 1]

        # MA contribution
        jmax = min(l - ncond, q)
        for j in range(jmax):
            if l - j - 1 >= 0:
                tmp -= theta[j] * resid[l - j - 1]

        resid[l] = tmp

        if not np.isnan(tmp):
            nu += 1
            ssq += tmp ** 2

    sigma2 = ssq / nu if nu > 0 else np.nan
    return sigma2, resid

def initialize_arima_state(
    phi: np.ndarray,
    theta: np.ndarray,
    Delta: np.ndarray,
    kappa: float = 1e6,
    SSinit: str = "Gardner1980",
    tol: float = np.finfo(float).eps
) -> DictType[str, Any]:
    """
    Create and initialize the state-space representation of an ARIMA model.

    Parameters
    ----------
    phi : np.ndarray
        AR coefficients.
    theta : np.ndarray
        MA coefficients.
    Delta : np.ndarray
        Differencing polynomial coefficients.
    kappa : float
        Prior variance for diffuse states.
    SSinit : str
        Method for Q0: "Gardner1980" or "Rossignol2011".
    tol : float
        Tolerance for Rossignol method.

    Returns
    -------
    model : dict
        ArimaStateSpace dictionary.
    """
    phi = np.asarray(phi, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    Delta = np.asarray(Delta, dtype=np.float64)

    p = len(phi)
    q = len(theta)
    r = max(p, q + 1)
    d = len(Delta)
    rd = r + d

    # Observation vector Z = [1, 0, ..., 0, Delta]
    Z = np.zeros(rd)
    Z[0] = 1.0
    Z[r:rd] = Delta

    # Transition matrix T
    T = np.zeros((rd, rd))
    if p > 0:
        T[:p, 0] = phi
    if r > 1:
        for i in range(1, r):
            T[i - 1, i] = 1.0
    if d > 0:
        T[r, :] = Z
        if d > 1:
            for i in range(1, d):
                T[r + i, r + i - 1] = 1.0

    # Pad theta if needed
    if q < r - 1:
        theta_padded = np.concatenate([theta, np.zeros(r - 1 - q)])
    else:
        theta_padded = theta

    # R vector and V = R R'
    R = np.concatenate([[1.0], theta_padded, np.zeros(d)])
    V = np.outer(R, R)

    h = 0.0
    a = np.zeros(rd)
    P = np.zeros((rd, rd))
    Pn = np.zeros((rd, rd))

    # Initialize Pn
    if r > 1:
        if SSinit == "Gardner1980":
            Pn[:r, :r] = compute_q0_covariance_matrix(phi, theta)
        elif SSinit == "Rossignol2011":
            Pn[:r, :r] = compute_q0_bis_covariance_matrix(phi, theta, tol)
        else:
            raise ValueError(f"Invalid SSinit: {SSinit}")
    else:
        if p > 0:
            Pn[0, 0] = 1.0 / (1.0 - phi[0] ** 2)
        else:
            Pn[0, 0] = 1.0

    # Diffuse prior for differencing states
    if d > 0:
        for i in range(r, rd):
            Pn[i, i] = kappa

    return {
        'phi': phi,
        'theta': theta,
        'Delta': Delta,
        'Z': Z,
        'a': a,
        'P': P,
        'T': T,
        'V': V,
        'h': h,
        'Pn': Pn
    }


def update_arima(
    mod: DictType[str, Any],
    phi: np.ndarray,
    theta: np.ndarray,
    ss_g: bool = True
) -> DictType[str, Any]:
    """
    Update ARIMA model with new AR and MA coefficients.

    Parameters
    ----------
    mod : dict
        ArimaStateSpace dictionary.
    phi : np.ndarray
        New AR coefficients.
    theta : np.ndarray
        New MA coefficients.
    ss_g : bool
        Use Gardner (True) or Rossignol (False) for Q0.

    Returns
    -------
    mod : dict
        Updated model dictionary.
    """
    phi = np.asarray(phi, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)

    p = len(phi)
    q = len(theta)
    r = max(p, q + 1)

    mod['phi'] = phi
    mod['theta'] = theta

    # Update transition matrix
    if p > 0:
        mod['T'][:p, 0] = phi

    # Update Pn
    if r > 1:
        if ss_g:
            mod['Pn'][:r, :r] = compute_q0_covariance_matrix(phi, theta)
        else:
            mod['Pn'][:r, :r] = compute_q0_bis_covariance_matrix(phi, theta, 0.0)
    else:
        if p > 0:
            mod['Pn'][0, 0] = 1.0 / (1.0 - phi[0] ** 2)
        else:
            mod['Pn'][0, 0] = 1.0

    # Reset state
    mod['a'][:] = 0.0

    return mod


def ar_check(ar: np.ndarray) -> bool:
    """
    Check if AR polynomial is stationary (all roots outside unit circle).

    Parameters
    ----------
    ar : np.ndarray
        AR coefficients.

    Returns
    -------
    bool
        True if stationary.
    """
    if len(ar) == 0:
        return True

    v = np.concatenate([[1.0], -np.asarray(ar)])
    last_nz = np.where(v != 0.0)[0]
    p = last_nz[-1] if len(last_nz) > 0 else 0

    if p == 0:
        return True

    coeffs = np.concatenate([[1.0], -ar[:p]])
    rts = np.roots(coeffs[::-1])

    return np.all(np.abs(rts) > 1.0)


def ma_invert(ma: np.ndarray) -> np.ndarray:
    """
    Invert MA polynomial to ensure invertibility.

    Parameters
    ----------
    ma : np.ndarray
        MA coefficients.

    Returns
    -------
    np.ndarray
        Inverted MA coefficients (roots inside unit circle reflected outside).
    """
    ma = np.asarray(ma)
    q = len(ma)
    if q == 0:
        return ma

    cdesc = np.concatenate([[1.0], ma])
    nz = np.where(cdesc != 0.0)[0]
    q0 = nz[-1] if len(nz) > 0 else 0

    if q0 == 0:
        return ma

    cdesc_q = cdesc[:q0 + 1]
    rts = np.roots(cdesc_q[::-1])

    ind = np.abs(rts) < 1.0
    if not np.any(ind):
        return ma

    if q0 == 1:
        return np.concatenate([[1.0 / ma[0]], np.zeros(q - q0)])

    # Reflect roots inside unit circle
    rts[ind] = 1.0 / rts[ind]

    # Reconstruct polynomial from roots
    x = np.array([1.0], dtype=complex)
    for root in rts:
        x = np.concatenate([x, [0.0 + 0j]]) - np.concatenate([[0.0 + 0j], x]) / root

    result = np.real(x[1:])
    if len(result) < q:
        result = np.concatenate([result, np.zeros(q - len(result))])

    return result[:q]


@njit(cache=True)
def kalman_forecast_core(
    n_ahead: int,
    phi: np.ndarray,
    theta: np.ndarray,
    delta: np.ndarray,
    Z: np.ndarray,
    a: np.ndarray,
    P: np.ndarray,
    Pn: np.ndarray,
    h: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # pragma: no cover
    """
    Core Kalman forecast computation (Numba-compatible).

    Parameters
    ----------
    n_ahead : int
        Number of steps to forecast.
    phi, theta, delta, Z, a, P, Pn, h : arrays
        Model components.

    Returns
    -------
    forecasts : np.ndarray
        Point forecasts.
    variances : np.ndarray
        Forecast variances.
    a_final : np.ndarray
        Final state vector.
    P_final : np.ndarray
        Final state covariance.
    """
    p = len(phi)
    q = len(theta)
    d = len(delta)
    rd = len(a)
    r = rd - d

    forecasts = np.zeros(n_ahead)
    variances = np.zeros(n_ahead)

    a_curr = a
    P_curr = P

    for l in range(n_ahead):
        # State prediction
        anew = state_prediction(a_curr, p, r, d, rd, phi, delta)
        a_curr = anew
        # Forecast
        fc = 0.0
        for i in range(rd):
            fc += Z[i] * a_curr[i]
        forecasts[l] = fc

        # Covariance prediction
        if d == 0:
            Pnew = predict_covariance_nodiff(P_curr, r, p, q, phi, theta)
        else:
            Pnew = predict_covariance_with_diff(P_curr, r, d, p, q, rd, phi, delta, theta)
        P_curr = Pnew

        # Forecast variance: h + Z' P Z
        tmpvar = h
        for i in range(rd):
            for j in range(rd):
                tmpvar += Z[i] * P_curr[i, j] * Z[j]
        variances[l] = tmpvar

    return forecasts, variances, a_curr, P_curr


def kalman_forecast(
    n_ahead: int,
    mod: DictType[str, Any],
    update: bool = False
) -> DictType[str, Any]:
    """
    Forecast n steps ahead from current state.

    Parameters
    ----------
    n_ahead : int
        Number of forecast steps.
    mod : dict
        ArimaStateSpace dictionary.
    update : bool
        If True, also return updated model.

    Returns
    -------
    result : dict
        Dictionary with 'pred', 'var', and optionally 'mod'.
    """
    phi = mod['phi'].astype(np.float64)
    theta = mod['theta'].astype(np.float64)
    delta = mod['Delta'].astype(np.float64)
    Z = mod['Z'].astype(np.float64)
    a = mod['a'].astype(np.float64)
    P = mod['P'].astype(np.float64)
    h = float(mod['h'])

    forecasts, variances, a_final, P_final = kalman_forecast_core(
        n_ahead, phi, theta, delta, Z, a, P, P, h
    )

    result = {'pred': forecasts, 'var': variances}

    if update:
        updated_mod = mod.copy()
        updated_mod['a'] = a_final
        updated_mod['P'] = P_final
        result['mod'] = updated_mod

    return result


def make_pdq(p: int, d: int, q: int) -> Tuple[int, int, int]:
    """
    Create PDQ tuple with validation.

    Parameters
    ----------
    p : int
        AR order.
    d : int
        Differencing order.
    q : int
        MA order.

    Returns
    -------
    tuple
        (p, d, q) tuple.

    Raises
    ------
    ValueError
        If any parameter is negative.
    """
    if p < 0 or d < 0 or q < 0:
        raise ValueError(f"All PDQ parameters must be non-negative. Got: p={p}, d={d}, q={q}")
    return (p, d, q)



def na_omit(x: np.ndarray) -> np.ndarray:
    """
    Remove NaN values from array.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Array with NaN values removed.
    """
    return x[~np.isnan(x)]


def na_omit_pair(x: np.ndarray, xreg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle missing values by converting to NaN.

    Parameters
    ----------
    x : np.ndarray
        Target variable.
    xreg : np.ndarray
        Regressor matrix.

    Returns
    -------
    x, xreg : tuple
        Arrays with missing converted to NaN.
    """
    x = np.asarray(x, dtype=np.float64)
    xreg = np.asarray(xreg, dtype=np.float64)
    return x, xreg


def diff(x: np.ndarray, lag: int = 1, differences: int = 1) -> np.ndarray:
    """
    Compute lagged differences of a time series.

    Parameters
    ----------
    x : np.ndarray
        Input array (1D or 2D).
    lag : int
        Lag for differencing.
    differences : int
        Order of differencing.

    Returns
    -------
    np.ndarray
        Differenced array.
    """
    result = x
    for _ in range(differences):
        if result.ndim == 1:
            result = result[lag:] - result[:-lag]
        else:
            result = result[lag:, :] - result[:-lag, :]
    return result


def match_arg(arg: str, choices: List[str]) -> str:
    """
    Match argument to valid choices (like R's match.arg).

    Parameters
    ----------
    arg : str
        Argument to match.
    choices : list
        Valid choices.

    Returns
    -------
    str
        Matched choice.

    Raises
    ------
    ValueError
        If no match found.
    """
    # First try exact match
    if arg in choices:
        return arg
    # Then try prefix match
    for choice in choices:
        if choice.startswith(arg):
            return arg
    raise ValueError(f"'{arg}' should be one of {choices}")


def process_xreg(
    xreg: Union[pd.DataFrame, np.ndarray, None],
    n: int
) -> Tuple[np.ndarray, int, List[str]]:
    """
    Process exogenous regressors.

    Parameters
    ----------
    xreg : DataFrame, ndarray, or None
        Exogenous regressors.
    n : int
        Expected number of rows.

    Returns
    -------
    xreg_mat : np.ndarray
        Regressor matrix (n x ncxreg).
    ncxreg : int
        Number of regressors.
    nmxreg : list
        Regressor names.

    Raises
    ------
    ValueError
        If row count doesn't match n.
    """
    if xreg is None:
        xreg_mat = np.zeros((n, 0))
        ncxreg = 0
        nmxreg = []
    elif isinstance(xreg, pd.DataFrame):
        if len(xreg) != n:
            raise ValueError("Lengths of x and xreg do not match!")
        xreg_mat = xreg.values.astype(np.float64)
        ncxreg = xreg_mat.shape[1]
        nmxreg = list(xreg.columns)
    else:
        xreg = np.asarray(xreg, dtype=np.float64)
        if xreg.ndim == 1:
            xreg = xreg.reshape(-1, 1)
        if xreg.shape[0] != n:
            raise ValueError("Lengths of x and xreg do not match!")
        xreg_mat = xreg
        ncxreg = xreg_mat.shape[1]
        nmxreg = [f"xreg{i+1}" for i in range(ncxreg)]

    return xreg_mat, ncxreg, nmxreg


def add_drift_term(
    xreg: Union[pd.DataFrame, np.ndarray, None],
    drift: np.ndarray,
    name: str = "intercept"
) -> pd.DataFrame:
    """
    Add a drift/intercept term to xreg.

    Parameters
    ----------
    xreg : DataFrame, ndarray, or None
        Existing regressors.
    drift : np.ndarray
        Drift term to add.
    name : str
        Name for the drift column.

    Returns
    -------
    pd.DataFrame
        DataFrame with drift term added.
    """
    if xreg is None or (isinstance(xreg, np.ndarray) and xreg.size == 0):
        return pd.DataFrame({name: drift})

    if isinstance(xreg, pd.DataFrame):
        df = xreg.copy()
        df.insert(0, name, drift)
        return df
    else:
        xreg = np.asarray(xreg)
        if xreg.ndim == 1:
            xreg = xreg.reshape(-1, 1)
        combined = np.column_stack([drift, xreg])
        cols = [name] + [f"xreg{i+1}" for i in range(xreg.shape[1])]
        return pd.DataFrame(combined, columns=cols)


def regress_and_update(
    x: np.ndarray,
    xreg: np.ndarray,
    mask: np.ndarray,
    narma: int,
    ncxreg: int,
    order_d: int,
    seasonal_d: int,
    m: int,
    Delta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int, bool, Optional[Any]]:
    """
    Regression block for exogenous regressors with coefficient scaling.

    Parameters
    ----------
    x : np.ndarray
        Target variable (can contain NaN).
    xreg : np.ndarray
        Regressor matrix.
    mask : np.ndarray
        Boolean mask for free parameters.
    narma : int
        Number of ARMA parameters.
    ncxreg : int
        Number of exogenous regressors.
    order_d : int
        Non-seasonal differencing order.
    seasonal_d : int
        Seasonal differencing order.
    m : int
        Seasonal period.
    Delta : np.ndarray
        Differencing polynomial.

    Returns
    -------
    init0 : np.ndarray
        Initial parameter values.
    parscale : np.ndarray
        Parameter scaling.
    n_used : int
        Number of observations used.
    orig_xreg : bool
        Whether original xreg was used.
    S : SVD or None
        SVD decomposition if transformed.
    """
    init0 = np.zeros(narma)
    parscale = np.ones(narma)

    x, xreg = na_omit_pair(x, xreg)

    orig_xreg = (ncxreg == 1) or np.any(~mask[narma:narma + ncxreg])

    if not orig_xreg:
        # Use SVD to orthogonalize regressors
        rows_good = np.array([np.all(np.isfinite(row)) for row in xreg])
        if np.sum(rows_good) > 0:
            U, s, Vt = np.linalg.svd(xreg[rows_good, :], full_matrices=False)
            S = {'V': Vt.T, 's': s}
            xreg = xreg @ S['V']
        else:
            S = None
    else:
        S = None

    # Difference series and regressors
    dx = x
    dxreg = xreg

    if order_d > 0:
        dx = diff(dx, lag=1, differences=order_d)
        dxreg = diff(dxreg, lag=1, differences=order_d)
        dx, dxreg = na_omit_pair(dx, dxreg)

    if m > 1 and seasonal_d > 0:
        dx = diff(dx, lag=m, differences=seasonal_d)
        dxreg = diff(dxreg, lag=m, differences=seasonal_d)
        dx, dxreg = na_omit_pair(dx, dxreg)

    # OLS regression
    fit = None
    fit_rank = 0

    if len(dx) > dxreg.shape[1] and dxreg.shape[1] > 0:
        try:
            # Simple OLS: beta = (X'X)^-1 X'y
            valid = ~np.isnan(dx) & np.all(np.isfinite(dxreg), axis=1)
            if np.sum(valid) > dxreg.shape[1]:
                X = dxreg[valid]
                y = dx[valid]
                beta, residuals, rank, sing = np.linalg.lstsq(X, y, rcond=None)
                fit = {'coef': beta, 'rank': rank}
                fit_rank = rank if rank is not None else len(beta)
        except Exception as e:
            warnings.warn(f"Fitting OLS to difference data failed: {e}")
            fit = None
            fit_rank = 0

    if fit_rank == 0 and ncxreg > 0:
        # Fall back to regression on original data
        x, xreg = na_omit_pair(x, xreg)
        valid = ~np.isnan(x) & np.all(np.isfinite(xreg), axis=1)
        if np.sum(valid) > xreg.shape[1]:
            X = xreg[valid]
            y = x[valid]
            beta, residuals, rank, sing = np.linalg.lstsq(X, y, rcond=None)
            fit = {'coef': beta, 'rank': rank}

    # Compute n_used
    isna = np.isnan(x) | np.array([np.any(np.isnan(row)) for row in xreg])
    n_used = int(np.sum(~isna)) - len(Delta)

    if fit is not None:
        model_coefs = fit['coef']
        init0 = np.concatenate([init0, model_coefs])

        # Estimate standard errors (approximate)
        if ncxreg > 0:
            valid = ~np.isnan(x) & np.all(np.isfinite(xreg), axis=1)
            X = xreg[valid]
            y = x[valid]
            y_pred = X @ model_coefs
            resid = y - y_pred
            mse = np.sum(resid**2) / max(len(resid) - len(model_coefs), 1)
            try:
                XtX_inv = np.linalg.inv(X.T @ X)
                ses = np.sqrt(np.diag(XtX_inv) * mse)
            except np.linalg.LinAlgError:
                ses = np.ones(ncxreg)
            parscale = np.concatenate([parscale, 10 * ses])
    else:
        init0 = np.concatenate([init0, np.zeros(ncxreg)])
        parscale = np.concatenate([parscale, np.ones(ncxreg)])

    return init0, parscale, n_used, orig_xreg, S


def prep_coefs(
    arma: List[int],
    coef: np.ndarray,
    cn: List[str],
    ncxreg: int
) -> pd.DataFrame:
    """
    Construct a DataFrame representing model coefficients.

    Parameters
    ----------
    arma : list
        ARIMA order [p, q, P, Q, s, d, D].
    coef : np.ndarray
        Coefficient values.
    cn : list
        Exogenous regressor names.
    ncxreg : int
        Number of exogenous regressors.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with named coefficients.
    """
    names = []

    if arma[0] > 0:
        names.extend([f"ar{i+1}" for i in range(arma[0])])
    if arma[1] > 0:
        names.extend([f"ma{i+1}" for i in range(arma[1])])
    if arma[2] > 0:
        names.extend([f"sar{i+1}" for i in range(arma[2])])
    if arma[3] > 0:
        names.extend([f"sma{i+1}" for i in range(arma[3])])
    if ncxreg > 0:
        names.extend(cn)

    return pd.DataFrame([coef], columns=names)


def optim_hessian(func, x, eps=None):
    """
    Compute numerical Hessian matrix using scipy.optimize.approx_fprime.

    Uses scipy's gradient approximation, then applies central differences
    to the gradient to obtain the Hessian matrix.

    Parameters
    ----------
    func : callable
        Objective function f(x) -> scalar.
    x : np.ndarray
        Point at which to compute Hessian.
    eps : float or None
        Step size for finite differences. If None, uses 1e-2 which works well
        for ARIMA parameters in transformed space. The default sqrt(machine epsilon)
        (~1.49e-8) is too small and causes numerical instability.

    Returns
    -------
    H : np.ndarray
        Hessian matrix (symmetric).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    if eps is None:
        eps = 1e-2

    # Compute gradient at perturbed points using scipy.optimize.approx_fprime
    # H[i,j] = (grad_j(x + eps*e_i) - grad_j(x - eps*e_i)) / (2*eps)
    def grad(x0):
        return opt.approx_fprime(x0, func, eps)

    H = np.zeros((n, n))
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps

        grad_plus = grad(x_plus)
        grad_minus = grad(x_minus)

        H[i, :] = (grad_plus - grad_minus) / (2 * eps)

    # Symmetrize for numerical stability
    H = 0.5 * (H + H.T)

    return H


def _make_numerical_gradient(func, eps=1e-2):
    """
    Create a numerical gradient function with specified step size.

    Parameters
    ----------
    func : callable
        Objective function f(x) -> scalar.
    eps : float
        Step size for finite differences. Default 1e-2 works well for ARIMA
        parameters in transformed space.

    Returns
    -------
    grad_func : callable
        Gradient function that returns numerical gradient at point x.
    """
    def grad_func(x):
        return opt.approx_fprime(x, func, eps)
    return grad_func


def arima(
    x: np.ndarray,
    m: int = 1,
    order: Tuple[int, int, int] = (0, 0, 0),
    seasonal: Tuple[int, int, int] = (0, 0, 0),
    xreg: Union[pd.DataFrame, np.ndarray, None] = None,
    include_mean: bool = True,
    transform_pars: bool = True,
    fixed: Optional[np.ndarray] = None,
    init: Optional[np.ndarray] = None,
    method: Optional[str] = "CSS-ML",
    n_cond: Optional[int] = None,
    SSinit: str = "Gardner1980",
    optim_method: str = "BFGS",
    opt_options: DictType = {'maxiter': 1000},
    kappa: float = 1e6
) -> DictType[str, Any]:
    """
    Fit an ARIMA model using maximum likelihood or conditional sum of squares.

    Parameters
    ----------
    x : np.ndarray
        Time series data.
    m : int
        Seasonal period (1 for non-seasonal).
    order : tuple
        (p, d, q) - AR order, differencing, MA order.
    seasonal : tuple
        (P, D, Q) - Seasonal AR, differencing, MA orders.
    xreg : DataFrame, ndarray, or None
        Exogenous regressors.
    include_mean : bool
        Include mean/intercept term.
    transform_pars : bool
        Transform parameters for stationarity.
    fixed : np.ndarray or None
        Fixed parameter values (NaN = free).
    init : np.ndarray or None
        Initial parameter values.
    method : str
        Estimation method: "CSS-ML", "ML", or "CSS".
    n_cond : int or None
        Number of conditioning observations.
    SSinit : str
        State-space initialization: "Gardner1980" or "Rossignol2011".
    optim_method : str
        Optimization method for scipy.optimize.minimize.
    opt_options : dict, default {'maxiter': 1000}
        Additional options for optimizer.
    kappa : float
        Prior variance for diffuse states.

    Returns
    -------
    result : dict
        Dictionary with fitted model results including:
        - 'y': Original data
        - 'fitted': Fitted values
        - 'coef': Coefficient DataFrame
        - 'sigma2': Innovation variance
        - 'var_coef': Variance-covariance matrix
        - 'loglik': Log-likelihood
        - 'aic': AIC
        - 'arma': ARIMA specification
        - 'residuals': Model residuals
        - 'converged': Convergence status
        - 'model': State-space model dict
        - 'method': Estimation method string
    """

    SSinit = match_arg(SSinit, ["Gardner1980", "Rossignol2011"])
    if method is None:
        method = "CSS-ML"
    method = match_arg(method, ["CSS-ML", "ML", "CSS"])
    SS_G = SSinit == "Gardner1980"

    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    y = x.copy()

    # Build arma specification: [p, q, P, Q, s, d, D]
    arma = [order[0], order[2], seasonal[0], seasonal[2], m, order[1], seasonal[1]]
    narma = sum(arma[:4])

    # Build Delta (differencing polynomial)
    Delta = np.array([1.0])

    for _ in range(order[1]):
        Delta = time_series_convolution(Delta, np.array([1.0, -1.0]))

    for _ in range(seasonal[1]):
        seasonal_filter = np.zeros(m + 1)
        seasonal_filter[0] = 1.0
        seasonal_filter[m] = -1.0
        Delta = time_series_convolution(Delta, seasonal_filter)

    Delta = -Delta[1:]

    nd = order[1] + seasonal[1]
    n_used = len(na_omit(x)) - len(Delta)

    xreg_original = xreg

    # Add intercept if needed
    if include_mean and nd == 0:
        xreg = add_drift_term(xreg, np.ones(n), "intercept")

    xreg_mat, ncxreg, nmxreg = process_xreg(xreg, n)

    # Check for missing values
    if method == "CSS-ML":
        has_missing = lambda xi: np.isnan(xi) if np.isscalar(xi) else np.any(np.isnan(xi))
        anyna = np.any([has_missing(xi) for xi in x])
        if ncxreg > 0:
            anyna = anyna or np.any(np.isnan(xreg_mat))
        if anyna:
            method = "ML"

    # Determine conditioning observations
    if method in ["CSS", "CSS-ML"]:
        ncond = order[1] + seasonal[1] * m
        ncond1 = order[0] + seasonal[0] * m
        if n_cond is None:
            ncond += ncond1
        else:
            ncond += max(n_cond, ncond1)
    else:
        ncond = 0

    # Handle fixed parameters
    if fixed is None:
        fixed = np.full(narma + ncxreg, np.nan)
    elif len(fixed) != narma + ncxreg:
        raise ValueError("Wrong length for 'fixed'")

    mask = np.isnan(fixed)
    no_optim = not np.any(mask)

    if no_optim:
        transform_pars = False

    if transform_pars:
        ind = np.arange(arma[0] + arma[1], arma[0] + arma[1] + arma[2])
        if np.any(~mask[:arma[0]]) or (len(ind) > 0 and np.any(~mask[ind])):
            warnings.warn("Some AR parameters were fixed: Setting transform_pars = False")
            transform_pars = False

    # Estimate initial values and scaling
    if ncxreg > 0:
        init0, parscale, n_used, orig_xreg, S = regress_and_update(
            x, xreg_mat, mask, narma, ncxreg, order[1], seasonal[1], m, Delta
        )
    else:
        init0 = np.zeros(narma)
        parscale = np.ones(narma)
        orig_xreg = True
        S = None

    if n_used <= 0:
        raise ValueError("Too few non-missing observations")

    if init is not None:
        if len(init) != len(init0):
            raise ValueError("'init' is of the wrong length")
        ind_na = np.isnan(init)
        init[ind_na] = init0[ind_na]

        if method == "ML":
            if arma[0] > 0 and not ar_check(init[:arma[0]]):
                raise ValueError("non-stationary AR part")
            if arma[2] > 0:
                sa_start = arma[0] + arma[1]
                sa_stop = arma[0] + arma[1] + arma[2]
                if not ar_check(init[sa_start:sa_stop]):
                    raise ValueError("non-stationary seasonal AR part")
    else:
        init = init0.copy()

    coef = fixed.astype(np.float64).copy()

    # Create working copies for optimization
    arma_arr = np.array(arma, dtype=np.float64)

    # Define objective functions
    def armafn(p, trans):
        """ML objective function."""
        nonlocal mod
        par = coef.copy()
        par[mask] = p

        try:
            trarma = transform_arima_parameters(par, arma_arr, trans)
            phi_t, theta_t = trarma
        except Exception:
            return 1e10

        try:
            mod = update_arima(mod, phi_t, theta_t, ss_g=SS_G)
        except Exception:
            return 1e10

        if ncxreg > 0:
            xxi = x - xreg_mat @ par[narma:narma + ncxreg]
        else:
            xxi = x

        try:
            resss = compute_arima_likelihood(xxi, mod, update_start=0, give_resid=False)
        except Exception:
            return 1e10

        s2 = resss['ssq'] / resss['nu'] if resss['nu'] > 0 else 1e10
        if s2 <= 0:
            return 1e10

        return 0.5 * (np.log(s2) + resss['sumlog'] / resss['nu'])

    def armaCSS(p):
        """CSS objective function."""
        par = fixed.copy()
        par[mask] = p

        try:
            trarma = transform_arima_parameters(par, arma_arr, False)
            phi_t, theta_t = trarma
        except Exception:
            return 1e10

        if ncxreg > 0:
            x_in = x - xreg_mat @ par[narma:narma + ncxreg]
        else:
            x_in = x

        try:
            sigma2, _ = compute_css_residuals(x_in, arma_arr, phi_t, theta_t, ncond)
        except Exception:
            return 1e10

        if sigma2 <= 0 or np.isnan(sigma2):
            return 1e10

        return 0.5 * np.log(sigma2)

    # Initialize state-space model
    init_phi, init_theta = transform_arima_parameters(init, arma_arr, transform_pars)
    mod = initialize_arima_state(init_phi, init_theta, Delta, kappa=kappa, SSinit=SSinit)

    # Optimization - use BFGS matching R's optim default
    if method == "CSS":
        if no_optim:
            res = {'converged': True, 'x': np.zeros(0), 'fun': armaCSS(np.zeros(0))}
        else:
            opt_result = opt.minimize(
                armaCSS,
                init[mask],
                method=optim_method,
                options=opt_options
            )
            res = {'converged': opt_result.success, 'x': opt_result.x, 'fun': opt_result.fun}

        if not res['converged']:
            warnings.warn(
                "CSS optimization convergence issue. Try to increase 'maxiter' or change the optimization method."
            )

        coef[mask] = res['x']

        trarma = transform_arima_parameters(coef, arma_arr, False)
        mod = initialize_arima_state(trarma[0], trarma[1], Delta, kappa=kappa, SSinit=SSinit)

        if ncxreg > 0:
            x_work = x - xreg_mat @ coef[narma:narma + ncxreg]
        else:
            x_work = x

        compute_arima_likelihood(x_work, mod, update_start=0, give_resid=True)
        sigma2, resid = compute_css_residuals(x_work, arma_arr, trarma[0], trarma[1], ncond)

        if no_optim:
            var = np.zeros((0, 0))
        else:
            hessian = optim_hessian(armaCSS, res['x'])
            try:
                var = np.linalg.inv(hessian * n_used)
            except np.linalg.LinAlgError:
                var = np.zeros((np.sum(mask), np.sum(mask)))

    else:
        # CSS-ML or ML
        if method == "CSS-ML":
            if no_optim:
                res = {'converged': True, 'x': init[mask], 'fun': armaCSS(np.zeros(np.sum(mask)))}
            else:
                opt_result = opt.minimize(
                    armaCSS,
                    init[mask],
                    method=optim_method,
                    options=opt_options
                )
                res = {'converged': opt_result.success, 'x': opt_result.x, 'fun': opt_result.fun}

            if res['converged']:
                # Check stationarity before accepting CSS results
                css_params = init.copy()
                css_params[mask] = res['x']
                
                # Check if CSS produced stationary parameters
                if (arma[0] > 0 and not ar_check(css_params[:arma[0]])) or \
                   (arma[2] > 0 and not ar_check(css_params[sum(arma[:2]):sum(arma[:3])])):
                    warnings.warn(
                        "CSS optimization produced non-stationary parameters. "
                        "Falling back to ML estimation with zero starting values."
                    )
                    # Reset AR parameters to zeros (like statsmodels does)
                    if arma[0] > 0:
                        init[:arma[0]] = 0.0
                    if arma[2] > 0:
                        init[sum(arma[:2]):sum(arma[:3])] = 0.0
                else:
                    # Use CSS results only if stationary
                    init[mask] = res['x']

            ncond = 0

        if transform_pars:
            init = inverse_arima_parameter_transform(init, np.array(arma[:3]))

            if arma[1] > 0:
                ind = slice(arma[0], arma[0] + arma[1])
                init[ind] = ma_invert(init[ind])

            if arma[3] > 0:
                ind = slice(sum(arma[:3]), sum(arma[:4]))
                init[ind] = ma_invert(init[ind])

        trarma = transform_arima_parameters(init, arma_arr, transform_pars)
        mod = initialize_arima_state(trarma[0], trarma[1], Delta, kappa=kappa, SSinit=SSinit)

        if no_optim:
            res = {'converged': True, 'x': np.zeros(0), 'fun': armafn(np.zeros(0), transform_pars)}
        else:
            ml_obj_func = lambda p: armafn(p, transform_pars)
            opt_result = opt.minimize(
                ml_obj_func,
                init[mask],
                method=optim_method,
                options=opt_options
            )
            res = {'converged': opt_result.success, 'x': opt_result.x, 'fun': opt_result.fun}

        if not res['converged']:
            warnings.warn(
                "Possible convergence problem. "
                "Try to increase 'maxiter' or change the optimization method."
            )

        coef[mask] = res['x']

        if transform_pars:
            if arma[1] > 0:
                ind = slice(arma[0], arma[0] + arma[1])
                if np.all(mask[ind]):
                    coef[ind] = ma_invert(coef[ind])

            if arma[3] > 0:
                ind = slice(sum(arma[:3]), sum(arma[:4]))
                if np.all(mask[ind]):
                    coef[ind] = ma_invert(coef[ind])

            hessian = optim_hessian(lambda p: armafn(p, True), coef[mask])
            A = compute_arima_transform_gradient(coef, arma_arr)
            A = A[np.ix_(mask, mask)]
            try:
                var = A.T @ np.linalg.solve(hessian * n_used, A)
            except np.linalg.LinAlgError:
                var = np.zeros((np.sum(mask), np.sum(mask)))

            coef = undo_arima_parameter_transform(coef, arma_arr)
        else:
            if no_optim:
                var = np.zeros((0, 0))
            else:
                hessian = optim_hessian(lambda p: armafn(p, transform_pars), res['x'])
                try:
                    var = np.linalg.inv(hessian * n_used)
                except np.linalg.LinAlgError:
                    var = np.zeros((np.sum(mask), np.sum(mask)))

        trarma = transform_arima_parameters(coef, arma_arr, False)
        mod = initialize_arima_state(trarma[0], trarma[1], Delta, kappa=kappa, SSinit=SSinit)

        if ncxreg > 0:
            x_work = x - xreg_mat @ coef[narma:narma + ncxreg]
        else:
            x_work = x

        val = compute_arima_likelihood(x_work, mod, update_start=0, give_resid=True)
        sigma2 = val['ssq'] / n_used
        resid = val['resid']
        
        # Update model state with final filtered state
        mod['a'] = val['a']
        mod['P'] = val['P']

    # Final computations
    value = 2 * n_used * res['fun'] + n_used + n_used * np.log(2 * np.pi)

    if method != "CSS":
        aic = value + 2 * np.sum(mask) + 2
    else:
        aic = np.nan

    loglik = -0.5 * value

    # Transform xreg coefficients back if needed
    if ncxreg > 0 and not orig_xreg and S is not None:
        ind = slice(narma, narma + ncxreg)
        coef[ind] = S['V'] @ coef[ind]
        A = np.eye(narma + ncxreg)
        A[np.ix_(range(narma, narma + ncxreg), range(narma, narma + ncxreg))] = S['V']
        A = A[np.ix_(mask, mask)]
        var = A @ var @ A.T

    arima_coef = prep_coefs(arma, coef, nmxreg, ncxreg)
    fitted_vals = y - resid

    if ncxreg > 0:
        fit_method = f"Regression with ARIMA({order[0]},{order[1]},{order[2]})({seasonal[0]},{seasonal[1]},{seasonal[2]})[{m}] errors"
    else:
        fit_method = f"ARIMA({order[0]},{order[1]},{order[2]})({seasonal[0]},{seasonal[1]},{seasonal[2]})[{m}]"

    return {
        'y': y,
        'fitted': fitted_vals,
        'coef': arima_coef,
        'sigma2': float(np.sum(resid**2) / n_used),
        'var_coef': var,
        'mask': mask,
        'loglik': loglik,
        'aic': aic,
        'bic': None,
        'aicc': None,
        'arma': arma,
        'residuals': resid,
        'converged': res['converged'],
        'n_cond': ncond,
        'nobs': n_used,
        'model': mod,
        'xreg': xreg_original,
        'method': fit_method,
        'lambda': None,
        'biasadj': None,
        'offset': None
    }


def predict_arima(
    model: DictType[str, Any],
    n_ahead: int = 1,
    newxreg: Union[pd.DataFrame, np.ndarray, None] = None,
    se_fit: bool = True,
    level: Union[List[float], np.ndarray, None] = None
) -> DictType[str, Any]:
    """
    Generate forecasts from a fitted ARIMA model.

    Parameters
    ----------
    model : dict
        Fitted ARIMA model from arima().
    n_ahead : int
        Number of periods to forecast.
    newxreg : DataFrame, ndarray, or None
        New exogenous regressors for forecast period.
    se_fit : bool
        Whether to compute standard errors.
    level : list of float, default None
        Confidence levels for prediction intervals (default [80, 95]).
        Values can be percentages (80, 95) or proportions (0.80, 0.95).

    Returns
    -------
    result : dict
        Dictionary with:
        - 'mean': Point forecasts
        - 'lower': Lower bounds of prediction intervals
        - 'upper': Upper bounds of prediction intervals
        - 'level': Confidence levels used
        - 'se': Standard errors
        - 'y': Original data
        - 'fitted': In-sample fitted values
        - 'residuals': Model residuals
        - 'method': Model description
    """
    arma = model['arma']
    coef_df = model['coef']
    coefs = coef_df.values.flatten()
    coef_names = list(coef_df.columns)
    narma = sum(arma[:4])
    ncoefs = len(coefs)

    if level is not None:
        levels = list(level)
        if min(levels) > 0 and max(levels) < 1:
            levels = [l * 100 for l in levels]
        if min(levels) < 0 or max(levels) > 99.99:
            raise ValueError("Confidence level out of range")
        levels = sorted(levels)
    else:
        levels = []

    # Check for intercept
    intercept_idx = coef_names.index("intercept") if "intercept" in coef_names else None
    has_intercept = intercept_idx is not None

    # Handle xreg
    if model['xreg'] is not None:
        if isinstance(model['xreg'], pd.DataFrame):
            ncxreg = model['xreg'].shape[1]
        else:
            ncxreg = model['xreg'].shape[1] if model['xreg'].ndim > 1 else 1
    else:
        ncxreg = 0

    if newxreg is not None:
        if isinstance(newxreg, pd.DataFrame):
            newxreg = newxreg.values
        newxreg = np.asarray(newxreg, dtype=np.float64)
        if newxreg.ndim == 1:
            newxreg = newxreg.reshape(-1, 1)

    # Compute xreg contribution to forecasts
    xm = np.zeros(n_ahead)

    if ncoefs > narma:
        if has_intercept and coef_names[narma] == "intercept":
            intercept_col = np.ones((n_ahead, 1))
            if newxreg is None:
                usexreg = intercept_col
            else:
                usexreg = np.column_stack([intercept_col, newxreg])
            reg_coef_inds = slice(narma, ncoefs)
        else:
            usexreg = newxreg
            reg_coef_inds = slice(narma, ncoefs)

        if usexreg is not None:
            if narma == 0:
                xm = usexreg @ coefs
            else:
                xm = usexreg @ coefs[reg_coef_inds]

    # Kalman forecast
    forecast_result = kalman_forecast(n_ahead, model['model'], update=False)
    pred = forecast_result['pred'] + xm

    if se_fit:
        se = np.sqrt(forecast_result['var'] * model['sigma2'])
    else:
        se = np.full(len(pred), np.nan)

    lower = None
    upper = None
    if levels:
        alpha_lvls = 1.0 - np.asarray(levels, dtype=np.float64) / 100.0
        z_scores = norm.ppf(1.0 - alpha_lvls / 2.0)
        se_expanded = se[:, np.newaxis]
        mean_expanded = pred[:, np.newaxis]
        lower = mean_expanded - z_scores * se_expanded  # lower bounds
        upper = mean_expanded + z_scores * se_expanded  # upper bounds

    return {
        'mean': pred,
        'lower': lower,
        'upper': upper,
        'level': levels,
        'se': se,
        'y': model['y'],
        'fitted': model['fitted'],
        'residuals': model['residuals'],
        'method': model['method']
    }


def fitted_values(model: DictType[str, Any]) -> np.ndarray:
    """
    Extract fitted values from ARIMA model.

    Parameters
    ----------
    model : dict
        Fitted ARIMA model.

    Returns
    -------
    np.ndarray
        Fitted values.
    """
    return model['fitted']


def residuals_arima(model: DictType[str, Any]) -> np.ndarray:
    """
    Extract residuals from ARIMA model.

    Parameters
    ----------
    model : dict
        Fitted ARIMA model.

    Returns
    -------
    np.ndarray
        Model residuals.
    """
    return model['residuals']

