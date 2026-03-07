# Deep Analysis: ARIMA Refactor (`feature__ai_enhancements` vs `master`)

## 1. Bugs



### BUG-1: Kalman filter `update_start` off-by-one (MEDIUM)

**File**: `_arima_base.py`, `_arima_kalman_core()` (line 1211)

```python
if t > update_start:
    # Covariance prediction: Pnew = T @ P @ T' + V
```

`_arima_kalman_core` is a `@njit(cache=True)` function. `compute_arima_likelihood()` always calls it with `update_start=0`. The filter loop runs from `t=0` to `t=n-1`:

1. At **t=0**: state prediction `anew = T @ a` and innovation/update of `P` happen correctly. Then `if t > 0` is `False`, so `Pnew` is **never propagated** — it stays as `Pn_init` (the Lyapunov P₀ from `initialize_arima_state`).
2. At **t=1**: `Pnew` is **still** the initial Lyapunov P₀ (not `T @ P_filtered_0 @ T' + V`). The t=0 filtered covariance update is lost.
3. From **t=2** onward the propagation runs normally (the off-by-one is limited to one step).

R's `stats::KalmanLike` uses `if(t >= antefirst)` which is inclusive — propagation happens right after the first observation update.

**Impact**: After t=0, the filtered covariance `P` is updated but never propagated to `Pnew`. At t=1, `Pnew` still equals the initial Lyapunov solution, so the innovation variance `F₁` and the Kalman gain `K₁` are computed from the wrong predicted covariance. For well-identified, stationary models with rapidly converging P the error is near-zero; for near-unit-root models or long differencing chains with diffuse initialisation (large κ) the first two innovations can be significantly mis-weighted, biasing the log-likelihood.

**Fix**: `if t >= update_start:`

---

### BUG-2: Fitted values use standardized innovations instead of raw innovations (MEDIUM)

**File**: `_arima_base.py`, `_build_arima_result()` (line 2679)

```python
fitted_vals = c.y - fit.resid
```

`c.y` is the original undifferenced series (`y = x.copy()` in `_prepare_arima_config`). The content of `fit.resid` differs by path:

- **ML / CSS-ML**: `fit.resid = kf_final['resid']` from `compute_arima_likelihood(..., give_resid=True)`, which calls `_arima_kalman_core` with `give_resid=True`. Inside the `@njit` core: `std_residuals[t] = innovation / np.sqrt(F)`. These are **standardized** one-step innovations `vₜ / √Fₜ`, not `yₜ − ŷₜ`. So `c.y[t] − vₜ/√Fₜ ≠ ŷₜ`.
- **CSS**: `fit.resid` from `compute_css_residuals()` are raw ARMA recursion errors `eₜ = wₜ − φ₁wₜ₋₁ − …` where `wₜ = Δᵈ Δₛᴰ yₜ` is the **differenced** series. For models with `d > 0` or `D > 0`, `c.y[t] − eₜ` mixes the original (integrated) domain with differenced-domain residuals and is also incorrect. For the pure ARMA case (`d=D=0`) it happens to be correct.

The correct fitted value for both paths is `ŷₜ = Z' aₜ|ₜ₋₁` (predicted observation from filtered state) plus the exogenous contribution, which is equivalent to `yₜ − vₜ` (raw innovation, not standardized).

**Impact**: `ArimaResult.fitted_values`, stored as `Arima.fitted_values_`, is incorrect for ML/CSS-ML and for any CSS model with differencing. This propagates to `get_fitted_values()`, `get_score()`, and `summary()`.

**Fix**: For ML path, raw innovations `vₜ` are not currently stored. Options: (a) store raw innovations alongside standardized ones in `_arima_kalman_core`, or (b) back-compute as `vₜ = std_resid_t * √Fₜ` (requires storing Fₜ too), or (c) compute fitted values as `yₜ − (yₜ − predicted_obs)` by running a second Kalman pass that captures `Z' anew[t]` at each step.

---

### BUG-3: `_build_arima_result` σ² recomputed with wrong denominator for CSS (MEDIUM)

**File**: `_arima_base.py`, `_build_arima_result()` (line 2693)

```python
sigma2=float(np.sum(fit.resid**2) / c.n_used),
```

The analysis of which path is affected is more subtle than originally stated:

**ML path — numerically correct (coincidence)**: `fit.resid` are standardized innovations `vₜ/√Fₜ`, so `Σ(vₜ/√Fₜ)² = Σvₜ²/Fₜ = kf['ssq']`. Dividing by `c.n_used` gives `kf['ssq'] / n_used`, which equals `fit.sigma2` as set in `_fit_ml` (`kf_final['ssq'] / c.n_used`). The computation is conceptually wrong but numerically matches the correct value.

**CSS path — numerically wrong**: `fit.resid` from `compute_css_residuals()` contains zeros for the conditioning period (`0 … n_cond-1`). `c.n_used = n − len(Δ) = n − (d + Ds)`, while the CSS σ² uses `n_valid = n_cond_to_n` non-zero residuals (denominator is smaller). Concretely, `c.n_used > n_valid` whenever `p + Ps > 0` (i.e., the model has any AR terms), so `_build_arima_result` stores a **smaller σ²** than the CSS estimate. This produces narrower prediction intervals for CSS-only models.

**Root cause**: `_build_arima_result` ignores `fit.sigma2` (which is correctly computed per path in `_fit_css` and `_fit_ml`) and recomputes σ² redundantly with a different denominator.

**Impact**: Prediction intervals for CSS-only models (`method="CSS"`) are too narrow. For CSS-ML and ML the stored σ² is numerically correct.

**Fix**: Use `fit.sigma2` directly instead of recomputing: `sigma2=float(fit.sigma2)`.

---

### BUG-4: `fit_custom_arima` σ² recalculation doubly incorrect for ML (LOW)

**File**: `_auto_arima.py`, `fit_custom_arima()` (lines ~453–455)

```python
resid_valid = fit['residuals'][~np.isnan(fit['residuals'])]
if len(resid_valid) > npar - 1:
    fit['sigma2'] = np.sum(resid_valid**2) / (nstar_adj - npar + 1)
```

This recalculation has two compounding problems:

1. **Wrong residuals for ML**: `fit['residuals']` is `ArimaResult.residuals`, which is `fit.resid` from `_FitResult` — i.e. standardized Kalman innovations `vₜ/√Fₜ` for the ML path. `Σ(vₜ/√Fₜ)²` equals `ssq` (the Kalman sum-of-squares) which, divided by `n_used`, equals the correct ML σ². However, filtering out NaNs and then computing with denominator `nstar_adj - npar + 1` breaks this coincidence. Conditioning-period residuals for CSS are zeros (not NaN), so they are **not** filtered out by the `~np.isnan` mask — they remain in `resid_valid`, artificially inflating `Σ resid²`.

2. **Wrong denominator**: `nstar_adj - npar + 1` is a degrees-of-freedom adjusted denominator (OLS convention), while ML σ² uses maximum-likelihood denominator `n_used` and CSS σ² uses `n_valid` (number of non-conditioning residuals). None of these match.

The net effect for a CSS-ML or ML model: σ² is overwritten with a value that has neither the correct numerator nor the correct denominator, corrupting the prediction interval widths computed downstream by `predict_arima()` (which scales forecast variances by `model['sigma2']`).

Similarly in `arima_rjh()` (line ~1 613):
```python
fit['sigma2'] = ss / (nstar - np_ + 1)
```
This overwrites the ML-estimated σ² with a CSS-style degrees-of-freedom–adjusted version **after** the ML fit has been performed, for the same reasons.

---

## 3. Optimization Opportunities


### OPT-1 / BUG: `_fit_css` discards Kalman result → wrong forecast state for CSS models (MEDIUM)

**File**: `_arima_base.py`, `_fit_css()` (line ~2 572)

```python
state_space = initialize_arima_state(phi_final, theta_final, c.Delta, kappa=c.kappa)
adjusted_series = c.x - c.exog_matrix @ params[...] if c.n_exog > 0 else c.x
compute_arima_likelihood(adjusted_series, state_space, update_start=0, give_resid=True)  # ← return value discarded
sigma2, resid = compute_css_residuals(...)
```

Verified: `_arima_kalman_core` works on internal copies — `a = a_init.copy()`, `P = P_init.copy()`, `Pnew = Pn_init.copy()` — so calling `compute_arima_likelihood()` does **not** mutate `state_space`. The return value contains `kf['a']` (final filtered state) and `kf['P']` (final filtered covariance), but both are discarded.

The consequences are:

1. **Wasted computation**: a full O(n·r²) Kalman pass is run unnecessarily.

2. **Wrong forecast state for `method="CSS"` models** (actual bug): After `_fit_css`, `state_space.filtered_state` is the zero vector and `state_space.filtered_covariance` is the zero matrix (both set by `initialize_arima_state`). This `state_space` is stored in the returned `_FitResult`, passed through `_build_arima_result`, and eventually used by `predict_arima()` → `kalman_forecast()`. Forecasting from a zero state ignores all information about the most recently observed values — it predicts as if the series ended at an all-zero state, which is wrong for any non-trivial AR model.

   For **ML and CSS-ML** (the default) this is not a problem: `_fit_ml` explicitly does `ss_final.filtered_state = kf_final['a']` and `ss_final.filtered_covariance = kf_final['P']` after its Kalman call. The CSS stage in CSS-ML only provides a warm start; the final model always comes from `_fit_ml`.

   For **`method="CSS"` only**: the zero state is used for all forecasts, producing incorrect predictions.

**Fix**: Use the Kalman result to update the state (as `_fit_ml` does), not remove the call:
```python
kf = compute_arima_likelihood(adjusted_series, state_space, update_start=0, give_resid=False)
state_space.filtered_state = kf['a']
state_space.filtered_covariance = kf['P']
```
