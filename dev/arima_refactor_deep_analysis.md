# Deep Analysis: ARIMA Refactor (`feature__ai_enhancements` vs `master`)

> **Date**: 2025  
> **Branch**: `feature__ai_enhancements`  
> **Scope**: `skforecast/stats/arima/_arima_base.py`, `_auto_arima.py`, `skforecast/stats/_arima.py`, `__init__.py`  
> **Diff summary**: 14 files changed, ~3 550 insertions, ~2 623 deletions

---

## 1. Architecture Overview

The refactor replaces the original dict-centric R-port with a **dataclass pipeline**:

| Old | New |
|-----|-----|
| Loose dicts (`{'phi': …, 'theta': …}`) | `StateSpaceArrays` dataclass |
| Tuple returns from `arima()` | `ArimaResult` dataclass (with backward-compat `__getitem__`) |
| Monolithic `arima()` function (~600 lines) | Pipeline: `_prepare_arima_config()` → `_fit_css()` / `_fit_ml()` / `_fit_css_ml()` → `_build_arima_result()` |
| `include_mean`, `transform_pars`, `xreg`, `SSinit` | `fit_intercept`, `enforce_stationarity`, `exog`, removed `SSinit` |
| No Numba | `@njit(cache=True)` on Kalman core, CSS residuals, parameter transforms, convolution |
| Single Lyapunov solver | Adaptive: Smith doubling (r ≤ 6) / scipy Kronecker (r > 6) |

New internal dataclasses:
- **`SARIMAOrder`** (frozen) — model orders + helpers (`n_arma_params`, `to_arma_list()`)
- **`StateSpaceArrays`** — all state-space matrices in one mutable bundle
- **`ArimaResult`** — fitted model output with `_KEY_MAP` for legacy dict access
- **`_ArimaConfig`** — immutable configuration for fitting pipeline
- **`_FitResult`** — intermediate optimization result

---

## 2. Bugs

### BUG-1: Operator precedence in ternary exogenous adjustment (HIGH)

**Files**: `_arima_base.py` — `_fit_css._css_objective()` (line ~2 570), `_fit_ml._ml_objective()` (line ~2 670)

```python
adjusted = c.x - c.exog_matrix @ _par[c.n_arma_params:c.n_arma_params + c.n_exog] if c.n_exog > 0 else c.x
```

Python parses this as:

```python
adjusted = c.x - (c.exog_matrix @ _par[...] if c.n_exog > 0 else c.x)
```

When `c.n_exog == 0` the expression becomes `c.x - c.x = 0`, zeroing the entire series.

**Impact**: Only manifests when a model with **no** exogenous regressors enters the code path where `c.n_exog > 0` was expected — which can't happen with the current flow because the intercept is handled upstream. But the operator precedence is still **technically wrong** and fragile. If a future refactor moves the intercept logic, this would become a silent data-zeroing bug.

**Fix**:
```python
adjusted = (c.x - c.exog_matrix @ _par[c.n_arma_params:c.n_arma_params + c.n_exog]) if c.n_exog > 0 else c.x
```

This same pattern appears **4 times** in `_fit_css` and `_fit_ml` (both in objective closures and final residual computation).

---

### BUG-2: Kalman filter `update_start` off-by-one (MEDIUM)

**File**: `_arima_base.py`, `_arima_kalman_core()` (line ~1 330)

```python
if t > update_start:
    # Covariance prediction: Pnew = T @ P @ T' + V
```

With `update_start=0` (the standard call from both `_fit_css` and `_fit_ml`), the covariance prediction **skips t=0 entirely**. At t=0, Pnew retains the initial `Pn_init` for the **next** iteration at t=1. This means the predicted covariance is never updated at the first step.

R's `stats::KalmanLike` uses `if(t >= antefirst)` which is inclusive. The `>=` vs `>` difference is equivalent to starting the covariance update one step later.

**Impact**: The first innovation variance F₀ is computed from `Pn_init` (the Lyapunov solution) which is correct. The bug means the update `Pnew = T P T' + V` for **t=1** uses the initial Pn instead of the updated one. For stable models this is near-zero error because P converges rapidly, but for near-unit-root or highly diffuse initialization it could bias the likelihood.

**Fix**: `if t >= update_start:`

---

### BUG-3: Fitted values use raw residuals from ML path (MEDIUM)

**File**: `_arima_base.py`, `_build_arima_result()` (line ~2 907)

```python
fitted_vals = c.y - fit.resid
```

For the **ML** and **CSS-ML** paths, `fit.resid` comes from Kalman standardized innovations (`innovation / sqrt(F)`) — not raw residuals (`y - ŷ`). Therefore `y - standardized_residuals ≠ fitted_values`.

For the **CSS** path, `fit.resid` comes from `compute_css_residuals()` which returns genuine raw residuals, so the formula is correct there.

**Impact**: Fitted values returned by ML-estimated models are **incorrect**. The `Arima.fit()` method stores these as `self.fitted_values_`, exposed via `get_fitted_values()`, `get_score()`, and `summary()`.

**Fix**: For ML path, run `compute_arima_likelihood(..., give_resid=True)` and use the structural residuals, or compute fitted values via Kalman filtered states: `fitted = Z @ a_filtered`.

---

### BUG-4: `_build_arima_result` σ² uses standardized residuals for ML (MEDIUM)

**File**: `_arima_base.py`, `_build_arima_result()` (line ~2 911)

```python
sigma2=float(np.sum(fit.resid**2) / c.n_used),
```

Same root cause as BUG-3: for ML, `fit.resid` are standardized innovations, so `Σ(v_t/√F_t)² / n` is not the correct innovation variance σ². The ML σ² should be `ssq / nu` from the Kalman filter output (`kf['ssq'] / kf['nu']`), which is what the optimizer already computes.

**Impact**: The `sigma2` stored in the result (and used by `fit_custom_arima` for IC recomputation) is wrong for ML-estimated models. This propagates to prediction intervals via `predict_arima()` which scales forecast variances by `model['sigma2']`.

---

### BUG-5: `forecast_arima` silently overwrites `level` when `fan=True` (LOW)

**File**: `_auto_arima.py`, `forecast_arima()` (line ~1 748)

```python
if level is not None:
    if fan:
        levels = np.arange(51, 100, 3).tolist()
    else:
        levels = list(level)
```

When `level` is not None AND `fan=True`, the user-supplied `level` is silently discarded and replaced with `[51, 54, ..., 99]`. No warning is emitted.

**Fix**: Raise `ValueError` or emit a warning when both `level` and `fan=True` are specified.

---

### BUG-6: `forecast_arima` returns empty list instead of default levels (LOW)

**File**: `_auto_arima.py`, `forecast_arima()` (line ~1 755)

```python
else:
    levels = []
```

When `level=None` and `fan=False` (both defaults), `levels` is set to `[]`. This means **no prediction intervals are computed by default**. The function's docstring says "default [80, 95]" but the code doesn't implement that default — it relies on `predict_arima()` to handle its own default, but the empty `levels` list means the `if levels:` block in `forecast_arima` is skipped and `lower`/`upper` remain `None`.

Meanwhile `predict_arima()` would return intervals if called with `level=None` directly, but here it receives `levels=[]` which is truthy emptiness. The interaction between these two functions means that `forecast_arima` **never returns prediction intervals unless `level` is explicitly provided**.

---

### BUG-7: `fit_custom_arima` σ² recalculation inconsistency (LOW)

**File**: `_auto_arima.py`, `fit_custom_arima()` (line ~455)

```python
resid_valid = fit['residuals'][~np.isnan(fit['residuals'])]
if len(resid_valid) > npar - 1:
    fit['sigma2'] = np.sum(resid_valid**2) / (nstar_adj - npar + 1)
```

This uses `nstar_adj - npar + 1` as denominator (degrees-of-freedom adjusted), but the `arima()` function returns σ² with denominator `n_used` (ML convention). The recalculation mixes conventions, and the residuals used include conditioning-period residuals (which are zero for CSS), inflating the denominator relative to the actual number of valid residuals.

Similarly in `arima_rjh()` (line ~1 613):
```python
fit['sigma2'] = ss / (nstar - np_ + 1)
```

This overwrites the ML-estimated σ² with a CSS-style degrees-of-freedom-adjusted version **after** the ML fit has been performed.

---

## 3. Dead Code

### DEAD-1: `compute_arima_likelihood_core` — 108 lines (HIGH)

**File**: `_arima_base.py` (lines ~1 383–1 479)

A full generic Kalman filter implementation using dense matrix multiplications (`T @ a`, `T @ P @ T.T + V`). It is **never called** by any production code — `_arima_kalman_core` (the companion-structure version) is used everywhere instead.

The only code that references it is `kalman_update()`, which is also dead.

### DEAD-2: `kalman_update` — 45 lines

**File**: `_arima_base.py` (lines ~1 135–1 179)

Only called by `compute_arima_likelihood_core` (which is itself dead). Never used in the companion-structure path.

### DEAD-3: `_numerical_gradient_factory` — 20 lines

**File**: `_arima_base.py` (lines ~2 275–2 294)

Creates a closure around `scipy.optimize.approx_fprime`. Never called anywhere in the codebase — the optimizer uses its own numerical gradients.

### DEAD-4: `_validate_pdq` — accessed only in tests

**File**: `_arima_base.py` (line ~407)

Not used by any production code path. Only referenced in unit tests.

### DEAD-5: `if __name__ == "__main__"` block — ~140 lines

**File**: `_auto_arima.py` (lines ~1 840–1 980)

Script-style manual tests. These serve no purpose in a library module and will never execute when imported normally. They should be proper test cases in `tests/`.

### DEAD-6: `time_series_convolution` — only in tests

**File**: `_arima_base.py`

The `@njit` convolution function is only exercised by the test suite, not by any production code path.

---

## 4. Optimization Opportunities

### OPT-1: `_update_state_space` skips V matrix rebuild → Kalman forecast uses stale V (MEDIUM)

**File**: `_arima_base.py`, `_update_state_space()` (lines 1 630–1 634)

```python
# ML optimization uses the companion Kalman core directly from
# (phi, theta, Delta) and does not consume innovation_covariance.
# Skip rebuilding V here to avoid repeated O(rd²) allocations.
# A fully consistent state-space object is rebuilt at the optimum.
```

The comment correctly notes that `_arima_kalman_core` doesn't use `ss.innovation_covariance` — it inline-computes V = RR'. However, `kalman_forecast()` **does** use `ss.innovation_covariance` (via `kalman_forecast_core`). After optimization, `_build_arima_result()` calls `initialize_arima_state()` which builds a fresh `StateSpaceArrays` with correct V.

This is safe **only if** `kalman_forecast()` is never called with the intermediate `ss_holder[0]` from the optimization loop — which it isn't currently. But the inconsistency is fragile and should be documented more prominently or removed by always rebuilding V.

### OPT-2: `_fit_css` calls `compute_arima_likelihood` wastefully (MEDIUM)

**File**: `_arima_base.py`, `_fit_css()` (line ~2 572)

```python
state_space = initialize_arima_state(phi_final, theta_final, c.Delta, kappa=c.kappa)
adjusted_series = c.x - c.exog_matrix @ params[...] if c.n_exog > 0 else c.x
compute_arima_likelihood(adjusted_series, state_space, update_start=0, give_resid=True)  # ← return value discarded
sigma2, resid = compute_css_residuals(...)
```

The `compute_arima_likelihood()` call runs the full Kalman filter, but its return value is **discarded**. The actual residuals and σ² come from `compute_css_residuals()` on the next line. This is a wasted O(n·r²) computation.

The likely intent was to populate `state_space.filtered_state` and `filtered_covariance` (which the Kalman core updates in-place). However, the `_arima_kalman_core` function works on **copies** (`a_init.copy()`, `P_init.copy()`), so `state_space` is **not** mutated — making the call truly useless.

**Fix**: Remove the `compute_arima_likelihood(...)` line from `_fit_css`.

### OPT-3: `_fit_ml._ml_objective` builds Pnew redundantly (LOW)

Inside the ML objective closure:
```python
ss_holder[0] = _update_state_space(ss_holder[0], phi_exp, theta_exp)
```

`_update_state_space` calls `compute_q0_covariance_matrix()` which solves the Lyapunov equation on every objective evaluation. For the Smith doubling path (r ≤ 6) this is O(r³) per iteration — fast but called hundreds of times during optimization. For seasonal models (r > 6), the scipy Kronecker solver involves constructing an r²×r² matrix and solving it, which is significantly more expensive.

Consider caching P₀ when only the AR coefficients change by a small delta (warm-starting the Lyapunov solver from the previous solution), or providing an option to skip P₀ recomputation during CSS-ML warm start where it's less critical.

### OPT-4: Stepwise search `results` array fixed at 94 rows (LOW)

**File**: `_auto_arima.py`, `auto_arima()` (line ~1 025)

```python
results = np.full((nmodels, 8), np.nan)
```

The default `nmodels=94` is a hard limit. Once `k > nmodels`, the search stops with a warning. The allocation is static regardless of whether the search space is tiny (e.g., max_p=1, max_q=1 → ~10 models) or large.

This isn't a performance problem (94×8 floats is negligible), but the hard cap means large search spaces silently truncate. Consider dynamically growing the array or computing the actual upper bound from the search space.

### OPT-5: `ArimaResult.__setitem__` silently ignores writes to `'arma'` and `'model'` (LOW)

**File**: `_arima_base.py`, `ArimaResult.__setitem__()` (lines ~276–280)

```python
def __setitem__(self, key, value):
    if key == 'arma':
        return  # read-only computed property
    if key == 'model':
        return  # read-only computed property
```

Writes like `result['arma'] = something` silently succeed with no effect. This can hide bugs in downstream code that expects the write to persist. A more explicit behavior would be to raise `AttributeError` or `TypeError` indicating these keys are read-only.

### OPT-6: Hessian step size is global constant (LOW)

**File**: `_arima_base.py`, `_HESSIAN_STEP_SIZE = 1e-2`

The same step size is used for all parameters regardless of scale. ARIMA AR/MA parameters are typically O(1), but exogenous coefficients or intercepts can be much larger. A parameter-adaptive step size (e.g., `eps_i = max(1e-2, |x_i| * 1e-4)`) would improve Hessian accuracy for mixed-scale problems.

---

## 5. Naming / Consistency Issues

### NAME-1: Internal variables still use old naming conventions

In `predict_arima()` (line ~3 070):
```python
narma = order_spec.n_arma_params
ncxreg = ...
usexreg = ...
```

These `ncxreg`, `usexreg` names are holdovers from the R codebase. Now that the public API uses `exog`, `n_exog`, etc., the internal naming should match.

Similarly in `_initialize_regressor_params()` and `_process_exogenous()` — parameters are called `exog` externally but `xreg` patterns persist internally. The `intercept_idx` logic in `predict_arima()` also has R-style variable naming.

### NAME-2: `observation_variance` field is always 0.0

In `StateSpaceArrays`, `observation_variance` (h) is always set to `0.0` by `initialize_arima_state()`. The ARIMA observation equation has no separate measurement noise (the innovation is part of the state). But `kalman_forecast_core` adds `h` to forecast variances:

```python
variances[t] = h + np.dot(Z, P_curr @ Z)
```

Since h=0 always, this addition is harmless but confusing. Either remove `observation_variance` from the dataclass or document why it exists (forward-compatibility for structural models).

---

## 6. Testing Gaps

### TEST-1: No test coverage for `enforce_stationarity=False`

The Jones transform branch is well-tested, but the `enforce_stationarity=False` path (direct parameter estimation without constrained reparameterization) doesn't appear to have dedicated tests. Edge cases like near-unit-root models where the optimizer finds non-stationary parameters should be covered.

### TEST-2: No integration test for SVD rotation round-trip

`_prepare_arima_config` applies SVD whitening to exogenous regressors, and `_build_arima_result` undoes it. There should be an end-to-end test verifying that the final coefficients and covariance matrix match the non-SVD path for well-conditioned data.

### TEST-3: No test for `_fit_css_ml` CSS→ML fallback

The CSS-ML warm start path (`_fit_css_ml`) contains logic for:
- Detecting non-stationary CSS estimates and falling back to zero initialization
- Skipping CSS when all parameters are fixed

These branches aren't covered by the visible test names.

### TEST-4: `_create_error_model` builds a degenerate `StateSpaceArrays`

`_create_error_model` creates a 1×1 state-space with zero-length arrays. Any code that accesses `.ar_coefs` or `.ma_coefs` on this object gets empty arrays, which could silently break downstream comparisons. There should be guards preventing `predict_arima()` or `kalman_forecast()` from being called on error models.

---

## 7. Summary of Priorities

| ID | Type | Severity | Summary |
|----|------|----------|---------|
| BUG-1 | Bug | HIGH | Operator precedence in ternary makes exog adjustment fragile |
| BUG-2 | Bug | MEDIUM | `>` vs `>=` in Kalman `update_start` |
| BUG-3 | Bug | MEDIUM | Fitted values wrong for ML path (uses standardized residuals) |
| BUG-4 | Bug | MEDIUM | σ² in `_build_arima_result` wrong for ML path |
| BUG-5 | Bug | LOW | `fan=True` silently discards explicit `level` |
| BUG-6 | Bug | LOW | Default prediction intervals never computed in `forecast_arima` |
| BUG-7 | Bug | LOW | σ² recalculation mixes ML/CSS conventions |
| DEAD-1 | Dead code | HIGH | `compute_arima_likelihood_core` (108 lines) never called |
| DEAD-2 | Dead code | — | `kalman_update` only used by DEAD-1 |
| DEAD-3 | Dead code | — | `_numerical_gradient_factory` never called |
| DEAD-4 | Dead code | — | `_validate_pdq` only in tests |
| DEAD-5 | Dead code | — | `if __name__` block (~140 lines) |
| DEAD-6 | Dead code | — | `time_series_convolution` only in tests |
| OPT-1 | Optimization | MEDIUM | Stale V in `_update_state_space` — safe now but fragile |
| OPT-2 | Optimization | MEDIUM | Wasted Kalman filter call in `_fit_css` |
| OPT-3 | Optimization | LOW | Lyapunov P₀ solved every objective evaluation |
| OPT-4 | Optimization | LOW | Fixed 94-row stepwise results array |
| OPT-5 | Optimization | LOW | Silent write ignore in `ArimaResult.__setitem__` |
| OPT-6 | Optimization | LOW | Global Hessian step size for mixed-scale parameters |
