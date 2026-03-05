# ARIMA Implementation Code Review — Fix Plan

**Branch:** `feature_refactor_arima`  
**Files reviewed:**
- `skforecast/stats/arima/_arima_base.py`
- `skforecast/stats/arima/_auto_arima.py`
- `skforecast/stats/_arima.py`

---

## Bugs

### BUG 1 — Discarded Kalman filter call in `_fit_css`
**File:** `skforecast/stats/arima/_arima_base.py` ~L2573  
**Status:** ❌ Open

The full O(n·r²) Kalman pass is run inside `_fit_css` but its return value is silently discarded. The actual `sigma2` and `resid` are computed on the very next line by `compute_css_residuals`.

```python
# Line ~2573: return value ignored — DEAD COMPUTATION
compute_arima_likelihood(adjusted_series, state_space, update_start=0, give_resid=True)
# These are the values actually used:
sigma2, resid = compute_css_residuals(...)
```

**Fix:** Delete the `compute_arima_likelihood(...)` line entirely.

---

### BUG 2 — Off-by-one in Kalman covariance propagation
**File:** `skforecast/stats/arima/_arima_base.py` ~L1329 (`_arima_kalman_core`) and ~L1462 (`compute_arima_likelihood_core`)  
**Status:** ❌ Open

The covariance prediction step `Pnew = T P Tᵀ + V` is gated by `if t > update_start:`.  
When `update_start=0` (the value used by `compute_arima_likelihood`):
- t=0: update step runs; prediction step **skipped** (0 > 0 is false) ✓
- t=1: update step uses stale `Pₙ_init` (κ=1e6 for non-stationary) instead of propagated covariance ✗

For non-stationary/diffuse components this is a significant error.

**Fix:** Change `if t > update_start:` → `if t >= update_start:` in **both** `_arima_kalman_core` and `compute_arima_likelihood_core`.

---

### BUG 3 — Fitted values mix standardized and raw innovations
**File:** `skforecast/stats/arima/_arima_base.py` ~L2896 (`_build_arima_result`)  
**Status:** ❌ Open

```python
fitted_vals = c.y - fit.resid
```

For the ML path, `fit.resid` comes from `_arima_kalman_core` which stores **standardized** Kalman innovations `vₜ/√Fₜ` (not raw innovations `vₜ`). Correct fitted values require `ŷₜ = yₜ − vₜ`. The CSS path stores raw residuals and is correct. This makes `fitted_values_` and `in_sample_residuals_` wrong for ML-fitted models and incomparable between methods.

**Fix:** Also return the `Fₜ` array from `_arima_kalman_core`, then compute raw innovations as `std_resid[t] * sqrt(F[t])` in `_build_arima_result`, or add a separate raw-innovation array to the Kalman core output.

---

### BUG 4 — `fan=True` silently overwrites explicit `level` in `forecast_arima`
**File:** `skforecast/stats/arima/_auto_arima.py` ~L1711  
**Status:** ❌ Open

```python
if level is not None:
    if fan:
        levels = np.arange(51, 100, 3).tolist()  # silently discards user-supplied level
    else:
        levels = list(level)
```

When both `level=[80, 95]` and `fan=True` are passed, the explicit `level` is silently ignored.

**Fix:**
```python
if fan and level is not None:
    raise ValueError("Cannot specify both `level` and `fan=True`.")
if fan:
    levels = np.arange(51, 100, 3).tolist()
elif level is not None:
    levels = list(level)
```

---

### BUG 5 — Inconsistent σ² denominator in `fit_custom_arima`
**File:** `skforecast/stats/arima/_auto_arima.py` ~L456  
**Status:** ❌ Open

```python
resid_valid = fit['residuals'][~np.isnan(fit['residuals'])]
if len(resid_valid) > npar - 1:
    fit['sigma2'] = np.sum(resid_valid**2) / (nstar_adj - npar + 1)
```

For the ML path, `fit['residuals']` contains standardized Kalman innovations `vₜ/√Fₜ`, so `Σresid² = ssq`. Dividing by `nstar_adj - npar + 1` applies a bias-corrected OLS-style denominator on top of ML-standardized residuals, biasing σ² downward (and consequently biasing prediction intervals) when `npar > 1`.

**Fix:** Trust the σ² already set by the fitting routine and only apply the adjustment for the CSS method, guarded by `method == "CSS"`.

---

## Dead Code

### DEAD 1 — `compute_arima_likelihood_core` (~108 lines)
**File:** `skforecast/stats/arima/_arima_base.py` ~L1374–L1481  
**Status:** ❌ Open

A complete `@njit` generic-matrix Kalman filter. Zero production call sites (confirmed by grep). All production code uses `_arima_kalman_core`.

**Fix:** Delete the function.

---

### DEAD 2 — `kalman_update` only called by dead code
**File:** `skforecast/stats/arima/_arima_base.py` ~L1130–L1183  
**Status:** ❌ Open

Its only caller is `compute_arima_likelihood_core` (DEAD 1). Has no other production call sites.

**Fix:** Delete alongside `compute_arima_likelihood_core`. Remove import in tests if applicable.

---

### DEAD 3 — `_numerical_gradient_factory` never called
**File:** `skforecast/stats/arima/_arima_base.py` ~L2252–L2271  
**Status:** ❌ Open

Zero call sites confirmed by grep. Optimizer calls never pass a `jac=` argument.

**Fix:** Delete the function.

---

### DEAD 4 — `if __name__ == "__main__":` block (~135 lines)
**File:** `skforecast/stats/arima/_auto_arima.py` ~L1787–L1886  
**Status:** ❌ Open

A script-style integration test suite embedded in a library module. Unreachable from any import path. References functions like `arima_rjh`, `time_index`, `prepend_drift` which may also warrant review.

**Fix:** Move any useful regression cases to `skforecast/stats/tests/` and delete the block.

---

## Inefficiencies

### INEFF 1 — Redundant `_ensure_float64_pair` calls
**File:** `skforecast/stats/arima/_arima_base.py` ~L2070–L2110 (`_initialize_regressor_params`)  
**Status:** ❌ Open

`_ensure_float64_pair` is called once at function entry (correct), then called **three more times** on arrays already guaranteed to be `float64` — after `diff()` calls that preserve dtype, and again in an OLS branch.

**Fix:** Keep only the first call; remove the three subsequent ones.

---

### INEFF 2 — Duplicate CSS objective closure in `_fit_css_ml`
**File:** `skforecast/stats/arima/_arima_base.py` ~L2800–L2860  
**Status:** ❌ Open

`_fit_css_ml` defines its own `_css_objective` closure that is functionally identical (byte-for-byte) to the one inside `_fit_css`.

**Fix:** Extract a shared factory `_make_css_objective(config) -> Callable` that both `_fit_css` and `_fit_css_ml` use.

---

## Priority Order

| Priority | Item | Impact |
|----------|------|--------|
| 1 | BUG 3 — Fitted values wrong for ML path | High: wrong residuals/fitted values exposed to users |
| 2 | BUG 2 — Off-by-one Kalman covariance | High: affects likelihood accuracy for non-stationary models |
| 3 | BUG 5 — σ² denominator bias | Medium: biases prediction intervals |
| 4 | BUG 1 — Discarded Kalman call | Medium: wastes O(n·r²) compute on every CSS fit |
| 5 | BUG 4 — `fan=True` overwrites `level` | Low: silent wrong behavior for edge case |
| 6 | DEAD 1+2 — `compute_arima_likelihood_core` + `kalman_update` | Low: code bloat, ~170 lines |
| 7 | DEAD 3 — `_numerical_gradient_factory` | Low: ~20 lines |
| 8 | DEAD 4 — `__main__` block | Low: ~135 lines, minor maintenance risk |
| 9 | INEFF 1 — Redundant float64 casts | Very low: O(n) copies, minor |
| 10 | INEFF 2 — Duplicate CSS objective | Very low: code duplication |
