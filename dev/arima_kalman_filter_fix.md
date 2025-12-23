# ARIMA Kalman Filter Implementation Fix

**Date**: December 23, 2025  
**Component**: `skforecast/stats/arima/_arima_base.py`  
**Issue**: ARIMA model not learning from training data - predictions always zero

---

## Executive Summary

Two critical bugs were discovered in the ARIMA Kalman filter implementation that prevented the model from learning:

1. **State Vector Not Updated**: The filtered state vector remained at initialization zeros after fitting
2. **Wrong Covariance Matrix**: Forecasting used the initial prior covariance instead of the filtered covariance

These issues caused predictions to be zero and forecast uncertainties to be extremely large (~1,000,000).

---

## Issue 1: State Vector Not Updated After Fitting

### Problem Description

After fitting the ARIMA model, the state vector `a` remained at its initialization value of zeros `[0, 0, 0, ...]`. This meant that forecasts were generated from a zero-state, completely ignoring all information learned from the training data.

**Symptoms**:
- All predictions were exactly 0.0
- Model appeared not to learn from training data
- Reference implementations (aeon, nixtla, statsmodels) produced reasonable predictions (~450-500) while skforecast produced zeros

### Root Cause

The `compute_arima_likelihood_core` function performed Kalman filtering through all training observations, updating the state vector `a` at each iteration:

```python
for l in range(n):
    anew = state_prediction(a, p, r, d, rd, phi, delta)
    
    if not np.isnan(y[l]):
        a_upd, P_upd, resid, gain, ssq_c, sumlog_c = kalman_update(...)
        a = a_upd  # State updated here
        P = P_upd
```

However, the function **did not return** the final filtered state:

```python
# Original (INCORRECT)
return stats, rsResid
```

The final filtered state `a` was computed but discarded, and the model retained only the initial zero-state.

### Theoretical Background

In Kalman filtering for ARIMA models:

- **Filtering**: Process observations sequentially to produce `a(t|t)` - the state estimate at time t given observations up to time t
- **Final filtered state**: After processing all T training observations, we obtain `a(T|T)` which incorporates all information from the training data
- **Forecasting requirement**: Forecasts must start from `a(T|T)`, not from the initial prior `a(0|0) = 0`

The filtered state `a(T|T)` contains:
1. The current level and trend of the series
2. Recent MA innovations
3. Differencing states for integrated models

Starting forecasts from zeros means predicting as if we had observed nothing.

### Solution

Modified the return signature to include final states:

```python
# Fixed
def compute_arima_likelihood_core(
    ...
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    stats : np.ndarray
    residuals : np.ndarray
    a_final : np.ndarray        # NEW: Final filtered state
    P_final : np.ndarray        # NEW: Final filtered covariance
    """
    ...
    return stats, rsResid, a, P
```

Updated `compute_arima_likelihood` to propagate these values:

```python
stats, residuals, a_final, P_final = compute_arima_likelihood_core(...)

result = {
    'ssq': stats[0],
    'sumlog': stats[1],
    'nu': int(stats[2]),
    'a': a_final,      # NEW
    'P': P_final       # NEW
}
```

Updated the `arima` function to store the final filtered state in the model:

```python
val = compute_arima_likelihood(x_work, mod, update_start=0, give_resid=True)
sigma2 = val['ssq'] / n_used
resid = val['resid']

# NEW: Update model state with final filtered state
mod['a'] = val['a']
mod['P'] = val['P']
```

### Files Modified

- `skforecast/stats/arima/_arima_base.py`:
  - `compute_arima_likelihood_core()`: Changed return signature
  - `compute_arima_likelihood()`: Added state extraction and return
  - `arima()`: Added state storage after fitting

---

## Issue 2: Wrong Covariance Matrix Used for Forecasting

### Problem Description

Even after fixing Issue 1, the forecast variances were unrealistically large (~1,000,002 instead of ~2-5). Investigation revealed that the forecast function was using the wrong covariance matrix.

**Symptoms**:
- Forecast variances of ~1,000,000+
- Prediction intervals extremely wide
- Variance increased by exactly 1,000,000 (the diffuse prior value)

### Root Cause

The `kalman_forecast_core` function initialized the forecast covariance from `Pn`:

```python
# Original (INCORRECT)
def kalman_forecast_core(...):
    a_curr = a.copy()
    P_curr = Pn.copy()  # Using Pn - the initial prior!
```

Where `Pn` is the **initial prior covariance matrix** before observing any data, containing:
- Stationary covariance for ARMA states (from Q0 calculation)
- **Diffuse prior (kappa=1e6) for differencing states**

Example of `Pn` after initialization:
```
[[1.744,     0.873,     0.0      ]
 [0.873,     0.761,     0.0      ]
 [0.0,       0.0,       1000000.0]]  <- Diffuse prior
```

This diffuse prior represents infinite uncertainty about the differencing state, appropriate before seeing any data but **completely wrong** after filtering through all training observations.

### Theoretical Background

In the Kalman filter loop during fitting:

```python
for l in range(n):
    # 1. Predict: a(t|t-1), P(t|t-1)
    anew = state_prediction(a, ...)
    
    # 2. Update: a(t|t), P(t|t) using observation y(t)
    a, P = kalman_update(y[l], anew, Pnew, ...)
    
    # 3. Prepare for next iteration: P(t+1|t)
    if l > update_start:
        Pnew = predict_covariance(P, ...)
        P = Pnew.copy()
```

At the end of the loop (l = n-1, the last observation):
1. We obtain `a(n|n)` - filtered state after the last observation
2. The condition `if l > update_start` executes, computing `P(n+1|n)`
3. **P contains `P(n+1|n)` - the one-step-ahead predicted covariance**

For forecasting:
- First forecast at t=n+1: Start from `a(n|n)` with uncertainty `P(n+1|n)` ✓ (this is P)
- Second forecast at t=n+2: Predict from `a(n+1|n)` with uncertainty predicted from `P(n+1|n)` ✓

The matrix `P` returned from likelihood calculation is **exactly** what we need for forecasting, while `Pn` represents the uncertainty **before** seeing any data.

Example of `P` after filtering:
```
[[1.000,     0.873,    -0.000]
 [0.873,     0.761,     0.000]
 [0.024,     0.231,    -0.024]]  <- Updated from data!
```

The diffuse prior has been completely replaced by information from the training data.

### Solution

Changed `kalman_forecast_core` to use the filtered covariance `P`:

```python
# Fixed
def kalman_forecast_core(...):
    a_curr = a.copy()
    P_curr = P.copy()  # Use P - the filtered/predicted covariance
```

Updated `kalman_forecast` to pass `P` instead of `Pn`:

```python
# Original (INCORRECT)
def kalman_forecast(n_ahead, mod, update=False):
    ...
    Pn = mod['Pn'].astype(np.float64)
    
    forecasts, variances, a_final, P_final = kalman_forecast_core(
        n_ahead, phi, theta, delta, Z, a, P, Pn, h
        #                                      ^^ Wrong!
    )

# Fixed
def kalman_forecast(n_ahead, mod, update=False):
    ...
    # Removed: Pn = mod['Pn'].astype(np.float64)
    
    forecasts, variances, a_final, P_final = kalman_forecast_core(
        n_ahead, phi, theta, delta, Z, a, P, P, h
        #                                     ^^ Correct!
    )
```

### Files Modified

- `skforecast/stats/arima/_arima_base.py`:
  - `kalman_forecast_core()`: Changed initialization from `Pn` to `P`
  - `kalman_forecast()`: Removed `Pn` loading, pass `P` twice to core function

---

## Verification and Results

### Before Fix

```python
estimator = Arima(order=(1, 1, 1), m=1)
estimator.fit(y=data['y'].to_numpy())

print(estimator.model_['model']['a'])  # [0. 0. 0.]
predictions = estimator.predict(steps=12)
print(predictions[:3])  # [0. 0. 0.]
```

### After Fix

```python
estimator = Arima(order=(1, 1, 1), m=1)
estimator.fit(y=data['y'].to_numpy())

print(estimator.model_['model']['a'])  # [42.0, 64.96, 390.0]
predictions = estimator.predict(steps=12)
print(predictions[:3])  # [476.80, 455.30, 465.62]
```

### Comparison with Reference Implementations

| Step | skforecast | aeon      | nixtla    | statsmodels |
|------|------------|-----------|-----------|-------------|
| 1    | 476.800    | 476.988   | 476.988   | 475.735     |
| 2    | 455.296    | 455.268   | 455.268   | 454.996     |
| 3    | 465.618    | 465.754   | 465.754   | 464.830     |

Predictions now match reference implementations within numerical tolerance (< 1%).

### Forecast Variances

**Before Fix**:
```
[1000002.76, 1000003.97, 1000005.22, ...]  # Dominated by diffuse prior
```

**After Fix**:
```
[2.20, 3.46, 4.72, 6.01, 7.32, ...]  # Reasonable uncertainty growth
```

Forecast standard errors are now in the correct range (~1.5-3.0 for early horizons).

---

## Theoretical Implications

### Kalman Filter State Representation

For ARIMA(p,d,q) models, the state vector contains:
```
a = [y_t, y_{t-1}, ..., y_{t-r+1}, ∇^1 y_t, ∇^2 y_t, ..., ∇^d y_t]
```

Where:
- First r components: ARMA state (innovations form)
- Last d components: Differencing states

For ARIMA(1,1,1) with y[143]=432:
- `a[0] = 42.0`: Current innovation/level
- `a[1] = 64.96`: Lagged innovation (MA term)
- `a[2] = 390.0`: First difference state

### Covariance Matrix Evolution

The covariance matrix P tracks uncertainty:

**Initial (Pn)**:
- ARMA components: Stationary covariance from ARMA structure
- Differencing components: Diffuse (infinite uncertainty)

**After Filtering (P)**:
- ARMA components: Updated by observations, typically reduced
- Differencing components: No longer diffuse, informed by data
- Off-diagonal terms: Capture correlations between states

### Forecast Distribution

At horizon h, the forecast distribution is:

```
y_{T+h} ~ N(μ_h, σ²_h)

where:
μ_h = Z' a(T+h|T)           # From filtered state
σ²_h = σ² * (Z' P(T+h|T) Z) # From filtered covariance
```

Using the initial prior Pn instead of the filtered P would give:
```
σ²_h ≈ σ² * kappa = σ² * 1e6  # Nonsensical
```

---

## Impact Assessment

### Functional Impact

**Critical**: The model was completely non-functional for forecasting. All predictions were zero regardless of training data.

### Performance Impact

**Minimal**: The fixes add negligible computational overhead:
- Returning two additional arrays from likelihood calculation
- Using correct covariance matrix (no additional computation)

### Backward Compatibility

**Breaking**: Any existing pickled ARIMA models will need to be retrained with the fixed code. The model state structure has changed to include filtered states.

---

## Testing Recommendations

1. **Unit tests** for `compute_arima_likelihood_core`:
   - Verify final state is non-zero after filtering
   - Check state dimensions match model specification

2. **Integration tests** for forecasting:
   - Compare predictions with reference implementations
   - Verify forecast variances are reasonable (not ~1e6)
   - Test with various ARIMA specifications

3. **Regression tests**:
   - Test cases from multiple datasets
   - Compare with aeon, nixtla, statsmodels
   - Tolerance: < 1% difference in point forecasts

---

## Code Changes Summary

### Modified Functions

1. **`compute_arima_likelihood_core`** (line ~289)
   - Changed return type: `Tuple[ndarray, ndarray]` → `Tuple[ndarray, ndarray, ndarray, ndarray]`
   - Added: Return final `a` and `P`

2. **`compute_arima_likelihood`** (line ~384)
   - Added state extraction from core function
   - Added 'a' and 'P' to result dictionary

3. **`arima`** (line ~2511)
   - Added state update: `mod['a'] = val['a']`, `mod['P'] = val['P']`

4. **`kalman_forecast_core`** (line ~1620)
   - Changed: `P_curr = Pn.copy()` → `P_curr = P.copy()`

5. **`kalman_forecast`** (line ~1674)
   - Removed: Loading of `Pn` from model
   - Changed: Pass `P` instead of `Pn` to core function

### Lines of Code Changed

- Total additions: ~10 lines
- Total modifications: ~5 lines
- Total deletions: ~1 line

---

## Lessons Learned

1. **State Management**: In Kalman filters, carefully track which state (prior, predicted, filtered) is being used at each stage

2. **Return Values**: Critical intermediate values (like filtered states) must be propagated through the call stack

3. **Covariance Matrices**: Distinguish between:
   - Prior covariance (before data)
   - Filtered covariance (after observations)
   - Predicted covariance (for next time step)

4. **Testing**: Compare with multiple reference implementations to catch subtle bugs

---

## References

- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods* (2nd ed.)
- Harvey, A. C. (1990). *Forecasting, Structural Time Series Models and the Kalman Filter*
- R source code: `stats::arima()` and `stats:::KalmanForecast()`
- Python implementations: statsmodels, aeon, nixtla/statsforecast

---

## Authors

- **Bug Discovery**: Code review and testing against reference implementations
- **Fix Implementation**: December 23, 2025
- **Theoretical Validation**: Verified against Kalman filter theory for ARIMA models
