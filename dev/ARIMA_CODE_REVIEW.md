# ARIMA Implementation Code Review

**Date**: December 23, 2025  
**File Reviewed**: `skforecast/stats/arima/_arima_base.py`  
**Reviewer**: Code Quality Analysis

---

## Executive Summary

This document presents a comprehensive review of the ARIMA model implementation. The review identified **2 critical bugs** in the Kalman filter logic, **3 major statistical issues**, and several code quality concerns that should be addressed before production use.

**Severity Levels:**
- 🔴 **Critical**: Bugs that produce incorrect results
- 🟠 **Major**: Statistical correctness issues
- 🟡 **Minor**: Code quality and maintainability concerns

---

## Critical Issues 🔴

### 1. Incorrect Kalman Filter Covariance Update Logic

**Location**: Lines 373-377

**Current Code**:
```python
if l > update_start:  # ❌ Wrong comparison operator
    if d == 0:
        Pnew = predict_covariance_nodiff(P, r, p, q, phi, theta)
    else:
        Pnew = predict_covariance_with_diff(P, r, d, p, q, rd, phi, delta, theta)
    P = Pnew.copy()
```

**Problem**: Using `>` instead of `>=` causes the covariance prediction to skip the first eligible update step. This means the first likelihood update uses an improperly initialized covariance matrix.

**Impact**: 
- Incorrect log-likelihood values
- Biased parameter estimates
- Potentially affects all model selection criteria (AIC, BIC)

**Fix**:
```python
if l >= update_start:  # ✅ Correct
    if d == 0:
        Pnew = predict_covariance_nodiff(P, r, p, q, phi, theta)
    else:
        Pnew = predict_covariance_with_diff(P, r, d, p, q, rd, phi, delta, theta)
    P = Pnew.copy()
```

---

### 2. Broken Kalman Filter Flow - Immediate Covariance Overwrite

**Location**: Lines 354-377

**Current Code**:
```python
if not np.isnan(y[l]):
    a_upd, P_upd, resid, gain, ssq_c, sumlog_c = kalman_update(
        y[l], anew, delta, Pnew, d, r, rd
    )
    a = a_upd
    P = P_upd  # ✅ Updated covariance from observation
    
    # ... likelihood accumulation ...
else:
    a = anew.copy()

# ❌ IMMEDIATELY overwrites P_upd!
if l > update_start:
    if d == 0:
        Pnew = predict_covariance_nodiff(P, r, p, q, phi, theta)
    else:
        Pnew = predict_covariance_with_diff(P, r, d, p, q, rd, phi, delta, theta)
    P = Pnew.copy()  # Overwrites the carefully updated P!
```

**Problem**: The standard Kalman filter cycle is:
1. Predict state and covariance for time t
2. Update with observation at time t
3. **Wait for next iteration** to predict for time t+1

This implementation violates step 3 by immediately predicting again, overwriting the updated covariance. This breaks the mathematical foundation of the Kalman filter.

**Impact**:
- Fundamentally incorrect filtering
- Wrong likelihood values
- Invalid confidence intervals
- All downstream statistics affected

**Fix**: Restructure the loop to follow standard Kalman filter flow:
```python
for l in range(n):
    # Predict covariance for time l
    if l >= update_start:
        if d == 0:
            Pnew = predict_covariance_nodiff(P, r, p, q, phi, theta)
        else:
            Pnew = predict_covariance_with_diff(P, r, d, p, q, rd, phi, delta, theta)
    else:
        Pnew = Pn_init.copy()
    
    # State prediction
    anew = state_prediction(a, p, r, d, rd, phi, delta)
    
    # Update step (if observation available)
    if not np.isnan(y[l]):
        a, P, resid, gain, ssq_c, sumlog_c = kalman_update(
            y[l], anew, delta, Pnew, d, r, rd
        )
        # Accumulate likelihood
        if gain < 1e4:
            nu += 1
            ssq += ssq_c
            sumlog += sumlog_c
        if give_resid:
            rsResid[l] = resid / np.sqrt(gain) if gain > 0 else np.nan
    else:
        a = anew.copy()
        P = Pnew.copy()
        if give_resid:
            rsResid[l] = np.nan
```

---

## Major Statistical Issues 🟠

### 3. Questionable Likelihood Accumulation with Arbitrary Threshold

**Location**: Lines 360-363

**Current Code**:
```python
if gain < 1e4:  # ❌ Arbitrary threshold
    nu += 1
    ssq += ssq_c
    sumlog += sumlog_c
```

**Problem**: The threshold `1e4` is completely arbitrary and excludes observations from the likelihood calculation when the Kalman gain denominator is large. This is not standard statistical practice and introduces bias.

**Why This Is Wrong**:
- No theoretical justification for this threshold
- Excludes valid observations during periods of high uncertainty
- Biases parameter estimates toward periods of low variance
- Violates maximum likelihood theory
- Makes results non-reproducible across scales

**Impact**:
- Biased parameter estimates
- Invalid standard errors
- Wrong confidence intervals
- Incorrect model selection criteria

**Recommendation**: 
1. Remove the threshold entirely, OR
2. Use proper numerical stability checks (e.g., `gain > eps` where `eps = np.finfo(float).eps`)
3. If extreme gains occur, this indicates deeper model misspecification

**Better Code**:
```python
if not np.isnan(y[l]):
    a, P, resid, gain, ssq_c, sumlog_c = kalman_update(...)
    
    if gain > 0:  # Only check for numerical validity
        nu += 1
        ssq += ssq_c
        sumlog += sumlog_c
        if give_resid:
            rsResid[l] = resid / np.sqrt(gain)
    else:
        warnings.warn(f"Non-positive gain at observation {l}")
```

---

### 4. Suspicious Seasonal MA Coefficient Expansion

**Location**: Lines 1235-1239

**Current Code**:
```python
# Seasonal MA
for j in range(msq):
    theta[(j + 1) * ns - 1] += params[mp + mq + msp + j]
    for i in range(mq):
        theta[(j + 1) * ns + i] += params[mp + i] * params[mp + mq + msp + j]  # ❌ Using +=
```

**Compare with AR expansion**:
```python
# Seasonal AR
for j in range(msp):
    phi[(j + 1) * ns - 1] += params[mp + mq + j]
    for i in range(mp):
        phi[(j + 1) * ns + i] -= params[i] * params[mp + mq + j]  # ✅ Using -=
```

**Problem**: The AR expansion uses `-=` (subtraction) for the multiplicative interaction term, but MA uses `+=` (addition). In standard ARIMA notation, both should follow the same multiplicative pattern.

**Mathematical Background**:
- Non-seasonal MA: (1 - θ₁B - θ₂B² - ...)
- Seasonal MA: (1 - Θ₁Bˢ - Θ₂B²ˢ - ...)
- Multiplicative: (1 - θ₁B - ...)(1 - Θ₁Bˢ - ...)
- This expands to: 1 - θ₁B - ... - Θ₁Bˢ + θ₁Θ₁Bˢ⁺¹ + ...

The `+` in the expansion suggests this should use `-=` to account for the sign convention.

**Recommendation**: 
1. Verify against R's `arima()` function with seasonal MA models
2. Test with known seasonal MA processes
3. Check against `statsmodels.tsa.arima.ARIMA`

**Likely Fix**:
```python
for i in range(mq):
    theta[(j + 1) * ns + i] -= params[mp + i] * params[mp + mq + msp + j]  # Changed to -=
```

---

### 5. Ambiguous Diffuse Initialization

**Location**: Lines 1399-1416

**Current Code**:
```python
# Initialize Pn
if r > 1:
    if SSinit == "Gardner1980":
        Pn[:r, :r] = compute_q0_covariance_matrix(phi, theta)
    else:
        Pn[:r, :r] = compute_q0_bis_covariance_matrix(phi, theta, tol)
else:
    if len(phi) == 0:
        Pn[0, 0] = 1.0 / (1.0 - np.sum(theta) ** 2)
    else:
        Pn[0, 0] = 1.0 / (1.0 - phi[0] ** 2)

# Diffuse prior for differencing states
if d > 0:
    P[r:, r:] = kappa * np.eye(d)  # ❌ Setting P instead of Pn?
```

**Problems**:
1. The distinction between `P` and `Pn` is unclear
2. Diffuse states are set in `P` while stationary states are in `Pn`
3. No documentation explaining the dual covariance matrices
4. The formula for r=1 case may be incorrect for some parameter combinations

**Impact**: Incorrect initial state uncertainty, affecting early forecasts and filtered estimates.

**Recommendation**: 
- Follow Durbin & Koopman (2012) "Time Series Analysis by State Space Methods" for proper diffuse initialization
- Document the role of P vs Pn clearly
- Consider implementing exact diffuse initialization

---

## Code Quality Issues 🟡

### 6. Hardcoded Magic Numbers

**Location**: Line 2029

**Code**:
```python
parscale = np.concatenate([parscale, 10 * ses])  # ❌ Why 10?
```

**Problem**: Magic number `10` with no justification.

**Fix**: Use 1.0 or make it a parameter:
```python
XREG_SCALE_FACTOR = 1.0  # Document why this value is chosen
parscale = np.concatenate([parscale, XREG_SCALE_FACTOR * ses])
```

---

### 7. Dangerous Silent MA Inversion

**Location**: Lines 1525-1567 (`ma_invert` function)

**Problem**: The function automatically "fixes" non-invertible MA processes by reflecting roots inside the unit circle. This:
- Changes fundamental model properties
- Makes forecasts unreliable
- Hides model misspecification
- Violates the principle of least surprise

**Example Impact**:
```python
# User specifies MA(1) with θ = 0.95
# Function silently changes to θ ≈ 1.05 (inverted)
# Forecasts and interpretation are now wrong
```

**Recommendation**: Fail with informative error instead:
```python
def ma_check(ma: np.ndarray) -> bool:
    """Check if MA polynomial is invertible."""
    if len(ma) == 0:
        return True
    coeffs = np.concatenate([[1.0], ma])
    rts = np.roots(coeffs[::-1])
    return np.all(np.abs(rts) > 1.0)

# In arima():
if not ma_check(theta):
    raise ValueError(
        "Non-invertible MA coefficients detected. "
        "Consider reducing MA order or using different initial values."
    )
```

---

### 8. Inefficient Hessian Computation

**Location**: Lines 2081-2125

**Problem**: Nested finite differences are O(n³) in function evaluations.

**Current Approach**:
```python
def optim_hessian(func, x, eps=None):
    # For each parameter i:
    #   Compute gradient at x + eps*e_i  (n function evals)
    #   Compute gradient at x - eps*e_i  (n function evals)
    # Total: 2n² function evaluations
```

**Better Approaches**:
1. Use `scipy.optimize.approx_fprime` with forward differences (n² evals)
2. Use `numdifftools.Hessian` (optimized algorithms)
3. Implement analytical Hessian (best, but complex)

**Recommendation**:
```python
try:
    import numdifftools as nd
    def optim_hessian(func, x):
        return nd.Hessian(func)(x)
except ImportError:
    # Fallback to current implementation with warning
    warnings.warn("Install numdifftools for faster Hessian computation")
```

---

### 9. CSS Residual Computation Issue

**Location**: Lines 1293-1308

**Code**:
```python
for l in range(ncond, n):
    tmp = w[l]
    
    # AR contribution
    for j in range(p):
        if l - j - 1 >= 0:
            tmp -= phi[j] * w[l - j - 1]
    
    # MA contribution
    jmax = min(l - ncond, q)  # ❌ This grows with l
    for j in range(jmax):
        if l - j - 1 >= 0:
            tmp -= theta[j] * resid[l - j - 1]
```

**Problem**: `jmax = min(l - ncond, q)` should simply be `jmax = min(l, q)` or just `q` after sufficient observations are available. The current formulation may use fewer MA terms than appropriate.

**Fix**:
```python
jmax = min(l - ncond, q) if l < ncond + q else q
```

---

### 10. Missing Input Validation

**Multiple Locations**

**Missing Checks**:
1. No validation for multicollinearity in `xreg` (line 1950+)
2. No condition number checks after SVD (line 1959)
3. No warning when regression rank is deficient (line 1997)
4. No checks for NA/Inf in parameters
5. No validation that p, d, q are non-negative before array allocation

**Recommendation**: Add comprehensive input validation:
```python
def validate_arima_inputs(x, order, seasonal, xreg):
    """Validate all ARIMA inputs before fitting."""
    p, d, q = order
    P, D, Q = seasonal
    
    # Check orders
    if any(v < 0 for v in [p, d, q, P, D, Q]):
        raise ValueError("ARIMA orders must be non-negative")
    
    if p + P > 100:
        raise ValueError("Total AR order too large (>100)")
    
    # Check data
    if len(x) < p + d + (P + D) * m + 10:
        raise ValueError(f"Insufficient data: need at least {p + d + (P + D) * m + 10} observations")
    
    # Check xreg
    if xreg is not None:
        if np.linalg.cond(xreg.T @ xreg) > 1e10:
            warnings.warn("xreg matrix is nearly singular (multicollinearity)")
    
    return True
```

---

### 11. Incomplete Function Implementation

**Location**: Line 1752 (`na_omit_pair`)

**Code**:
```python
def na_omit_pair(x: np.ndarray, xreg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle missing values by converting to NaN.
    """
    x = np.asarray(x, dtype=np.float64)
    xreg = np.asarray(xreg, dtype=np.float64)
    return x, xreg  # ❌ Doesn't actually remove NaN!
```

**Problem**: Function claims to "handle missing values" but just converts types and returns unchanged arrays.

**Fix**: Either remove the function or implement proper NA handling:
```python
def na_omit_pair(x: np.ndarray, xreg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove rows where either x or any xreg column has NaN."""
    x = np.asarray(x, dtype=np.float64)
    xreg = np.asarray(xreg, dtype=np.float64)
    
    if xreg.size == 0:
        return x[~np.isnan(x)], xreg
    
    mask = ~np.isnan(x) & ~np.any(np.isnan(xreg), axis=1)
    return x[mask], xreg[mask]
```

---

### 12. Missing Information Criteria

**Location**: Lines 2547-2549

**Code**:
```python
return {
    'aic': aic,
    'bic': None,  # ❌ Not implemented
    'aicc': None,  # ❌ Not implemented
    # ...
}
```

**Problem**: BIC and AICc are standard model selection criteria but not computed.

**Fix**:
```python
npar = np.sum(mask) + 1  # +1 for sigma^2
bic = value + npar * np.log(n_used)
aicc = aic + 2 * npar * (npar + 1) / max(n_used - npar - 1, 1)

return {
    'aic': aic,
    'bic': bic,
    'aicc': aicc,
    # ...
}
```

---

## Testing Recommendations

### Critical Tests Needed

1. **Kalman Filter Validation**
   ```python
   # Compare against known implementations
   def test_kalman_filter_vs_statsmodels():
       from statsmodels.tsa.arima.model import ARIMA
       # Test that likelihood values match
   ```

2. **Seasonal Expansion Test**
   ```python
   def test_seasonal_expansion():
       # ARIMA(1,0,1)(1,0,1)[12]
       # Verify coefficients against R's arima()
   ```

3. **Numerical Stability Tests**
   ```python
   def test_extreme_parameters():
       # Near-unit-root AR
       # Near-cancellation MA
       # High-variance data
   ```

4. **Forecast Accuracy Tests**
   ```python
   def test_forecasts():
       # Compare against analytical solutions for ARMA(1,1)
       # Check forecast variance accumulation
   ```

---

## Comparison with Reference Implementations

### R's `arima()` vs This Implementation

| Feature | R | This Implementation | Status |
|---------|---|---------------------|---------|
| Kalman filter | Correct | **Broken** | 🔴 Fix Required |
| CSS estimation | Standard | **Potentially buggy** | 🟠 Verify |
| MA inversion | Manual | **Automatic (dangerous)** | 🟡 Change behavior |
| Seasonal models | Tested | **Unverified** | 🟠 Test needed |
| Hessian | Optimized | **Slow** | 🟡 Improve |

### Statsmodels vs This Implementation

| Feature | Statsmodels | This Implementation | Status |
|---------|------------|---------------------|---------|
| Diffuse init | Proper | **Unclear** | 🟠 Review |
| Missing data | Handled | **Partial** | 🟡 Improve |
| Diagnostics | Complete | **Missing BIC/AICc** | 🟡 Add |
| Documentation | Extensive | **Minimal** | 🟡 Improve |

---

## Immediate Action Items

### Priority 1 (Critical - Fix Before Any Use)
- [ ] Fix Kalman filter covariance update logic (Issue #1 & #2)
- [ ] Remove or justify arbitrary likelihood threshold (Issue #3)
- [ ] Test against reference implementations

### Priority 2 (Major - Fix Before Production)
- [ ] Verify seasonal MA expansion (Issue #4)
- [ ] Clarify diffuse initialization (Issue #5)
- [ ] Implement proper input validation (Issue #10)
- [ ] Add comprehensive unit tests

### Priority 3 (Code Quality - Improve Gradually)
- [ ] Remove magic numbers (Issue #6)
- [ ] Change MA inversion behavior (Issue #7)
- [ ] Optimize Hessian computation (Issue #8)
- [ ] Fix CSS residual computation (Issue #9)
- [ ] Complete BIC/AICc computation (Issue #12)
- [ ] Fix or remove `na_omit_pair` (Issue #11)

---

## Long-Term Recommendations

1. **Comprehensive Test Suite**: Create tests comparing against:
   - R's `forecast::Arima()`
   - Python's `statsmodels.tsa.arima.ARIMA`
   - Known analytical results for simple models

2. **Documentation**: Add detailed mathematical documentation:
   - State-space representation
   - Kalman filter equations
   - Parameter transformations
   - Initialization procedures

3. **Numerical Stability**: Implement robust numerical methods:
   - Use Cholesky decomposition where possible
   - Add condition number checks
   - Implement matrix square root for covariance
   - Use log-likelihood to avoid underflow

4. **Code Organization**: Consider refactoring:
   - Separate Kalman filter into its own module
   - Create a proper state-space class
   - Use dataclasses for model results
   - Add type hints throughout

5. **Performance**: Profile and optimize:
   - Vectorize where possible
   - Consider Cython for hot loops
   - Cache repeated computations
   - Parallelize parameter search

---

## References

1. Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods* (2nd ed.). Oxford University Press.

2. Harvey, A. C. (1990). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.

3. Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting* (3rd ed.). Springer.

4. R Core Team. (2023). *R: A Language and Environment for Statistical Computing*. https://www.R-project.org/

5. Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with python. *Proceedings of the 9th Python in Science Conference*.

---

## Conclusion

This ARIMA implementation has significant issues that must be addressed before production use. The critical Kalman filter bugs will produce incorrect results for all models. The statistical issues raise questions about the correctness of seasonal models and parameter estimation.

**Recommendation**: Do not use this implementation in its current state. Fix critical issues #1-3 immediately, then perform extensive validation against reference implementations before any production use.

**Estimated Effort**:
- Critical fixes: 2-3 days
- Major fixes + testing: 1-2 weeks  
- Full refactoring: 3-4 weeks

---

*This review was conducted on December 23, 2025. Some issues may have been addressed in subsequent commits.*
