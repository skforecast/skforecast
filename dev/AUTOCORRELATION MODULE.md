# Refactoring: `skforecast.stats.autocorrelation` submodule

**Branch:** `0.22.x`  
**Date:** 2026-05-03

---

## Summary

A new `skforecast/stats/autocorrelation/` submodule was created containing fast,
self-contained implementations of `acf`, `pacf`, and `calculate_lag_autocorrelation`.
The function `calculate_lag_autocorrelation` was simultaneously removed from
`skforecast.plot`, where it had historically lived alongside plotting utilities.

---

## New files

| File | Description |
|------|-------------|
| `skforecast/stats/autocorrelation/__init__.py` | Exports `acf`, `pacf`, `calculate_lag_autocorrelation` |
| `skforecast/stats/autocorrelation/_autocorrelation.py` | All four functions: `_fft_acf`, `acf`, `pacf`, `calculate_lag_autocorrelation` |
| `skforecast/stats/autocorrelation/tests/__init__.py` | Empty marker |
| `skforecast/stats/autocorrelation/tests/test_acf.py` | 13 tests for `acf` |
| `skforecast/stats/autocorrelation/tests/test_pacf.py` | 14 tests for `pacf` |
| `skforecast/stats/autocorrelation/tests/test_calculate_lag_autocorrelation.py` | 6 tests for `calculate_lag_autocorrelation` |

---

## Modified files

### `skforecast/stats/__init__.py`
- Added imports: `acf`, `pacf`, `calculate_lag_autocorrelation` from `.autocorrelation`
- Added `autocorrelation` to `__all__`

### `skforecast/plot/plot.py`
- Removed `calculate_lag_autocorrelation` function entirely (moved to `stats.autocorrelation`)
- Removed all imports related to it

### `skforecast/plot/__init__.py`
- Removed `calculate_lag_autocorrelation` from exports

### `docs/api/stats.md`
- Added three new mkdocstrings directives:
  ```
  ::: skforecast.stats.autocorrelation._autocorrelation.acf
  ::: skforecast.stats.autocorrelation._autocorrelation.pacf
  ::: skforecast.stats.autocorrelation._autocorrelation.calculate_lag_autocorrelation
  ```

### `docs/api/plot.md`
- Removed the `calculate_lag_autocorrelation` directive (function no longer in `skforecast.plot`)

---

## Implementation details

### `_fft_acf(x_centered, n, nlags)` — internal helper
- Biased ACF via FFT (denominator `n`), O(N log N)
- Padding size: `scipy.fft.next_fast_len(2 * n - 1)` (replaced manual `2**ceil(log2(...))` formula)
- Returns all-ones for constant series (zero variance guard)

### `acf(x, nlags, adjusted, alpha)`
- Wraps `_fft_acf`; supports `adjusted=True` for unbiased estimator
- Confidence intervals: Bartlett formula, `Var(ρ̂_k) = (1/n)(1 + 2·Σ ρ̂_j²)`
- Default `nlags`: `min(int(10·log10(n)), n - 1)` — statsmodels convention

### `pacf(x, nlags, alpha)`
- Stage 1: biased ACF via `_fft_acf`
- Stage 2: Levinson-Durbin recursion, O(p²) with O(p) memory (two rolling vectors, no matrix)
- Uses biased ACF internally (guarantees PSD Toeplitz → numerically stable recursion)
- Confidence intervals: asymptotic white-noise `±z_{α/2} / √n`
- Default `nlags`: `min(int(10·log10(n)), n // 2 - 1)`

### `calculate_lag_autocorrelation(data, n_lags, last_n_samples, sort_by)`
- Unchanged API vs. the previous `skforecast.plot` version
- Returns DataFrame with columns: `lag`, `partial_autocorrelation_abs`, `partial_autocorrelation`, `autocorrelation_abs`, `autocorrelation`

---

## Improvements applied during review

### 1. `scipy.fft.next_fast_len` (performance)

In `_fft_acf`, the manual power-of-two padding was replaced:

```python
# Before
n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))

# After
n_fft = next_fast_len(2 * n - 1)
```

`next_fast_len` finds the optimal FFT size for factors of 2, 3, 5, 7, 11. For typical
time series lengths (e.g. 365, 720, 8760) this is often smaller than the next power of
two, reducing both memory use and execution time.

### 2. Upfront `n_lags >= n // 2` guard in `calculate_lag_autocorrelation`

Added before the `pacf()` call (after the `last_n_samples` trim):

```python
n = len(data)
if n_lags >= n // 2:
    raise ValueError(
        f"`n_lags` ({n_lags}) must be less than len(data) // 2 ({n // 2}). "
        "Partial autocorrelation cannot be computed for more than half the "
        "sample size."
    )
```

This surfaces a user-facing error with the `n_lags` name (not the internal `nlags`)
before delegating to `pacf()`. A corresponding test was added to
`test_calculate_lag_autocorrelation.py`.

### 3. Docstring correction

In the `calculate_lag_autocorrelation` docstring, the example was corrected:

- Before: "if the series has 10 samples, `n_lags` must be less than or equal to **5**"
- After: "if the series has 10 samples, `n_lags` must be less than or equal to **4**"

The validation is `n_lags >= n // 2` (strict `>=`); for `n=10`, `n//2=5`, so `n_lags=5`
is rejected and the maximum allowed value is 4.

---

## Test design decisions

- **No statsmodels runtime dependency**: Three tests that verify numerical agreement
  with statsmodels use hardcoded expected arrays (generated once offline). Tolerances:
  - `test_acf_output_matches_statsmodels_to_machine_epsilon`: `atol=1e-14`
  - `test_acf_adjusted_true_matches_statsmodels`: `atol=1e-12`
  - `test_pacf_values_close_to_statsmodels_yw`: `atol=5e-2` (intentional — biased vs.
    unbiased ACF causes ~2e-2 systematic divergence from statsmodels `method='yw'`)

- **Old test file deleted**: `skforecast/plot/tests/tests_plot/test_calculate_lag_autocorrelation.py`
  was removed when the function left `skforecast.plot`.

---

## Decisions NOT taken (reviewed and rejected)

| Proposal | Decision | Reason |
|----------|----------|--------|
| Clamp `n_lags` automatically when `last_n_samples` makes it too large | Rejected | Silent behavior change; the new upfront guard already gives a clear error |
| Levinson-Durbin `abs(den) > 1e-12` tolerance instead of `den != 0.0` | Rejected | Biased ACF guarantees PSD Toeplitz → `den` is monotonically non-increasing and ≥ 0; a magic threshold has no principled basis |
| Numba JIT on the Levinson-Durbin loop | Rejected | Loop runs ≤ 50 iterations for typical `n_lags`; dominated by the O(N log N) FFT stage; JIT compilation overhead exceeds savings |
