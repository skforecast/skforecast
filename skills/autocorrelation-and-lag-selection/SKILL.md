---
name: autocorrelation-and-lag-selection
description: >
  Analyzes time series dynamics with the fast skforecast.stats functions
  acf, pacf and calculate_lag_autocorrelation. Covers reading ACF/PACF
  patterns to identify AR/MA orders and seasonality, ranking lags by
  partial autocorrelation, and feeding the result to the lags argument
  of any skforecast forecaster. Use when the user wants to understand
  the dynamics of a series, choose a candidate set of lags before
  hyperparameter tuning, or replace a slow statsmodels acf/pacf call.
---

# Autocorrelation and Lag Selection

## When to Use This Skill

Use this skill when the user wants to:

- Inspect the ACF / PACF of a series before training a forecaster.
- Pick a sensible starting set of lags for `ForecasterRecursive`, `ForecasterDirect`, or any multi-series forecaster.
- Detect seasonality (peaks at multiples of the period) from autocorrelation.
- Replace `statsmodels.tsa.stattools.acf` / `pacf` calls with a faster equivalent on long series.

### Related skills

- **After**: `feature-engineering` (turn the candidate lags into a full feature pipeline with rolling and calendar features)
- **After**: `hyperparameter-optimization` (refine the candidate set via cross-validation with `bayesian_search_forecaster`)
- **After**: `feature-selection` (prune redundant lags or exog variables with `select_features` once the forecaster is configured)

## Overview

| Function | Module | Purpose |
|----------|--------|---------|
| `acf` | `skforecast.stats` | Biased ACF via FFT; optional Bartlett confidence intervals |
| `pacf` | `skforecast.stats` | PACF via FFT + Levinson&ndash;Durbin; white-noise CI |
| `calculate_lag_autocorrelation` | `skforecast.stats` | Tabular ACF + PACF, ranked by &#124;PACF&#124; |

> The skforecast functions mirror the public API of `statsmodels.tsa.stattools.acf` / `pacf` but are reimplemented from scratch (FFT for the ACF, Levinson&ndash;Durbin for the PACF) to **maximise speed on long series**. Only the most common configuration options are exposed.
>
> If the user needs the wider catalogue offered by statsmodels — alternative estimators (`'ywunbiased'`, `'ols'`, `'ld'`, `'burg'`), Ljung&ndash;Box statistics, etc. — recommend calling `statsmodels.tsa` directly.
>
> For plotting, keep the statsmodels helpers with their fastest configuration: `plot_acf(..., alpha=0.05, fft=True)` and `plot_pacf(..., alpha=0.05, method='burg')`.

## Computing the ACF

```python
from skforecast.stats import acf

# Just the values (lag 0 included; output length = nlags + 1)
acf_vals = acf(y, nlags=20)

# With 95% Bartlett confidence intervals
acf_vals, confint = acf(y, nlags=20, alpha=0.05)

# Unbiased estimator (denominator n - k)
acf_vals = acf(y, nlags=20, adjusted=True)
```

Notes:
- `nlags` must satisfy `0 < nlags < len(x)`. If `None`, defaults to `min(int(10 * log10(n)), n - 1)`.
- `acf_vals[0]` is always `1.0`.
- Default is the **biased estimator** (`adjusted=False`, denominator `n`). It guarantees a positive semi-definite Toeplitz matrix and is required for downstream PACF computation.

## Computing the PACF

```python
from skforecast.stats import pacf

pacf_vals = pacf(y, nlags=20)
pacf_vals, confint = pacf(y, nlags=20, alpha=0.05)
```

Notes:
- `nlags` must satisfy `0 < nlags < len(x) // 2` — Levinson&ndash;Durbin becomes unreliable when the AR order approaches half the sample size; the function raises explicitly.
- Confidence intervals are asymptotic under the white-noise null: `±z_{α/2} / sqrt(n)` for all lags `≥ 1`. Lag 0 has no uncertainty.
- The implementation uses the biased ACF internally, so values can differ slightly (~2e-2 for typical series) from `statsmodels.pacf(method='yw')`, which uses the unbiased estimator.

## Reading ACF / PACF Patterns

| Pattern | Suggests |
|---------|----------|
| ACF cuts off after lag *q* | MA(q) component |
| ACF decays slowly | AR or mixed process — use PACF |
| PACF cuts off after lag *p* | AR(p) component (lags 1..*p* are the relevant ones) |
| PACF decays slowly | MA or mixed process |
| ACF peaks at lags *k*, *2k*, *3k* | Seasonality of period *k* |
| Persistent slow decay touching every lag | Trend — difference the series before reading orders |

```python
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(11, 3))
plot_acf(y, lags=36, alpha=0.05, fft=True, ax=axes[0])
plot_pacf(y, lags=36, alpha=0.05, method='burg', ax=axes[1])
plt.tight_layout()
```

## Ranking Lags with `calculate_lag_autocorrelation`

```python
from skforecast.stats import calculate_lag_autocorrelation

# Default: rows ordered by |PACF| descending, lags 1..n_lags
lag_table = calculate_lag_autocorrelation(
    data=y,
    n_lags=24,
    sort_by='partial_autocorrelation_abs',
)

# Other valid sort keys
calculate_lag_autocorrelation(data=y, n_lags=24, sort_by='lag')
calculate_lag_autocorrelation(data=y, n_lags=24, sort_by='autocorrelation_abs')
```

Returns a DataFrame with columns
`['lag', 'partial_autocorrelation_abs', 'partial_autocorrelation', 'autocorrelation_abs', 'autocorrelation']`
covering lags 1..`n_lags` (lag 0 dropped because it is always 1.0).

`last_n_samples` keeps only the most recent observations — useful when the series is very long and only the recent dynamics matter:

```python
calculate_lag_autocorrelation(data=y, n_lags=24, last_n_samples=2000)
```

`data` accepts a `pd.Series` or a single-column `pd.DataFrame`. `n_lags` must be strictly less than `len(data) // 2`.

## End-to-End: From PACF to a Forecaster

```python
import numpy as np
from skforecast.stats import calculate_lag_autocorrelation
from skforecast.recursive import ForecasterRecursive
from sklearn.ensemble import HistGradientBoostingRegressor

# 1. Rank lags by |PACF|
lag_table = calculate_lag_autocorrelation(data=y_train, n_lags=24)

# 2. Threshold against the asymptotic 95% white-noise band (1.96 / sqrt(n))
threshold = 1.96 / np.sqrt(len(y_train))
selected_lags = (
    lag_table
    .loc[lag_table['partial_autocorrelation_abs'] > threshold, 'lag']
    .astype(int)
    .sort_values()
    .tolist()
)

# 3. Feed directly into any forecaster
forecaster = ForecasterRecursive(
    estimator=HistGradientBoostingRegressor(random_state=963),
    lags=selected_lags,
)
forecaster.fit(y=y_train)
predictions = forecaster.predict(steps=12)
```

The selected lags are an *informed starting point*, not a final answer — refine with `bayesian_search_forecaster` (see `hyperparameter-optimization`) or prune with `select_features` (see `feature-selection`).

## Common Mistakes

1. **Using `nlags >= n // 2` for PACF.** Levinson&ndash;Durbin loses stability as the AR order approaches half the sample size; the function raises explicitly. Reduce `nlags` or use a longer series.
2. **Reading the ACF to choose AR order.** ACF is contaminated by indirect dependencies through earlier lags and rarely cuts off cleanly for AR processes — use the PACF for AR order, the ACF for MA order.
3. **Treating the confidence bands as joint tests.** Bartlett (ACF) and white-noise (PACF) bands are *pointwise*. Several lags will cross the band by chance; use them to spot candidates, not to formally reject white noise.
4. **Skipping detrending / differencing.** A strong trend dominates the ACF and masks higher-lag structure. If the series shows visible trend, read ACF / PACF on `y.diff().dropna()` (or use `differentiation=1` in the forecaster).
5. **Switching to `adjusted=True` by default.** The biased estimator is preferred for forecasting workflows; the unbiased one can produce `|ACF| > 1` at high lags and breaks the positive semi-definite property required by Levinson&ndash;Durbin.
6. **Expecting statsmodels-only options.** `method='ols'`, `method='burg'`, alternative confidence intervals, etc. are *not* exposed by `skforecast.stats.pacf`. For those, call `statsmodels.tsa.stattools.pacf` directly — the API is the same.
7. **Passing a multi-column DataFrame to `calculate_lag_autocorrelation`.** It accepts a `Series` or a single-column `DataFrame` and raises otherwise. Select the target column first.
