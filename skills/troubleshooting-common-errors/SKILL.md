---
name: troubleshooting-common-errors
description: >
  Diagnoses and fixes common errors when using skforecast, especially mistakes
  frequently made by LLMs generating skforecast code. Covers deprecated imports,
  wrong function names, missing parameters, and data format issues.
  Use when generated code produces errors or unexpected results.
---

# Troubleshooting Common Errors

## Deprecated Import Paths

The most frequent LLM error. Old import paths no longer exist.

| Wrong (Deprecated) | Correct (v0.14.0+) |
|-------|---------|
| `from skforecast.ForecasterAutoreg import ForecasterAutoreg` | `from skforecast.recursive import ForecasterRecursive` |
| `from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries` | `from skforecast.recursive import ForecasterRecursiveMultiSeries` |
| `from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect` | `from skforecast.direct import ForecasterDirect` |
| `from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate` | `from skforecast.direct import ForecasterDirectMultiVariate` |
| `from skforecast.model_selection_multiseries import backtesting_forecaster_multiseries` | `from skforecast.model_selection import backtesting_forecaster_multiseries` |

## Wrong Class/Function Names

| Wrong | Correct |
|-------|---------|
| `ForecasterAutoreg` | `ForecasterRecursive` |
| `ForecasterAutoregMultiSeries` | `ForecasterRecursiveMultiSeries` |
| `ForecasterAutoregDirect` | `ForecasterDirect` |
| `ForecasterAutoregMultiVariate` | `ForecasterDirectMultiVariate` |
| `ForecasterSarimax` | `ForecasterStats(estimator=Sarimax(...))` |

## Data Issues

### "ValueError: The index of the series must be a DatetimeIndex with frequency"

```python
# Fix: set the frequency
data = data.asfreq('h')       # Hourly
data = data.asfreq('D')       # Daily
data = data.asfreq('MS')      # Monthly start
data = data.asfreq('QS')      # Quarterly start
```

### "ValueError: y contains NaN values"

```python
# Fix: handle missing values before fitting
data = data.ffill()                      # Forward fill
data = data.interpolate(method='linear') # Linear interpolation
```

### "ValueError: exog must have the same index as y" / "exog does not cover forecast horizon"

```python
# Fix: exog for prediction must cover ALL future steps
# If predicting 10 steps ahead, exog_test must have at least 10 rows
# with dates matching the expected forecast dates
exog_test = exog.loc[forecast_start:forecast_end]
predictions = forecaster.predict(steps=10, exog=exog_test)
```

## Wrong Backtesting Function

```python
# ❌ WRONG: using backtesting_forecaster with ForecasterStats
from skforecast.model_selection import backtesting_forecaster
backtesting_forecaster(forecaster=forecaster_stats, ...)  # Error!

# ✅ CORRECT: use backtesting_stats for statistical models
from skforecast.model_selection import backtesting_stats
backtesting_stats(forecaster=forecaster_stats, ...)

# ❌ WRONG: using backtesting_forecaster with ForecasterRecursiveMultiSeries
backtesting_forecaster(forecaster=forecaster_multi, ...)  # Error!

# ✅ CORRECT: use backtesting_forecaster_multiseries
from skforecast.model_selection import backtesting_forecaster_multiseries
backtesting_forecaster_multiseries(forecaster=forecaster_multi, series=series, ...)
```

## Wrong Search Function

```python
# ❌ WRONG: grid_search_forecaster with ForecasterStats
grid_search_forecaster(forecaster=forecaster_stats, ...)

# ✅ CORRECT: grid_search_stats for statistical models
from skforecast.model_selection import grid_search_stats
grid_search_stats(forecaster=forecaster_stats, ...)

# ❌ WRONG: grid_search_forecaster with ForecasterRecursiveMultiSeries
grid_search_forecaster(forecaster=forecaster_multi, ...)

# ✅ CORRECT: grid_search_forecaster_multiseries
from skforecast.model_selection import grid_search_forecaster_multiseries
grid_search_forecaster_multiseries(forecaster=forecaster_multi, series=series, ...)
```

## Prediction Interval Errors

### "No in-sample residuals stored"

```python
# ❌ WRONG: fit without residuals, then call predict_interval
forecaster.fit(y=y_train)
forecaster.predict_interval(steps=10, method='bootstrapping')

# ✅ CORRECT: store residuals during fit
forecaster.fit(y=y_train, store_in_sample_residuals=True)
forecaster.predict_interval(steps=10, method='bootstrapping')
```

### Wrong interval method for a forecaster

| Forecaster | Supported Methods |
|------------|-------------------|
| `ForecasterRecursive` | `'bootstrapping'`, `'conformal'` |
| `ForecasterDirect` | `'bootstrapping'`, `'conformal'` |
| `ForecasterRecursiveMultiSeries` | `'bootstrapping'`, `'conformal'` (default: `'conformal'`) |
| `ForecasterDirectMultiVariate` | `'bootstrapping'`, `'conformal'` (default: `'conformal'`) |
| `ForecasterEquivalentDate` | `'conformal'` only |
| `ForecasterRnn` | `'conformal'` only |
| `ForecasterStats` | Built-in (uses `alpha` or `interval` parameter, no `method`) |
| `ForecasterRecursiveClassifier` | Not available — use `predict_proba()` |

## ETS Model API Confusion

```python
# ❌ WRONG (deprecated Ets API)
ets_model = Ets(error='add', trend='add', seasonal='add', seasonal_periods=12)

# ✅ CORRECT (current API)
ets_model = Ets(model='AAA', m=12)
# Model string: 1st char=Error, 2nd=Trend, 3rd=Seasonal
# A=Additive, M=Multiplicative, N=None, Z=Auto-select
```

## Function Mapping Reference

| Task | Single Series | Multi-Series | Statistical |
|------|--------------|-------------|-------------|
| **Backtesting** | `backtesting_forecaster` | `backtesting_forecaster_multiseries` | `backtesting_stats` |
| **Grid Search** | `grid_search_forecaster` | `grid_search_forecaster_multiseries` | `grid_search_stats` |
| **Random Search** | `random_search_forecaster` | `random_search_forecaster_multiseries` | `random_search_stats` |
| **Bayesian Search** | `bayesian_search_forecaster` | `bayesian_search_forecaster_multiseries` | N/A |
| **Feature Selection** | `select_features` | `select_features_multiseries` | N/A |
