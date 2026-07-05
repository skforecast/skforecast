---
name: troubleshooting-common-errors
description: >
  Diagnoses and fixes common errors when using skforecast, especially mistakes
  frequently made by LLMs generating skforecast code. Covers deprecated imports,
  wrong function names, missing parameters, and data format issues.
  Use when generated code produces errors or unexpected results.
---

# Troubleshooting Common Errors

This skill is the full pitfall catalog. The highest-risk skills also carry a focused `## Stop Conditions` table at the top (forecasting-single-series, forecasting-multiple-series, prediction-intervals, statistical-models, foundation-forecasting, feature-engineering, deep-learning-forecasting, hyperparameter-optimization); this skill remains the canonical source they point back to.

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

## Removed Arguments

| Removed (v0.22.0+) | Replacement |
|---------------------|-------------|
| `regressor=...` | `estimator=...` (in all Forecasters) |

## Categorical Exogenous Variables

```python
# ❌ WRONG: setting categorical features directly on the estimator
forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(categorical_feature=[0, 1]),
    lags=24,
)

# ✅ CORRECT: use categorical_features parameter on the forecaster
forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=24,
    categorical_features='auto',  # or ['col_name_1', 'col_name_2']
)
```

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
# Fix 1 (recommended for NaN-tolerant estimators): keep NaN rows
forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(verbose=-1),  # LightGBM handles NaN natively
    lags=14,
    dropna_from_series=False,  # Default — NaN rows kept in training matrices
)
forecaster.fit(y=data['target'], suppress_warnings=True)

# Fix 2: drop rows with NaN from training matrices
forecaster = ForecasterRecursive(
    estimator=RandomForestRegressor(),
    lags=14,
    dropna_from_series=True,  # Drop NaN rows before fitting
)

# Fix 3: impute missing values before fitting
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
backtesting_forecaster(forecaster=forecaster_stats, y=y, cv=cv, metric=metric)  # Error!

# ✅ CORRECT: use backtesting_stats for statistical models
from skforecast.model_selection import backtesting_stats
backtesting_stats(forecaster=forecaster_stats, y=y, cv=cv, metric=metric)

# ❌ WRONG: using backtesting_forecaster with ForecasterRecursiveMultiSeries
backtesting_forecaster(forecaster=forecaster_multi, y=y, cv=cv, metric=metric)  # Error!

# ✅ CORRECT: use backtesting_forecaster_multiseries
from skforecast.model_selection import backtesting_forecaster_multiseries
backtesting_forecaster_multiseries(
    forecaster=forecaster_multi, series=series, cv=cv, metric=metric
)
```

## Wrong Search Function

```python
# ❌ WRONG: grid_search_forecaster with ForecasterStats
grid_search_forecaster(forecaster=forecaster_stats, y=y, cv=cv, param_grid=param_grid)

# ✅ CORRECT: grid_search_stats for statistical models
from skforecast.model_selection import grid_search_stats
grid_search_stats(forecaster=forecaster_stats, y=y, cv=cv, param_grid=param_grid)

# ❌ WRONG: grid_search_forecaster with ForecasterRecursiveMultiSeries
grid_search_forecaster(forecaster=forecaster_multi, y=y, cv=cv, param_grid=param_grid)

# ✅ CORRECT: grid_search_forecaster_multiseries
from skforecast.model_selection import grid_search_forecaster_multiseries
grid_search_forecaster_multiseries(
    forecaster=forecaster_multi, series=series, cv=cv, param_grid=param_grid
)
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

## Loading Serialized Forecasters from Older Versions

Forecasters saved (pickled/joblib) with older skforecast versions may fail to load or behave unexpectedly after upgrading. Internal attributes, class structures, and default values change between releases.

```python
# ❌ Common error when loading a forecaster saved with an older version
import joblib
forecaster = joblib.load('forecaster_v0.13.pkl')
# AttributeError: 'ForecasterRecursive' object has no attribute 'new_attribute'
# or: ModuleNotFoundError: No module named 'skforecast.ForecasterAutoreg'

# ✅ CORRECT: retrain the forecaster with the current version
forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=24,
)
forecaster.fit(y=y_train)
joblib.dump(forecaster, 'forecaster_v0.22.pkl')
```

**Best practices:**
- Always retrain and re-save forecasters after upgrading skforecast.
- Store training code (not just the serialized object) so models can be reproduced.
- Pin skforecast version in `requirements.txt` for production deployments.
