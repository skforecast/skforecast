---
name: prediction-intervals
description: >
  Generates prediction intervals for time series forecasts using bootstrapping,
  conformal prediction, or built-in statistical model intervals. Covers
  interval configuration, residual management, and calibration.
  Use when the user needs uncertainty quantification for forecasts.
---

# Prediction Intervals

## When to Use

Use prediction intervals to quantify forecast uncertainty. Skforecast offers three methods:

| Method | Forecasters | Description |
|--------|-------------|-------------|
| **Bootstrapping** | Recursive, Direct | Resample from training residuals |
| **Conformal** | All ML forecasters | Distribution-free intervals via conformal prediction |
| **Built-in** | ForecasterStats (ARIMA, ETS) | Parametric intervals from the statistical model |

## Bootstrapping Method

```python
from skforecast.recursive import ForecasterRecursive
from sklearn.ensemble import RandomForestRegressor

forecaster = ForecasterRecursive(
    estimator=RandomForestRegressor(n_estimators=100, random_state=123),
    lags=24,
)

# IMPORTANT: store_in_sample_residuals=True is required for bootstrapping
forecaster.fit(y=y_train, store_in_sample_residuals=True)

predictions = forecaster.predict_interval(
    steps=10,
    interval=[10, 90],              # Percentiles → 80% interval
    method='bootstrapping',
    n_boot=500,                      # Number of bootstrap samples
    use_in_sample_residuals=True,    # Use training residuals
    use_binned_residuals=True,       # Better calibration: residuals binned by prediction level
    random_state=123,
)
# Returns: DataFrame with columns [pred, lower_bound, upper_bound]
```

## Conformal Prediction

```python
# No store_in_sample_residuals needed — uses backtesting residuals automatically
predictions = forecaster.predict_interval(
    steps=10,
    interval=[10, 90],
    method='conformal',
    use_in_sample_residuals=True,
    use_binned_residuals=True,
)
```

## Statistical Model Intervals

```python
from skforecast.recursive import ForecasterStats
from skforecast.stats import Arima

forecaster = ForecasterStats(
    estimator=Arima(order=(1, 1, 1), seasonal_order=(1, 1, 1), m=12)
)
forecaster.fit(y=y_train)

# Uses parametric intervals from statsmodels — different interface
predictions = forecaster.predict_interval(
    steps=12,
    interval=[10, 90],         # Or use alpha=0.05 for 95% interval
)
```

## During Backtesting

```python
from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold

cv = TimeSeriesFold(
    steps=10,
    initial_train_size=len(y_train),
    refit=False,
)

metric, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=data['target'],
    cv=cv,
    metric='mean_absolute_error',
    interval=[10, 90],
    interval_method='bootstrapping',   # or 'conformal'
    n_boot=250,
    use_in_sample_residuals=True,
    use_binned_residuals=True,
)
# predictions has columns: pred, lower_bound, upper_bound
```

## Multi-Series Intervals

```python
from skforecast.recursive import ForecasterRecursiveMultiSeries

# NOTE: default method for multi-series is 'conformal', not 'bootstrapping'
predictions = forecaster_multi.predict_interval(
    steps=10,
    levels=['series_1', 'series_2'],
    interval=[10, 90],
    method='conformal',
)
```

## Out-of-Sample Residuals (Better Calibration)

```python
# For better interval calibration, use out-of-sample residuals
# First, compute them via backtesting
metric, predictions_bt = backtesting_forecaster(
    forecaster=forecaster,
    y=data['target'],
    cv=cv,
    metric='mean_absolute_error',
)

# Set out-of-sample residuals on the forecaster
forecaster.set_out_sample_residuals(
    y_true=data['target'].loc[predictions_bt.index],
    y_pred=predictions_bt['pred'],
)

# Now use them for intervals
predictions = forecaster.predict_interval(
    steps=10,
    interval=[10, 90],
    method='bootstrapping',
    use_in_sample_residuals=False,  # Use out-of-sample residuals
)
```

## Evaluating Interval Quality

```python
from skforecast.metrics import calculate_coverage

coverage = calculate_coverage(
    y_true=y_test,
    lower_bound=predictions['lower_bound'],
    upper_bound=predictions['upper_bound'],
)
print(f"Coverage: {coverage:.2%}")  # Should be close to 0.80 for [10, 90] interval
```

## Common Mistakes

1. **Forgetting `store_in_sample_residuals=True`**: Required in `fit()` before using `predict_interval(method='bootstrapping')`.
2. **Wrong default method for multi-series**: `ForecasterRecursiveMultiSeries` and `ForecasterDirectMultiVariate` default to `method='conformal'`, not `'bootstrapping'`.
3. **Using `alpha` with ML forecasters**: Only `ForecasterStats` supports the `alpha` parameter. ML forecasters use `interval=[lo, hi]`.
4. **Not evaluating coverage**: Always check if actual coverage matches nominal interval width.
