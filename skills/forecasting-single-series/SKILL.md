---
name: forecasting-single-series
description: >
  Forecasts a single time series using ForecasterRecursive or ForecasterDirect.
  Covers data preparation, model creation, training, prediction, backtesting,
  and prediction intervals. Use when the user needs to predict future values
  of one time series.
---

# Forecasting a Single Time Series

## When to Use

Use this workflow when you have **one time series** and want to predict its future values.

- **ForecasterRecursive**: Default choice. Uses its own predictions as inputs for multi-step forecasting. Works with any sklearn-compatible regressor.
- **ForecasterDirect**: Trains one model per forecast step. Better when the relationship between lags and target changes across the horizon.

## Complete Workflow

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.preprocessing import RollingFeatures
from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold

# 1. Load and prepare data (MUST have DatetimeIndex with frequency)
data = pd.read_csv('data.csv', index_col='date', parse_dates=True)
data = data.asfreq('h')  # Set frequency — required

# 2. Train/test split (time-based, never random)
end_train = '2023-01-01'
y_train = data.loc[:end_train, 'target']
y_test = data.loc[end_train:, 'target']

# 3. Create forecaster with optional rolling features
rolling_features = RollingFeatures(
    stats=['mean', 'std'],
    window_sizes=24
)

forecaster = ForecasterRecursive(
    estimator=RandomForestRegressor(n_estimators=100, random_state=123),
    lags=24,
    window_features=rolling_features,
    transformer_y=None,          # e.g., StandardScaler() for scaling
    differentiation=None,        # e.g., 1 for first-order differencing
)

# 4. Train
forecaster.fit(y=y_train)

# 5. Predict
predictions = forecaster.predict(steps=10)

# 6. Backtesting (proper evaluation)
cv = TimeSeriesFold(
    steps=10,
    initial_train_size=len(y_train),
    refit=False,
    fixed_train_size=False,
)

metric, predictions_bt = backtesting_forecaster(
    forecaster=forecaster,
    y=data['target'],
    cv=cv,
    metric='mean_absolute_error',
)
print(f"MAE: {metric}")

# 7. Prediction intervals
# Default interval is [5, 95] (90%). Here [10, 90] creates an 80% interval.
forecaster.fit(y=y_train, store_in_sample_residuals=True)
predictions_interval = forecaster.predict_interval(
    steps=10,
    interval=[10, 90],          # 80% prediction interval
    method='bootstrapping',
    n_boot=500,
)
```

## With Exogenous Variables

```python
# Exogenous variables must cover the forecast horizon during prediction
forecaster = ForecasterRecursive(
    estimator=RandomForestRegressor(n_estimators=100, random_state=123),
    lags=24,
)
forecaster.fit(y=y_train, exog=exog_train)
predictions = forecaster.predict(steps=10, exog=exog_test)
```

## Using ForecasterDirect

```python
from skforecast.direct import ForecasterDirect

# Must specify `steps` at creation — trains one model per step
forecaster = ForecasterDirect(
    estimator=RandomForestRegressor(n_estimators=100, random_state=123),
    lags=24,
    steps=10,
)
forecaster.fit(y=y_train, exog=exog_train)
predictions = forecaster.predict(exog=exog_test)
```

## Common Mistakes

1. **Missing frequency on index**: Always call `data.asfreq('h')` (or `'D'`, `'MS'`, etc.).
2. **NaN in data**: Forecasters reject NaN by default. Impute missing values first.
3. **Exog not covering forecast horizon**: The exogenous DataFrame for `predict()` must have rows for every future step.
4. **Random train/test split**: Time series must be split chronologically, never shuffled.
5. **Forgetting `store_in_sample_residuals=True`**: Required before calling `predict_interval()` with `method='bootstrapping'` on a standalone forecaster. During backtesting, residuals are computed automatically.
