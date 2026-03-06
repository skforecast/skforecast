---
name: feature-engineering
description: >
  Creates features for time series forecasting using RollingFeatures,
  TimeSeriesDifferentiator, DateTimeFeatureTransformer, and custom exogenous
  variables. Covers rolling statistics, differencing, calendar features, and
  data preprocessing. Use when the user wants to improve model accuracy
  through feature engineering.
---

# Feature Engineering

## Overview

Skforecast provides three built-in feature engineering tools:

| Tool | Purpose |
|------|---------|
| `RollingFeatures` | Rolling window statistics (mean, std, min, max, etc.) |
| `TimeSeriesDifferentiator` | Make non-stationary series stationary |
| `DateTimeFeatureTransformer` | Extract calendar features from datetime index |

## Rolling Features (Window Statistics)

```python
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from lightgbm import LGBMRegressor

# Single window size for all stats
rolling = RollingFeatures(
    stats=['mean', 'std', 'min', 'max'],
    window_sizes=7,  # int applies same window to all stats
)

# Different window sizes per statistic
rolling = RollingFeatures(
    stats=['mean', 'std', 'min', 'max'],
    window_sizes=[7, 7, 14, 14],  # Must match length of stats
)

# Multiple RollingFeatures objects
rolling_short = RollingFeatures(stats=['mean', 'std'], window_sizes=7)
rolling_long = RollingFeatures(stats=['mean', 'std'], window_sizes=30)

forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=24,
    window_features=[rolling_short, rolling_long],  # List of RollingFeatures
)
```

### Available Rolling Statistics

Standard: `'mean'`, `'std'`, `'min'`, `'max'`, `'sum'`, `'median'`, `'ratio_min_max'`, `'coef_variation'`

Exponential weighted: `'ewm'` — requires `kwargs_stats`:
```python
rolling = RollingFeatures(
    stats=['ewm'],
    window_sizes=7,
    kwargs_stats={'ewm': {'alpha': 0.3}},
)
```

Custom names:
```python
rolling = RollingFeatures(
    stats=['mean', 'std'],
    window_sizes=7,
    features_names=['rolling_mean_7', 'rolling_std_7'],
)
```

## Differencing (Non-Stationary Series)

```python
# Option 1: Built-in — forecaster handles differencing and inverse transform
forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=24,
    differentiation=1,  # First-order differencing (removes linear trend)
    # differentiation=2,  # Second-order differencing (removes quadratic trend)
)
forecaster.fit(y=y_train)
predictions = forecaster.predict(steps=10)  # Auto inverse-transformed

# Option 2: Manual — for inspection or custom use
from skforecast.preprocessing import TimeSeriesDifferentiator

differentiator = TimeSeriesDifferentiator(order=1)
y_diff = differentiator.fit_transform(y_train.to_numpy())
# ... use y_diff for analysis ...
y_original = differentiator.inverse_transform(y_diff)
```

## Calendar / DateTime Features

```python
from skforecast.preprocessing import DateTimeFeatureTransformer

# Extract calendar features from DatetimeIndex
transformer = DateTimeFeatureTransformer(
    features=['year', 'month', 'week', 'day_of_week', 'hour'],
    encoding='cyclical',  # 'cyclical' (sin/cos), 'onehot', or None (raw integers)
)

# Transform data — input must have DatetimeIndex
exog_calendar = transformer.fit_transform(data)

# Use as exogenous variable
forecaster.fit(y=y_train, exog=exog_calendar.loc[y_train.index])
predictions = forecaster.predict(steps=10, exog=exog_calendar.loc[forecast_index])
```

### Standalone function alternative

```python
from skforecast.preprocessing import create_datetime_features

exog_calendar = create_datetime_features(data)
```

## Data Transformers (Scaling)

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Scale target variable — transformer applied automatically during fit/predict
forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=24,
    transformer_y=StandardScaler(),       # Applied to y
    transformer_exog=StandardScaler(),    # Applied to exogenous variables
)

# For multi-series, different transformers per series
from skforecast.recursive import ForecasterRecursiveMultiSeries

forecaster = ForecasterRecursiveMultiSeries(
    estimator=LGBMRegressor(),
    lags=24,
    transformer_series={
        'series_1': StandardScaler(),
        'series_2': MinMaxScaler(),
    },
)
```

## Combining Features

```python
# Best practice: combine lags + rolling features + calendar + exog
rolling = RollingFeatures(stats=['mean', 'std'], window_sizes=[7, 14])
calendar = DateTimeFeatureTransformer(features=['month', 'day_of_week', 'hour'])

exog_calendar = calendar.fit_transform(data)
exog_combined = pd.concat([exog_external, exog_calendar], axis=1)

forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=[1, 2, 3, 7, 14, 24],   # Custom lag selection
    window_features=rolling,
    transformer_y=StandardScaler(),
    differentiation=1,
)
forecaster.fit(y=y_train, exog=exog_combined.loc[y_train.index])
```

## Common Mistakes

1. **Rolling window size larger than lags**: Window sizes must be supported by the minimum lag configuration.
2. **Forgetting frequency on index**: `DateTimeFeatureTransformer` requires `DatetimeIndex` with frequency set.
3. **Not covering forecast horizon with exog**: Calendar features for `predict()` must include future dates.
4. **Over-engineering features**: Start with lags only, then add rolling features and calendar features incrementally. Validate each addition with backtesting.
