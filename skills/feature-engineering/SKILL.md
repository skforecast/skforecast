---
name: feature-engineering
description: >
  Creates features for time series forecasting: calendar features with
  feature_engine (DatetimeFeatures, CyclicalFeatures), rolling statistics
  with RollingFeatures, differencing, sunlight features, and data scaling.
  Use when the user wants to improve model accuracy through feature
  engineering or asks about exogenous variable creation.
---

# Feature Engineering

## References

See [references/rolling-stats-reference.md](references/rolling-stats-reference.md) for
the complete `RollingFeatures` constructor, all 9 available statistics,
feature name generation formula, window behavior, and `kwargs_stats` usage.

## When to Use This Skill

Use this skill when the user wants to create features to improve forecasting
accuracy: calendar/datetime features, rolling statistics, cyclical encoding,
sunlight features, differencing, or data scaling.

## Overview

| Tool | Package | Purpose |
|------|---------|---------|
| `DatetimeFeatures` | feature_engine | Extract calendar features from datetime index |
| `CyclicalFeatures` | feature_engine | Encode cyclical features with sin/cos |
| `RollingFeatures` | skforecast | Rolling window statistics (mean, std, min, max, etc.) |
| `differentiation` param | skforecast | Make non-stationary series stationary |
| `astral` | astral | Sunrise, sunset, daylight hours |

## Calendar Features with feature_engine

### Manual extraction (pandas)

```python
import pandas as pd

# Data must have a DatetimeIndex with frequency set
data = data.asfreq('h')

data['year'] = data.index.year
data['month'] = data.index.month
data['day_of_week'] = data.index.dayofweek
data['hour'] = data.index.hour
```

### Automated extraction (DatetimeFeatures)

```python
from feature_engine.datetime import DatetimeFeatures

features_to_extract = ['month', 'week', 'day_of_week', 'hour']
calendar_transformer = DatetimeFeatures(
    variables           = 'index',
    features_to_extract = features_to_extract,
    drop_original       = True,
)

calendar_features = calendar_transformer.fit_transform(data)
```

> `DatetimeFeatures` is sklearn-compatible and can be passed directly as
> `transformer_exog` in skforecast forecasters.

## Cyclical Encoding

Cyclical features (hour, day_of_week, month) should NOT be treated as linear
integers — hour 23 is only 1 hour from hour 0. Use sin/cos encoding to
preserve the cyclical relationship.

```python
from feature_engine.datetime import DatetimeFeatures
from feature_engine.creation import CyclicalFeatures

# Step 1: Extract calendar features
features_to_extract = ['month', 'week', 'day_of_week', 'hour']
calendar_transformer = DatetimeFeatures(
    variables           = 'index',
    features_to_extract = features_to_extract,
    drop_original       = True,
)
calendar_features = calendar_transformer.fit_transform(data)

# Step 2: Encode as cyclical (sin/cos)
features_to_encode = ['month', 'week', 'day_of_week', 'hour']
max_values = {
    'month': 12,
    'week': 52,
    'day_of_week': 7,
    'hour': 24,
}
cyclical_encoder = CyclicalFeatures(
    variables     = features_to_encode,
    max_values    = max_values,
    drop_original = True,
)
exog_calendar = cyclical_encoder.fit_transform(calendar_features)
# Produces columns: month_sin, month_cos, week_sin, week_cos, ...
```

## Sunlight Features

Sunrise/sunset times can be powerful features for energy, transport, or
activity-related series.

```python
from astral.sun import sun
from astral import LocationInfo

location = LocationInfo('Washington, D.C.', 'USA')
sunrise_hour = [sun(location.observer, date=date)['sunrise'] for date in data.index]
sunset_hour = [sun(location.observer, date=date)['sunset'] for date in data.index]

# Round to the nearest hour
sunrise_hour = pd.Series(sunrise_hour, index=data.index).dt.round('h').dt.hour
sunset_hour = pd.Series(sunset_hour, index=data.index).dt.round('h').dt.hour

sun_light_features = pd.DataFrame({
    'sunrise_hour': sunrise_hour,
    'sunset_hour': sunset_hour,
})
sun_light_features['daylight_hours'] = (
    sun_light_features['sunset_hour'] - sun_light_features['sunrise_hour']
)
```

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

## Differencing (Non-Stationary Series)

```python
# Built-in — forecaster handles differencing and inverse transform automatically
forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=24,
    differentiation=1,  # First-order differencing (removes linear trend)
    # differentiation=2,  # Second-order (removes quadratic trend)
)
forecaster.fit(y=y_train)
predictions = forecaster.predict(steps=10)  # Auto inverse-transformed
```

## Data Transformers (Scaling)

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Scale target variable — transformer applied automatically during fit/predict
forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=24,
    transformer_y=StandardScaler(),
    transformer_exog=StandardScaler(),
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

## Combining Features — Full Example

```python
import pandas as pd
from feature_engine.datetime import DatetimeFeatures
from feature_engine.creation import CyclicalFeatures
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

# 1. Calendar features with cyclical encoding
calendar_transformer = DatetimeFeatures(
    variables='index',
    features_to_extract=['month', 'day_of_week', 'hour'],
    drop_original=True,
)
cyclical_encoder = CyclicalFeatures(
    variables=['month', 'day_of_week', 'hour'],
    max_values={'month': 12, 'day_of_week': 7, 'hour': 24},
    drop_original=True,
)
exog_calendar = cyclical_encoder.fit_transform(
    calendar_transformer.fit_transform(data)
)

# 2. Combine with other exogenous variables
exog = pd.concat([exog_external, exog_calendar], axis=1)

# 3. Rolling features + lags + differencing
rolling = RollingFeatures(stats=['mean', 'std'], window_sizes=[7, 14])

forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=[1, 2, 3, 7, 14, 24],
    window_features=rolling,
    transformer_y=StandardScaler(),
    differentiation=1,
)
forecaster.fit(y=y_train, exog=exog.loc[y_train.index])
predictions = forecaster.predict(steps=10, exog=exog.loc[forecast_index])
```

## Common Mistakes

1. **Not encoding cyclical features**: Using raw integers for hour/month/day_of_week loses the cyclical relationship (hour 23 appears far from hour 0). Always use sin/cos encoding.
2. **Forgetting frequency on index**: Calendar transformers require `DatetimeIndex` with frequency set (`data.asfreq('h')`).
3. **Not covering forecast horizon with exog**: Calendar features for `predict()` must include future dates covering the entire forecast horizon.
4. **Over-engineering features**: Start with lags only, then add rolling features and calendar features incrementally. Validate each addition with backtesting.
