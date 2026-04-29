---
name: feature-engineering
description: >
    Creates features for time series forecasting: calendar features with
    feature_engine (DatetimeFeatures, CyclicalFeatures), rolling statistics
    with RollingFeatures, differencing, and categorical exogenous variables.
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


## Calendar Features with feature_engine

`feature-engine` is not a core skforecast dependency. Install it separately
before using `DatetimeFeatures` or `CyclicalFeatures`:

```bash
pip install feature-engine
```

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

## Categorical Exogenous Variables

All ML forecasters include a `categorical_features` parameter (default `'auto'`)
that automatically detects and encodes non-numeric exogenous columns using an
internal `OrdinalEncoder` (into float codes, since numpy arrays are used internally).
Native categorical support is configured automatically for LightGBM, CatBoost,
XGBoost, and HistGradientBoostingRegressor.

```python
forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=24,
    categorical_features='auto',  # Default — auto-detect non-numeric columns
)
```

**`categorical_features` options:**
- `'auto'` (default): Auto-detect non-numeric columns after `transformer_exog`.
- `list`: Explicit column names to treat as categorical (including numeric columns).
- `None`: No internal categorical encoding.

**Important:** When `categorical_features` is not `None`, do not set categorical
features directly on the estimator or via `fit_kwargs`. The forecaster manages
the configuration internally and overwrites estimator-level settings.

**Choosing an encoding strategy:**

| Method | API | Best for |
|--------|-----|----------|
| Built-in `categorical_features` | `categorical_features='auto'` or `list` | Gradient boosting (LightGBM, XGBoost, CatBoost, HistGBR) — simplest workflow |
| One-hot / Ordinal encoding | `transformer_exog` | Linear models, SVMs, non-gradient-boosting trees |
| Target encoding | Outside forecaster | High-cardinality features (applied manually to avoid leakage) |

**Combining `transformer_exog` and `categorical_features`:**
`transformer_exog` is applied **before** `categorical_features` detection. Scale
numeric columns with `transformer_exog` while `categorical_features='auto'` handles
the rest. Avoid applying both mechanisms to the same columns.

```python
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler

transformer_exog = make_column_transformer(
    (StandardScaler(), ['temp', 'hum']),
    remainder='passthrough',
    verbose_feature_names_out=False
).set_output(transform='pandas')

forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=24,
    transformer_exog=transformer_exog,
    categorical_features='auto',  # Detects remaining non-numeric columns
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
