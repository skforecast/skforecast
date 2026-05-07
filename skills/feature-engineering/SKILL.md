---
name: feature-engineering
description: >
    Creates features for time series forecasting: calendar features with
    skforecast's `DateTimeFeatureTransformer` (cyclical, onehot, or spline
    encoding), holiday distance features with `calculate_distance_from_holiday`,
    rolling statistics with `RollingFeatures`, differencing, and categorical
    exogenous variables. Use when the user wants to improve model accuracy
    through feature engineering or asks about exogenous variable creation.
---

# Feature Engineering

## References

See [references/rolling-stats-reference.md](references/rolling-stats-reference.md) for
the complete `RollingFeatures` constructor, all 9 available statistics,
feature name generation formula, window behavior, and `kwargs_stats` usage.

## When to Use This Skill

Use this skill when the user wants to create features to improve forecasting
accuracy: calendar/datetime features, cyclical / onehot / spline encoding,
holiday distance features, rolling statistics, differencing, or data scaling.

### Related skills

- **Before**: `autocorrelation-and-lag-selection` (use ACF/PACF to choose a candidate set of lags before adding rolling and calendar features)
- **After**: `feature-selection` (prune redundant features with `select_features` after engineering)
- **After**: `hyperparameter-optimization` (jointly tune the engineered configuration and the estimator hyperparameters)

## Overview

| Tool | Module | Purpose |
|------|--------|---------|
| `DateTimeFeatureTransformer` | `skforecast.preprocessing` | Sklearn-compatible transformer: extract calendar features from a `DatetimeIndex` and (optionally) encode them. Use as `transformer_exog` or in a `Pipeline`. |
| `create_datetime_features` | `skforecast.preprocessing` | Function form of the same logic, for one-shot use without a transformer. |
| `calculate_distance_from_holiday` | `skforecast.preprocessing` | Periods to next / since last holiday |
| `RollingFeatures` | `skforecast.preprocessing` | Rolling window statistics (mean, std, min, max, etc.) |
| `differentiation` param | skforecast forecasters | Make non-stationary series stationary |

> All calendar tools are built into skforecast — no `feature_engine` (or any
> other extra) dependency is required.

## Calendar Features

`DateTimeFeatureTransformer` extracts features from a `DatetimeIndex` and, in
the same `fit_transform` call, applies the chosen encoding. The result is a
single DataFrame ready to pass as `exog`. Because it is sklearn-compatible, it
can be used as `transformer_exog` in any forecaster or inside a `Pipeline` /
`ColumnTransformer`.

### Supported features

`'year'`, `'month'`, `'week'`, `'day_of_week'`, `'day_of_month'`,
`'day_of_year'`, `'weekend'`, `'hour'`, `'minute'`, `'second'`, `'quarter'`.

By default all are extracted; pass `features=[...]` to subset.

### Supported encodings

| `encoding` | Output | Notes |
|-----------|--------|-------|
| `'cyclical'` (default) | `{feature}_sin`, `{feature}_cos` | sin/cos pair per cyclical feature |
| `'onehot'` | One column per known category (e.g. `month_1` … `month_12`) | Stable schema across train / predict |
| `'spline'` | `≈ max_val` columns per feature, periodic B-splines | Smooth alternative to onehot |
| `None` | Raw integer columns | No transformation |

`'year'` and `'weekend'` are **never** encoded (they are not cyclical) and are
always kept as raw integers regardless of `encoding`.

### Basic usage — `DateTimeFeatureTransformer`

```python
import pandas as pd
from skforecast.preprocessing import DateTimeFeatureTransformer

# Data must have a DatetimeIndex with frequency set
data = data.asfreq('h')

calendar_transformer = DateTimeFeatureTransformer(
    features=['month', 'week', 'day_of_week', 'hour'],
    encoding='cyclical',                 # 'cyclical' | 'onehot' | 'spline' | None
    keep_original_columns=False,         # True merges with X's columns
)
exog_calendar = calendar_transformer.fit_transform(data)
# Columns: month_sin, month_cos, week_sin, week_cos,
#          day_of_week_sin, day_of_week_cos, hour_sin, hour_cos

# After fit, the resulting column names are available via the sklearn API
calendar_transformer.get_feature_names_out()
```

Defaults for `max_values` (`{'month': 12, 'week': 53, 'day_of_week': 7,
'day_of_month': 31, 'day_of_year': 366, 'hour': 24, 'minute': 60, 'second': 60,
'quarter': 4}`) handle leap years and ISO week 53 correctly. Override only the
keys you need:

```python
calendar_transformer = DateTimeFeatureTransformer(
    features=['month', 'hour'],
    encoding='cyclical',
    max_values={'month': 6},   # Custom semester period; hour keeps default 24
)
exog_calendar = calendar_transformer.fit_transform(data)
```

### Onehot and spline encoding

```python
# Onehot — stable column schema (e.g. month_1 … month_12 always present)
calendar_transformer = DateTimeFeatureTransformer(
    features=['month', 'day_of_week'],
    encoding='onehot',
    keep_original_columns=False,
)
exog_calendar = calendar_transformer.fit_transform(data)

# Spline — smooth periodic B-splines, ≈ max_val columns per feature
calendar_transformer = DateTimeFeatureTransformer(
    features=['day_of_year'],
    encoding='spline',
    spline_kwargs={'degree': 3, 'n_knots': 12},   # optional
    keep_original_columns=False,
)
exog_calendar = calendar_transformer.fit_transform(data)
```

`spline_kwargs` accepts any argument of `sklearn.preprocessing.SplineTransformer`
**except** `knots` (computed internally from `max_values`) and `sparse_output`
(incompatible with the DataFrame output).

### Encoding only some features

Use `features_to_encode` to extract a feature but leave it as a raw integer:

```python
calendar_transformer = DateTimeFeatureTransformer(
    features=['year', 'month', 'hour'],
    features_to_encode=['month', 'hour'],   # 'year' kept as raw int
    encoding='cyclical',
)
exog_calendar = calendar_transformer.fit_transform(data)
```

### Function form — `create_datetime_features`

For one-shot use without instantiating a transformer, the same logic is
exposed as a function:

```python
from skforecast.preprocessing import create_datetime_features

exog_calendar = create_datetime_features(
    X=data,
    features=['month', 'day_of_week', 'hour'],
    encoding='cyclical',
    keep_original_columns=False,
)
```

All parameters match `DateTimeFeatureTransformer`. Prefer the transformer when
you want to fit / re-apply the same configuration, plug it into a `Pipeline`,
or pass it as `transformer_exog`.

## Holiday Distance Features

`calculate_distance_from_holiday` computes the number of periods to the next
and since the last holiday. The time unit is inferred from the index frequency
(days, hours, minutes, …) when `date_column=None`, and is always days when a
date column is used.

```python
from skforecast.preprocessing import calculate_distance_from_holiday

# Index-based (preferred): unit follows the index frequency
holiday_dist = calculate_distance_from_holiday(
    X=data[['is_holiday']],          # boolean / 0-1 column
    holiday_column='is_holiday',
    fill_na=0,                       # value for rows before first / after last holiday
)
# Columns: time_to_holiday, time_since_holiday

# Series form: values used directly as the holiday indicator
holiday_dist = calculate_distance_from_holiday(
    X=data['is_holiday'],
    fill_na=0,
)
```

Combine with calendar features into a single `exog`:

```python
exog = pd.concat([exog_calendar, holiday_dist], axis=1)
```

## Cyclical / Onehot / Spline — Choosing an Encoding

| Encoding | Columns per feature | Best for |
|----------|---------------------|----------|
| `'cyclical'` | 2 (sin + cos) | Compact, smooth — good default for tree and linear models |
| `'onehot'` | `max_val` | Stable schema; tree models can split on individual categories |
| `'spline'` | `≈ max_val` (dense) | Smooth + flexible; useful for high-cardinality features (`day_of_year`) when memory allows |
| `None` | 1 | Tree models that benefit from raw ordinal values |

For high-cardinality features (`day_of_year` → 366, `day_of_month` → 31),
`'cyclical'` or `'spline'` are typically more memory-efficient than `'onehot'`.

## Rolling Features (Window Statistics)

```python
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from lightgbm import LGBMRegressor

# Single window size for all stats
rolling = RollingFeatures(
    stats=['mean', 'std', 'min', 'max'],
    window_sizes=7,                  # int applies same window to all stats
)

# Different window sizes per statistic
rolling = RollingFeatures(
    stats=['mean', 'std', 'min', 'max'],
    window_sizes=[7, 7, 14, 14],     # Must match length of stats
)

# Multiple RollingFeatures objects
rolling_short = RollingFeatures(stats=['mean', 'std'], window_sizes=7)
rolling_long = RollingFeatures(stats=['mean', 'std'], window_sizes=30)

forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=24,
    window_features=[rolling_short, rolling_long],   # List of RollingFeatures
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
    differentiation=1,               # First-order differencing (removes linear trend)
    # differentiation=2,             # Second-order (removes quadratic trend)
)
forecaster.fit(y=y_train)
predictions = forecaster.predict(steps=10)   # Auto inverse-transformed
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
    categorical_features='auto',     # Default — auto-detect non-numeric columns
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
    categorical_features='auto',     # Detects remaining non-numeric columns
)
```

## Combining Features — Full Example

```python
import pandas as pd
from skforecast.preprocessing import (
    DateTimeFeatureTransformer,
    calculate_distance_from_holiday,
    RollingFeatures,
)
from skforecast.recursive import ForecasterRecursive
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

# 1. Calendar features with cyclical encoding (no extra deps)
calendar_transformer = DateTimeFeatureTransformer(
    features=['month', 'day_of_week', 'hour'],
    encoding='cyclical',
    keep_original_columns=False,
)
exog_calendar = calendar_transformer.fit_transform(data)

# 2. Holiday distance features (unit inferred from index frequency)
exog_holiday = calculate_distance_from_holiday(
    X=data[['is_holiday']],
    holiday_column='is_holiday',
    fill_na=0,
)

# 3. Combine with other exogenous variables
exog = pd.concat([exog_external, exog_calendar, exog_holiday], axis=1)

# 4. Rolling features + lags + differencing
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

1. **Not encoding cyclical features**: Using raw integers for hour/month/day_of_week loses the cyclical relationship (hour 23 appears far from hour 0). Use `encoding='cyclical'` (or `'onehot'` / `'spline'`).
2. **Forgetting frequency on index**: `DateTimeFeatureTransformer` requires a `DatetimeIndex`; `calculate_distance_from_holiday` (index mode) infers the time unit from the frequency. Always set `data.asfreq('h')` (or similar) first.
3. **Not covering forecast horizon with exog**: Calendar / holiday features for `predict()` must include future dates covering the entire forecast horizon.
4. **Overriding `max_values` for `'week'` or `'day_of_year'`**: The defaults (53, 366) are intentional — they handle ISO week 53 and leap years correctly. Use the smaller value (52 / 365) only if you have verified your data never reaches the maximum.
5. **Over-engineering features**: Start with lags only, then add rolling features and calendar features incrementally. Validate each addition with backtesting.
