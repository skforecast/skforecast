---
name: feature-engineering
description: >
    Creates features for time series forecasting: calendar features with
    skforecast's `CalendarFeatures` (cyclical, onehot, or spline encoding) —
    either delegated to the forecaster via the `calendar_features` parameter or
    built manually as exog — holiday distance features with
    `calculate_distance_from_holiday`, rolling statistics with `RollingFeatures`,
    differencing, and categorical exogenous variables. Use when the user wants to
    improve model accuracy through feature engineering or asks about exogenous
    variable creation.
---

# Feature Engineering

## References

See [references/calendar-features-reference.md](references/calendar-features-reference.md)
for the complete `CalendarFeatures` constructor, all supported features and
encodings, the delegated (`calendar_features` parameter) vs. manual (`exog`)
workflows, per-forecaster support, and gotchas.

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
| `calendar_features` param | skforecast ML forecasters | **Delegated** calendar features: pass a `CalendarFeatures` instance and the forecaster generates them at train and predict — no manual exog. **New in 0.23.0.** |
| `CalendarFeatures` | `skforecast.preprocessing` | Sklearn-compatible transformer: extract calendar features from a `DatetimeIndex` and (optionally) encode them. Pass to `calendar_features`, use as `transformer_exog`, or in a `Pipeline`. |
| `create_calendar_features` | `skforecast.preprocessing` | Function form of the same logic, for one-shot use without a transformer. |
| `calculate_distance_from_holiday` | `skforecast.preprocessing` | Periods to next / since last holiday |
| `RollingFeatures` | `skforecast.preprocessing` | Rolling window statistics (mean, std, min, max, etc.) |
| `differentiation` param | skforecast forecasters | Make non-stationary series stationary |

## Calendar Features

There are **two ways** to add calendar features (month, day of week, hour, …).
Both use the `CalendarFeatures` class; they differ in who builds the features.

| Workflow | How | When to use |
|----------|-----|-------------|
| **Delegated** (new in 0.23.0) | Pass a `CalendarFeatures` instance to the forecaster's `calendar_features` parameter. The forecaster generates the features at train **and** predict — no manual exog. | The 4 supported forecasters: `ForecasterRecursive`, `ForecasterRecursiveMultiSeries`, `ForecasterDirect`, `ForecasterDirectMultiVariate`. |
| **Manual** | Build features with `CalendarFeatures.fit_transform` / `create_calendar_features` and pass them as `exog` (or wire `CalendarFeatures` as `transformer_exog`). | Forecasters without `calendar_features` support (`ForecasterRecursiveClassifier`, `ForecasterRnn`, `ForecasterStats`, `ForecasterFoundation`, `ForecasterEquivalentDate`), or when the features live inside a `Pipeline` / `ColumnTransformer`. |

> Full constructor, all features/encodings, per-forecaster support, and gotchas:
> [references/calendar-features-reference.md](references/calendar-features-reference.md).

**Supported features:** `'year'`, `'month'`, `'week'`, `'day_of_week'`,
`'day_of_month'`, `'day_of_year'`, `'weekend'`, `'hour'`, `'minute'`,
`'second'`, `'quarter'`. **Encodings:** `'cyclical'` (default, sin/cos pair),
`'onehot'`, `'spline'`, `None`. `'year'` and `'weekend'` are never encoded.

### Delegated — `calendar_features` parameter (preferred when supported)

The forecaster builds the calendar features from the datetime index during both
training and prediction. You do **not** build or pass a calendar `exog`, and you
do **not** need to cover the forecast horizon manually.

```python
from lightgbm import LGBMRegressor
from skforecast.preprocessing import CalendarFeatures
from skforecast.recursive import ForecasterRecursive

calendar = CalendarFeatures(
    features=['month', 'day_of_week', 'hour'],
    encoding='cyclical',                 # 'cyclical' | 'onehot' | 'spline' | None
    keep_original_columns=False,
)

forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=24,
    calendar_features=calendar,          # requires a pandas DatetimeIndex
)
forecaster.fit(y=y_train)                # calendar features built automatically
predictions = forecaster.predict(steps=24)   # calendar features built automatically
```

Requires a `DatetimeIndex` (otherwise `TypeError`). Calendar features are added
as predictors alongside lags, window features, and any `exog`.

### Manual — build calendar features as exog

For forecasters without the `calendar_features` parameter, or for use in a
`Pipeline`:

```python
import pandas as pd
from skforecast.preprocessing import CalendarFeatures

data = data.asfreq('h')                  # DatetimeIndex with frequency set

calendar_transformer = CalendarFeatures(
    features=['month', 'week', 'day_of_week', 'hour'],
    encoding='cyclical',
    keep_original_columns=False,
)
exog_calendar = calendar_transformer.fit_transform(data)
# Columns: month_sin, month_cos, week_sin, week_cos, ...

# The exog passed to predict() must cover the entire forecast horizon.
```

The function form `create_calendar_features(X=data, ...)` does the same in one
shot without a transformer. See the reference for `max_values`, `spline_kwargs`,
and `features_to_encode` options.

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

### The `window_features` parameter

The forecaster's `window_features` argument accepts **any object (or list of objects)** that implements the contract:

```python
def transform_batch(y: pd.Series) -> pd.DataFrame:
    # Returns one row per training sample, indexed to align with y[max_window_size:]
    ...
```

`RollingFeatures` is the built-in implementation; user-defined classes that follow the same contract are accepted. See [references/rolling-stats-reference.md](references/rolling-stats-reference.md#custom-window-feature-classes-protocol) for the full protocol and a custom-class example.

**Effective `window_size`.** The forecaster sets `window_size = max(max_lag, max_size_window_features) (+ differentiation)`. A `RollingFeatures(window_sizes=30)` combined with `lags=7` raises `window_size` to 30 — `last_window` must hold ≥ 30 values and the first 30 rows of training data are dropped. This is a frequent source of "not enough data to train" errors when the rolling window is much larger than the lags.

**Lags-free configuration.** `lags=None, window_features=...` is valid (and useful for purely smoothed-history models). At least one of `lags` / `window_features` must be non-`None` — passing both as `None` raises `ValueError`.

**Multiple `RollingFeatures` instances.** When passing a list, each instance is processed independently and their feature names are concatenated. Names must be unique across instances; collisions silently overwrite columns. Use the `features_names=` argument of `RollingFeatures` to disambiguate when reusing the same `stat`/`window_size` combination.

**Runtime mutation.** After construction, use `forecaster.set_window_features(new_window_features)` to swap the configuration (analogous to `set_lags`). This recomputes `window_size`, `max_size_window_features`, `window_features_names`, and the differentiator's window. Useful inside manual hyperparameter loops.

**Interaction with `differentiation`.** Window features are computed on the **differenced** series, not the original one. With `differentiation=1`, `roll_mean_7` is the mean of seven consecutive *changes*, not seven raw values. Take this into account when interpreting feature importances.

**Multi-series.** In `ForecasterRecursiveMultiSeries`, the same `RollingFeatures` instance is applied independently to each series — configuration is global but the computed values are per-series (no leakage between series).

**Classifier variant.** For `ForecasterRecursiveClassifier`, use `RollingFeaturesClassification` (also in `skforecast.preprocessing`) instead of `RollingFeatures`.

### Choosing window features

**When window features help:**
- Noisy series — a smoothed signal (`mean`, `ewm`) reduces variance the lags cannot.
- Capturing a longer scale than the lags without exploding column count: with hourly data and `lags=24`, a `roll_mean_168` represents the weekly level with one feature instead of adding 144 more lags.
- Heteroscedastic or regime-changing series (energy, finance, traffic) — `roll_std`, `coef_variation`.
- When extreme behaviour drives decisions (peak demand, drawdowns) — `roll_min`, `roll_max`.

**When they don't help:**
- Short series — each large window discards training rows (`window_size` grows).
- Highly periodic, low-noise series — lags + calendar features are usually enough.
- Redundant ranges — a `roll_mean_24` alongside `lags=range(1, 25)` adds little for a gradient boosting model that can already construct the average from splits.

**Which statistic for which signal:**

| Stat | Adds signal when… | Notes |
|------|------------------|-------|
| `mean` | Default smoothing — recent level. | Start here. |
| `ewm` | Recent observations should dominate. | Tune `alpha` in `kwargs_stats` (0.1 slow, 0.5 reactive). |
| `std` | Volatility / heteroscedasticity matters. | Near-mandatory in finance and energy demand. |
| `median` | Frequent outliers. | Replaces `mean`, do not combine. |
| `min` / `max` | Threshold crossings, peaks, operational floors. | Useful in capacity planning. |
| `sum` | Cumulative quantities (rainfall, sales) rather than levels. | Scale-sensitive — pair with `transformer_y` or scaling. |
| `ratio_min_max` | Bounded regime indicator. | Often noisy on stationary series. |
| `coef_variation` | Scale-free volatility. | Useful in `ForecasterRecursiveMultiSeries` to compare series of different magnitudes. |

**Window sizes:**
- Tie them to actual seasonality: hourly → 24, 168; daily → 7, 30; weekly → 4, 13, 52; monthly → 12.
- Multi-scale usually wins: combine one short (reactive) and one long (trend) window — e.g. `stats=['mean', 'std', 'mean', 'std'], window_sizes=[7, 7, 30, 30]`.
- Keep `max_window_size < ~25-30% of len(y_train)` to preserve enough training rows.

**Practical recipe:**
1. **Baseline**: lags only (from ACF/PACF) + calendar features if applicable. Measure via backtesting.
2. **If residuals track recent level** → add `roll_mean_<seasonal_period>`.
3. **If errors scale with magnitude** → add `roll_std_<seasonal_period>`.
4. **If outliers/peaks matter** → add `roll_max` or `roll_min` with the same window.
5. **One addition per iteration**, validated with backtesting. Drop it if MAE/RMSE doesn't improve on independent folds.
6. **Prune with `select_features`** at the end — typically 2-4 of 6-8 candidates survive.

**Combinations that tend to work:**
- **Hourly electricity demand**: `lags=[1..24, 168]` + `RollingFeatures(['mean', 'std'], [24, 24])` + cyclical calendar.
- **Daily sales with promotions**: `lags=[1..14]` + `RollingFeatures(['mean', 'median'], [7, 28])` + holiday distance + promo exog.
- **Financial series**: `lags=[1..5]` + `RollingFeatures(['std', 'coef_variation'], [5, 20])` + `differentiation=1` (window features compute on returns, which is usually what you want).
- **Multi-series with heterogeneous scales**: prefer `coef_variation` and `ratio_min_max` (scale-free), or apply per-series `transformer_series`.

**Anti-patterns:**
- Adding 6-8 statistics at once "just in case" — dilutes signal and increases overfitting risk.
- `roll_sum` without scaling the target — its magnitude dominates the lags and destabilises training.
- `min` / `max` with very long windows on trending series — the feature becomes near-constant or monotone.
- Duplicating information: keeping `lags=range(1, 31)` together with `roll_mean_30` rarely helps a gradient boosting model — shrink the lag set when you add the rolling stat.

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
    CalendarFeatures,
    calculate_distance_from_holiday,
    RollingFeatures,
)
from skforecast.recursive import ForecasterRecursive
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

# 1. Calendar features with cyclical encoding (no extra deps)
calendar_transformer = CalendarFeatures(
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

### Variant — delegate calendar features to the forecaster

On a forecaster that supports `calendar_features`, drop the manual calendar exog
and let the forecaster build it at train and predict time. Only the
non-calendar exog (holiday + external) is passed manually:

```python
calendar = CalendarFeatures(
    features=['month', 'day_of_week', 'hour'],
    encoding='cyclical',
    keep_original_columns=False,
)

exog = pd.concat([exog_external, exog_holiday], axis=1)   # no calendar columns here

forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=[1, 2, 3, 7, 14, 24],
    window_features=rolling,
    calendar_features=calendar,          # built automatically at fit/predict
    transformer_y=StandardScaler(),
    differentiation=1,
)
forecaster.fit(y=y_train, exog=exog.loc[y_train.index])
predictions = forecaster.predict(steps=10, exog=exog.loc[forecast_index])
```

## Common Mistakes

1. **Not encoding cyclical features**: Using raw integers for hour/month/day_of_week loses the cyclical relationship (hour 23 appears far from hour 0). Use `encoding='cyclical'` (or `'onehot'` / `'spline'`).
2. **Forgetting frequency on index**: `CalendarFeatures` (and the `calendar_features` parameter) requires a `DatetimeIndex`; `calculate_distance_from_holiday` (index mode) infers the time unit from the frequency. Always set `data.asfreq('h')` (or similar) first.
3. **Not covering forecast horizon with calendar/holiday exog (manual path only)**: When building calendar features manually as `exog`, the `predict()` exog must include future dates covering the entire forecast horizon. With the delegated `calendar_features` parameter this is handled automatically.
4. **Mixing the delegated and manual calendar paths for the same features**: Passing `calendar_features=` *and* also adding manually-built calendar columns with the same names to `exog` raises `ValueError: Duplicated feature names detected in X_train`. Pick one path per feature.
5. **Using `calendar_features` on an unsupported forecaster**: Only `ForecasterRecursive`, `ForecasterRecursiveMultiSeries`, `ForecasterDirect`, and `ForecasterDirectMultiVariate` accept it. For `ForecasterRecursiveClassifier`, `ForecasterRnn`, `ForecasterStats`, `ForecasterFoundation`, and `ForecasterEquivalentDate`, build calendar features manually as `exog`.
6. **Overriding `max_values` for `'week'` or `'day_of_year'`**: The defaults (53, 366) are intentional — they handle ISO week 53 and leap years correctly. Use the smaller value (52 / 365) only if you have verified your data never reaches the maximum.
7. **Over-engineering features**: Start with lags only, then add rolling features and calendar features incrementally. Validate each addition with backtesting.
8. **Rolling window larger than lags without checking training size**: The forecaster's `window_size` is `max(max_lag, max_size_window_features) (+ differentiation)`. A 30-step rolling window with `lags=7` drops the first 30 rows of training data and requires `last_window` of length ≥ 30 at predict time.
9. **Colliding feature names across multiple `RollingFeatures`**: When passing `window_features=[wf1, wf2]`, names like `roll_mean_7` produced by both instances overwrite each other. Override with `features_names=[...]` on at least one instance.
10. **Forgetting that window features run on the differenced series**: With `differentiation=1`, `roll_mean_7` is the mean of seven changes, not seven raw values. Adjust interpretation accordingly.
