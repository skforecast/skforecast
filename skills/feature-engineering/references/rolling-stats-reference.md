# Rolling Features — Statistics Reference

## RollingFeatures Constructor

```python
from skforecast.preprocessing import RollingFeatures

rolling = RollingFeatures(
    stats,                           # str | list[str] (required)
    window_sizes,                    # int | list[int] (required)
    min_periods=None,                # int | list[int] | None — defaults to window_sizes
    features_names=None,             # list[str] | None — auto-generated if None
    fillna=None,                     # str | float | None — fill NaN in transform_batch
    kwargs_stats=                    # dict | None — extra args per statistic
        {'ewm': {'alpha': 0.3}},     # default
)
```

## Available Statistics

| Statistic | String | Description | Requires `kwargs_stats` |
|-----------|--------|-------------|:-:|
| Mean | `'mean'` | Arithmetic mean of the window | — |
| Std deviation | `'std'` | Standard deviation of the window | — |
| Minimum | `'min'` | Minimum value in the window | — |
| Maximum | `'max'` | Maximum value in the window | — |
| Sum | `'sum'` | Sum of values in the window | — |
| Median | `'median'` | Median value in the window | — |
| Min/Max ratio | `'ratio_min_max'` | `min / max` of the window | — |
| Coef. variation | `'coef_variation'` | `std / mean` of the window | — |
| Exp. weighted mean | `'ewm'` | Exponentially weighted mean | ✓ `{'ewm': {'alpha': ...}}` |

## Feature Name Generation

Default names follow the pattern: `roll_{stat}_{window_size}`

| Stats | Window | Generated name |
|-------|--------|---------------|
| `'mean'` | `7` | `roll_mean_7` |
| `'std'` | `14` | `roll_std_14` |
| `'ewm'` | `7` (alpha=0.3) | `roll_ewm_7_alpha_0.3` |

Override with custom names:

```python
rolling = RollingFeatures(
    stats=['mean', 'std'],
    window_sizes=[7, 14],
    features_names=['weekly_avg', 'biweekly_std'],
)
```

## Window Behavior

Rolling windows use `closed='left'` and `center=False` to avoid data leakage.
The last point in the window is **excluded** from calculations:

```
Window size = 3, calculating for time t:
Uses values: [t-3, t-2, t-1]  (NOT t itself)
```

## Configuration Patterns

### Same window for all stats

```python
rolling = RollingFeatures(
    stats=['mean', 'std', 'min', 'max'],
    window_sizes=7,                   # int → applied to all 4 stats
)
# Features: roll_mean_7, roll_std_7, roll_min_7, roll_max_7
```

### Different windows per stat

```python
rolling = RollingFeatures(
    stats=['mean', 'std', 'min', 'max'],
    window_sizes=[7, 7, 14, 14],      # list → must match length of stats
)
# Features: roll_mean_7, roll_std_7, roll_min_14, roll_max_14
```

### Repeated stats with different windows

```python
rolling = RollingFeatures(
    stats=['mean', 'mean', 'std', 'std'],
    window_sizes=[7, 30, 7, 30],
)
# Features: roll_mean_7, roll_mean_30, roll_std_7, roll_std_30
```

### Multiple RollingFeatures objects

```python
rolling_short = RollingFeatures(stats=['mean', 'std'], window_sizes=7)
rolling_long = RollingFeatures(stats=['mean', 'std'], window_sizes=30)

forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=24,
    window_features=[rolling_short, rolling_long],  # list of RollingFeatures
)
```

### Exponentially weighted mean

```python
rolling = RollingFeatures(
    stats=['ewm'],
    window_sizes=7,
    kwargs_stats={'ewm': {'alpha': 0.3}},
)
# Feature: roll_ewm_7_alpha_0.3

# Multiple ewm with different alphas (use separate objects)
ewm_fast = RollingFeatures(stats=['ewm'], window_sizes=7, kwargs_stats={'ewm': {'alpha': 0.5}})
ewm_slow = RollingFeatures(stats=['ewm'], window_sizes=7, kwargs_stats={'ewm': {'alpha': 0.1}})

forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=24,
    window_features=[ewm_fast, ewm_slow],
)
```

## min_periods Parameter

Controls the minimum number of observations required to compute a statistic.

```python
# Default: min_periods = window_sizes (requires full window)
rolling = RollingFeatures(stats=['mean'], window_sizes=7)
# Equivalent to: min_periods=7 → first 6 values are NaN

# Allow partial windows
rolling = RollingFeatures(stats=['mean'], window_sizes=7, min_periods=1)
# min_periods=1 → computes mean even with 1 observation
```

## fillna Parameter

Used only in `transform_batch()` method (not in recursive prediction):

| Value | Behavior |
|-------|----------|
| `None` | No filling (NaN remains) |
| `'mean'` | Fill with mean of the feature |
| `'median'` | Fill with median of the feature |
| `'ffill'` | Forward fill |
| `'bfill'` | Backward fill |
| `float` | Fill with specific value (e.g., `0.0`) |

## Forecaster Compatibility

| Forecaster | `window_features` supported |
|------------|:--:|
| ForecasterRecursive | ✓ |
| ForecasterDirect | ✓ |
| ForecasterRecursiveMultiSeries | ✓ |
| ForecasterDirectMultiVariate | ✓ |
| ForecasterRecursiveClassifier | ✓ |
| ForecasterRnn | — |
| ForecasterStats | — |
| ForecasterEquivalentDate | — |

## Feature Selection Interaction

When using `select_features()`, the returned `selected_window_features` is a
**list of feature name strings** (e.g., `['roll_mean_7', 'roll_std_14']`),
NOT the `RollingFeatures` object. The original `RollingFeatures` instance
should still be passed to the forecaster.

```python
selected_lags, selected_wf, selected_exog = select_features(
    forecaster=forecaster,
    selector=RFECV(...),
    y=y_train,
)
# selected_wf = ['roll_mean_7', 'roll_std_14']  ← names, not objects
# The forecaster still uses the original RollingFeatures object
```
