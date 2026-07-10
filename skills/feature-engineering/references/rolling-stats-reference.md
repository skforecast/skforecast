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

| Forecaster | `window_features` supported | Notes |
|------------|:--:|-------|
| ForecasterRecursive | ✓ | |
| ForecasterDirect | ✓ | |
| ForecasterRecursiveMultiSeries | ✓ | Same instance applied independently per series |
| ForecasterDirectMultiVariate | ✓ | Same instance applied independently per input series |
| ForecasterRecursiveClassifier | ✓ | Use `RollingFeaturesClassification` (not `RollingFeatures`) |
| ForecasterRnn | — | |
| ForecasterStats | — | |
| ForecasterEquivalentDate | — | |
| ForecasterFoundation | — | |

## How the forecaster uses `window_features`

When the forecaster builds the training matrix, it calls `wf.transform_batch(y)` on each instance once. The returned DataFrame is appended to the lag matrix as additional predictors. At prediction time, the forecaster slices the last `max_size_window_features` values of the running history and calls `wf.transform(...)` (numpy fast path) for each step.

**Effective `window_size`:**

```
window_size = max(max_lag, max_size_window_features) [+ differentiation]
```

Consequences:
- The first `window_size` rows of `y_train` cannot be used as targets (no full history available to compute predictors).
- `last_window` passed to `predict(..., last_window=...)` must contain at least `window_size` observations.
- A large rolling window combined with short series may leave too few training samples — check `len(y) - window_size` before fitting.

**Differentiation interaction:** the forecaster differentiates the series first, then computes window features on the differenced values. `roll_mean_7` with `differentiation=1` therefore averages seven *changes*, not seven raw observations. Inverse-differentiation only applies to the predictions, not to the predictors themselves.

**Multiple instances:** `window_features=[wf1, wf2]` calls `transform_batch` on each instance and concatenates the columns. Feature names must be unique across the combined set; duplicates silently collide. Use `RollingFeatures(features_names=[...])` to disambiguate.

**Lags-free mode:** `lags=None, window_features=...` is supported. `window_size` falls back to `max_size_window_features`. At least one of the two must be non-`None` — passing both as `None` raises `ValueError` from `set_lags` / `set_window_features`.

**Runtime mutation:** `forecaster.set_window_features(new_wf)` (analogous to `set_lags`) replaces the configuration after init and recomputes `window_size`, `window_features_names`, `window_features_class_names`, and the differentiator's window size. Useful in manual hyperparameter loops where the forecaster instance is reused across configurations.

## Custom window feature classes (protocol)

`window_features` accepts any object implementing the following contract — `RollingFeatures` is one such implementation, but user-defined classes are accepted as long as they conform.

**Required methods:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `transform_batch` | `(y: pd.Series) -> pd.DataFrame` | Computes features for the full training series. Used during `fit`. |
| `transform` | `(X: np.ndarray) -> np.ndarray` | Computes features for a single (or batched) prediction window. Used during `predict`. |

**Required attributes:**

| Attribute | Type | Purpose |
|-----------|------|---------|
| `max_window_size` | `int` | Largest window the class needs — drives the forecaster's `window_size`. |
| `features_names` | `list[str]` | Output column / feature names (must match `transform_batch` columns and the order produced by `transform`). |

**Validation enforced by the forecaster** (see `_create_window_features`):
- `transform_batch` must return a `pd.DataFrame` (`TypeError` otherwise).
- The DataFrame length must equal `len(y) - max_window_size` (`ValueError` otherwise).
- The DataFrame index must equal the forecaster's `train_index` (`ValueError` otherwise).

**Minimal custom example:**

```python
import numpy as np
import pandas as pd

class LastDeltaFeature:
    """Feature: difference between the last and the first value of the window."""

    def __init__(self, window_size: int):
        self.max_window_size = window_size
        self.features_names = [f'last_minus_first_{window_size}']

    def transform_batch(self, y: pd.Series) -> pd.DataFrame:
        # closed='left' to avoid leakage — last point excluded
        last = y.shift(1).rolling(self.max_window_size).apply(lambda w: w[-1], raw=True)
        first = y.shift(1).rolling(self.max_window_size).apply(lambda w: w[0], raw=True)
        out = (last - first).iloc[self.max_window_size:]
        return out.to_frame(name=self.features_names[0])

    def transform(self, X: np.ndarray) -> np.ndarray:
        # X shape: (window_length,) or (window_length, n_samples)
        if X.ndim == 1:
            return np.array([X[-1] - X[0]])
        return (X[-1, :] - X[0, :]).reshape(-1, 1)

# Use it like a RollingFeatures instance
forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=7,
    window_features=LastDeltaFeature(window_size=14),
)
```

Combine custom and built-in instances freely:

```python
forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=7,
    window_features=[
        RollingFeatures(stats=['mean', 'std'], window_sizes=14),
        LastDeltaFeature(window_size=14),
    ],
)
```

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
