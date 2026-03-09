---
name: feature-selection
description: >
  Selects the most relevant lags, window features, and exogenous variables
  using sklearn feature selectors (RFECV, SelectFromModel). Covers single-series
  and multi-series selection with force inclusion and subsampling.
  Use when the user has many features and wants to identify the most
  important ones.
---

# Feature Selection

## When to Use

Use feature selection when:
- You have many lags or exogenous variables and want to reduce overfitting
- You want to identify which features matter most
- You need to speed up training by removing irrelevant features

## Single Series

`select_features` works with `ForecasterRecursive` and `ForecasterDirect`.

```python
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.preprocessing import RollingFeatures
from skforecast.feature_selection import select_features

# Create forecaster with many candidate features
rolling = RollingFeatures(stats=['mean', 'std', 'min', 'max'], window_sizes=[7, 14])

forecaster = ForecasterRecursive(
    estimator=RandomForestRegressor(n_estimators=100, random_state=123),
    lags=48,  # Many lags — feature selection will reduce
    window_features=rolling,
)

# Run feature selection
selected_lags, selected_window_features, selected_exog = select_features(
    forecaster=forecaster,
    selector=RFECV(
        estimator=RandomForestRegressor(n_estimators=50, random_state=123),
        step=1,
        cv=3,
    ),
    y=y_train,
    exog=exog_train,
    select_only=None,          # 'autoreg' (lags+window), 'exog', or None (all)
    force_inclusion=None,      # Features to always keep (list or regex string)
    subsample=0.5,             # Use 50% of data for faster selection
    random_state=123,
    verbose=True,
)

# Apply selected lags to the same forecaster (simplest approach)
forecaster.set_lags(selected_lags)

# selected_window_features is a list of names (strings), not the RollingFeatures
# object. Use the names to verify which window features were selected.
print(f'Selected window features: {selected_window_features}')
print(f'Selected exog variables: {selected_exog}')
```

## Multi-Series

`select_features_multiseries` works with `ForecasterRecursiveMultiSeries` and `ForecasterDirectMultiVariate`.

> **Note:** When used with `ForecasterDirectMultiVariate`, `selected_lags` is returned as a `dict` (one entry per series) instead of a `list`.

```python
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.feature_selection import select_features_multiseries

forecaster = ForecasterRecursiveMultiSeries(
    estimator=RandomForestRegressor(n_estimators=100, random_state=123),
    lags=48,
    encoding='ordinal',
)

selected_lags, selected_window_features, selected_exog = select_features_multiseries(
    forecaster=forecaster,
    selector=RFECV(
        estimator=RandomForestRegressor(n_estimators=50, random_state=123),
        step=1,
        cv=3,
    ),
    series=series_df,
    exog=exog_df,
    select_only=None,
    force_inclusion=None,
    subsample=0.5,
    random_state=123,
    verbose=True,
)
```

## Force Inclusion

```python
# Always keep specific features regardless of selection
selected_lags, selected_wf, selected_exog = select_features(
    forecaster=forecaster,
    selector=selector,
    y=y_train,
    exog=exog_train,
    force_inclusion=['temperature', 'holiday'],  # Always keep these exog columns
)

# Regex pattern to force include
selected_lags, selected_wf, selected_exog = select_features(
    forecaster=forecaster,
    selector=selector,
    y=y_train,
    exog=exog_train,
    force_inclusion='^lag_',  # Keep all lag features
)
```

## Select Only Specific Feature Types

```python
# Only select among exogenous variables (keep all lags)
selected_lags, selected_wf, selected_exog = select_features(
    forecaster=forecaster,
    selector=selector,
    y=y_train,
    exog=exog_train,
    select_only='exog',  # Only select exog, keep all autoregressive features
)

# Only select among autoregressive features (keep all exog)
selected_lags, selected_wf, selected_exog = select_features(
    forecaster=forecaster,
    selector=selector,
    y=y_train,
    exog=exog_train,
    select_only='autoreg',  # Only select lags+window features, keep all exog
)
```

## Common Mistakes

1. **Using the wrong selector**: RFECV works best for recursive feature elimination. For faster selection, use `SelectFromModel`.
2. **Too small subsample**: If `subsample` is too small, selection may be unreliable. Use at least 0.3.
3. **Not updating forecaster**: After selection, update the forecaster with `forecaster.set_lags(selected_lags)` — the original is not modified in place by `select_features`.
4. **Running on full dataset**: Always run on training data only (`y_train`, `exog_train`).
5. **Confusing `selected_window_features` with `RollingFeatures`**: The returned `selected_window_features` is a list of **feature name strings** (e.g. `['mean_7', 'std_14']`), not the `RollingFeatures` object itself. Use these names to verify which window features were kept, but pass the original `RollingFeatures` instance to the forecaster.
