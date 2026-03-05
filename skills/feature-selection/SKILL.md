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

# Apply selected features to forecaster
forecaster = ForecasterRecursive(
    estimator=RandomForestRegressor(n_estimators=100, random_state=123),
    lags=selected_lags,
    window_features=rolling,  # Will only use selected window features
)
```

## Multi-Series

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
3. **Not updating forecaster**: After selection, create a new forecaster with the selected lags — the original is not modified.
4. **Running on full dataset**: Always run on training data only (`y_train`, `exog_train`).
