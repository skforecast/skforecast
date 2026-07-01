# Prediction Intervals — Compatibility Reference

## Interval Scale: Quantiles (Changed in 0.23.0)

Since skforecast 0.23.0, `interval` is expressed as **quantiles in `[0, 1]`**,
not percentiles in `[0, 100]`.

| Input | Behavior |
|-------|----------|
| All values in `[0, 1]`, e.g. `[0.05, 0.95]` | Treated as quantiles (recommended). |
| Single `float`, e.g. `0.9` | Nominal coverage; expands to symmetric quantiles `[0.05, 0.95]`. |
| All values in `(1, 100]`, e.g. `[5, 95]` | Legacy percentiles. Converted to quantiles with a `FutureWarning`. Removed in 0.25.0. |
| Mixed scale, e.g. `[0.05, 95]` | `ValueError` — ambiguous scale. |

`predict_quantiles(quantiles=...)` already used the 0-1 scale and is unaffected.

## Method Compatibility by Forecaster

| Forecaster | `'bootstrapping'` | `'conformal'` | Built-in (parametric) | Default method |
|------------|:------------------:|:-------------:|:---------------------:|:--------------|
| `ForecasterRecursive` | ✓ | ✓ | — | `'bootstrapping'` |
| `ForecasterDirect` | ✓ | ✓ | — | `'bootstrapping'` |
| `ForecasterRecursiveMultiSeries` | ✓ | ✓ | — | `'conformal'` |
| `ForecasterDirectMultiVariate` | ✓ | ✓ | — | `'conformal'` |
| `ForecasterEquivalentDate` | — | ✓ | — | `'conformal'` |
| `ForecasterRnn` | — | ✓ | — | `'conformal'` |
| `ForecasterStats` | — | — | ✓ | N/A (uses `alpha` or `interval`) |
| `ForecasterRecursiveClassifier` | — | — | — | N/A (use `predict_proba()`) |

## Parameter Differences by Forecaster

### ML Forecasters (Recursive, Direct, RecursiveMultiSeries, DirectMultiVariate)

```python
forecaster.predict_interval(
    steps,
    method='bootstrapping',          # or 'conformal'
    interval=[0.05, 0.95],           # quantiles → 90% interval (or a float, e.g. 0.9)
    n_boot=250,                      # only used with method='bootstrapping'
    use_in_sample_residuals=True,
    use_binned_residuals=True,
    random_state=123,
)
```

### ForecasterStats (different interface)

```python
forecaster.predict_interval(
    steps,
    alpha=0.05,                      # significance level → 95% interval
    interval=None,                   # list[float] | None — quantiles, alternative to alpha
)
# NOTE: No `method`, `n_boot`, `use_in_sample_residuals`, `use_binned_residuals`,
#       or `random_state` parameters.
# Use EITHER `alpha` OR `interval`, not both.
```

### ForecasterEquivalentDate (conformal only)

```python
forecaster.predict_interval(
    steps,
    method='conformal',              # only valid value
    interval=[0.05, 0.95],
    use_in_sample_residuals=True,
    use_binned_residuals=True,
    random_state=None,               # Any, accepted but ignored
    exog=None,                       # Any, accepted but ignored
    n_boot=None,                     # Any, accepted but ignored
)
# NOTE: `random_state`, `exog`, and `n_boot` exist for API compatibility but are ignored.
```

### ForecasterRnn (conformal only)

```python
forecaster.predict_interval(
    steps=None,
    levels=None,
    method='conformal',              # only valid value
    interval=[0.05, 0.95],
    use_in_sample_residuals=True,
    use_binned_residuals=True,       # bool, selects residuals by predicted value level
    n_boot=None,                     # Any, accepted but ignored
    random_state=None,               # Any, accepted but ignored
)
# NOTE: `n_boot` and `random_state` exist for API compatibility but are ignored.
```

## Backtesting with Intervals

| Backtesting function | `interval_method` default | Supports `n_boot` | Supports `alpha` |
|---------------------|:------------------------:|:-----------------:|:----------------:|
| `backtesting_forecaster` | `'bootstrapping'` | ✓ | — |
| `backtesting_forecaster_multiseries` | `'conformal'` | ✓ | — |
| `backtesting_stats` | N/A | — | ✓ |

## Prerequisites for Bootstrapping

Bootstrapping requires stored residuals. Two approaches:

### In-sample residuals (simpler)

```python
forecaster.fit(y=y_train, store_in_sample_residuals=True)
forecaster.predict_interval(
    steps=10,
    method='bootstrapping',
    use_in_sample_residuals=True,
)
```

### Out-of-sample residuals (better calibration)

```python
# Step 1: Get predictions via backtesting
metric, preds = backtesting_forecaster(forecaster=forecaster, y=data, cv=cv, metric='mae')

# Step 2: Store residuals
forecaster.set_out_sample_residuals(
    y_true=data.loc[preds.index],
    y_pred=preds['pred'],
)

# Step 3: Use out-of-sample residuals
forecaster.predict_interval(
    steps=10,
    method='bootstrapping',
    use_in_sample_residuals=False,  # Use out-of-sample
)
```

## Binned Residuals

When `use_binned_residuals=True`, residuals are selected based on the predicted
value level (using KBinsDiscretizer). This produces better-calibrated intervals
because residual variance often depends on the prediction magnitude.

| Forecaster | Supports `use_binned_residuals` |
|------------|:-------------------------------:|
| ForecasterRecursive | ✓ |
| ForecasterDirect | ✓ |
| ForecasterRecursiveMultiSeries | ✓ |
| ForecasterDirectMultiVariate | ✓ |
| ForecasterEquivalentDate | ✓ |
| ForecasterRnn | ✓ |
| ForecasterStats | — |

## Probabilistic Prediction Methods Beyond Intervals

| Method | Available in | Description |
|--------|-------------|-------------|
| `predict_interval()` | All except Classifier | Lower/upper bounds at given quantiles |
| `predict_quantiles()` | Recursive, Direct, RecursiveMultiSeries, DirectMultiVariate | Arbitrary quantile predictions |
| `predict_dist()` | Recursive, Direct, RecursiveMultiSeries, DirectMultiVariate | Fit a scipy distribution to bootstrapped predictions |
| `predict_proba()` | RecursiveClassifier only | Class probabilities |

### predict_quantiles signature

```python
forecaster.predict_quantiles(
    steps,
    quantiles=[0.05, 0.5, 0.95],    # any list of quantiles between 0 and 1
    n_boot=250,
    use_in_sample_residuals=True,
    use_binned_residuals=True,
    random_state=123,
)
# Returns DataFrame with one column per quantile: q_0.05, q_0.5, q_0.95
```

### predict_dist signature

```python
from scipy.stats import norm

forecaster.predict_dist(
    steps,
    distribution=norm,               # any scipy.stats distribution
    n_boot=250,
    use_in_sample_residuals=True,
    use_binned_residuals=True,
    random_state=123,
)
# Returns DataFrame with fitted distribution parameters (loc, scale, etc.)
```

## Common Error Patterns

| Error | Cause | Fix |
|-------|-------|-----|
| "No in-sample residuals stored" | `store_in_sample_residuals=False` in `fit()` | Set `store_in_sample_residuals=True` |
| `FutureWarning` about percentiles | `interval` passed as percentiles, e.g. `[5, 95]` | Use quantiles, e.g. `interval=[0.05, 0.95]` (removed in 0.25.0) |
| `ValueError` about ambiguous scale | `interval` mixes quantile and percentile values, e.g. `[0.05, 95]` | Use a single scale, e.g. `interval=[0.05, 0.95]` |
| `method='bootstrapping'` on ForecasterRnn | Bootstrapping not supported | Use `method='conformal'` |
| `method='bootstrapping'` on ForecasterEquivalentDate | Bootstrapping not supported | Use `method='conformal'` |
| Using `alpha` on ML forecasters | Only `ForecasterStats` accepts `alpha` | Use `interval=[lo, hi]` instead |
| Using `method` on ForecasterStats | Stats models use built-in parametric intervals | Remove `method`, use `alpha` or `interval` |
