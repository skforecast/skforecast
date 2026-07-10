---
name: metric-selection
description: >
  Guides selection of the most appropriate evaluation metric(s) for a
  forecasting task based on forecaster type, prediction output type, data
  characteristics, and multi-series aggregation needs. Use when the user
  asks which metric to use, how to evaluate forecast quality, or needs help
  configuring the `metric` parameter in backtesting or hyperparameter search.
---

# Metric Selection

## When to Use This Skill

Use this skill when the user needs help choosing an evaluation metric, comparing
metrics, understanding trade-offs between error measures, or configuring the
`metric` parameter in `backtesting_forecaster`, `bayesian_search_forecaster`,
or any other model selection function.

### Related skills

- **After**: `choosing-a-forecaster` (the forecaster type determines which metrics apply)
- **Before**: `hyperparameter-optimization` (the chosen metric drives the search objective)
- **Before**: `prediction-intervals` (probabilistic metrics evaluate interval quality)

## Quick Recommendations

If unsure, start here:

**Point forecast metrics** (pass as `metric=` in backtesting/search):

| Task | Default metric | Why |
|------|---------------|-----|
| General purpose | `'mean_absolute_error'` | Interpretable, robust to outliers, works at any scale |
| Compare across series | `'mean_absolute_scaled_error'` | Scale-independent — the only fair comparison across differently-scaled series |
| Classification | `'balanced_accuracy_score'` | Handles class imbalance common in time series classification |

**Probabilistic metrics** (computed post-hoc on interval/quantile predictions):

| Task | Metric | Why |
|------|--------|-----|
| Interval calibration | `calculate_coverage` | Check if actual coverage matches the nominal level |
| Interval quality (overall) | `crps_from_predictions` | Evaluates sharpness and calibration together |
| Quantile quality (foundation models) | `crps_from_quantiles` | Proper scoring rule for quantile predictions |
| Single-quantile optimization | `create_mean_pinball_loss(alpha)` | Can be passed as `metric=` to optimize for a specific quantile |

## Step 1 — What Are You Evaluating?

```
What is your forecaster producing?
│
├─► Point forecasts (single value per step)
│   └─► Go to Step 2
│
├─► Prediction intervals (lower_bound, upper_bound)
│   └─► Go to Step 3
│
├─► Quantile predictions (multiple quantile levels)
│   └─► Go to Step 3
│
├─► Class labels (ForecasterRecursiveClassifier)
│   └─► Go to Step 4
│
└─► Multi-series with aggregation needs
    └─► Go to Step 5 (then return to Step 2 or 3 for the base metric)
```

## Step 2 — Point Forecast Metrics

### Decision Table

| Criterion | Recommended metric | Avoid |
|-----------|-------------------|-------|
| **General purpose** | MAE (`'mean_absolute_error'`) | — |
| **Penalize large errors more** | MSE (`'mean_squared_error'`) | — |
| **Robust to outliers** | MAE or MedAE (`'median_absolute_error'`) | MSE (inflated by outliers) |
| **Need percentage interpretation** | SMAPE (`'symmetric_mean_absolute_percentage_error'`) | MAPE if data has zeros |
| **Data contains zeros or near-zero values** | MAE, MASE, RMSSE | MAPE (divides by y_true → infinite) |
| **Compare across different-scale series** | MASE (`'mean_absolute_scaled_error'`) or RMSSE (`'root_mean_squared_scaled_error'`) | MAE, MSE (scale-dependent) |
| **Target is always positive** | MSLE (`'mean_squared_log_error'`) | — (use when relative errors matter more than absolute) |
| **Interpretable baseline comparison** | MASE (value < 1 means better than naive forecast) | — |

### Metric Properties

| Metric | Scale-independent | Robust to outliers | Handles zeros | Requires `y_train` | Range |
|--------|:-----------------:|:------------------:|:-------------:|:-------------------:|-------|
| MAE | — | ✓ | ✓ | — | [0, ∞) |
| MSE | — | — | ✓ | — | [0, ∞) |
| MedAE | — | ✓✓ | ✓ | — | [0, ∞) |
| MAPE | ✓ | — | — | — | [0, ∞) |
| SMAPE | ✓ | — | ✓ | — | [0, 200] % |
| MSLE | — | — | ✓ (if ≥ 0) | — | [0, ∞) |
| MASE | ✓ | ✓ | ✓ | ✓ | [0, ∞) |
| RMSSE | ✓ | — | ✓ | ✓ | [0, ∞) |

### Using Metrics in Code

#### Direct computation (train/test split)

For a simple train/test evaluation, import the metric and call it directly.
Sklearn metrics are imported from `sklearn.metrics`; skforecast-specific metrics
(`mean_absolute_scaled_error`, `root_mean_squared_scaled_error`,
`symmetric_mean_absolute_percentage_error`) from `skforecast.metrics`.

```python
from sklearn.metrics import mean_squared_error
from skforecast.metrics import mean_absolute_scaled_error

forecaster.fit(y=data_train)
predictions = forecaster.predict(steps=36)

# Sklearn metrics: func(y_true, y_pred)
error_mse = mean_squared_error(y_true=data_test, y_pred=predictions)

# Skforecast metrics that need y_train: func(y_true, y_pred, y_train)
error_mase = mean_absolute_scaled_error(
    y_true=data_test, y_pred=predictions, y_train=data_train
)
```

#### Inside backtesting

When using backtesting or hyperparameter search, pass metrics as strings —
no import needed, and `y_train` is handled automatically:

```python
from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold

cv = TimeSeriesFold(steps=12, initial_train_size=len(y_train), refit=False)

# Single metric (pass as string)
metric, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=data['target'],
    cv=cv,
    metric='mean_absolute_error',
)

# Multiple metrics at once
metric, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=data['target'],
    cv=cv,
    metric=['mean_absolute_error', 'mean_absolute_scaled_error'],
)
# metric is a DataFrame with one column per metric
```

**MASE and RMSSE** require training data to compute the naive forecast baseline.
When passed as a string or callable, skforecast handles `y_train` automatically —
it detects whether the function signature includes a `y_train` parameter and
provides the training data if so.

```python
# Both options work identically in backtesting — y_train is handled internally
metric = 'mean_absolute_scaled_error'            # Option 1: as string
metric = mean_absolute_scaled_error              # Option 2: as callable (import from skforecast.metrics)
```

## Step 3 — Probabilistic Forecast Metrics

Use these metrics when evaluating prediction intervals or quantile forecasts.

| Metric | Evaluates | Input | Use case |
|--------|-----------|-------|----------|
| `calculate_coverage` | Calibration | y_true, lower_bound, upper_bound | Check if actual coverage matches nominal level |
| `crps_from_predictions` | Calibration + sharpness | y_true (scalar), y_pred (array of bootstrap samples) | Evaluate bootstrapped interval quality |
| `crps_from_quantiles` | Calibration + sharpness | y_true (scalar), pred_quantiles, quantile_levels | Evaluate quantile predictions (foundation models) |
| `create_mean_pinball_loss(alpha)` | Single-quantile accuracy | y_true, y_pred (at quantile alpha) | Evaluate a specific quantile forecast |

### Coverage

Coverage measures the proportion of true values that fall within the predicted
interval. **Target: match the nominal level** (e.g., 90% interval should have ~90% coverage).

```python
from skforecast.metrics import calculate_coverage

coverage = calculate_coverage(
    y_true=y_test,
    lower_bound=predictions['lower_bound'],
    upper_bound=predictions['upper_bound'],
)
# Ideal: coverage ≈ 0.90 for a 90% interval
# coverage >> 0.90 → intervals too wide (not sharp)
# coverage << 0.90 → intervals too narrow (miscalibrated)
```

### CRPS (Continuous Ranked Probability Score)

CRPS is a proper scoring rule that rewards both calibration and sharpness.
Lower is better.

```python
from skforecast.metrics import crps_from_predictions, crps_from_quantiles
import numpy as np

# From bootstrap samples (e.g., predict_bootstrapping output)
crps = crps_from_predictions(
    y_true=100.0,
    y_pred=np.array([98.2, 101.5, 99.8, 102.1, 97.5])  # Bootstrap predictions
)

# From quantile predictions (e.g., foundation model output)
crps = crps_from_quantiles(
    y_true=100.0,
    pred_quantiles=np.array([90.0, 95.0, 100.5, 105.0, 110.0]),
    quantile_levels=np.array([0.1, 0.25, 0.5, 0.75, 0.9]),
)
```

### Pinball Loss (Quantile Loss)

Evaluates a single quantile forecast. Use `create_mean_pinball_loss(alpha)` to
create a metric function for a specific quantile level.

```python
from skforecast.metrics import create_mean_pinball_loss

# Create metric for the 90th percentile
pinball_90 = create_mean_pinball_loss(alpha=0.9)

# Use in backtesting (as callable)
metric, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=data['target'],
    cv=cv,
    metric=pinball_90,
)
```

## Step 4 — Classification Metrics

For `ForecasterRecursiveClassifier` only. These metrics evaluate predicted
class labels, not continuous values.

| Metric | Best for |
|--------|----------|
| `'accuracy_score'` | Balanced classes (equal importance per class) |
| `'balanced_accuracy_score'` | Imbalanced classes (common in time series — e.g., rare events) |
| `'f1_score'` | When both precision and recall matter |
| `'precision_score'` | When false positives are costly |
| `'recall_score'` | When false negatives are costly (e.g., missing a spike) |

```python
from skforecast.recursive import ForecasterRecursiveClassifier
from sklearn.ensemble import RandomForestClassifier
from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold

forecaster = ForecasterRecursiveClassifier(
    estimator=RandomForestClassifier(n_estimators=100, random_state=123),
    lags=24,
)

# y contains class labels (e.g., 'low', 'medium', 'high'); encoding is handled internally
cv = TimeSeriesFold(steps=10, initial_train_size=len(y) - 100, refit=False)

metric, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=y,
    cv=cv,
    metric='balanced_accuracy_score',
)
```

## Step 5 — Multi-Series Metric Aggregation

When predicting multiple series, skforecast computes per-series metrics and
can aggregate them into a single score. The aggregation options are:

| Aggregation | Formula | Use when |
|-------------|---------|----------|
| `'average'` | Arithmetic mean of per-series metrics | All series equally important |
| `'weighted_average'` | Weighted by number of predicted values per level | Series with more observations contribute more |
| `'pooling'` | Metric computed on all predictions pooled together | Want a single global error regardless of series identity |

### In backtesting

Use `add_aggregated_metric=True` (default) to include all three aggregations:

```python
from skforecast.model_selection import backtesting_forecaster_multiseries

metric, predictions = backtesting_forecaster_multiseries(
    forecaster=forecaster_multi,
    series=series_df,
    cv=cv,
    metric='mean_absolute_error',
    add_aggregated_metric=True,  # Default: includes average, weighted_average, pooling
)
# metric DataFrame has rows for each level + aggregated rows
```

### In hyperparameter search

Use the `aggregate_metric` parameter to select which aggregations to report:

```python
from skforecast.model_selection import bayesian_search_forecaster_multiseries

results, study = bayesian_search_forecaster_multiseries(
    forecaster=forecaster_multi,
    series=series_df,
    cv=cv,
    search_space=search_space,
    metric='mean_absolute_error',
    aggregate_metric=['weighted_average', 'average', 'pooling'],  # Select which to report
)
```

**Tip:** Use `'pooling'` when you care about overall accuracy regardless
of which series contributes the error. Use `'average'` when all series are
equally important regardless of their length.

## Custom Metrics

Any callable with signature `func(y_true, y_pred)` can be passed as a metric.
If your metric also needs training data, include a `y_train` parameter:

```python
# Custom metric without y_train
def mean_bias_error(y_true, y_pred):
    return np.mean(y_pred - y_true)

# Custom metric WITH y_train (detected automatically)
def relative_mase(y_true, y_pred, y_train):
    naive_mae = np.mean(np.abs(np.diff(y_train)))
    return np.mean(np.abs(y_true - y_pred)) / naive_mae

# Both work directly in backtesting
metric, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=data['target'],
    cv=cv,
    metric=[mean_bias_error, relative_mase],
)
```

Skforecast automatically wraps callables via `add_y_train_argument`. If the function
already has a `y_train` parameter, it will receive the training data. If not, the
argument is silently ignored.

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Using MAPE with data containing zeros | Division by zero → infinite error | Use SMAPE, MAE, or MASE instead |
| Using MSE when data has outliers | A few large errors dominate the metric | Use MAE or MedAE for robustness |
| Comparing MAE across differently-scaled series | MAE = 10 on sales vs MAE = 10 on temperature means nothing | Use MASE or RMSSE for cross-series comparison |
| Interpreting high coverage as "good" | 99% coverage with a 90% interval means intervals are too wide | Target coverage ≈ nominal level; combine with CRPS for sharpness |
| Using only point metrics for probabilistic forecasts | Ignores uncertainty quality entirely | Add `calculate_coverage` and/or CRPS alongside point metrics |
| Passing MASE as callable without understanding y_train | Works fine — skforecast detects the `y_train` parameter automatically | Just pass the callable or string; no extra work needed |

## Metric Compatibility Reference

See [references/metric-compatibility.md](references/metric-compatibility.md) for the
complete matrix of all 18 metrics with their properties, compatible forecasters,
and usage guidance.
