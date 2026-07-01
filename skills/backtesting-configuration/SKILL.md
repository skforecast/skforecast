---
name: backtesting-configuration
description: >
  Configures TimeSeriesFold parameters for backtesting based on deployment
  scenarios. Maps business requirements (retraining frequency, forecast
  horizon, data budget) to cross-validation strategy parameters.
  Use when the user describes how they plan to deploy or evaluate a model.
---

# Backtesting Configuration

## When to Use

Use this skill to translate a deployment scenario (retraining cadence, forecast
horizon, data budget, ingestion delay) into `TimeSeriesFold` parameters.
`TimeSeriesFold` is the cross-validation strategy passed via the `cv` argument to
`backtesting_forecaster` (and its multi-series / stats variants) and to the
hyperparameter search functions.

```python
from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold

cv = TimeSeriesFold(
    steps=7,                       # forecast horizon
    initial_train_size=365,        # first training window (required)
    refit=False,                   # train once (default)
    fixed_train_size=True,         # rolling window (default)
    gap=0,
)

metric, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=data['target'],
    cv=cv,
    metric='mean_absolute_error',
)
```

For fast hyperparameter tuning where multi-step realism is not required, use
`OneStepAheadFold` instead: it validates one step ahead (no recursive
prediction), so it is much faster but less representative of multi-step
performance. Use `TimeSeriesFold` for realistic multi-step backtesting.

### Related skills

- **Before**: `forecasting-single-series` / `forecasting-multiple-series` (have a fitted forecaster before backtesting)
- **With**: `metric-selection` (choose the metric(s) backtesting reports)
- **With**: `hyperparameter-optimization` (the same `cv` object drives the search functions)
- **After**: `prediction-intervals` (add `interval=` / `interval_method=` to backtest uncertainty)

## Stop Conditions

Scan before writing code. Each row lists a rule, the symptom when it is broken, and the recovery. Full pitfall catalog: the `troubleshooting-common-errors` skill.

| Rule | Symptom | Recovery |
|------|---------|----------|
| `initial_train_size` must be provided (the API default is `None`, which only works when reusing an already-fitted forecaster) | `ValueError` / no training in the first fold | Pass an int, date string, or Timestamp, e.g. `initial_train_size=len(y) - 100` |
| `initial_train_size` does not accept a float fraction | `ValueError`: must be int, date string, Timestamp, or None | Convert fractions to an int, e.g. `int(len(data) * 0.7)` |
| Configuration must yield at least 2 folds | Single-fold or empty backtest | Ensure `initial_train_size + gap + 2 * steps <= n_observations` |
| `refit=True` retrains every fold (slow path); the default is `refit=False` | Backtest much slower than expected | Use `refit=False` or an int cadence (e.g. `refit=7`) unless per-fold retraining is required |
| `gap`, `steps`, and `fold_stride` count observations, not calendar time | Off-by-frequency delays | Convert the delay to the series frequency (e.g. 2 days at daily freq = `gap=2`) |

## TimeSeriesFold Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_train_size` | int, str, pd.Timestamp | None (required) | Observations for initial training. Int = count, str/Timestamp = last training date. A common starting point is `int(len(data) * 0.7)`. |
| `refit` | bool, int | False | Refit every fold (True), train once (False, default), or every n folds (int). |
| `fixed_train_size` | bool | True | Rolling fixed window (True, default) vs expanding window (False). |
| `gap` | int | 0 | Observations between training end and test start. |
| `fold_stride` | int, None | None (= steps) | Distance between consecutive test set starts. |
| `skip_folds` | int, list, None | None | Skip folds to reduce compute. Int = keep every n-th. |
| `allow_incomplete_fold` | bool | True | Allow final fold with fewer observations than steps. |

## Constraints

- Must produce **at least 2 folds**: `initial_train_size + gap + 2 * steps <= n_observations`
- `initial_train_size` must be large enough for the model to learn patterns (at minimum 2× the lag order for ML models, or 2× steps for statistical models)
- `gap` simulates real-world delay between data availability and forecast usage
- When `fixed_train_size=True`, the window rolls forward (oldest data discarded). Use for concept drift or when old data is less relevant.
- When `fixed_train_size=False`, the window expands (all history retained). Use when more data always helps.

## Business Scenario Mapping

### Retraining frequency

| Scenario | Configuration |
|----------|--------------|
| Retrain every time new data arrives | `refit=True` |
| Retrain weekly (with daily forecasts, steps=1) | `refit=7` |
| Retrain monthly (with daily forecasts, steps=7) | `refit=4` (every 4 folds × 7 steps ≈ monthly) |
| Never retrain / train once, evaluate across time | `refit=False` (`fixed_train_size` has no effect without refit) |

### Data freshness vs volume

| Scenario | Configuration |
|----------|--------------|
| Recent data more relevant (concept drift) | `fixed_train_size=True` |
| All historical data valuable | `fixed_train_size=False` (expanding) |
| Limited compute budget | `refit=False` or `skip_folds=3` (keep every 3rd fold) |

### Deployment gap

| Scenario | Configuration |
|----------|--------------|
| Real-time predictions (no delay) | `gap=0` |
| 1-day delay between data collection and forecast | `gap=1` (if freq=daily) |
| Forecast must be ready before weekend (2-day gap) | `gap=2` |

### Initial training size

| Scenario | Configuration |
|----------|--------------|
| Default (balanced) | `int(len(data) * 0.7)` |
| Maximize evaluation coverage | Minimum viable: `2 * max_lag` or `2 * steps` |
| Maximize training data | `n_observations - gap - 2 * steps` (minimum 2 folds) |
| Start from specific date | `initial_train_size="2023-01-01"` |
| Conservative (large training set) | `int(len(data) * 0.8)` |

### Fold stride (test set overlap)

| Scenario | Configuration |
|----------|--------------|
| Non-overlapping evaluation (default) | `fold_stride=None` (equals steps) |
| Sliding window evaluation (1-step shift) | `fold_stride=1` |
| Sparse evaluation (save compute) | `fold_stride=steps * 2` |

## Examples

### "I retrain my model every Monday and forecast the next 7 days"
```
refit = True
fixed_train_size = False  # Keep all history
gap = 0  # No delay
fold_stride = None  # Non-overlapping weeks
```

### "I want to simulate deploying once and seeing how the model degrades"
```
refit = False  # Train once; fixed_train_size has no effect without refit
gap = 0
```

### "There's a 2-day lag between data ingestion and when forecasts are needed"
```
gap = 2
refit = True
```

### "I have limited compute — evaluate every other week"
```
refit = True
skip_folds = 2  # Keep every 2nd fold
```

### "I want maximum evaluation coverage with a 12-step horizon"
```
initial_train_size = <minimum viable>  # 2 * max_lag
refit = False  # Faster; trains once, fixed_train_size has no effect
fold_stride = 1  # Sliding window (many overlapping folds)
```
