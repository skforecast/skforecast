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

- **Prerequisite**: `forecasting-single-series` / `forecasting-multiple-series` (have a fitted forecaster before backtesting)
- **Alongside**: `metric-selection` (choose the metric(s) backtesting reports)
- **Alongside**: `hyperparameter-optimization` (the same `cv` object drives the search functions)
- **Next**: `prediction-intervals` (add `interval=` / `interval_method=` to backtest uncertainty)

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
- **Hard minimum**: `initial_train_size` must be greater than `forecaster.window_size`, the number of past observations needed to build one training row (`max(max_lag, max_size_window_features)` plus the `differentiation` order). Direct forecasters (`ForecasterDirect`, `ForecasterDirectMultiVariate`, `ForecasterRnn`) need at least `window_size + steps`, using the forecaster's `steps`. In multi-series forecasters, each series with data in the first window needs more than `window_size` observations there.
- Forecasters without lags: `ForecasterEquivalentDate` has `window_size = offset * n_offsets` (same rule); in `ForecasterStats` the estimator sets the minimum (`window_size` is fixed at 1); `ForecasterFoundation` accepts any value ≥ 1 (`context_length` is a maximum, not a minimum).
- **Recommended**: leave a margin above the hard minimum so the model has enough rows to learn from (e.g. two or more seasonal cycles), while keeping at least 2 folds
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
| Retrain at each month end and forecast the whole next month (daily data) | `steps=31`, `fold_stride=1`, `skip_folds` = folds not starting on day 1, `refit=True` (see example below) |
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
| Maximize evaluation coverage | Hard minimum (`window_size + 1`; `window_size + steps` for direct) plus a learning margin, see Constraints |
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

### "I retrain at the end of every month and forecast the whole next month" (daily data)

Folds count observations, not calendar months. Create one candidate fold per day
and skip those that do not start predicting on day 1 of a month:

```python
gap = 0
first_origin = '2025-09-30'  # month end (minus `gap` days): fold 0 cannot be skipped
n_train = y.index.get_loc(first_origin) + 1
skip_folds = [
    i for i, date in enumerate(y.index[n_train + gap:]) if not date.is_month_start
]  # fold i predicts from position n_train + gap + i

cv = TimeSeriesFold(
    steps=31,                  # longest month
    initial_train_size=first_origin,
    fold_stride=1,
    gap=gap,
    skip_folds=skip_folds,
    refit=True,
    fixed_train_size=False,
    allow_incomplete_fold=True,
)
```

- Needs a regular index without missing dates. With sub-daily data, `is_month_start`
  is `True` for every hour of day 1: also require `date == date.normalize()` and set
  `steps` to the observations of the longest month (744 for hourly data).
- `ForecasterDirect` and `ForecasterDirectMultiVariate` need `steps >= 31 + gap`.
- Use `refit=True` (retrain every month) or `refit=False` (train once). An integer
  `refit` counts the daily candidate folds, not months: `refit=3` does not retrain
  every 3 months.
- The metric needs no correction: overlapping folds are scored with the last
  prediction of each date, the one of its own month. It works the same in the
  multi-series backtesting and in the hyperparameter search functions.
- `predictions` repeats the dates where months shorter than 31 days spill into the
  next month. Remove them before grouping by month or plotting:
  `predictions[~predictions.index.duplicated(keep='last')]` for a single series;
  with multiple series (long format with a `level` column), deduplicate by date and
  level, `predictions[~predictions.set_index('level', append=True).index.duplicated(keep='last')]`.
- The `fold` column keeps the numbering of the daily folds (0, 31, 61, ...).
- Use `verbose=False`: `verbose=True` prints every skipped fold.

### "I want maximum evaluation coverage with a 12-step horizon"
```
# Hard minimum: forecaster.window_size + 1 (window_size + 12 for ForecasterDirect with steps=12)
initial_train_size = forecaster.window_size + margin  # margin: rows to learn from, e.g. 2+ seasonal cycles
refit = False  # Faster; trains once, fixed_train_size has no effect
fold_stride = 1  # Sliding window (many overlapping folds)
```
