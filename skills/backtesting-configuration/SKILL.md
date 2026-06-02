---
name: backtesting-configuration
description: >
  Configures TimeSeriesFold parameters for backtesting based on deployment
  scenarios. Maps business requirements (retraining frequency, forecast
  horizon, data budget) to cross-validation strategy parameters.
  Use when the user describes how they plan to deploy or evaluate a model.
---

# Backtesting Configuration

## TimeSeriesFold Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_train_size` | int, float, str | 70% of data | Observations for initial training. Float = fraction, str = date. |
| `refit` | bool, int | True | Refit every fold (True), never (False), or every n folds (int). |
| `fixed_train_size` | bool | False | Fixed (rolling) window vs expanding window. |
| `gap` | int | 0 | Observations between training end and test start. |
| `fold_stride` | int, None | None (= steps) | Distance between consecutive test set starts. |
| `skip_folds` | int, list, None | None | Skip folds to reduce compute. Int = keep every n-th. |
| `allow_incomplete_fold` | bool | True | Allow final fold with fewer observations than steps. |

## Constraints

- Must produce **at least 2 folds**: `initial_train_size + 2 * steps <= n_observations`
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
| Never retrain (fixed model) | `refit=False` |
| Train once, evaluate across time | `refit=False, fixed_train_size=False` |

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
| Default (balanced) | 70% of data |
| Maximize evaluation coverage | Minimum viable: `2 * max_lag` or `2 * steps` |
| Maximize training data | `n_observations - 2 * steps` (minimum 2 folds) |
| Start from specific date | `initial_train_size="2023-01-01"` |
| Conservative (large training set) | 80% of data |

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
refit = False
fixed_train_size = True  # Fixed training window
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
refit = False  # Faster
fixed_train_size = False
fold_stride = 1  # Sliding window (many overlapping folds)
```
