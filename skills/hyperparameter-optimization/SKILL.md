---
name: hyperparameter-optimization
description: >
  Optimizes forecaster hyperparameters using grid search, random search, or
  Bayesian search (Optuna). Covers single-series and multi-series search,
  cross-validation configuration, and search space definition.
  Use when the user wants to find the best model configuration.
---

# Hyperparameter Optimization

## When to Use

Use hyperparameter search after establishing a baseline forecaster to improve prediction accuracy. Skforecast supports three strategies:

| Strategy | When to Use | Speed |
|----------|-------------|-------|
| **Grid Search** | Small parameter space, exhaustive exploration | Slowest |
| **Random Search** | Large parameter space, limited compute budget | Medium |
| **Bayesian Search** | Best results, smart exploration via Optuna | Fastest to converge |

## Grid Search

```python
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import grid_search_forecaster, TimeSeriesFold
from lightgbm import LGBMRegressor

forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(random_state=123),
    lags=24,
)

cv = TimeSeriesFold(
    steps=12,
    initial_train_size=len(data) - 100,
    refit=False,
)

# Different lag configurations to try
lags_grid = [3, 10, 24, [1, 2, 3, 23, 24]]

# Estimator hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1],
}

results = grid_search_forecaster(
    forecaster=forecaster,
    y=data['target'],
    exog=exog,
    cv=cv,
    lags_grid=lags_grid,
    param_grid=param_grid,
    metric='mean_absolute_error',
    return_best=True,        # Automatically updates forecaster with best params
    n_jobs='auto',
    show_progress=True,
)
# results is a DataFrame sorted by metric (best first)
```

## Bayesian Search (Recommended)

```python
from skforecast.model_selection import bayesian_search_forecaster, TimeSeriesFold

cv = TimeSeriesFold(
    steps=12,
    initial_train_size=len(data) - 100,
    refit=False,
)

# Define search space as a function — lags CAN be included here
def search_space(trial):
    return {
        'lags': trial.suggest_categorical('lags', [12, 24, [1, 2, 3, 23, 24]]),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
    }

results, best_trial = bayesian_search_forecaster(
    forecaster=forecaster,
    y=data['target'],
    exog=exog,
    cv=cv,
    search_space=search_space,
    metric='mean_absolute_error',
    n_trials=50,
    random_state=123,
    return_best=True,
    n_jobs='auto',
    show_progress=True,
    output_file='search_results.csv',  # Save results incrementally
)
```

## Multi-Series Search

```python
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import bayesian_search_forecaster_multiseries

forecaster = ForecasterRecursiveMultiSeries(
    estimator=LGBMRegressor(random_state=123),
    lags=24,
    encoding='ordinal',
)

cv = TimeSeriesFold(
    steps=12,
    initial_train_size=len(series) - 100,
    refit=False,
)

results, best_trial = bayesian_search_forecaster_multiseries(
    forecaster=forecaster,
    series=series,
    exog=exog,
    cv=cv,
    search_space=search_space,
    metric='mean_absolute_error',
    n_trials=50,
    return_best=True,
    n_jobs='auto',
    show_progress=True,
)
```

## Statistical Models Search

```python
from skforecast.recursive import ForecasterStats
from skforecast.stats import Arima
from skforecast.model_selection import grid_search_stats

forecaster = ForecasterStats(estimator=Arima(order=(1, 1, 1)))

param_grid = {
    'order': [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 1)],
    'seasonal_order': [(0, 0, 0), (1, 1, 1)],
    'm': [12],
}

results = grid_search_stats(
    forecaster=forecaster,
    y=data['target'],
    cv=cv,
    param_grid=param_grid,
    metric='mean_absolute_error',
    return_best=True,
)
```

## Fast Tuning with OneStepAheadFold

```python
from skforecast.model_selection import OneStepAheadFold

# Much faster than TimeSeriesFold — no recursive predictions needed
cv_fast = OneStepAheadFold(
    initial_train_size=len(data) - 100,
)

results, best_trial = bayesian_search_forecaster(
    forecaster=forecaster,
    y=data['target'],
    cv=cv_fast,
    search_space=search_space,
    metric='mean_absolute_error',
    n_trials=100,
    return_best=True,
)
```

## Common Mistakes

1. **Not setting `return_best=True`**: The forecaster is not updated with the best parameters unless this is True.
2. **Too few trials in Bayesian search**: Start with at least 20-50 trials for meaningful exploration.
3. **Using TimeSeriesFold for initial tuning**: Use `OneStepAheadFold` first for fast screening, then validate the top candidates with `TimeSeriesFold`.
4. **Forgetting to include lags in search space**: For Bayesian search, lags can be included in `search_space()` — this is often the most impactful parameter.
