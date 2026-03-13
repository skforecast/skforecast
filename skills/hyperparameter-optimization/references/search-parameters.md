# Hyperparameter Search — Parameter Reference

## Function Routing

| Forecaster | Grid Search | Random Search | Bayesian Search |
|------------|-------------|---------------|-----------------|
| ForecasterRecursive | `grid_search_forecaster` | `random_search_forecaster` | `bayesian_search_forecaster` |
| ForecasterDirect | `grid_search_forecaster` | `random_search_forecaster` | `bayesian_search_forecaster` |
| ForecasterRecursiveMultiSeries | `grid_search_forecaster_multiseries` | `random_search_forecaster_multiseries` | `bayesian_search_forecaster_multiseries` |
| ForecasterDirectMultiVariate | `grid_search_forecaster_multiseries` | `random_search_forecaster_multiseries` | `bayesian_search_forecaster_multiseries` |
| ForecasterRnn | `grid_search_forecaster_multiseries` | `random_search_forecaster_multiseries` | `bayesian_search_forecaster_multiseries` |
| ForecasterStats | `grid_search_stats` | `random_search_stats` | N/A |
| ForecasterEquivalentDate | N/A | N/A | N/A |
| ForecasterRecursiveClassifier | `grid_search_forecaster` | `random_search_forecaster` | `bayesian_search_forecaster` |

## Parameter Comparison Across Search Functions

### Single-Series Functions

| Parameter | `grid_search_forecaster` | `random_search_forecaster` | `bayesian_search_forecaster` |
|-----------|:-:|:-:|:-:|
| `forecaster` | ✓ | ✓ | ✓ |
| `y` | ✓ | ✓ | ✓ |
| `cv` | TimeSeriesFold \| OneStepAheadFold | TimeSeriesFold \| OneStepAheadFold | TimeSeriesFold \| OneStepAheadFold |
| `param_grid` | ✓ | — | — |
| `param_distributions` | — | ✓ | — |
| `search_space` | — | — | ✓ (Callable) |
| `metric` | ✓ | ✓ | ✓ |
| `exog` | ✓ | ✓ | ✓ |
| `lags_grid` | ✓ | ✓ | — (included in `search_space`) |
| `n_iter` | — | ✓ (default: 10) | — |
| `n_trials` | — | — | ✓ (default: 20) |
| `random_state` | — | ✓ (default: 123) | ✓ (default: 123) |
| `return_best` | ✓ (default: True) | ✓ (default: True) | ✓ (default: True) |
| `n_jobs` | ✓ (default: 'auto') | ✓ (default: 'auto') | ✓ (default: 'auto') |
| `verbose` | ✓ | ✓ | ✓ |
| `show_progress` | ✓ | ✓ | ✓ |
| `suppress_warnings` | ✓ | ✓ | ✓ |
| `output_file` | ✓ | ✓ | ✓ |
| `kwargs_create_study` | — | — | ✓ (default: None) |
| `kwargs_study_optimize` | — | — | ✓ (default: None) |
| **Returns** | `pd.DataFrame` | `pd.DataFrame` | `tuple[pd.DataFrame, object]` |

### Multi-Series Functions (additional parameters)

These functions have all the parameters above plus:

| Parameter | grid | random | bayesian |
|-----------|:----:|:------:|:--------:|
| `series` (replaces `y`) | ✓ | ✓ | ✓ |
| `aggregate_metric` | ✓ | ✓ | ✓ |
| `levels` | ✓ | ✓ | ✓ |

Default `aggregate_metric = ['weighted_average', 'average', 'pooling']`

### Stats Functions (limited parameters)

| Parameter | `grid_search_stats` | `random_search_stats` |
|-----------|:---:|:---:|
| `forecaster` | ✓ | ✓ |
| `y` | ✓ | ✓ |
| `cv` | **TimeSeriesFold only** | **TimeSeriesFold only** |
| `param_grid` / `param_distributions` | ✓ | ✓ |
| `metric` | ✓ | ✓ |
| `exog` | ✓ | ✓ |
| `lags_grid` | — | — |
| `n_iter` | — | ✓ (default: 10) |
| `random_state` | — | ✓ (default: 123) |
| `return_best` | ✓ | ✓ |
| `n_jobs` | ✓ | ✓ |
| **Returns** | `pd.DataFrame` | `pd.DataFrame` |

> **Note:** Stats search does NOT support `OneStepAheadFold`, `lags_grid`,
> or Bayesian search.

## How `lags_grid` Works

For grid and random search, `lags_grid` is a list of lag configurations to try:

```python
# List format — each element is a configuration
lags_grid = [3, 10, 24, [1, 2, 3, 23, 24]]
# Tries: lags=3, lags=10, lags=24, lags=[1,2,3,23,24]

# Dict format — keys become labels in the results
lags_grid = {
    'short': 3,
    'medium': 12,
    'long': 24,
    'custom': [1, 2, 3, 23, 24],
}
```

For Bayesian search, lags are included in the `search_space` function:

```python
def search_space(trial):
    return {
        'lags': trial.suggest_categorical('lags', [3, 12, 24, [1, 2, 3, 23, 24]]),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
    }
```

## How `param_grid` vs `param_distributions` vs `search_space` Work

### Grid Search: `param_grid`

All combinations are evaluated (Cartesian product):

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10],
}
# Evaluates: 3 × 2 = 6 combinations
```

### Random Search: `param_distributions`

Random sample of `n_iter` combinations:

```python
param_distributions = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [3, 5, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
}
# Evaluates n_iter=10 random combinations (default)
```

### Bayesian Search: `search_space`

Optuna trial function with suggest methods:

```python
def search_space(trial):
    return {
        'lags': trial.suggest_categorical('lags', [12, 24]),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
    }
# Optuna methods: suggest_int, suggest_float, suggest_categorical
# Evaluates n_trials=20 trials (default), guided by TPE sampler
```

## Stats Model param_grid

Parameters in `param_grid` for stats models are passed to the model constructor:

```python
# Arima
param_grid = {
    'order': [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 1)],
    'seasonal_order': [(0, 0, 0), (1, 1, 1)],
    'm': [12],
}

# Ets
param_grid = {
    'model': ['AAA', 'ANA', 'MAM', 'ZZZ'],
    'm': [12],
}
```

## Optuna kwargs

```python
# Advanced: customize Optuna study
results, best_trial = bayesian_search_forecaster(
    ...,
    kwargs_create_study={
        'sampler': optuna.samplers.TPESampler(seed=123),
        'direction': 'minimize',
    },
    kwargs_study_optimize={
        'timeout': 600,  # seconds
        'gc_after_trial': True,
    },
)
```

## Return Values

| Function | Returns | Best trial object |
|----------|---------|:-:|
| `grid_search_*` | `pd.DataFrame` sorted by metric | — |
| `random_search_*` | `pd.DataFrame` sorted by metric | — |
| `bayesian_search_*` | `tuple[pd.DataFrame, optuna.FrozenTrial]` | ✓ |
| `*_stats` | `pd.DataFrame` sorted by metric | — |

When `return_best=True`, the forecaster is automatically updated with the
best parameters found. The results DataFrame always has rows sorted by
metric (best first).
