# Hyperparameter Search — Parameter Reference

## Contents

- Function routing (which search function per forecaster)
- Parameter comparison across search functions
- How `lags_grid` works
- How `param_grid` vs `param_distributions` vs `search_space` work
- Stats model `param_grid`
- Optuna kwargs
- Return values

## Function Routing

| Forecaster | Grid Search | Random Search | Bayesian Search |
|------------|-------------|---------------|-----------------|
| ForecasterRecursive | `grid_search_forecaster` | `random_search_forecaster` | `bayesian_search_forecaster` |
| ForecasterDirect | `grid_search_forecaster` | `random_search_forecaster` | `bayesian_search_forecaster` |
| ForecasterRecursiveMultiSeries | `grid_search_forecaster_multiseries` | `random_search_forecaster_multiseries` | `bayesian_search_forecaster_multiseries` |
| ForecasterDirectMultiVariate | `grid_search_forecaster_multiseries` | `random_search_forecaster_multiseries` | `bayesian_search_forecaster_multiseries` |
| ForecasterRnn | `grid_search_forecaster_multiseries` | `random_search_forecaster_multiseries` | `bayesian_search_forecaster_multiseries` |
| ForecasterStats | `grid_search_stats` | `random_search_stats` | N/A |
| ForecasterEquivalentDate | `grid_search_equivalent_date` | N/A | N/A |
| ForecasterFoundation | N/A | N/A | `bayesian_search_foundation` |
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

### Foundation Function (zero-shot inference-time tuning)

`bayesian_search_foundation` tunes the **inference-time** configuration of a
`ForecasterFoundation`. Foundation models are zero-shot, so no weights are
trained; only how the pre-trained model is queried is optimized. Each trial is
evaluated with `backtesting_foundation`.

| Parameter | `bayesian_search_foundation` |
|-----------|:---:|
| `forecaster` | ✓ (must be `ForecasterFoundation`) |
| `series` | ✓ (`pd.Series`, `pd.DataFrame` or `dict`) |
| `cv` | **TimeSeriesFold only** |
| `search_space` | ✓ (Callable) |
| `metric` | ✓ |
| `aggregate_metric` | ✓ (default: `['weighted_average', 'average', 'pooling']`) |
| `levels` | ✓ |
| `exog` | ✓ |
| `n_trials` | ✓ (default: 20) |
| `random_state` | ✓ (default: 123) |
| `return_best` | ✓ (default: True) |
| `n_jobs` | — |
| `verbose` / `show_progress` / `suppress_warnings` | ✓ |
| `output_file` | ✓ (optuna log, not results TSV) |
| `kwargs_create_study` / `kwargs_study_optimize` | ✓ (default: None) |
| **Returns** | `tuple[pd.DataFrame, optuna Study]` |

> **Note:** No `n_jobs`, no `lags` / `lags_grid` (foundation models have no lag
> concept), no `OneStepAheadFold` (raises `TypeError`), and no grid/random
> variant. Passing a `ForecasterFoundation` to `bayesian_search_forecaster*`
> raises `TypeError`.

Every key returned by `search_space` is validated against
`forecaster.estimator.adapter.get_params()`; an unknown key raises `ValueError`.
Being accepted is not the same as being worth searching:

| Model | Worth searching |
|-------|-----------------|
| All models | `context_length` |
| Amazon Chronos-2 | `cross_learning` |
| TabICL, Prior Labs TabPFN-TS | `point_estimate`, `temporal_features` |
| Synthefy Nori | `point_estimate`, `add_calendar_features`, `n_fourier_terms` |
| Moirai-2, T0 | `context_length` only |

Do not search `device`, `device_map`, `torch_dtype`, `mode`, `show_progress`,
`allow_auto_download` or `max_horizon`: they are accepted but cannot improve
accuracy. `model_id` and `checkpoint_version` do, but they select a different
pre-trained model — compare those with separate searches.

Cost warning: `model_id` and device/dtype arguments force a full model reload,
and `context_length` does too on TimesFM 2.5, Moirai-2, TabICL and TabPFN-TS.
Full matrix: the `foundation-forecasting` skill
(`references/adapter-parameters.md`).

```python
from skforecast.foundation import FoundationModel, ForecasterFoundation
from skforecast.model_selection import bayesian_search_foundation, TimeSeriesFold

forecaster = ForecasterFoundation(
    estimator=FoundationModel(model_id='autogluon/chronos-2-small', device_map='auto')
)
cv = TimeSeriesFold(steps=24, initial_train_size=len(series) - 200, refit=False)

def search_space(trial):
    return {
        'context_length': trial.suggest_categorical('context_length', [512, 1024, 2048, 4096]),
        'cross_learning': trial.suggest_categorical('cross_learning', [True, False]),
    }

results, study = bayesian_search_foundation(
    forecaster=forecaster,
    series=series,
    cv=cv,
    search_space=search_space,
    metric='mean_absolute_error',
    n_trials=30,
    return_best=True,
)
# results columns: trial_number, levels, params, metric/s, one column per searched param
```

### Equivalent Date Function (baseline tuning)

`grid_search_equivalent_date` selects the best `ForecasterEquivalentDate`
baseline configuration. Only grid search exists (random/Bayesian are
unnecessary for such a small, discrete space).

| Parameter | `grid_search_equivalent_date` |
|-----------|:---:|
| `forecaster` | ✓ (must be `ForecasterEquivalentDate`) |
| `y` | ✓ |
| `cv` | **TimeSeriesFold only** |
| `param_grid` | ✓ (dict **or** list of dicts, see below) |
| `metric` | ✓ |
| `return_best` | ✓ (default: True) |
| `n_jobs` | ✓ (default: 'auto') |
| `output_file` | ✓ |
| **Returns** | `pd.DataFrame` |

> **Note:** No `exog` (baseline does not use exogenous variables), no
> `lags_grid`, no `OneStepAheadFold`, no random/Bayesian variant.

`param_grid` behaves differently from the other search functions because
`offset` and `n_offsets` are **coupled** (together they define the equivalent
window). A third parameter, `agg_func`, sets how the `n_offsets` values are
aggregated (when `n_offsets > 1`) and is searchable too, so remember to include
it when the aggregation matters:

```python
from skforecast.recursive import ForecasterEquivalentDate
from skforecast.model_selection import grid_search_equivalent_date, TimeSeriesFold
import numpy as np

forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)
cv = TimeSeriesFold(steps=7, initial_train_size=len(y) - 70, refit=False)

# dict -> Cartesian product (use only when every combination is meaningful)
param_grid = {'offset': [1, 7], 'n_offsets': [1, 2], 'agg_func': [np.mean, np.median]}

# list of dicts -> one explicit configuration per dict (no cross product).
# Values must be scalar. Add one dict per configuration to avoid the coupled
# offset/n_offsets footgun. Each dict may carry an optional `alias` label.
param_grid = [
    {'alias': '7-day moving average',     'offset': 1, 'n_offsets': 7, 'agg_func': np.mean},
    {'alias': 'mean of lag-7 and lag-14', 'offset': 7, 'n_offsets': 2, 'agg_func': np.mean},
]

results = grid_search_equivalent_date(
    forecaster=forecaster, y=y, cv=cv, param_grid=param_grid,
    metric='mean_absolute_error', return_best=True,
)
```

When at least one configuration defines `alias`, an `alias` column is added as
the first column of the results table. The `alias` key is stripped before the
forecaster is configured (it never reaches `set_params`).

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

For `bayesian_search_foundation` the same `search_space` contract applies, but
`lags` must NOT be included and every key must be a valid adapter parameter:

```python
def search_space(trial):
    return {
        'context_length': trial.suggest_categorical('context_length', [512, 2048, 8192]),
    }
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
results, study = bayesian_search_forecaster(
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
# Access the best trial with study.best_trial
```

## Return Values

| Function | Returns | Study object |
|----------|---------|:-:|
| `grid_search_*` | `pd.DataFrame` sorted by metric | — |
| `random_search_*` | `pd.DataFrame` sorted by metric | — |
| `bayesian_search_*` | `tuple[pd.DataFrame, optuna Study]` | ✓ |
| `*_stats` | `pd.DataFrame` sorted by metric | — |
| `bayesian_search_foundation` | `tuple[pd.DataFrame, optuna Study]`, no `lags` column | ✓ |

When `return_best=True`, the forecaster is automatically updated with the
best parameters found. The results DataFrame always has rows sorted by
metric (best first). Access the best Optuna trial with `study.best_trial`.
