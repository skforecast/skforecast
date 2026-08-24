# Model Selection Signatures

Backtesting, hyperparameter search, cross-validation and feature selection signatures.

## Contents

- Backtesting functions
- Hyperparameter search functions (single series, multi-series, statistical, foundation)
- Cross-validation classes
- Feature selection functions

## Backtesting Functions

```python
backtesting_forecaster(
    forecaster,                         # ForecasterRecursive | ForecasterDirect |
                                        # ForecasterEquivalentDate | ForecasterRecursiveClassifier
    y,                                  # pd.Series with DatetimeIndex
    cv,                                 # TimeSeriesFold
    metric,                             # str | Callable | list[str | Callable]
    exog=None,                          # pd.Series | pd.DataFrame | None
    interval=None,                      # float | list[float] | tuple[float] | str | distribution | None
    interval_method='bootstrapping',    # 'bootstrapping' | 'conformal'
    n_boot=250,                         # int
    use_in_sample_residuals=True,       # bool
    use_binned_residuals=True,          # bool
    random_state=123,                   # int
    return_predictors=False,            # bool
    n_jobs='auto',                      # int | str
    verbose=False,                      # bool
    show_progress=True,                 # bool
    suppress_warnings=False             # bool
) -> tuple[pd.DataFrame, pd.DataFrame]

backtesting_forecaster_multiseries(
    forecaster,                         # ForecasterRecursiveMultiSeries |
                                        # ForecasterDirectMultiVariate | ForecasterRnn
    series,                             # pd.DataFrame | dict[str, pd.Series | pd.DataFrame]
    cv,                                 # TimeSeriesFold
    metric,                             # str | Callable | list[str | Callable]
    levels=None,                        # str | list[str] | None
    add_aggregated_metric=True,         # bool
    exog=None,                          # pd.Series | pd.DataFrame | dict | None
    interval=None,                      # float | list[float] | tuple[float] | str | distribution | None
    interval_method='conformal',        # 'bootstrapping' | 'conformal' (NOTE: default 'conformal')
    n_boot=250,                         # int
    use_in_sample_residuals=True,       # bool
    use_binned_residuals=True,          # bool
    random_state=123,                   # int
    return_predictors=False,            # bool
    n_jobs='auto',                      # int | str
    verbose=False,                      # bool
    show_progress=True,                 # bool
    suppress_warnings=False             # bool
) -> tuple[pd.DataFrame, pd.DataFrame]

backtesting_stats(
    forecaster,                         # ForecasterStats
    y,                                  # pd.Series with DatetimeIndex
    cv,                                 # TimeSeriesFold
    metric,                             # str | Callable | list[str | Callable]
    exog=None,                          # pd.Series | pd.DataFrame | None
    alpha=None,                         # float | None, significance level
    interval=None,                      # list[float] | tuple[float] | None
    freeze_params=True,                 # bool, if True only first fold fits the model
    n_jobs='auto',                      # int | str
    verbose=False,                      # bool
    show_progress=True,                 # bool
    suppress_warnings=False             # bool
) -> tuple[pd.DataFrame, pd.DataFrame]

backtesting_foundation(
    forecaster,                         # ForecasterFoundation
    series,                             # pd.Series | pd.DataFrame | dict
    cv,                                 # TimeSeriesFold (refit / fixed_train_size overridden internally)
    metric,                             # str | Callable | list[str | Callable]
    levels=None,                        # str | list[str] | None
    add_aggregated_metric=True,         # bool
    exog=None,                          # pd.Series | pd.DataFrame | dict | None
    quantiles=None,                     # list[float] | None, native model quantiles
    verbose=False,                      # bool
    show_progress=True,                 # bool
    suppress_warnings=False             # bool
) -> tuple[pd.DataFrame, pd.DataFrame]
```

## Hyperparameter Search Functions

### Single Series

```python
grid_search_forecaster(
    forecaster,              # ForecasterRecursive | ForecasterDirect
    y,                       # pd.Series with DatetimeIndex
    cv,                      # TimeSeriesFold | OneStepAheadFold
    param_grid,              # dict, sklearn-style parameter grid
    metric,                  # str | Callable | list[str | Callable]
    exog=None,               # pd.Series | pd.DataFrame | None
    lags_grid=None,          # list[int | list | np.ndarray | range] | dict | None
    return_best=True,        # bool
    n_jobs='auto',           # int | str
    verbose=False,           # bool
    show_progress=True,      # bool
    suppress_warnings=False, # bool
    output_file=None         # str | None, path to save results incrementally
) -> pd.DataFrame

random_search_forecaster(
    forecaster,              # ForecasterRecursive | ForecasterDirect
    y,                       # pd.Series with DatetimeIndex
    cv,                      # TimeSeriesFold | OneStepAheadFold
    param_distributions,     # dict, parameter distributions for sampling
    metric,                  # str | Callable | list[str | Callable]
    exog=None,               # pd.Series | pd.DataFrame | None
    lags_grid=None,          # list[int | list | np.ndarray | range] | dict | None
    n_iter=10,               # int, number of random parameter combinations
    random_state=123,        # int
    return_best=True,        # bool
    n_jobs='auto',           # int | str
    verbose=False,           # bool
    show_progress=True,      # bool
    suppress_warnings=False, # bool
    output_file=None         # str | None
) -> pd.DataFrame

bayesian_search_forecaster(
    forecaster,              # ForecasterRecursive | ForecasterDirect
    y,                       # pd.Series with DatetimeIndex
    cv,                      # TimeSeriesFold | OneStepAheadFold
    search_space,            # Callable, Optuna trial search space function
    metric,                  # str | Callable | list[str | Callable]
    exog=None,               # pd.Series | pd.DataFrame | None
    n_trials=20,             # int, number of Optuna trials
    random_state=123,        # int
    return_best=True,        # bool
    n_jobs='auto',           # int | str
    verbose=False,           # bool
    show_progress=True,      # bool
    suppress_warnings=False, # bool
    output_file=None,        # str | None
    kwargs_create_study=None,     # dict | None, kwargs for optuna.create_study()
    kwargs_study_optimize=None    # dict | None, kwargs for study.optimize()
) -> tuple[pd.DataFrame, object]
```

### Multi-Series

```python
grid_search_forecaster_multiseries(
    forecaster,              # ForecasterRecursiveMultiSeries | ForecasterDirectMultiVariate | ForecasterRnn
    series,                  # pd.DataFrame | dict[str, pd.Series | pd.DataFrame]
    cv,                      # TimeSeriesFold | OneStepAheadFold
    param_grid,              # dict
    metric,                  # str | Callable | list[str | Callable]
    aggregate_metric=['weighted_average', 'average', 'pooling'],  # str | list[str]
    levels=None,             # str | list[str] | None
    exog=None,               # pd.Series | pd.DataFrame | dict | None
    lags_grid=None,          # list | dict | None
    return_best=True,        # bool
    n_jobs='auto',           # int | str
    verbose=False,           # bool
    show_progress=True,      # bool
    suppress_warnings=False, # bool
    output_file=None         # str | None
) -> pd.DataFrame

random_search_forecaster_multiseries(
    forecaster,              # ForecasterRecursiveMultiSeries | ForecasterDirectMultiVariate | ForecasterRnn
    series,                  # pd.DataFrame | dict[str, pd.Series | pd.DataFrame]
    cv,                      # TimeSeriesFold | OneStepAheadFold
    param_distributions,     # dict
    metric,                  # str | Callable | list[str | Callable]
    aggregate_metric=['weighted_average', 'average', 'pooling'],  # str | list[str]
    levels=None,             # str | list[str] | None
    exog=None,               # pd.Series | pd.DataFrame | dict | None
    lags_grid=None,          # list | dict | None
    n_iter=10,               # int
    random_state=123,        # int
    return_best=True,        # bool
    n_jobs='auto',           # int | str
    verbose=False,           # bool
    show_progress=True,      # bool
    suppress_warnings=False, # bool
    output_file=None         # str | None
) -> pd.DataFrame

bayesian_search_forecaster_multiseries(
    forecaster,              # ForecasterRecursiveMultiSeries | ForecasterDirectMultiVariate | ForecasterRnn
    series,                  # pd.DataFrame | dict[str, pd.Series | pd.DataFrame]
    cv,                      # TimeSeriesFold | OneStepAheadFold
    search_space,            # Callable, Optuna trial search space function
    metric,                  # str | Callable | list[str | Callable]
    aggregate_metric=['weighted_average', 'average', 'pooling'],  # str | list[str]
    levels=None,             # str | list[str] | None
    exog=None,               # pd.Series | pd.DataFrame | dict | None
    n_trials=20,             # int
    random_state=123,        # int
    return_best=True,        # bool
    n_jobs='auto',           # int | str
    verbose=False,           # bool
    show_progress=True,      # bool
    suppress_warnings=False, # bool
    output_file=None,        # str | None
    kwargs_create_study=None,     # dict | None
    kwargs_study_optimize=None    # dict | None
) -> tuple[pd.DataFrame, object]
```

### Statistical Models

```python
grid_search_stats(
    forecaster,              # ForecasterStats
    y,                       # pd.Series with DatetimeIndex
    cv,                      # TimeSeriesFold
    param_grid,              # dict
    metric,                  # str | Callable | list[str | Callable]
    exog=None,               # pd.Series | pd.DataFrame | None
    return_best=True,        # bool
    n_jobs='auto',           # int | str
    verbose=False,           # bool
    show_progress=True,      # bool
    suppress_warnings=False, # bool
    output_file=None         # str | None
) -> pd.DataFrame

random_search_stats(
    forecaster,              # ForecasterStats
    y,                       # pd.Series with DatetimeIndex
    cv,                      # TimeSeriesFold
    param_distributions,     # dict
    metric,                  # str | Callable | list[str | Callable]
    exog=None,               # pd.Series | pd.DataFrame | None
    n_iter=10,               # int
    random_state=123,        # int
    return_best=True,        # bool
    n_jobs='auto',           # int | str
    verbose=False,           # bool
    show_progress=True,      # bool
    suppress_warnings=False, # bool
    output_file=None         # str | None
) -> pd.DataFrame

grid_search_equivalent_date(
    forecaster,              # ForecasterEquivalentDate
    y,                       # pd.Series with DatetimeIndex
    cv,                      # TimeSeriesFold
    param_grid,              # dict | list[dict] (coupled offset/n_offsets, plus searchable agg_func; list config supports optional 'alias')
    metric,                  # str | Callable | list[str | Callable]
    return_best=True,        # bool
    n_jobs='auto',           # int | str
    verbose=False,           # bool
    show_progress=True,      # bool
    suppress_warnings=False, # bool
    output_file=None         # str | None
) -> pd.DataFrame
```

### Foundation Models

```python
bayesian_search_foundation(
    forecaster,              # ForecasterFoundation
    series,                  # pd.Series | pd.DataFrame | dict
    cv,                      # TimeSeriesFold (OneStepAheadFold raises TypeError)
    search_space,            # Callable, keys validated against adapter.get_params(); no 'lags'
    metric,                  # str | Callable | list[str | Callable]
    aggregate_metric=None,   # str | list[str] | None
    levels=None,             # str | list[str] | None
    exog=None,               # pd.Series | pd.DataFrame | dict | None
    n_trials=20,             # int
    random_state=123,        # int
    return_best=True,        # bool
    verbose=False,           # bool
    show_progress=True,      # bool
    suppress_warnings=False, # bool
    output_file=None,        # str | None, optuna log
    kwargs_create_study=None,     # dict | None
    kwargs_study_optimize=None    # dict | None
) -> tuple[pd.DataFrame, object]
```

## Cross-Validation Classes

```python
TimeSeriesFold(
    steps,                          # int (required), forecast horizon
    initial_train_size=None,        # int | str | pd.Timestamp | None
    fold_stride=None,               # int | None, if None equals steps
    window_size=None,               # int | None, set automatically by forecaster
    differentiation=None,           # int | None, set automatically by forecaster
    refit=False,                    # bool | int, refit model each fold or every n folds
    fixed_train_size=True,          # bool, fixed vs expanding window
    gap=0,                          # int, observations between train end and test start
    skip_folds=None,                # int | list[int] | None
    allow_incomplete_fold=True,     # bool
    return_all_indexes=False,       # bool
    verbose=True                    # bool
)

OneStepAheadFold(
    initial_train_size,             # int | str | pd.Timestamp (required)
    window_size=None,               # int | None, set automatically by forecaster
    differentiation=None,           # int | None, set automatically by forecaster
    return_all_indexes=False,       # bool
    verbose=True                    # bool
)
```

## Feature Selection Functions

```python
select_features(
    forecaster,              # ForecasterRecursive | ForecasterDirect
    selector,                # sklearn feature selector (RFECV, SelectFromModel, etc.)
    y,                       # pd.Series | pd.DataFrame
    exog=None,               # pd.Series | pd.DataFrame | None
    select_only=None,        # 'autoreg' | 'exog' | None (select all)
    force_inclusion=None,    # list[str] | str (regex) | None
    subsample=0.5,           # int | float, proportion or number of samples
    random_state=123,        # int
    verbose=True             # bool
) -> tuple[list[int], list[str], list[str]]
# Returns: (selected_lags, selected_window_features, selected_exog)

select_features_multiseries(
    forecaster,              # ForecasterRecursiveMultiSeries
    selector,                # sklearn feature selector
    series,                  # pd.DataFrame | dict[str, pd.Series | pd.DataFrame]
    exog=None,               # pd.Series | pd.DataFrame | dict | None
    select_only=None,        # 'autoreg' | 'exog' | None
    force_inclusion=None,    # list[str] | str (regex) | None
    subsample=0.5,           # int | float
    random_state=123,        # int
    verbose=True             # bool
) -> tuple[list[int] | dict[str, int], list[str], list[str]]
```

