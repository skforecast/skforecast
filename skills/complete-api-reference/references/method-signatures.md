# Method Signatures Reference

Complete constructor and method signatures for all skforecast public API.

## Forecaster Constructors

### ForecasterRecursive

```python
ForecasterRecursive(
    estimator=None,           # sklearn-compatible regressor
    lags=None,                # int | list[int] | np.ndarray | range | None
    window_features=None,     # RollingFeatures | list[RollingFeatures] | None
    transformer_y=None,       # sklearn transformer for target variable
    transformer_exog=None,    # sklearn transformer | ColumnTransformer for exog
    weight_func=None,         # Callable to weight training samples by index position
    differentiation=None,     # int, differencing order applied before training
    fit_kwargs=None,          # dict, extra kwargs passed to estimator.fit()
    binner_kwargs=None,       # dict, kwargs for KBinsDiscretizer (binned residuals)
    forecaster_id=None,       # str | int, optional identifier
)
```

### ForecasterRecursiveMultiSeries

```python
ForecasterRecursiveMultiSeries(
    estimator=None,                # sklearn-compatible regressor
    lags=None,                     # int | list[int] | np.ndarray | range | None
    window_features=None,          # RollingFeatures | list[RollingFeatures] | None
    encoding='ordinal',            # 'ordinal' | 'ordinal_category' | 'onehot' | None
    transformer_series=None,       # sklearn transformer | dict[str, transformer] | None
    transformer_exog=None,         # sklearn transformer | ColumnTransformer | None
    weight_func=None,              # Callable | dict[str, Callable] | None
    series_weights=None,           # dict[str, float] | None, relative weight of each series
    differentiation=None,          # int | dict[str, int | None] | None
    dropna_from_series=False,      # bool, allow NaN in individual series
    fit_kwargs=None,               # dict, extra kwargs passed to estimator.fit()
    binner_kwargs=None,            # dict, kwargs for KBinsDiscretizer (binned residuals)
    forecaster_id=None,            # str | int, optional identifier
)
```

### ForecasterDirect

```python
ForecasterDirect(
    steps,                    # int (required), number of steps to forecast
    estimator=None,           # sklearn-compatible regressor
    lags=None,                # int | list[int] | np.ndarray | range | None
    window_features=None,     # RollingFeatures | list[RollingFeatures] | None
    transformer_y=None,       # sklearn transformer for target variable
    transformer_exog=None,    # sklearn transformer | ColumnTransformer for exog
    weight_func=None,         # Callable to weight training samples by index position
    differentiation=None,     # int, differencing order applied before training
    fit_kwargs=None,          # dict, extra kwargs passed to estimator.fit()
    binner_kwargs=None,       # dict, kwargs for KBinsDiscretizer (binned residuals)
    n_jobs='auto',            # int | str, parallel jobs for training one model per step
    forecaster_id=None,       # str | int, optional identifier
)
```

### ForecasterDirectMultiVariate

```python
ForecasterDirectMultiVariate(
    level,                         # str (required), name of the target series to predict
    steps,                         # int (required), number of steps to forecast
    estimator=None,                # sklearn-compatible regressor
    lags=None,                     # int | list | np.ndarray | range | dict[str, int|list] | None
    window_features=None,          # RollingFeatures | list[RollingFeatures] | None
    transformer_series=StandardScaler(),  # sklearn transformer | dict[str, transformer] | None
    transformer_exog=None,         # sklearn transformer | ColumnTransformer | None
    weight_func=None,              # Callable to weight training samples by index position
    differentiation=None,          # int, differencing order applied before training
    fit_kwargs=None,               # dict, extra kwargs passed to estimator.fit()
    binner_kwargs=None,            # dict, kwargs for KBinsDiscretizer (binned residuals)
    n_jobs='auto',                 # int | str, parallel jobs for training one model per step
    forecaster_id=None,            # str | int, optional identifier
)
```

### ForecasterRecursiveClassifier

```python
ForecasterRecursiveClassifier(
    estimator,                # sklearn-compatible classifier (required, not optional)
    lags=None,                # int | list[int] | np.ndarray | range | None
    window_features=None,     # RollingFeatures | list[RollingFeatures] | None
    features_encoding='auto', # str, encoding for categorical exog features
    transformer_exog=None,    # sklearn transformer | ColumnTransformer for exog
    weight_func=None,         # Callable to weight training samples by index position
    fit_kwargs=None,          # dict, extra kwargs passed to estimator.fit()
    forecaster_id=None,       # str | int, optional identifier
)
# NOTE: No transformer_y, differentiation, or binner_kwargs.
# NOTE: Uses predict_proba() instead of predict_interval().
```

### ForecasterStats

```python
ForecasterStats(
    estimator=None,           # Arima | Sarimax | Ets | Arar | list of these
    transformer_y=None,       # sklearn transformer for target variable
    transformer_exog=None,    # sklearn transformer | ColumnTransformer for exog
    forecaster_id=None,       # str | int, optional identifier
)
```

### ForecasterEquivalentDate

```python
ForecasterEquivalentDate(
    offset,                   # int | pd.tseries.offsets.DateOffset (required)
    n_offsets=1,              # int, number of past offsets to aggregate
    agg_func=np.mean,         # Callable, function to aggregate multiple offsets
    binner_kwargs=None,       # dict, kwargs for KBinsDiscretizer (binned residuals)
    forecaster_id=None,       # str | int, optional identifier
)
```

### ForecasterRnn

```python
ForecasterRnn(
    levels,                            # str | list[str] (required), target series names
    lags,                              # int | list[int] | np.ndarray | range (required)
    estimator=None,                    # Keras model (use create_and_compile_model)
    transformer_series=MinMaxScaler(feature_range=(0, 1)),  # transformer | dict | None
    transformer_exog=MinMaxScaler(feature_range=(0, 1)),    # transformer | None
    fit_kwargs=None,                   # dict, extra kwargs passed to model.fit()
    forecaster_id=None,                # str | int, optional identifier
)
```

## Forecaster Methods: fit()

```python
# ForecasterRecursive, ForecasterDirect
forecaster.fit(
    y,                                # pd.Series with DatetimeIndex (required)
    exog=None,                        # pd.Series | pd.DataFrame | None
    store_last_window=True,           # bool
    store_in_sample_residuals=False,  # bool, set True before using predict_interval()
    random_state=123,                 # int, seed for residual sampling
    suppress_warnings=False           # bool
)

# ForecasterRecursiveMultiSeries
forecaster.fit(
    series,                           # pd.DataFrame | dict[str, pd.Series|pd.DataFrame] (required)
    exog=None,                        # pd.Series | pd.DataFrame | dict[str, pd.Series|pd.DataFrame] | None
    store_last_window=True,           # bool | list[str], True stores all, list stores specific series
    store_in_sample_residuals=False,  # bool, set True before using predict_interval()
    random_state=123,                 # int
    suppress_warnings=False           # bool
)

# ForecasterDirectMultiVariate, ForecasterRnn
forecaster.fit(
    series,                           # pd.DataFrame with multiple columns (required)
    exog=None,                        # pd.Series | pd.DataFrame | None
    store_last_window=True,           # bool
    store_in_sample_residuals=False,  # bool
    random_state=123,                 # int
    suppress_warnings=False           # bool
)

# ForecasterRecursiveClassifier
forecaster.fit(
    y,                                # pd.Series with DatetimeIndex (required)
    exog=None,                        # pd.Series | pd.DataFrame | None
    store_last_window=True,           # bool
    suppress_warnings=False           # bool
)
# NOTE: No store_in_sample_residuals or random_state.

# ForecasterStats
forecaster.fit(
    y,                                # pd.Series with DatetimeIndex (required)
    exog=None,                        # pd.Series | pd.DataFrame | None
    store_last_window=True,           # bool
    suppress_warnings=False           # bool
)

# ForecasterEquivalentDate
forecaster.fit(
    y,                                # pd.Series with DatetimeIndex (required)
    store_in_sample_residuals=False,  # bool
    random_state=123,                 # int
    suppress_warnings=False           # bool
)
# NOTE: No exog parameter (uses date offsets, not exogenous variables).
```

## Forecaster Methods: predict()

```python
# ForecasterRecursive
forecaster.predict(
    steps,                    # int | str | pd.Timestamp (required)
    last_window=None,         # pd.Series | pd.DataFrame | None
    exog=None,                # pd.Series | pd.DataFrame | None
    check_inputs=True,        # bool
    suppress_warnings=False   # bool
) -> pd.Series

# ForecasterRecursiveMultiSeries
forecaster.predict(
    steps,                    # int (required)
    levels=None,              # str | list[str] | None, which series to predict
    last_window=None,         # pd.DataFrame | None
    exog=None,                # pd.Series | pd.DataFrame | dict | None
    suppress_warnings=False,  # bool
    check_inputs=True         # bool
) -> pd.DataFrame

# ForecasterDirect
forecaster.predict(
    steps=None,               # int | list[int] | None, subset of trained steps
    last_window=None,         # pd.Series | pd.DataFrame | None
    exog=None,                # pd.Series | pd.DataFrame | None
    check_inputs=True,        # bool
    suppress_warnings=False   # bool
) -> pd.Series

# ForecasterDirectMultiVariate
forecaster.predict(
    steps=None,               # int | list[int] | None, subset of trained steps
    last_window=None,         # pd.DataFrame | None
    exog=None,                # pd.Series | pd.DataFrame | None
    suppress_warnings=False,  # bool
    check_inputs=True         # bool
) -> pd.DataFrame

# ForecasterRecursiveClassifier
forecaster.predict(
    steps,                    # int | str | pd.Timestamp (required)
    last_window=None,         # pd.Series | pd.DataFrame | None
    exog=None                 # pd.Series | pd.DataFrame | None
) -> pd.Series
# Also: predict_proba(steps, last_window=None, exog=None) -> pd.DataFrame

# ForecasterStats
forecaster.predict(
    steps,                    # int (required)
    last_window=None,         # pd.Series | None
    last_window_exog=None,    # pd.Series | pd.DataFrame | None, exog for last_window period
    exog=None,                # pd.Series | pd.DataFrame | None, exog for forecast period
    suppress_warnings=False   # bool
) -> pd.Series | pd.DataFrame

# ForecasterEquivalentDate
forecaster.predict(
    steps,                    # int (required)
    last_window=None,         # pd.Series | None
    check_inputs=True,        # bool
    suppress_warnings=False   # bool
) -> pd.Series

# ForecasterRnn
forecaster.predict(
    steps=None,               # int | list[int] | None, subset of trained steps
    levels=None,              # str | list[str] | None, which series to predict
    last_window=None,         # pd.DataFrame | None
    exog=None,                # pd.Series | pd.DataFrame | None
    suppress_warnings=False,  # bool
    check_inputs=True         # bool
) -> pd.DataFrame
```

## Forecaster Methods: predict_interval()

```python
# ForecasterRecursive
forecaster.predict_interval(
    steps,                              # int | str | pd.Timestamp (required)
    last_window=None,                   # pd.Series | pd.DataFrame | None
    exog=None,                          # pd.Series | pd.DataFrame | None
    method='bootstrapping',             # 'bootstrapping' | 'conformal'
    interval=[5, 95],                   # float | list[float] | tuple[float]
    n_boot=250,                         # int, number of bootstrap samples
    use_in_sample_residuals=True,       # bool
    use_binned_residuals=True,          # bool
    random_state=123,                   # int
    suppress_warnings=False             # bool
) -> pd.DataFrame

# ForecasterRecursiveMultiSeries (NOTE: default method='conformal')
forecaster.predict_interval(
    steps,                              # int (required)
    levels=None,                        # str | list[str] | None
    last_window=None,                   # pd.DataFrame | None
    exog=None,                          # pd.Series | pd.DataFrame | dict | None
    method='conformal',                 # 'bootstrapping' | 'conformal'
    interval=[5, 95],                   # float | list[float] | tuple[float]
    n_boot=250,                         # int
    use_in_sample_residuals=True,       # bool
    use_binned_residuals=True,          # bool
    random_state=123,                   # int
    suppress_warnings=False             # bool
) -> pd.DataFrame

# ForecasterDirect
forecaster.predict_interval(
    steps=None,                         # int | list[int] | None
    last_window=None,                   # pd.Series | pd.DataFrame | None
    exog=None,                          # pd.Series | pd.DataFrame | None
    method='bootstrapping',             # 'bootstrapping' | 'conformal'
    interval=[5, 95],                   # float | list[float] | tuple[float]
    n_boot=250,                         # int
    use_in_sample_residuals=True,       # bool
    use_binned_residuals=True,          # bool
    random_state=123,                   # int
    suppress_warnings=False             # bool
) -> pd.DataFrame

# ForecasterDirectMultiVariate (NOTE: default method='conformal')
forecaster.predict_interval(
    steps=None,                         # int | list[int] | None
    last_window=None,                   # pd.DataFrame | None
    exog=None,                          # pd.Series | pd.DataFrame | None
    method='conformal',                 # 'bootstrapping' | 'conformal'
    interval=[5, 95],                   # float | list[float] | tuple[float]
    n_boot=250,                         # int
    use_in_sample_residuals=True,       # bool
    use_binned_residuals=True,          # bool
    random_state=123,                   # int
    suppress_warnings=False             # bool
) -> pd.DataFrame

# ForecasterStats (NOTE: different interface — uses alpha, no method/n_boot)
forecaster.predict_interval(
    steps,                              # int (required)
    last_window=None,                   # pd.Series | None
    last_window_exog=None,              # pd.Series | pd.DataFrame | None
    exog=None,                          # pd.Series | pd.DataFrame | None
    alpha=0.05,                         # float, significance level
    interval=None,                      # list[float] | tuple[float] | None
    suppress_warnings=False             # bool
) -> pd.DataFrame

# ForecasterEquivalentDate (NOTE: only 'conformal' method supported)
forecaster.predict_interval(
    steps,                              # int (required)
    last_window=None,                   # pd.Series | None
    method='conformal',                 # only 'conformal' supported
    interval=[5, 95],                   # float | list[float] | tuple[float]
    use_in_sample_residuals=True,       # bool
    use_binned_residuals=True,          # bool
    suppress_warnings=False             # bool
) -> pd.DataFrame

# ForecasterRnn (NOTE: only 'conformal' method supported)
forecaster.predict_interval(
    steps=None,                         # int | list[int] | None
    levels=None,                        # str | list[str] | None
    last_window=None,                   # pd.DataFrame | None
    exog=None,                          # pd.Series | pd.DataFrame | None
    method='conformal',                 # only 'conformal' supported
    interval=[5, 95],                   # float | list[float] | tuple[float]
    use_in_sample_residuals=True,       # bool
    suppress_warnings=False             # bool
) -> pd.DataFrame

# ForecasterRecursiveClassifier: No predict_interval(). Use predict_proba() instead.
```

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

## Drift Detection Classes

```python
# RangeDriftDetector — lightweight out-of-range detector
RangeDriftDetector()  # No constructor parameters

RangeDriftDetector.fit(
    series=None,             # pd.DataFrame | pd.Series | dict | None
    exog=None,               # pd.DataFrame | pd.Series | dict | None
)

RangeDriftDetector.predict(
    last_window=None,        # pd.Series | pd.DataFrame | dict | None
    exog=None,               # pd.Series | pd.DataFrame | dict | None
    verbose=True,            # bool
    suppress_warnings=False  # bool
) -> tuple[bool, list[str], list[str] | dict[str, list[str]]]

# PopulationDriftDetector — statistical tests for distribution drift
PopulationDriftDetector(
    chunk_size=None,                    # int | str | None
    threshold=3,                        # int | float
    threshold_method='std',             # 'std' | 'quantile'
    max_out_of_range_proportion=0.1     # float
)

PopulationDriftDetector.fit(X)          # Reference dataset
PopulationDriftDetector.predict(X) -> tuple[pd.DataFrame, pd.DataFrame]
```

## Preprocessing Classes

```python
RollingFeatures(
    stats,                   # str | list[str], e.g. ['mean', 'std', 'min', 'max']
    window_sizes,            # int | list[int], int applies to all stats
    min_periods=None,        # int | list[int] | None
    features_names=None,     # list[str] | None, custom names for features
    fillna=None,             # str | float | None
    kwargs_stats={'ewm': {'alpha': 0.3}}  # dict | None, kwargs for specific stats
)

TimeSeriesDifferentiator(
    order=1,                 # int, differencing order
    window_size=None         # int | None
)

DateTimeFeatureTransformer(
    features=None,           # list[str] | None, e.g. ['year', 'month', 'day_of_week', 'hour']
    encoding='cyclical',     # 'cyclical' | 'onehot' | None
    max_values=None          # dict[str, int] | None, max values for cyclical encoding
)
```
