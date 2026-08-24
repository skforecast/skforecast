# Forecaster Constructors

Constructor signatures for every skforecast forecaster.

## Contents

- ForecasterRecursive, ForecasterRecursiveMultiSeries
- ForecasterDirect, ForecasterDirectMultiVariate
- ForecasterRecursiveClassifier
- ForecasterStats, ForecasterEquivalentDate
- ForecasterRnn, ForecasterFoundation

## Forecaster Constructors

### ForecasterRecursive

```python
ForecasterRecursive(
    estimator=None,               # sklearn-compatible regressor
    lags=None,                    # int | list[int] | np.ndarray | range | None
    window_features=None,         # RollingFeatures | list[RollingFeatures] | None
    transformer_y=None,           # sklearn transformer for target variable
    transformer_exog=None,        # sklearn transformer | ColumnTransformer for exog
    categorical_features='auto',  # 'auto' | list[str] | None, categorical exog handling
    weight_func=None,             # Callable to weight training samples by index position
    differentiation=None,         # int, differencing order applied before training
    dropna_from_series=False,     # bool, drop NaN rows from training matrices
    fit_kwargs=None,              # dict, extra kwargs passed to estimator.fit()
    binner_kwargs=None,           # dict, kwargs for KBinsDiscretizer (binned residuals)
    forecaster_id=None,           # str | int, optional identifier
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
    categorical_features='auto',   # 'auto' | list[str] | None, categorical exog handling
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
    steps,                        # int (required), number of steps to forecast
    estimator=None,               # sklearn-compatible regressor
    lags=None,                    # int | list[int] | np.ndarray | range | None
    window_features=None,         # RollingFeatures | list[RollingFeatures] | None
    transformer_y=None,           # sklearn transformer for target variable
    transformer_exog=None,        # sklearn transformer | ColumnTransformer for exog
    categorical_features='auto',  # 'auto' | list[str] | None, categorical exog handling
    weight_func=None,             # Callable to weight training samples by index position
    differentiation=None,         # int, differencing order applied before training
    dropna_from_series=False,     # bool, drop NaN rows from training matrices
    fit_kwargs=None,              # dict, extra kwargs passed to estimator.fit()
    binner_kwargs=None,           # dict, kwargs for KBinsDiscretizer (binned residuals)
    n_jobs='auto',                # int | str, parallel jobs for training one model per step
    forecaster_id=None,           # str | int, optional identifier
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
    categorical_features='auto',   # 'auto' | list[str] | None, categorical exog handling
    weight_func=None,              # Callable to weight training samples by index position
    differentiation=None,          # int, differencing order applied before training
    dropna_from_series=False,      # bool, drop NaN rows from training matrices
    fit_kwargs=None,               # dict, extra kwargs passed to estimator.fit()
    binner_kwargs=None,            # dict, kwargs for KBinsDiscretizer (binned residuals)
    n_jobs='auto',                 # int | str, parallel jobs for training one model per step
    forecaster_id=None,            # str | int, optional identifier
)
```

### ForecasterRecursiveClassifier

```python
ForecasterRecursiveClassifier(
    estimator,                    # sklearn-compatible classifier (required, not optional)
    lags=None,                    # int | list[int] | np.ndarray | range | None
    window_features=None,         # RollingFeatures | list[RollingFeatures] | None
    features_encoding='auto',     # str, encoding for categorical exog features
    transformer_exog=None,        # sklearn transformer | ColumnTransformer for exog
    categorical_features='auto',  # 'auto' | list[str] | None, categorical exog handling
    weight_func=None,             # Callable to weight training samples by index position
    dropna_from_series=False,     # bool, drop NaN rows from training matrices
    fit_kwargs=None,              # dict, extra kwargs passed to estimator.fit()
    forecaster_id=None,           # str | int, optional identifier
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
    estimator=None,                    # Keras model (use create_and_compile_model)
    levels,                            # str | list[str] (required), target series names
    lags,                              # int | list[int] | np.ndarray | range (required)
    transformer_series=MinMaxScaler(feature_range=(0, 1)),  # transformer | dict | None
    transformer_exog=MinMaxScaler(feature_range=(0, 1)),    # transformer | None
    fit_kwargs=None,                   # dict, extra kwargs passed to model.fit()
    binner_kwargs=None,                # dict, kwargs for KBinsDiscretizer (binned residuals)
    forecaster_id=None,                # str | int, optional identifier
)
```

### ForecasterFoundation

```python
FoundationModel(
    model_id,                  # str (required), e.g. 'autogluon/chronos-2-small'
    **kwargs,                  # Forwarded to the resolved adapter. Common keys:
                               #   context_length : int
                               #   device_map / device : 'auto' | 'cuda' | 'mps' | 'cpu'
                               #   torch_dtype : object (Chronos-2, T0)
                               #   cross_learning : bool (Chronos-2 only)
                               #   max_horizon, forecast_config_kwargs (TimesFM 2.5)
                               #   point_estimate, tabicl_config, temporal_features (TabICL)
                               #   mode, point_estimate, tabpfn_model_config, temporal_features (TabPFN-TS)
                               #   (T0 uses only context_length, device_map, torch_dtype)
                               #   point_estimate, add_calendar_features, n_fourier_terms, nori_config (Nori)
)

ForecasterFoundation(
    estimator,                 # FoundationModel (required)
    forecaster_id=None,        # str | int, optional identifier
)
```

