# Statistical Models — Parameter Reference

## Constructor Comparison

| Parameter | `Arima` | `Sarimax` | `Ets` | `Arar` |
|-----------|:-------:|:---------:|:-----:|:------:|
| `order` | ✓ `(1,0,0)` | ✓ `(1,0,0)` | — | — |
| `seasonal_order` | ✓ `(0,0,0)` 3-tuple | ✓ `(0,0,0,0)` **4-tuple** | — | — |
| `m` | ✓ `1` | — (in seasonal_order) | ✓ `1` | — |
| `model` | — | — | ✓ `'ZZZ'` | — |
| Auto-selection | `order=None` | — | `model='ZZZ'` or `None` | — |

> **Critical difference:** `Arima` uses 3-tuple `seasonal_order=(P,D,Q)` + separate `m`.
> `Sarimax` uses 4-tuple `seasonal_order=(P,D,Q,s)` with season period embedded.

## Arima

```python
Arima(
    # --- Manual ARIMA (set order explicitly) ---
    order=(1, 0, 0),                  # tuple(p, d, q) | None → auto ARIMA
    seasonal_order=(0, 0, 0),         # tuple(P, D, Q) — 3 elements, NOT 4
    m=1,                              # int, seasonal period (1 = no seasonality)
    include_mean=True,                # bool
    transform_pars=True,              # bool
    method='CSS-ML',                  # str, fitting method
    n_cond=None,                      # int | None
    SSinit='Gardner1980',             # str
    optim_method='BFGS',              # str
    optim_kwargs=None,                # dict | None
    kappa=1e6,                        # float

    # --- Auto ARIMA (only used when order=None) ---
    max_p=5,                          # int
    max_q=5,                          # int
    max_P=2,                          # int
    max_Q=2,                          # int
    max_order=5,                      # int
    max_d=2,                          # int
    max_D=1,                          # int
    start_p=2,                        # int
    start_q=2,                        # int
    start_P=1,                        # int
    start_Q=1,                        # int
    stationary=False,                 # bool
    seasonal=True,                    # bool
    ic='aicc',                        # 'aic' | 'aicc' | 'bic'
    stepwise=True,                    # bool
    nmodels=94,                       # int
    trace=False,                      # bool
    approximation=None,               # bool | None
    truncate=None,                    # int | None
    test='kpss',                      # str
    test_kwargs=None,                 # dict | None
    seasonal_test='seas',             # str
    seasonal_test_kwargs=None,        # dict | None
    allowdrift=True,                  # bool
    allowmean=True,                   # bool

    # --- Box-Cox transformation ---
    lambda_bc=None,                   # float | str | None
    biasadj=False,                    # bool
)
```

### Arima usage modes

| Mode | How to activate | Description |
|------|----------------|-------------|
| **Manual** | `order=(p,d,q)` | User specifies exact order |
| **Auto** | `order=None` | Automatic order selection via stepwise algorithm |

After fitting (auto mode), selected order available in:
- `forecaster.estimator.best_params_['order']`
- `forecaster.estimator.best_params_['seasonal_order']`

## Sarimax

```python
Sarimax(
    order=(1, 0, 0),                  # tuple(p, d, q)
    seasonal_order=(0, 0, 0, 0),      # tuple(P, D, Q, s) — 4 elements
    trend=None,                       # str | None
    measurement_error=False,          # bool
    time_varying_regression=False,    # bool
    mle_regression=True,              # bool
    simple_differencing=False,        # bool
    enforce_stationarity=True,        # bool
    enforce_invertibility=True,       # bool
    hamilton_representation=False,    # bool
    concentrate_scale=False,          # bool
    trend_offset=1,                   # int
    use_exact_diffuse=False,          # bool
    dates=None,
    freq=None,
    missing='none',                   # str
    validate_specification=True,      # bool
    method='lbfgs',                   # str, fitting method
    maxiter=50,                       # int
    start_params=None,                # np.ndarray | None
    disp=False,                       # bool
    sm_init_kwargs={},                # dict, extra kwargs for statsmodels init
    sm_fit_kwargs={},                 # dict, extra kwargs for statsmodels fit
    sm_predict_kwargs={},             # dict, extra kwargs for statsmodels predict
)
```

> **Note:** `Sarimax` is the only model that supports exogenous variables natively.

## Ets

```python
Ets(
    m=1,                              # int, seasonal period (1 = no seasonality)
    model='ZZZ',                      # str | None, model specification string
    damped=None,                      # bool | None
    alpha=None,                       # float | None, smoothing level
    beta=None,                        # float | None, smoothing trend
    gamma=None,                       # float | None, smoothing seasonal
    phi=None,                         # float | None, damping parameter
    lambda_param=None,                # float | None, Box-Cox lambda
    lambda_auto=False,                # bool
    bias_adjust=True,                 # bool
    bounds='both',                    # str
    seasonal=True,                    # bool
    trend=None,                       # bool | None
    ic='aicc',                        # 'aic' | 'aicc' | 'bic'
    allow_multiplicative=True,        # bool
    allow_multiplicative_trend=False, # bool
)
```

### Ets model string format

The `model` parameter is a 3-character string: `Error`, `Trend`, `Seasonal`.

| Character | Meaning | Position |
|-----------|---------|----------|
| `A` | Additive | Any |
| `M` | Multiplicative | Any |
| `N` | None (not present) | Trend or Seasonal |
| `Z` | Auto-select | Any |

| Example | Error | Trend | Seasonal | Description |
|---------|-------|-------|----------|-------------|
| `'AAN'` | Additive | Additive | None | Simple exponential smoothing with trend |
| `'AAA'` | Additive | Additive | Additive | Holt-Winters additive |
| `'MAM'` | Multiplicative | Additive | Multiplicative | Holt-Winters multiplicative seasonal |
| `'ANA'` | Additive | None | Additive | Seasonal without trend |
| `'ZZZ'` | Auto | Auto | Auto | Auto-ETS (selects best) |

After fitting (auto mode), selected configuration available in:
- `forecaster.estimator.best_params_`

## Arar

```python
Arar(
    max_ar_depth=None,                # int | None
    max_lag=None,                     # int | None
    safe=True,                        # bool
)
```

> `Arar` is the simplest model — no configuration of order or seasonality needed.
> The algorithm automatically determines the best AR structure via memory shortening.

## ForecasterStats Constructor

```python
ForecasterStats(
    estimator=None,           # Arima | Sarimax | Ets | Arar | list of these
    transformer_y=None,       # sklearn transformer for target variable
    transformer_exog=None,    # sklearn transformer | ColumnTransformer for exog
    forecaster_id=None,       # str | int, optional identifier
)
```

### Multiple models simultaneously

```python
# Pass a list to fit multiple models at once
models = [
    Arima(order=(1, 1, 1), seasonal_order=(1, 1, 1), m=12),
    Ets(model='AAA', m=12),
    Arar(),
]
forecaster = ForecasterStats(estimator=models)
forecaster.fit(y=data)

# predict() returns DataFrame with one column per model
predictions = forecaster.predict(steps=12)
```

## Grid Search param_grid Examples

```python
# Arima manual orders
param_grid = {
    'order': [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 1)],
    'seasonal_order': [(0, 0, 0), (1, 1, 1)],
    'm': [12],
}

# Ets models
param_grid = {
    'model': ['AAA', 'ANA', 'MAM', 'MNM', 'ZZZ'],
    'm': [12],
}

# Sarimax
param_grid = {
    'order': [(1, 1, 1), (2, 1, 1)],
    'seasonal_order': [(0, 0, 0, 12), (1, 1, 1, 12)],
}
```

## Backtesting with freeze_params

```python
backtesting_stats(
    forecaster=forecaster,
    y=data,
    cv=cv,
    metric='mean_absolute_error',
    freeze_params=True,     # Default. Reuse params from first fit in all folds.
    # freeze_params=False,  # Re-run auto selection each fold (slow but more realistic).
    #                       # Output includes extra 'estimator_params' column.
)
```

## Exogenous Variables Support

| Model | Exog in fit/predict | Notes |
|-------|:--:|------|
| `Arima` | — | No exog support |
| `Sarimax` | ✓ | Full exog support via statsmodels SARIMAX |
| `Ets` | — | No exog support |
| `Arar` | — | No exog support |

For `Sarimax` with exog, pass `exog` to both `fit()` and `predict()`:

```python
forecaster = ForecasterStats(estimator=Sarimax(order=(1,1,1), seasonal_order=(1,1,1,12)))
forecaster.fit(y=y_train, exog=exog_train)
predictions = forecaster.predict(steps=12, exog=exog_test)
```
