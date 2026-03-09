---
name: statistical-models
description: >
  Forecasts time series using classical statistical models (ARIMA, SARIMAX, ETS,
  ARAR) wrapped in ForecasterStats. Covers model selection, Auto-ARIMA,
  backtesting statistical models, and parameter tuning.
  Use when the user wants traditional statistical forecasting methods.
---

# Statistical Models (ARIMA, ETS, SARIMAX, ARAR)

## References

See [references/model-parameters.md](references/model-parameters.md) for
complete constructor signatures of all statistical models (Arima, Sarimax,
Ets, Arar), the Ets model string format, Auto-ARIMA parameters, seasonal_order
differences between Arima and Sarimax, and grid search param_grid examples.

## When to Use

Use statistical models when:
- The series is short (< 200 observations)
- Interpretability is important (ARIMA coefficients, ETS components)
- You need built-in prediction intervals without residual bootstrapping
- As a baseline to compare against ML models

## Available Models

| Model | Class | Description |
|-------|-------|-------------|
| **ARIMA** | `Arima` | AutoRegressive Integrated Moving Average |
| **Auto-ARIMA** | `Arima(order=None)` | Automatic order selection |
| **SARIMAX** | `Sarimax` | ARIMA with exogenous variables (seasonal) |
| **ETS** | `Ets` | Exponential Smoothing (Error-Trend-Seasonal) |
| **ARAR** | `Arar` | Autoregressive model with memory shortening |

## Complete Workflow: ARIMA

```python
import pandas as pd
from skforecast.recursive import ForecasterStats
from skforecast.stats import Arima
from skforecast.model_selection import backtesting_stats, TimeSeriesFold

# 1. Load data
data = pd.read_csv('data.csv', index_col='date', parse_dates=True)
data = data.asfreq('MS')  # Monthly Start frequency

# 2. Manual ARIMA: specify order and seasonal_order
arima_model = Arima(
    order=(1, 1, 1),              # (p, d, q)
    seasonal_order=(1, 1, 1),     # (P, D, Q)
    m=12,                         # Seasonal period
)
forecaster = ForecasterStats(estimator=arima_model)
forecaster.fit(y=data['target'])
predictions = forecaster.predict(steps=12)

# 3. Prediction intervals (all stat models support this natively,
#    no bootstrapping needed). Accepts both `interval` and `alpha`.
predictions_interval = forecaster.predict_interval(
    steps=12,
    interval=[10, 90],  # or use alpha=0.2 for 80% interval
)
```

## Auto-ARIMA (Automatic Order Selection)

```python
# Set order=None to enable automatic order selection
auto_arima = Arima(order=None, seasonal=True, m=12)
forecaster = ForecasterStats(estimator=auto_arima)
forecaster.fit(y=data['target'])

# Check selected order
print(forecaster.estimator.best_params_['order'])
print(forecaster.estimator.best_params_['seasonal_order'])

predictions = forecaster.predict(steps=12)
```

## ETS (Exponential Smoothing)

```python
from skforecast.stats import Ets

# Model string: 1st=Error, 2nd=Trend, 3rd=Seasonal
# A=Additive, M=Multiplicative, N=None, Z=Auto-select
ets_model = Ets(model='AAA', m=12)
forecaster = ForecasterStats(estimator=ets_model)
forecaster.fit(y=data['target'])
predictions = forecaster.predict(steps=12)
```

## Auto-ETS (Automatic Model Selection)

```python
# Use model='ZZZ' (or model=None) to let ETS automatically select
# the best Error, Trend, and Seasonal components
auto_ets = Ets(model='ZZZ', m=12)
forecaster = ForecasterStats(estimator=auto_ets)
forecaster.fit(y=data['target'])

# Check the selected model configuration
print(forecaster.estimator.best_params_)

predictions = forecaster.predict(steps=12)
```

## SARIMAX (with Exogenous Variables)

```python
from skforecast.stats import Sarimax

sarimax_model = Sarimax(
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),  # (P, D, Q, seasonal_period)
)
forecaster = ForecasterStats(estimator=sarimax_model)
forecaster.fit(y=data['target'], exog=exog_train)

# For prediction, exog must cover the forecast horizon
predictions = forecaster.predict(steps=12, exog=exog_test)
```

## ARAR

```python
from skforecast.stats import Arar

arar_model = Arar()
forecaster = ForecasterStats(estimator=arar_model)
forecaster.fit(y=data['target'])
predictions = forecaster.predict(steps=12)
```

## Backtesting Statistical Models

```python
cv = TimeSeriesFold(
    steps=12,
    initial_train_size=len(data) - 60,
    refit=False,
)

metric, predictions_bt = backtesting_stats(
    forecaster=forecaster,
    y=data['target'],
    cv=cv,
    metric='mean_absolute_error',
    freeze_params=True,  # Params from first fit reused in refits (avoids re-running auto selection)
)
# If freeze_params=False, auto selection runs independently each fold and output
# includes an extra 'estimator_params' column with the parameters selected per fold.
```

## Multiple Models Simultaneously

```python
# ForecasterStats accepts a list of models — fits each independently
from skforecast.stats import Arima, Ets

models = [
    Arima(order=(1, 1, 1), seasonal_order=(1, 1, 1), m=12),
    Ets(model='AAA', m=12),
]
forecaster = ForecasterStats(estimator=models)
forecaster.fit(y=data['target'])

# predict returns DataFrame with one column per model
predictions = forecaster.predict(steps=12)
```

## Common Mistakes

1. **Using deprecated `Ets(error=, trend=, seasonal=)` syntax**: Use `Ets(model='AAA', m=12)` with a model string instead.
2. **Forgetting `m` parameter**: ARIMA and ETS seasonal models require `m` (seasonal period).
3. **Not using `backtesting_stats`**: Use `backtesting_stats()` for statistical models, NOT `backtesting_forecaster()`.
4. **Using grid_search_forecaster for stats**: Use `grid_search_stats()` or `random_search_stats()` instead.
5. **Passing `seasonal_order=(1,1,1,12)` to `Arima`**: `Arima` uses a 3-tuple `seasonal_order=(P,D,Q)` plus a separate `m=12` parameter. The 4-tuple `(P,D,Q,s)` format is only for `Sarimax`.
