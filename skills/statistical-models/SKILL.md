---
name: statistical-models
description: >
  Forecasts time series using classical statistical models (ARIMA, SARIMAX, ETS,
  ARAR) wrapped in ForecasterStats. Covers model selection, Auto-ARIMA,
  backtesting statistical models, and parameter tuning.
  Use when the user wants traditional statistical forecasting methods.
---

# Statistical Models (ARIMA, ETS, SARIMAX, ARAR)

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

# 3. Prediction intervals (built-in, no bootstrapping needed)
predictions_interval = forecaster.predict_interval(
    steps=12,
    interval=[10, 90],
)
```

## Auto-ARIMA (Automatic Order Selection)

```python
# Set order=None to enable automatic order selection
auto_arima = Arima(order=None, seasonal=True, m=12)
forecaster = ForecasterStats(estimator=auto_arima)
forecaster.fit(y=data['target'])

# Check selected order
print(forecaster.estimator.order)
print(forecaster.estimator.seasonal_order)

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
    freeze_params=True,  # If True, only first fold fits; faster but less accurate
)
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
